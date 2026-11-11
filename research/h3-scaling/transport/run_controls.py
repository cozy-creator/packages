"""Ordinary-CLI transport runs, with an explicit pause after each warmup.

This runner never rents, resets a worker, changes software, or resubmits an
uncertain accepted request. Root owns lifecycle and the GPU lease.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
COUNTS = {'sparse':192, 'dense_step':192, 'dense_path':32, 'dense_prefix':192}


def now(): return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def read_command(argv, path):
    result = subprocess.run(argv, text=True, capture_output=True)
    path.write_text(result.stdout)
    path.with_suffix(path.suffix + '.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Command failed with exit {result.returncode}; see {path}')
    return json.loads(result.stdout)


def receipt_facts(show):
    if show.get('status') != 'completed' or show.get('degree') != 4:
        raise ValueError('Expected a completed four-GPU request')
    gpus = show['gpus']
    if len(gpus) != 4 or len({g['pid'] for g in gpus}) != 4 or len({g['uuid'] for g in gpus}) != 4:
        raise ValueError('Receipt must identify four distinct GPU processes and UUIDs')
    attention = {}
    for gpu in gpus:
        if gpu['arch'] != 'sm_90':
            raise ValueError('This controlled study is for four H100s')
        a = gpu['attention']
        if a['requested'] != '' or 'fl2va_dit=sol-attn' not in a['observed']:
            raise ValueError(f'H100 default selection differs: {a}')
        if a['sol'] != COUNTS:
            raise ValueError(f'Unexpected physical dispatch/replay counts: {a["sol"]}')
        if 'sol-attn:cute_sm90' not in a['impl']:
            raise ValueError('Actual Hopper Sol implementation was not recorded')
        attention[gpu['uuid']] = {k:a[k] for k in ('requested','observed','impl','sol')}
    denoise = next(s for s in show['steps'] if s['name'] == 'denoise')
    if denoise['count'] != 8 or len(denoise['series']) != 8:
        raise ValueError('The request did not retain all eight denoising steps')
    return {'pids': sorted(g['pid'] for g in gpus),
            'pid_by_uuid':{g['uuid']:g['pid'] for g in gpus},
            'attention_by_uuid':attention,
            'denoise_ms':next(s['ms'] for s in show['stages'] if s['name']=='denoise'),
            'denoise_step_total_ms':denoise['total_ms']}


def submit(mode, index, rental, host):
    folder = host / f'native-{mode}' / ('warmup-0' if index == 0 else f'timed-{index}')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'accepted.json').exists():
        return json.loads((folder/'accepted.json').read_text()), folder
    # A crash after this claim is an investigation, never an automatic second submit.
    fd = os.open(folder/'submitted.claim', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    source = ROOT/f'package-{mode}'
    argv = ['cozy','run',str(source/'fl2va_turbo'),
            '--input='+str(BASE/'native'/f'input-turbo-{index}.json'),
            '--rental='+rental,
            f'--idempotency-key=h100-transport-{host.name}-{mode}-{index}-v1',
            '--out='+str(folder/'outputs'),'--json']
    save(folder/'command.json', {'argv':argv,'started_utc':now(),'mode':mode,'index':index})
    result = read_command(argv, folder/'submit-result.json')
    number = result.get('number',result.get('run'))
    if number is None:
        raise RuntimeError('Accepted identity missing; retain the claim and investigate, never resubmit')
    show = read_command(['cozy','run','watch',str(number),'--once','--json','--full'],folder/'accepted-show.json')
    accepted = {'number':show['number'],'request_id':show['request_id'],'mode':mode,'index':index,'at':now()}
    save(folder/'accepted.json',accepted)
    print(json.dumps({'event':'accepted',**accepted}),flush=True)
    return accepted,folder


def observe(record, folder, warm=None):
    if not (folder/'validation.json').exists():
        with (folder/'watch.jsonl').open('w') as out, (folder/'watch.stderr').open('w') as err:
            result = subprocess.run(['cozy','run','watch',str(record['number']),'--json','--full'],stdout=out,stderr=err)
        save(folder/'watch-exit.json',{'exit_code':result.returncode,'at':now()})
    show = read_command(['cozy','run','watch',str(record['number']),'--once','--json','--full'],folder/'show-final.json')
    facts = receipt_facts(show)
    if warm is not None:
        for field in ('pid_by_uuid','attention_by_uuid'):
            if facts[field] != warm[field]:
                raise ValueError(f'Warmup-to-timed {field} changed; hold this result for review')
    video = next(item for item in show['output'] if item['type']=='video')
    path = Path(video['path'])
    with path.open('rb') as stream: digest = hashlib.file_digest(stream,'sha256').hexdigest()
    if digest != video['sha256'].removeprefix('sha256:') or path.stat().st_size != video['length']:
        raise ValueError('Video custody differs from the output receipt')
    probe = read_command(['ffprobe','-v','error','-show_entries',
         'stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,duration,sample_rate,channels:format=duration',
         '-of','json',str(path)],folder/'ffprobe.json')
    v = next(s for s in probe['streams'] if s['codec_type']=='video')
    a = next(s for s in probe['streams'] if s['codec_type']=='audio')
    if (v['width'],v['height'],v['avg_frame_rate'],v['nb_frames']) != (1344,768,'24/1','362'):
        raise ValueError('Full video geometry differs')
    if (a['sample_rate'],a['channels']) != ('32000',2): raise ValueError('Audio geometry differs')
    with (folder/'av-decode.log').open('w') as log:
        decoded = subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(path),
                    '-map','0:v:0','-map','0:a:0','-f','null','-'],stdout=log,stderr=log)
    if decoded.returncode: raise ValueError('Full AV decode failed')
    summary = {**record, **facts, 'validation':'pass','execution_ms':show['execution_ms'],
               'stages':show['stages'],'steps':show['steps'],'video_sha256':digest,'video_path':str(path),
               'validated_utc':now(),'timing_role':'warmup' if record['index']==0 else 'timed'}
    save(folder/'validation.json',summary)
    print(json.dumps({'event':'validated','number':record['number'],'mode':record['mode'],
                     'index':record['index'],'execution_ms':show['execution_ms']}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('peer','host'),required=True)
    parser.add_argument('--phase',choices=('warmup','timed'),required=True)
    parser.add_argument('--rental',required=True)
    parser.add_argument('--host-dir',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();host=args.host_dir.resolve()
    global ROOT, BASE
    BASE=args.campaign.resolve();ROOT=BASE/'native-transport-control'
    if args.phase=='warmup':
        record,folder=submit(args.mode,0,args.rental,host)
        observe(record,folder)
        print('Warmup complete. Hold for observed NCCL route validation before timed submission.',flush=True)
        return
    warm_folder=host/f'native-{args.mode}'/'warmup-0'
    warm=json.loads((warm_folder/'validation.json').read_text())
    proof=json.loads((warm_folder/'transport-proof.json').read_text())
    if not proof.get('complete') or proof.get('mode')!=args.mode or proof.get('run')!=warm['number'] or proof.get('pids')!=warm['pids']:
        raise ValueError('Complete transport proof for this exact warmup group is required')
    # Both timed requests enter the ordinary FIFO before client-side artifact transfer.
    pending=[submit(args.mode,i,args.rental,host) for i in (1,2)]
    summaries=[observe(record,folder,warm) for record,folder in pending]
    save(host/f'native-{args.mode}'/'results.json',{'mode':args.mode,'warmup':warm,'timed':summaries,
         'mean_execution_ms':sum(r['execution_ms'] for r in summaries)/2,
         'mean_denoise_ms':sum(r['denoise_ms'] for r in summaries)/2,
         'next':'Stop here. Root owns the fresh lifecycle boundary before another transport mode.'})

if __name__=='__main__': main()
