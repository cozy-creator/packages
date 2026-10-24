"""HiDream's Reasoning-Driven Prompt Agent: Gemma-4-31B-it with HiDream's system prompt, served from
its as-is upload (its vision tower is never built).

Inference follows upstream's `prompt_agent.py` local backend (HiDream-ai/HiDream-O1-Image, main
2c2d29f): the chat template without thinking, sampling at temperature 1.0 with Gemma's own top-k 64
and top-p 0.95, at most 4096 new tokens, and the JSON reply's `prompt`; a reply that does not parse
falls back to the request with upstream's quality suffix.
"""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from cozy_runtime.author import Config, Loader, Model, ModelAssets, uses_components
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedTokenizerFast
from transformers.initialization import no_init_weights

COMPONENT = "model"
MAX_NEW_TOKENS = 4096
#: Gemma-4-31B-it's generation config, which upstream's `generate(do_sample=True, temperature=1.0)`
#: inherits.
TEMPERATURE, TOP_K, TOP_P = 1.0, 64, 0.95
FALLBACK_SUFFIX = ", highly detailed, masterpiece, best quality, sharp focus"

#: Upstream's `REWRITE_SYSTEM_PROMPT`, verbatim, with the two U+FFFD its file carries where "合" was.
SYSTEM_PROMPT = """\
你是专业的AI图像生成Prompt工程师的Prompt Engineering Engine,也是一名拥有百科知识和视觉导演能力的创意总监.你的任务是分析用户的原始图像需求,推理出隐含知识和最佳视觉方案,并改写成一个**明确,详细,可直接用于图像生成的英文prompt**.

## 核心目标

图像生成模型只能执行直接的视觉描述,不能自行补全背景知识,逻辑关系或文字内容.因此,你必须提前完成知识解析,空间规划和视觉导演,把结果显式写入prompt中.

使用 SCALIST 框架扩写每个画面:
- **Subject**: 主体的身份,外观,颜色,材质,纹理,动作,表情,服饰.
- **Composition**: 镜头景别,视角,主体位置,前景/中景/背景层次,留白和视觉焦点.
- **Action**: 主体正在做什么,动作方向,姿态,互动关系.
- **Location**: 场景地点,室内/室外,时代,天气,时间段,环境细节.
- **Image style**: photorealistic, cinematic, oil painting, watercolor, anime, 3D render 等,并匹配合适的光线和色彩氛围.
- **Specs**: 摄影/渲染参数,如 85mm lens, low-angle shot, shallow depth of field, soft diffused light, dramatic backlighting, matte texture, sharp focus.
- **Text rendering**: 如果用户要求文字,必须把准确文字放在英文双引号中,并说明字体风格,颜色,大小,材质和精确位置.

1. **知识解析与显式化**: 凡是诗词,歌词,名言,公式,历史人物,科学概念,地标,名画,文化符号,历史事件,UI布局或现实世界对象,都要先解析出具体答案和可见特征,再写入prompt.不要只写 "Mona Lisa","Dunkirk evacuation","freedom" 这类需要模型自行理解的词.
2. **空间与逻辑锚定**: 把模糊关系改写为明确布局,例如 top left corner, centered in the foreground, slightly behind the main subject, background out of focus, text aligned along the bottom edge.不要使用"旁边""一些""好看"等含糊表达.
3. **文字排版精度**: 中文,英文,公式,多语言文本都必须逐字保留在引号中,例如 "床前明月光,疑是地上霜.举头望明月,低头思故乡." 或 "E = mc²";同时指定字体(calligraphy, serif, sans-serif, handwritten),颜色,材质和位置.
4. **真实世界落地**: 如果用户要求事实准确的内容,例如历史文物,天气现象,人物肖像,建筑,仪表盘或应用界面,要使用你的内部知识补全准确视觉细节.
5. **抽象概念具象化**: 把"自由,孤独,未来感,治愈"等抽象词转成可见场景,符号和氛围,例如飞鸟,断裂锁链,辽阔天空,冷色霓虹,柔和晨光等.

## 示例合并学习

- 用户说"李白的静夜思写在墙上",prompt 应写出完整中文诗句,并指定它以优雅中国书法写在古旧石墙的哪个位置.
- 用户说"三大力学的奠基人"或"爱因斯坦写质能方程",prompt 应解析出 Isaac Newton 或 Albert Einstein,并描述人物外貌,时代服饰,黑板,公式 "E = mc²" 等可见内容.
- 用户说"蒙娜丽莎""比萨斜塔""福字""敦刻尔克大撤退",prompt 应描述对应画面特征: 神秘微笑与交叠双手,倾斜白色大理石钟楼与拱廊,红底金色/黑色书法 "福",1940年海滩上等待撤离的士兵和海面船只.

## 输出prompt要求

- prompt 必须是一个英文的,连贯自然的单段落,像 Creative Director's Brief,而不是关键词堆砌或 tag soup.
- 长度通常为 80-220 词;简单需求可以更短,复杂画面可以更长.
- 最重要的主体和画面意图放在开头,然后自然展开构图,动作,地点,风格,技术参数和文字渲染.
- 使用完整句子,丰富但准确的形容词,摄影/绘画/设计术语.
- 不要包含任何需要图像模型继续推理才能理解的表达.
- prompt 必须自包含,仅凭prompt本身就能准确生成图片.

## 执行步骤

1. **Analyze**: 识别核心主体,用户意图,文字要求,参考限制和需要解析的隐含知识.
2. **Reason**: 选择最适��画面的光线,镜头,角度,纹理,风格,空间布局和事实细节.
3. **Rewrite**: 输出最终增强后的英文单段落prompt.

只输出JSON,不加任何其他文字:
{"prompt": "英文单段落prompt", "reasoning": "你的推理和知识解析过程(中文简述)", "resolved_knowledge": "你解析了哪些隐含知识(中文,如果没有隐含知识写'无')"}\
"""


class AgentGraph:
    def __init__(self, config: Config) -> None:
        with no_init_weights():
            model = AutoModelForImageTextToText.from_config(
                AutoConfig.for_model(**config.mapping()), dtype=torch.bfloat16
            )
        model.model.vision_tower = None
        model.model.embed_vision = None
        self.components: dict[str, torch.nn.Module] = {COMPONENT: model.eval()}


def build_tokenizer(assets: ModelAssets) -> tuple[PreTrainedTokenizerFast, str]:
    """Gemma's tokenizer and chat template from the checkpoint's files, parsed in memory."""
    config = json.loads(assets.read("tokenizer_config.json"))
    for key in ("tokenizer_file", "name_or_path", "added_tokens_decoder", "tokenizer_class", "processor_class",
                "chat_template", "backend", "response_schema", "response_template", "extra_special_tokens"):
        config.pop(key, None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer.json").decode()), **config
    )
    return tokenizer, assets.read("chat_template.jinja").decode()


def _json_block(text: str) -> str | None:
    """The first balanced `{...}` outside strings (upstream's `_extract_json_block`)."""
    depth, start, quoted, escaped = 0, None, False, False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            quoted = not quoted
        elif not quoted and ch == "{":
            start = i if depth == 0 else start
            depth += 1
        elif not quoted and ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _escape_newlines(text: str) -> str:
    """Raw newlines inside JSON strings escaped (upstream's `_fix_unescaped_newlines`)."""
    out, quoted, escaped = [], False, False
    for ch in text:
        if escaped:
            escaped = False
        elif ch == "\\" and quoted:
            escaped = True
        elif ch == '"':
            quoted = not quoted
        elif quoted and ch in "\r\n":
            out.append("\\n" if ch == "\n" else "")
            continue
        out.append(ch)
    return "".join(out)


def caption(reply: str, request: str) -> str:
    """The rewritten prompt of a reply (upstream's `parse_json` and `_wrap_result`)."""
    text = reply.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL) if "```" in text else None
    text = fenced.group(1).strip() if fenced else text
    block = _json_block(text) or text
    for candidate in (block, _escape_newlines(block)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("prompt"), str):
            return str(value["prompt"])
        break
    return request + FALLBACK_SUFFIX


class PromptAgent(Model[AgentGraph], encoded_leaves="accept", phase="decode"):
    """Rewrites a request into the dense English prompt HiDream-O1 was trained on."""

    graph: AgentGraph
    tokenizer: Any
    template: str

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(AgentGraph, factory=AgentGraph)
        self.tokenizer, self.template = build_tokenizer(loader.assets)

    @uses_components(COMPONENT, memoize=True, memo_dependencies=(caption,))
    def rewrite(self, request: str, seed: int) -> str:
        model = self.graph.components[COMPONENT]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
        text = self.tokenizer.apply_chat_template(
            messages, chat_template=self.template, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        device = next(model.parameters()).device
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
                top_p=TOP_P, pad_token_id=self.tokenizer.pad_token_id,
            )
        reply = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return caption(reply, request)
