"""`rc-floor.py <pyproject.toml> <wheel URL>`: a package built for the open RC's Runtime (`task publish -- --rc`, cl-952)
declares that Runtime as its floor and locks it by URL, the RC's pure wheel: `cozy-runtime[media]>=0.46.0.dev312,<0.47`
and `[tool.uv.sources] cozy-runtime = { url = "…/cozy_runtime-0.46.0.dev312+g1a2b3c4d-py3-none-any.whl" }`."""

import re
import sys
from pathlib import Path

path, url = Path(sys.argv[1]), sys.argv[2]
version = url.rsplit("/", 1)[-1].split("-")[1].replace("%2B", "+")
public = version.split("+")[0]
major, minor = public.split(".")[:2]
text = path.read_text()
required = re.compile(r'"cozy-runtime(\[[^\]]*\])?[^"]*"')
if not required.search(text):
    sys.exit(f"{path} requires no cozy-runtime")
text = required.sub(lambda found: f'"cozy-runtime{found.group(1) or ""}>={public},<{major}.{int(minor) + 1}"', text, count=1)
source = f'cozy-runtime = {{ url = "{url}" }}'
table = re.search(r"^\[tool\.uv\.sources\]\n((?:(?!\[).*\n?)*)", text, re.MULTILINE)
if table is None:
    text = text.rstrip("\n") + f"\n\n[tool.uv.sources]\n{source}\n"
elif re.search(r"^cozy-runtime\s*=.*$", table.group(1), re.MULTILINE):
    body = re.sub(r"^cozy-runtime\s*=.*$", lambda _: source, table.group(1), count=1, flags=re.MULTILINE)
    text = text[: table.start(1)] + body + text[table.end(1) :]
else:
    text = text[: table.start(1)] + source + "\n" + text[table.start(1) :]
path.write_text(text)
print(f"{path.parent.name}: cozy-runtime>={public},<{major}.{int(minor) + 1} from {url}")
