#!/usr/bin/env python
from pathlib import Path
import compileall, json
ROOT=Path(__file__).resolve().parents[1]
ok=compileall.compile_dir(ROOT/'src',quiet=1) and compileall.compile_dir(ROOT/'scripts',quiet=1)
print(json.dumps({'python_syntax_ok':bool(ok)},indent=2))
raise SystemExit(0 if ok else 1)
