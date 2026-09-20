"""Small, self-contained paper data exports, never model/checkpoint archives."""

import json
import os
from pathlib import Path


def write_export(kind, data, table="", target=None):
    target = Path(target) if target else Path.home() / f"{kind}-results.txt"
    if target.suffix != ".txt":
        raise ValueError("paper export output must end in .txt")
    content = (f"{kind.upper()} PAPER DATA\n"
               "Missing values are not zero, infinite cost, or completed experiments.\n"
               "Partial measurements are explicitly labeled; do not pool them as final scores.\n\n"
               f"TABLE\n{table}\n\nDATA_JSON\n"
               + json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n")
    if len(content.encode("utf-8")) > 1900000:
        raise ValueError("paper export exceeds the 1.9 MB TXT limit; no truncated file was written")
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[{kind}-results] TXT saved: {target} ({target.stat().st_size} bytes)")
    return target
