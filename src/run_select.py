"""수확 스크립트 공통 run 선택기.

배경: 판독기들이 각자 `root.glob("v2-*")`를 하드코딩하고 있어서 세대가 v3으로
넘어가자 대상이 0건이 되고, 출력이 조용히 빈 파일로 나왔다(0820 수확의
READOUT.md 0바이트). 세대 접두사는 `v<숫자>-`로 일반화하고, 완주 판정과
필수 산출물 검사를 한곳에서 한다.
"""

from __future__ import annotations

import re
from pathlib import Path

GEN_RE = re.compile(r"^v\d+-")          # v2-, v3-, v10- …
LEGACY_PREFIXES = ("gate-", "drift")     # v1 계열: DONE 파일이 없다


def is_generation_run(name: str) -> bool:
    return bool(GEN_RE.match(name))


def iter_runs(root: Path, *, need: tuple[str, ...] = (),
              require_done: bool = True, include_legacy: bool = False,
              skip_smoke: bool = True) -> list[Path]:
    """조건을 만족하는 run 디렉터리 목록 (이름순).

    need: 반드시 존재해야 하는 산출물 파일명들.
    require_done: 세대 run(v<N>-)에 DONE 마커를 요구할지. 레거시 run은
                  DONE을 만들지 않으므로 산출물 존재로만 판정한다.
    """
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        name = d.name
        if skip_smoke and "smoke" in name:
            continue
        gen = is_generation_run(name)
        if not gen:
            if not (include_legacy and name.startswith(LEGACY_PREFIXES)):
                continue
        if gen and require_done and not (d / "DONE").exists():
            continue
        if any(not (d / f).exists() for f in need):
            continue
        out.append(d)
    return out


def describe_skips(root: Path, chosen: list[Path]) -> list[str]:
    """선택되지 않은 디렉터리와 그 이유 — 조용한 0건을 막기 위한 진단용."""
    if not root.is_dir():
        return [f"{root}: 디렉터리 없음"]
    picked = {p.name for p in chosen}
    lines = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name in picked or "smoke" in d.name:
            continue
        why = []
        if not is_generation_run(d.name) and not d.name.startswith(LEGACY_PREFIXES):
            why.append("세대 접두사 아님")
        if is_generation_run(d.name) and not (d / "DONE").exists():
            why.append("DONE 없음")
        for f in ("scores_oracle.json", "scores_offpolicy.json",
                  "oracle_micro_groups.pt", "val_gradient.pt"):
            if not (d / f).exists():
                why.append(f"{f} 없음")
        lines.append(f"{d.name}: {', '.join(why) or '조건 미상'}")
    return lines
