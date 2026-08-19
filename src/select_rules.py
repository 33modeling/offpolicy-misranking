"""top-k 크기 규칙 단일화 — P0-5 (docs/PAPER_REVIEW_2026-08-19.md).

int()/round() 혼재로 같은 n=256, frac=0.1에서 스크립트마다 k=25/26이 갈려
본문 표(precision·chance)와 floor·통계가 다른 k를 섞어 쓰고 있었다.
규칙은 k = max(1, floor(frac·n)) 하나로 고정한다 — 본문 표 계열(int)과 일치.
"""

from __future__ import annotations


def topk_count(n: int, frac: float = 0.10) -> int:
    return max(1, int(n * frac))
