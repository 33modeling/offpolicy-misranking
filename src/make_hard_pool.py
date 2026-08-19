"""β pass-rate로 hard-slice 풀 생성 — 전량 정답/전량 오답 제외(live 프롬프트만).

    python3 src/make_hard_pool.py <run_dir> <out.jsonl> [lo=0.0] [hi=1.0]

<run_dir>: prep + rollout-behavior까지 끝난 폴더
  (prompts.json + rollouts_behavior_train*.jsonl — 샤드/병합본 모두 지원)
lo < mean-reward < hi 인 프롬프트만 {"question","answer"} jsonl로 내보낸다.
산출 파일은 OM_POOL_FILE로 본실행에 주입한다 (27B G블록의 난이도 매칭).
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    lo = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    hi = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    prompts = json.loads((run / "prompts.json").read_text())["train"]
    shards = sorted(run.glob("rollouts_behavior_train*.jsonl"))
    if not shards:
        print(f"[abort] rollout 파일 없음: {run}/rollouts_behavior_train*.jsonl")
        return 1
    # 병합본이 있으면 그것만 (샤드와 중복 집계 방지)
    merged = run / "rollouts_behavior_train.jsonl"
    if merged in shards:
        shards = [merged]

    acc: dict[int, list[float]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for sh in shards:
        for line in sh.open():
            r = json.loads(line)
            key = (r["prompt_idx"], r.get("rollout_idx", len(acc[r["prompt_idx"]])))
            if key in seen:
                continue
            seen.add(key)
            acc[r["prompt_idx"]].append(float(r["reward"]))

    rows, hist = [], defaultdict(int)
    for i, item in enumerate(prompts):
        rs = acc.get(i)
        if not rs:
            continue
        rate = sum(rs) / len(rs)
        hist[round(rate, 1)] += 1
        if lo < rate < hi:
            rows.append(item)

    covered = len(acc)
    if covered < len(prompts):
        print(f"[warn] rollout 커버리지 {covered}/{len(prompts)} — 샤드 누락/부분 완료 의심")
        if not rows:
            # 커버리지가 불완전한 0건은 '전 문제 포화'의 증거가 아니다 — pool을
            # 기록하면 go_27b의 -f 게이트가 포화로 오진하므로 기록 없이 실패 반환
            print("[abort] hard-slice 0건 + 커버리지 불완전 — 포화 단정 불가. "
                  "프리스크린 rollout 샤드 완주 후 재실행할 것 (pool 미기록)")
            return 3

    out.parent.mkdir(parents=True, exist_ok=True)
    # 원자적 기록 — 중간 사망이 남긴 부분/0바이트 pool이 '있음'으로 채택되거나
    # 포화로 오진되는 것 방지 (완성된 파일만 최종 이름을 가진다)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print("pass-rate 분포:", dict(sorted(hist.items())))
    print(f"hard-slice: {len(rows)}/{len(prompts)} → {out}")
    if not rows:
        print("[warn] hard-slice 0건 — β pass-rate가 전 프롬프트에서 경계 밖(전량 정답/오답). "
              "POOL_N 증량으로는 해결되지 않음 — 데이터셋 난이도 재검토 필요")
    elif len(rows) < 620:
        print(f"[warn] 풀 {len(rows)} < 620 (n=512+val 100+여유) — POOL_N을 키워 재실행 권장")
    return 0


if __name__ == "__main__":
    sys.exit(main())
