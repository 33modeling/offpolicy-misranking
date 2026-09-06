#!/usr/bin/env python3
"""Turn a failed launcher session log into one Korean diagnosis + one action.

    python3 src/diagnose_launch_failure.py SESSION_LOG

Reads only the log; no torch. Exit 0 always (diagnosis is advisory).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# (pattern, diagnosis, action). First match from the END of the log wins.
RULES: list[tuple[str, str, str]] = [
    (r"math-verify is required|No module named 'math_verify'",
     "math-verify 번들이 PYTHONPATH에 없음 — 004f5f8 이전 코드로 실행됨",
     "git pull --ff-only 후 다시 실행 (fa28ce5 이상이어야 함)"),
    (r"\[locate-abort\]",
     "MODELS_DIR 아래에서 Qwen3.5-9B 폴더를 못 찾음 (config.json model_type으로 식별)",
     "업로드 폴더가 $MODELS_DIR 바로 아래(또는 한 단계 아래)에 있는지, config.json이 있는지 확인"),
    (r"missing model\.safetensors-0000\d-of-0000\d\.safetensors \(no unique",
     "업로드된 shard 파일 크기가 공식 파일과 다름 — 업로드가 잘렸거나 다른 revision",
     "근거 줄의 shard를 HF pinned revision에서 다시 올리기"),
    (r"safetensors shard set incomplete|weight index contains no shards|neither a safetensors index",
     "모델 가중치(.safetensors)가 로컬에 없음 — 다운로드 패턴 버그(dbee3d1 이전)로 config/tokenizer만 받힘",
     "인터넷 되는 머신에서 git pull 후 `bash scripts/run_qwen35_9b.sh prepare` 재실행 (19GB)"),
    (r"file is not registered for the pinned model|model file missing: |model file (size|hash) mismatch",
     "업로드된 모델 파일이 pinned revision과 다름/누락 (main에서 받았거나 잘림) — 근거 줄의 파일명 확인",
     "그대로 쓰려면: OM_ALLOW_UNPINNED_SNAPSHOT=1 bash scripts/run_qwen35_9b.sh  (재현성 표기 'unverified')"),
    (r"missing files: .*\.om_snapshot\.json|model snapshot missing|snapshot provenance mismatch|cannot prove local Hub revision",
     "모델 스냅샷이 준비/봉인되지 않음",
     "인터넷 되는 머신에서 `bash scripts/run_qwen35_9b.sh prepare`"),
    (r"Qwen3\.5 needs transformers>=5|No module named 'transformers\.models\.qwen3_5'|cannot import name 'AutoModelForMultimodalLM'|Unrecognized configuration class.*qwen3_5",
     "venv의 transformers가 Qwen3.5 클래스를 모름 (5.x 필요)",
     "$VENV_DIR/bin/pip install -U 'transformers>=5' 후 재실행"),
    (r"expected FLA 0\.5\.2|fla-core .* is not installed|No module named 'fla'|FLA fused recurrent/chunk kernels are unavailable",
     "flash-linear-attention(fla-core 0.5.2) 미설치/버전 불일치 — GatedDeltaNet 커널 없음",
     "$VENV_DIR/bin/pip install 'flash-linear-attention[cuda]==0.5.2' 후 재실행"),
    (r"CUDA out of memory|OutOfMemoryError",
     "GPU 메모리 부족",
     "다른 프로세스 점유 확인(nvidia-smi). 반복되면 config의 runtime.generation_batch를 낮춘 새 계약 필요"),
    (r"unspecified launch failure|illegal memory access|CUBLAS_STATUS|device-side assert",
     "CUDA 런타임 오류 (커널/드라이버 수준) — 코드 버그 아님",
     "같은 명령 재실행; 같은 GPU에서 반복되면 노드 교체"),
    (r"exactly four H100 GPUs required",
     "이 노드가 4×H100이 아님",
     "4×H100 노드에서 실행"),
    (r"GPUs are already in use|four GPUs did not become idle|GPU memory did not clear",
     "다른 프로세스가 GPU를 점유 중",
     "nvidia-smi로 확인 후 정리, 또는 빈 노드에서 실행"),
    (r"additional suite already queued on this physical node",
     "이 노드에 이미 같은 런처가 떠 있음",
     "기존 런처를 쓰거나 종료 후 재실행"),
    (r"queued behind local primary",
     "이 노드는 OLMo primary 런처가 잠금(primary.lock)을 쥐고 있어 대기 중 — 에러 아님, OLMo가 끝나야 시작됨",
     "OLMo가 안 도는 노드에서 실행"),
    (r"checkout is dirty|worktree is dirty",
     "src/scripts/configs에 커밋 안 된 변경이 있음",
     "git status 확인 → commit 또는 git checkout -- . 후 재실행"),
    (r"venv missing|venv python 없음",
     "VENV_DIR에 python이 없음",
     "scripts/provision.sh 또는 VENV_DIR 지정"),
    (r"qualification does not match|content fingerprint|official row count|dataset .* (missing|not found)|로컬 사본을 못 찾았고",
     "데이터셋(MATH-500/MBPP) 스냅샷이 없거나 검증 실패",
     "인터넷 되는 머신에서 `bash scripts/run_qwen35_9b.sh prepare`"),
    (r"chat_template missing",
     "tokenizer chat_template 없음 — 스냅샷 파일 누락",
     "prepare 재실행 (chat_template.jinja 포함 다운로드)"),
    (r"LoRA targets missing",
     "config의 lora_targets가 모델 모듈명과 안 맞음",
     "configs/*.json lora_targets 확인"),
    (r"HF_HUB_OFFLINE|Cannot reach|ConnectionError|Repository Not Found|401 Client Error|Max retries exceeded",
     "Hugging Face에 접근 불가 (오프라인/인증)",
     "prepare는 인터넷 되는 머신에서만; run/check는 미리 받은 스냅샷 사용"),
    (r"prompts differ from the qualified matrix|prompts\.json differs",
     "프롬프트 분할이 계약과 다름 — 다른 데이터 사본으로 초기화된 run",
     "run 디렉터리가 자동 격리됨; 재실행"),
    (r"first policy-loss evaluation is not on-policy",
     "GRPO 첫 ratio가 1이 아님 — 수치 불안정(bf16/커널)",
     "재실행; 반복되면 attn=eager인지, FLA 버전 확인"),
    (r"invalid policy loss/gradient|zero or non-finite gradient norm",
     "GRPO step에서 NaN/0 gradient — optimizer 미적용, 체크포인트에서 재개됨",
     "같은 명령 재실행"),
]

EXC_RE = re.compile(r"^(?:\S+Error|RuntimeError|ValueError|KeyError|OSError|ImportError|TypeError|AssertionError)\b.*|.*abort\].*")


def diagnose(text: str) -> tuple[str, str, str] | None:
    lines = text.splitlines()
    for line in reversed(lines):
        for pattern, why, what in RULES:
            if re.search(pattern, line):
                return why, what, line.strip()
    return None


def last_exception(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if EXC_RE.match(stripped) and not stripped.startswith("[exit]") and not stripped.startswith("[error]"):
            return stripped[:300]
    return None


def snapshot_listing(text: str) -> list[str]:
    """For weight/snapshot failures, echo the locator's on-disk comparison lines."""
    lines = [line.strip() for line in text.splitlines() if line.startswith("[locate]")]
    return lines[-40:]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: diagnose_launch_failure.py SESSION_LOG", file=sys.stderr)
        return 0
    path = Path(sys.argv[1])
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"진단 불가: 로그를 읽을 수 없음 ({exc})")
        return 0
    found = diagnose(text)
    if found:
        why, what, line = found
        print(f"진단: {why}")
        print(f"조치: {what}")
        print(f"근거: {line[:300]}")
        if re.search(r"가중치|스냅샷|shard|모델 파일|폴더를 못 찾음", why):
            for extra in snapshot_listing(text):
                print(f"  {extra}")
        return 0
    exc = last_exception(text)
    if exc:
        print("진단: 알려진 패턴 아님 — 아래 예외가 원인")
        print(f"근거: {exc}")
    else:
        print("진단: 로그에 예외/abort 줄이 없음 — 프로세스가 신호로 죽었거나(SIGKILL/OOM-killer) 외부 종료")
        print("조치: dmesg -T | tail 또는 같은 명령 재실행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
