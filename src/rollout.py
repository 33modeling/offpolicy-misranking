"""rollout 수집과 drift 체크포인트 생성 (LoRA RFT).

행동 정책 β = base instruct 모델.
현재 정책 π = β의 정답 rollout으로 LoRA SFT(RFT)를 n step 돌린 모델.
drift 수준은 step 수(50/100/200)로 제어한다 — concept 10절의 drift 축.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from data import build_user_msg, reward

# torch 2.7 + Hopper: cuDNN SDPA가 산발적 'unspecified launch failure'를 낸다
# (비동기라 보고 지점은 attention이 아닐 수도 있음 — modeling_qwen2.py:47 사례).
# load_model 안에만 두면 drift 학습(rollout.py 직접 로드)·downstream이 빠지므로
# import 시점 전역 비활성 — flash/efficient/math 커널로 폴백된다.
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except AttributeError:
    pass

# 샘플링 분포 — 감사 blocker A: top-p 절단 분포에서 표본을 뽑고 raw-softmax로
# ratio를 계산하면 정의한 IS estimand가 아니다. 기본을 full softmax(top_p=1.0)로
# 통일한다. 예전 조건 재현 시에만 OM_TOP_P=0.95 지정.
SAMPLING = {"top_p": float(os.environ.get("OM_TOP_P", "1.0"))}


def _eta(done: int, total: int, t_start: float) -> str:
    import time as _t
    if done == 0:
        return "?"
    rem = (total - done) * (_t.time() - t_start) / done
    h, m = int(rem // 3600), int(rem % 3600 // 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _attn_kwargs() -> dict:
    """OM_ATTN=eager|sdpa|flash_attention_2 — CUDA 커널 문제 시 코드 수정 없이 우회."""
    import os
    attn = os.environ.get("OM_ATTN")
    return {"attn_implementation": attn} if attn else {}


def load_model(name_or_path: str, device: str | None = None, dtype: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or auto_device()
    if dtype is None:
        dtype = "bfloat16" if device == "cuda" else "float32"
    tok = AutoTokenizer.from_pretrained(name_or_path)
    # device_map 대신 CPU 로드 → .to(cuda) 2단계: 신아키텍처(Qwen3.8 등)가
    # meta-init을 못 타면 device_map 경로가 GPU에 스켈레톤+체크포인트 이중
    # 상주(27B에서 ~52+49GB)로 OOM — CPU 경유는 GPU에 정확히 한 벌만 올린다.
    # (GPU 선택은 CUDA_VISIBLE_DEVICES가 담당하므로 기능 동일)
    want = getattr(torch, dtype)
    kw = dict(low_cpu_mem_usage=True, **_attn_kwargs())

    def _load(cls):
        try:
            return cls.from_pretrained(name_or_path, dtype=want, **kw)
        except TypeError:  # 구버전 transformers: dtype 인자 미지원 → 옛 이름
            return cls.from_pretrained(name_or_path, torch_dtype=want, **kw)

    try:
        model = _load(AutoModelForCausalLM)
    except (ValueError, KeyError):
        # Qwen3.6/3.8 계열은 멀티모달 automap이라 CausalLM 매핑에 없다 —
        # MM 클래스로 폴백 (텍스트 전용 사용, vision 경로는 안 탄다)
        from transformers import AutoModelForMultimodalLM
        model = _load(AutoModelForMultimodalLM)
    # dtype 인자가 무시된 버전 방어 — fp32(27B=111GB)로 GPU에 올리면 .to에서 즉사
    if next(model.parameters()).dtype != want:
        print(f"[load_model] dtype {next(model.parameters()).dtype} → {want} 강제 캐스트", flush=True)
        model = model.to(want)
    if device == "cuda":
        model.to("cuda")
    model.eval()
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"model loaded: {name_or_path} → {device} ({gpu}, {dtype})", flush=True)
    return model, tok


def _lora_targets() -> list | str:
    """OM_LORA_TARGETS: 콤마 목록 또는 'all-linear' — DeltaNet류 신아키텍처는
    q/v_proj가 일부 층에만 있어 all-linear가 안전하다 (27B G블록용)."""
    v = os.environ.get("OM_LORA_TARGETS", "")
    if not v:
        return ["q_proj", "v_proj"]
    return v if v == "all-linear" else [s.strip() for s in v.split(",") if s.strip()]


def _gen_batch_size(total: int) -> int:
    """OM_GEN_BATCH: generate 배치 상한 (0/미설정 = 전체 한 배치 — 기존 동작 그대로).
    27B처럼 가중치가 GPU를 거의 채우는 모델은 8 정도로 제한해 KV 캐시/프리필
    활성값 OOM을 막는다 — 긴 프롬프트에서 한참 돌다 터지는 그 경로."""
    try:
        v = int(os.environ.get("OM_GEN_BATCH", "0"))
    except ValueError:
        v = 0
    return total if v <= 0 else max(1, min(total, v))


def chat_ids(tok, question: str) -> torch.Tensor:
    msgs = [{"role": "user", "content": build_user_msg(question)}]
    # transformers 4/5 양쪽에서 안전: 템플릿은 텍스트로 뽑고 별도로 토크나이즈.
    # thinking 계열(Qwen3+)은 기본 OFF — 짧은 rollout이라는 추정량 정의를 보존.
    # (해당 인자를 모르는 템플릿에서는 TypeError 폴백)
    kw = {} if os.environ.get("OM_THINKING") == "on" else {"enable_thinking": False}
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


@torch.no_grad()
def collect_rollouts(
    model,
    tok,
    prompts: list[dict],
    k: int,
    max_new_tokens: int,
    temperature: float,
    out_path: Path,
    batch_prompts: int = 8,
    idx_offset: int = 0,
) -> None:
    """프롬프트별 K개 응답 생성 → jsonl (token id·reward 저장, logp는 나중에 재계산)."""
    import time

    from grads import ts

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] rollout 시작: {len(prompts)} prompts × K={k}, "
          f"max_new={max_new_tokens}, temp={temperature} → {out_path.name}", flush=True)
    t_start = time.time()
    tmp_path = out_path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        for i, item in enumerate(prompts):
            t0 = time.time()
            ids = chat_ids(tok, item["question"]).to(model.device)
            resp_start = ids.numel()
            n_correct = 0
            bs = _gen_batch_size(k)
            j = 0
            for s in range(0, k, bs):
                nb = min(bs, k - s)
                batch_ids = ids.unsqueeze(0).expand(nb, -1)
                gen = model.generate(
                    batch_ids,
                    attention_mask=torch.ones_like(batch_ids),
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tok.eos_token_id,
                    **SAMPLING,
                )
                for seq in gen:
                    # 뒤쪽 padding(eos 반복) 제거
                    text = tok.decode(seq[resp_start:], skip_special_tokens=True)
                    r = reward(text, item["answer"])
                    n_correct += r > 0.5
                    f.write(
                        json.dumps(
                            {
                                "prompt_idx": idx_offset + i,
                                "rollout_idx": j,
                                "input_ids": seq.tolist(),
                                "resp_start": resp_start,
                                "reward": r,
                            }
                        )
                        + "\n"
                    )
                    j += 1
            print(f"[{ts()}]  rollout {i + 1}/{len(prompts)} "
                  f"({100 * (i + 1) // len(prompts)}%, {time.time() - t0:.0f}s/개, "
                  f"정답 {n_correct}/{k}, ETA {_eta(i + 1, len(prompts), t_start)})", flush=True)
    tmp_path.rename(out_path)  # 원자적 완료 표시 — 중단된 부분 파일은 .tmp로 남는다


def train_drift_lora(
    base: str,
    rollout_path: Path,
    out_dir: Path,
    steps: int,
    lr: float = 1e-4,
    batch_size: int = 4,
    device: str | None = None,
) -> None:
    """정답 rollout에 대한 LoRA SFT — checkpoint를 out_dir에 저장 (병합 없이 adapter)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or auto_device()
    # 로드는 load_model로 일원화 — dtype 보장·CPU 경유 단일 사본·MM 폴백 전부 공유
    model, tok = load_model(base, device=device)
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, target_modules=_lora_targets(), lora_dropout=0.0),
    )
    # 활성값 메모리 절감 — 7B 학습 OOM 방지
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    rows = [json.loads(l) for l in rollout_path.open()]
    correct = [r for r in rows if r["reward"] > 0.5]
    if not correct:
        # 게이트 본실행에서는 있어선 안 되는 상황 — 스모크 완주용 폴백.
        print("경고: 정답 rollout 0개 — 전체 rollout으로 drift SFT (스모크 전용 폴백)")
        correct = rows
    print(f"drift SFT: rollout {len(correct)}개, {steps} steps")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    i = 0
    for step in range(steps):
        opt.zero_grad()
        loss_acc = 0.0
        for _ in range(batch_size):
            r = correct[i % len(correct)]
            i += 1
            ids = torch.tensor(r["input_ids"][:1280], device=model.device).unsqueeze(0)
            labels = ids.clone()
            labels[0, : min(r["resp_start"], 1279)] = -100
            loss = model(ids, labels=labels).loss / batch_size
            loss.backward()
            loss_acc += float(loss)
        opt.step()
        if (step + 1) % 20 == 0:
            from grads import ts; print(f"[{ts()}]  drift step {step + 1}/{steps} loss={loss_acc:.4f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def load_policy(base: str, adapter: Path | None, device: str | None = None):
    """π 로드 — adapter가 있으면 base+LoRA 병합, 없으면 base 그대로 (=β)."""
    model, tok = load_model(base, device=device)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
        model.eval()
    return model, tok
