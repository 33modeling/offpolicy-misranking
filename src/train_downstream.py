"""downstream 비교 — 선택된 프롬프트 부분집합으로 짧은 policy-gradient 학습 후 val 정확도.

게이트 마지막 조건: "선택된 데이터로 짧은 200-step 학습을 했을 때 같은 총연산
GradAlign보다 나쁘지 않다." 방법 간 차이는 오직 프롬프트 부분집합이다.
학습은 LoRA + LOO-baseline REINFORCE (clip 없는 GRPO-lite, 인증 estimator와 동일 관측치).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from data import reward
from grads import loo_advantages
from rollout import SAMPLING, _lora_targets, auto_device, chat_ids
from rollout_contract import eos_ids_of, gen_kwargs, resp_end_index
from select_rules import jittered_topk, topk_count


def grpo_lite_train(
    base: str,
    prompts: list[dict],
    selected_idx: list[int],
    steps: int,
    k: int,
    max_new_tokens: int,
    temperature: float,
    out_dir: Path,
    lr: float = 1e-5,
    seed: int = 0,
) -> None:
    from peft import LoraConfig, get_peft_model
    device = auto_device()
    from rollout import load_model
    # 로드는 load_model로 일원화 — dtype 보장·CPU 경유 단일 사본·MM 폴백 공유
    model, tok = load_model(base, device=device)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules=_lora_targets(), lora_dropout=0.0))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    pool = [prompts[i] for i in selected_idx]
    rng = random.Random(seed)
    from grads import ts
    import time as _time
    print(f"[{ts()}] downstream 학습 시작: {steps} steps, 선택 {len(pool)}개, K={k}", flush=True)
    t0 = _time.time()

    for step in range(steps):
        item = rng.choice(pool)
        ids = chat_ids(tok, item["question"]).to(model.device)
        with torch.no_grad():
            batch = ids.unsqueeze(0).expand(k, -1)
            gen = model.generate(
                batch, attention_mask=torch.ones_like(batch),
                **gen_kwargs(temperature, SAMPLING["top_p"],
                             max_new_tokens, tok.eos_token_id))
        # P0-2: 행별 응답 끝 — padding 토큰을 보상 디코드·logp 어디에도 넣지 않는다
        eos_set = eos_ids_of(model, tok, pad_id=tok.eos_token_id)
        ends = [resp_end_index(gen[j], ids.numel(), eos_set) for j in range(k)]
        rewards = torch.tensor([
            reward(tok.decode(gen[j, ids.numel():ends[j]], skip_special_tokens=True),
                   item["answer"])
            for j in range(k)
        ])
        advs = loo_advantages(rewards)
        if float(advs.abs().sum()) == 0.0:
            continue  # 전부 같은 보상 — 신호 없음
        opt.zero_grad()
        for j in range(k):
            if float(advs[j]) == 0.0:
                continue
            seq = gen[j, :ends[j]].unsqueeze(0)
            logits = model(seq).logits[0, :-1].float()
            logp = torch.log_softmax(logits, dim=-1).gather(
                -1, seq[0, 1:].unsqueeze(-1)).squeeze(-1)
            resp = logp[ids.numel() - 1:]
            (-(float(advs[j]) / k) * resp.sum()).backward()
        opt.step()
        if (step + 1) % 5 == 0:
            print(f"[{ts()}]  train {step + 1}/{steps} mean_r={float(rewards.mean()):.2f} "
                  f"({(_time.time() - t0) / (step + 1):.0f}s/step)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)


@torch.no_grad()
def eval_accuracy(base: str, adapter: Path | None, prompts: list[dict],
                  max_new_tokens: int) -> float:
    from rollout import load_policy

    model, tok = load_policy(base, adapter)
    from grads import ts
    print(f"[{ts()}] eval 시작: {len(prompts)} prompts (greedy)", flush=True)
    correct = 0
    for item in prompts:
        ids = chat_ids(tok, item["question"]).unsqueeze(0).to(model.device)
        # repetition_penalty는 greedy에도 적용된다 — generation_config 기본값 차단
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                             repetition_penalty=1.0, no_repeat_ngram_size=0,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        correct += reward(text, item["answer"]) > 0.5
    return correct / len(prompts)


def selection_rollout_cost(run: Path, source: str) -> int:
    """그 선택 방법이 소비한 fresh rollout 수 (총연산 통일용).

    stale 점수(g00/g10/g01/g11)와 random은 0, oracle은 fresh pool 전체,
    certagrad는 report.json의 fresh_groups × micro_group.
    """
    if source == "oracle":
        fresh = run / "rollouts_fresh_train.jsonl"
        return sum(1 for _ in fresh.open()) if fresh.exists() else 0
    if source == "certagrad":
        rep = run / "report.json"
        if rep.exists():
            cg = json.loads(rep.read_text()).get("certagrad", {})
            micro_path = run / "oracle_micro_groups.pt"
            if not micro_path.exists():
                raise ValueError("certagrad cost requires oracle_micro_groups.pt")
            micro = torch.load(micro_path, map_location="cpu", weights_only=True)
            groups = min(stack.shape[0] for stack in micro.values())
            from kcurve_floor import find_fresh_k
            group_size = find_fresh_k(run, groups) // groups
            cand_groups = int(cg.get("candidate_groups", cg.get("fresh_groups", 0)))
            val_groups = int(cg.get("validation_groups", 0))
            val_k = 0
            val_rollouts = run / "rollouts_fresh_val.jsonl"
            if val_groups and val_rollouts.exists():
                counts: dict[int, int] = {}
                for line in val_rollouts.open():
                    row = json.loads(line)
                    idx = int(row["prompt_idx"])
                    counts[idx] = counts.get(idx, 0) + 1
                if counts:
                    val_k = min(counts.values())
            if val_groups and val_k == 0:
                raise ValueError("cannot infer validation rollout cost for certagrad")
            return cand_groups * group_size + val_groups * val_k
    return 0


def run_downstream(run: Path, base: str, source: str, steps: int, k: int,
                   max_new_tokens: int, temperature: float, topk_frac: float,
                   budget_rollouts: int = 0, seed: int = 0) -> dict:
    """source ∈ {oracle, g00, g10, g01, g11, random} 선택으로 학습→val 정확도.

    budget_rollouts > 0 이면 '선택+학습 총 rollout 예산'을 통일한다 —
    선택에 쓴 rollout을 차감한 나머지로 학습 스텝 수를 정한다 (concept 5절
    '같은 총연산' 조건의 구현).
    """
    from gate_rules import has_valid_analysis_protocol
    if not has_valid_analysis_protocol(run):
        raise ValueError("downstream requires corrected score and oracle protocols")
    prompts = json.loads((run / "prompts.json").read_text())
    n = len(prompts["train"])
    kk = topk_count(n, topk_frac)
    if source == "random":
        sel = random.Random(seed).sample(range(n), kk)
    elif source == "certagrad":
        report = json.loads((run / "report.json").read_text()).get("certagrad", {})
        sel = [int(idx) for idx in report.get("selected", [])]
        if len(sel) != kk:
            raise ValueError(
                f"certagrad selection missing or wrong size: expected {kk}, got {len(sel)}"
            )
    elif source == "oracle":
        scores = {int(i): v for i, v in json.loads((run / "scores_oracle.json").read_text()).items()}
        sel = sorted(jittered_topk(
            {idx: value["score"] for idx, value in scores.items()}, kk, seed
        ))
    else:
        off = json.loads((run / "scores_offpolicy.json").read_text())[source]
        scores = {int(i): v for i, v in off.items()}
        sel = sorted(jittered_topk(
            {idx: value["score"] for idx, value in scores.items()}, kk, seed
        ))
    sel_cost = selection_rollout_cost(run, source)
    if budget_rollouts > 0:
        remaining = budget_rollouts - sel_cost
        if remaining < k:
            raise ValueError(
                f"selection cost {sel_cost} leaves {remaining} rollouts, below one train step K={k}"
            )
        steps = remaining // k
        print(f"[budget] 총예산 {budget_rollouts} rollouts — 선택 소비 {sel_cost} → 학습 {steps} steps")
    out_dir = run / f"downstream_{source}"
    grpo_lite_train(
        base, prompts["train"], sel, steps, k, max_new_tokens, temperature, out_dir,
        seed=seed,
    )
    acc = eval_accuracy(base, out_dir, prompts["val"], max_new_tokens)
    base_acc = eval_accuracy(base, None, prompts["val"], max_new_tokens)
    result = {"source": source, "selected": sel, "val_acc": acc, "base_acc": base_acc,
              "selection_rollout_cost": sel_cost, "train_steps": steps,
              "budget_rollouts": budget_rollouts, "seed": seed}
    (run / f"downstream_{source}.json").write_text(json.dumps(result, indent=1))
    print(f"downstream[{source}]: val {base_acc:.3f} → {acc:.3f}")
    return result
