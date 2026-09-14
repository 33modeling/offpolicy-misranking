"""Figures for observed switching outcomes. Missing states are never interpolated."""

import argparse
from pathlib import Path

import selection_gate as core


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = core.read(root / "test-report.json")
    rows = report["rows"]
    if not rows:
        print("[figures pending] no completed held-out states; no synthetic replacement")
        return
    seeds = sorted({r["seed"] for r in rows})
    target = root / "figures"
    target.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), constrained_layout=True)
    for i, seed in enumerate(seeds):
        for row in [r for r in rows if r["seed"] == seed]:
            x = row["step"]+(i-(len(seeds)-1)/2)*2
            action = row["intended_action"]
            axes[0].scatter(x, 100*row["audit"]["delta"], marker="^" if action == "select" else "v", s=65,
                            color=f"C{i}", facecolors="none" if row["fallback"] else f"C{i}")
            for arm, offset, color in (("gated", -1, "#16826d"), ("selection_full", 0, "#b35e30"), ("random_full", 1, "#666666")):
                axes[1].scatter(x+offset, 100*row["means"][arm], color=color, s=28)
            axes[2].scatter(x-1, 100*row["audit"]["gate_minus_random"], color="#16826d", s=35)
            axes[2].scatter(x+1, 100*row["audit"]["gate_minus_selection"], color="#b35e30", s=35)
    axes[0].set_title("Paid continuation advantage; triangle up: retain selection, down: switch")
    axes[0].set_ylabel("Continue_D - Switch_D (pp)")
    axes[1].set_title("Executed reward: gate (green), continue (orange), random (gray)")
    axes[1].set_ylabel("Independent test reward (%)")
    axes[2].set_title("Gate minus random (green) and continue (orange)")
    axes[2].set_ylabel("Paired reward contrast (pp)")
    for ax in axes:
        ax.set_xticks([25, 50, 100])
        ax.grid(axis="y", alpha=.2)
        ax.set_xlabel("Selected-prefix updates (independent branch points)")
    for ax in (axes[0], axes[2]):
        ax.axhline(0, color="black", linewidth=.8)
    fig.savefig(target / "switch-outcomes.pdf")
    fig.savefig(target / "switch-outcomes.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
    for i, row in enumerate(rows):
        audit = row["audit"]
        axes[0].bar(i-.2, 100*audit["wrong_switch_loss"], width=.2, color="#16826d")
        axes[0].bar(i, 100*audit["wrong_retention_loss"], width=.2, color="#b35e30")
        if "checkpoint_only_audit" in row:
            axes[0].bar(i+.2, 100*row["checkpoint_only_audit"]["decision_regret"], width=.2, color="#777777")
        axes[1].bar(i, row["measurement_gpu_seconds"], color="#555555")
    for ax in axes:
        ax.set_xticks(range(len(rows)), [f"s{r['seed']} t{r['step']}" for r in rows], rotation=30)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_title("Wrong switch (green), wrong retention (orange), checkpoint-only regret (gray)")
    axes[0].set_ylabel("Observed reward loss (pp)")
    axes[1].set_ylabel("One-time diagnostic GPU-seconds")
    fig.savefig(target / "switch-regret-cost.pdf")
    fig.savefig(target / "switch-regret-cost.png", dpi=180)
    plt.close(fig)
    print(f"[figures] {target}; descriptive seed marks, not a continuous switching trajectory")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    plot(p.parse_args().root)
