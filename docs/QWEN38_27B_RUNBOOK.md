# Qwen3.8-27B post-trained replication

Prepared 2026-09-06. No GPU experiment has been launched by this code change.
Local tests use Transformers 5.14.1 and PEFT 0.20.0. The generic requirements
minimum is not evidence of Qwen3.8 support: the compute environment must expose
the Qwen3.5 multimodal model classes and pass the FLA/runtime checks below.

## Scientific design

`configs/qwen38_27b_grpo.json` pins `Qwen/Qwen3.8-27B` to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The official model card identifies
this as a post-trained model. Step zero is this released checkpoint, not a
pretrained base. Existing historical SFT artifacts must not be resumed.

The experiment uses MATH-500 (400 candidate / 100 validation) and MBPP
(512 / 100), seeds 0--4, and continuous GRPO checkpoints 0/25/100/400:
10 families, 40 points. Sampling counts, full-softmax temperature 1, 2048-token
cap, one-epoch GRPO, learning rate 1e-5, rank-16 LoRA, R/A/B separation and
10,000 FIRST replicates match the OLMo main configuration.

Qwen uses its pinned official chat template with thinking disabled. OLMo uses
its RL-Zero completion format, so prompt wrappers are not identical. The same
dataset items and verifier are used, but prior post-training, architecture,
capacity, template and trainable parameter count differ. This tests replication
in another model setting and cannot identify the effect of any one difference.

LoRA targets are `q_proj,v_proj` in attention blocks and
`in_proj_qkv,in_proj_z,in_proj_b,in_proj_a` in DeltaNet blocks. These explicit
names avoid adapting unused vision layers. The fused QKV projection also
adapts keys: this is not an exactly matched Q/V-only parameterization. Ranking
uses the final four text decoder blocks and text normalization; vision,
embedding and LM output head parameters are excluded. Trainable/ranking
parameter counts are recorded in the benchmark report.

## Commands

Commit the source before launching. Do not update a shared checkout while a
worker is active. The existing environment and pinned snapshot machinery are
reused. On the network-enabled preparation machine:

```bash
bash scripts/run_qwen38_27b.sh prepare
```

On an idle four-H100 node, first perform offline compatibility checks:

```bash
bash scripts/run_qwen38_27b.sh check
```

This checks snapshot hashes, dataset qualification, FLA 0.5.2 kernels, short
generation, rank-16 LoRA backward, adapter save/reload/merge and an actual
4096-dimensional dense ranking projection. Timing and peak allocated CUDA
memory are saved next to the smoke marker in `contracts/*.benchmark.json`.
The report is a **single short prompt** smoke: it does not certify 2048-token
peak memory, four-rank GRPO updates/resume, reward signal or total matrix ETA.
On the target cluster, qualify those workloads before committing a long
allocation; short-prompt throughput cannot be extrapolated reliably to long
responses. The trainer's first-pass ratio check still aborts above 5e-3.

After qualification, start the explicit replication on each four-H100 node:

```bash
bash scripts/run_qwen38_27b.sh run
```

Compute modes are offline and remove inherited Hugging Face tokens. A shared
family queue and node-local lock serialize each continuous seed lineage.
Initial generation, gradient and training log-prob microbatches are all one;
group size stays eight. This favors memory headroom over speed. Changing these
registered runtime settings changes the configuration hash and requires a new
contract, rather than silently adopting existing results.

Results are isolated under
`$OM_WORK/results/qwen38-27b-posttrained-math-code-grpo-v1/` and runs under
the corresponding `runs/` directory. `check` does not enqueue training or
write a final regime bundle. Calling the wrapper without a mode prints usage
and exits. The legacy extension's default entry point remains unchanged.

Before interpreting outcomes, check truncation and reward saturation, realized
KL/ESS, complete seed coverage, corrected oracle/regime schemas and final
artifact hashes. A failed reliability gate yields an inconclusive comparison.
The manuscript's old registered extension must be amended explicitly before
claiming this new configuration as its completed replication.

Source: [official model card](https://huggingface.co/Qwen/Qwen3.8-27B).
