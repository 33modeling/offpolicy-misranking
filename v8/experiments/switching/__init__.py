"""V8 switching experiments (E0-E10) layered over the existing SR-GC engine.

The GRPO trainer, On-policy gradient-alignment selector, cached-SR selector,
SR-GC contrast and verifiers live in the code repository
``33modeling/offpolicy-misranking`` (``srgc_rebuttal/``). This package never
re-implements them: it imports that package from ``V8_CODE_ROOT`` and adds
selector/transition/cache separation, checkpoint branching, the cost ledger,
budget endpoints, seed/matrix generation, tuning, freezing and analysis.
"""

SCHEMA_VERSION = 1
STUDY_ID = "rlvr_switching_rebuttal_v8"
