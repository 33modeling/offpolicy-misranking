# Gemma switch가 25번 검사 단계를 반복하던 문제 (2026-10-31)

사용자 보고: Gemma switch만 SELECT 25/275에서 24시간 넘게 결과가 나오지 않고, 같은 단계를
반복한다. 다른 arm은 느리지만 결과가 나온다.

원인. switch는 25, 50, … 업데이트에서 새 후보 40개와 SR 비교 40개(최대 80문제)를 채점하고,
이어서 검증 50문제를 채점한 뒤 업데이트 25를 학습하고 체크포인트를 저장한다. 고정된
`ResumableRolloutMixin`(scripts/srgc_resumable_rollouts.py)은 채점 묶음 하나가 끝나는 즉시
그 묶음의 rollout 캐시를 지운다. 그래서 후보 묶음이 끝난 뒤 검증 채점이나 업데이트 25 학습
도중 작업이 끝나면, 재시작은 업데이트 25 체크포인트에서 후보 80문제의 응답 640개를 처음부터
다시 생성한다. Gemma에서는 이 생성만 몇 시간이 걸리므로 그 사이 한 번이라도 끊기면 같은
단계를 반복한다. on_policy는 같은 시점에 후보 40개만 채점하므로 이 구간이 절반이다.
작업이 끝난 이유(노드 회수, 1800초 진행 정지 판정, NCCL 대기 한도 등)는 클러스터 로그가
없어 확인하지 못했다.

수정. 운영 rank 래퍼(`srgc_research/dispatch/rank_runtime.py`)가 Gemma(gemma4, gemma4_text,
gemma4_it) rank에서만 `srgc_research/dispatch/selection_resume.py`를 설치한다.

- 채점 묶음이 끝나도 rollout 캐시를 지우지 않고, 모은 projected gradient를
  `gathered-gradients.npz`로 함께 저장한다. 학습 가능한 가중치의 SHA-256(`policy.sha256`)과
  요청한 문제, 응답 수, 그룹 크기가 같을 때만 다시 쓴다.
- 다시 쓸 때는 저장된 rollout을 다시 읽어 원시 응답 기록(reward-curves rollouts)에도 그대로
  남긴다. 생성과 역전파는 하지 않는다.
- 가중치가 바뀐 다음 학습 업데이트가 시작될 때(앞 업데이트의 체크포인트가 저장된 뒤) 지난
  묶음을 지운다. 다른 가중치에서 만든 묶음은 다시 쓰지 않는다.
- 분산 rank는 rank 0의 판단을 broadcast로 함께 따른다.

바꾸지 않은 것. plan, 입력, seed, sampling, 보상, 선택 규칙, 학습 코드, 체크포인트 형식.
Gemma plan의 adapter·engine 지문(`adapter_sha256`, `engine_sha256`)에 들어가는 파일은 건드리지
않았으므로 기존 root를 그대로 이어서 쓴다. Qwen·Llama rank에는 설치하지 않는다.

검사(CPU, torch 없는 가짜 backend). 실제 SRGC `Engine`으로 switch 검사 시점까지 진행한 뒤
검증 채점 중 한 번, 업데이트 학습 중 한 번 끊고 재시작한다. 두 재시작 모두 후보·SR 문제를
다시 생성하지 않고, d·전환 여부·선택 문제·점수가 끊지 않은 실행과 같으며 다음 업데이트로
넘어간다. 같은 검사를 고치기 전 mixin에 돌리면 재시작이 후보를 다시 생성해 실패한다.
원격 Gemma에서 실제로 25를 넘었는지는 미확인이다.

적용: 클러스터에서 `git pull --ff-only`. 다음에 뜨는 switch 시도부터 적용된다. 이미 끝난
후보 묶음은 고치기 전 코드가 지웠으므로 첫 시도는 후보를 한 번 더 생성할 수 있다. 진행 중인
묶음의 rollout은 이어서 쓴다.
