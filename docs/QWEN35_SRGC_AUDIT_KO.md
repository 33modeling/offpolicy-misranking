# Qwen3.5-9B 실행 코드 추가 감사

## 2026-10-06 재점검

- 후속 시작 오류: `math-verify`가 설치되지 않은 Python에서 package metadata
  검사 전에 저장소의 오프라인 verifier를 연결하도록 순서를 수정했다. 패키지가
  없는 `python -S`에서 같은 오류를 재현하고, 수정 후 실제 verifier import,
  metadata 조회, 수식 동치 채점, 자식 프로세스 경로 전달을 검사했다.
  기존 admission과 같은 고정 wheel을 사용하며 pip·공유 venv 수정은 없다.
  후속 회귀 검사: 시작·shell·runtime identity·누락 의존성 **28개 및 8개
  subtest 통과**. 로그: `/tmp/qwen-math-verify-fix-20261006.log`.
- 수동 `download`가 자동 시작과 같은 모델 잠금을 사용하도록 통일했다.
  명시적 모델 경로를 존중하고, 검증에 실패한 기존 snapshot은 교체하지 않는다.
- 마지막 endpoint 저장 뒤 중단된 worker를 다시 실행하면, GPU admission이나
  학습 없이 완료 manifest와 결과·비용 요약을 복구한다. 중단 표시된 queue는 유지한다.
- 여러 dataset queue에 참여할 때 모든 기존 환경 정보를 먼저 검사한다.
  하나의 환경이 다르다는 이유로 거부된 노드가 다른 queue에 환경 정보를 남기지 않는다.
- 수동 `prepare`도 기존 run의 plan이 사라졌으면 입력이나 plan을 새로 만들지 않는다.
  기존 plan 복구가 필요하며 checkpoint·결과 파일은 그대로 보존한다.
- 실제 실행에 쓰이는 rollout 재개, attention 설정, 방향 진단 기록 helper를
  Qwen adapter hash에 포함했다. helper가 바뀌면 기존 plan으로 학습 재개는
  거부하지만, 저장된 identity로 과거 결과를 조회하는 기능은 유지한다.

검증: `/home/nsh/.venvs/proto-ml/bin/python -m pytest srgc_rebuttal/tests -q`
전체 **696개 및 313개 subtest 통과, 실패·skip 없음**. 두 프로세스의 60개 작업
배정, 다운로드 잠금, 완료 결과 복구, 작은 실제 Qwen 모델의 생성·gradient·LoRA
update를 포함한다. Python 3.12의 multiprocessing fork 관련 경고 6건이 있으며
로그는 `/tmp/qwen-debug-final-20261006.log`에 남겼다. shell 구문, Python 구문,
`git diff --check`도 통과했다. 환경은 PyTorch 2.13.0+cpu / Transformers 5.14.1 /
PEFT 0.20.0이며 실제 9B GPU 검증은 아니다.

선별 규칙·응답 수·학습 업데이트 수는 바꾸지 않았다. 아래 9월 검증 기록은
당시 코드의 기록이다. 이번 수정은 adapter hash를 바꾸므로 기존 Qwen 실험은
원래 checkout으로 유지한다. 원격 실험을 시작·중단하지 않았으며 실제 H100
9B 메모리·FLA/CUDA·NCCL 검증은 여전히 worker admission에서 수행해야 한다.

## 2026-09-28 감사

2026-09-28. 대상은 `scripts/run_srgc_qwen35.py`에서 시작하는 Qwen 온라인
전환 확장과 이 실행이 사용하는 process guard다. 실행 방법은
[Qwen 실행 안내](QWEN35_SRGC_KO.md)에 있다.

## 발견한 문제와 수정

| 영역 | 발견한 문제 또는 미방어 경로 | 수정 |
| --- | --- | --- |
| 생성 메모리 | 8응답 동시 생성의 KV 메모리와 큰 vocabulary projection을 제한하지 않음 | 2응답씩 생성하고 마지막 토큰만 vocabulary projection. CUDA OOM이면 난수 상태를 복원해 1응답씩 전체 호출을 재시도하며 응답 수·길이는 유지 |
| 역전파 메모리 | scoring/학습의 활성값과 GPU optimizer snapshot이 메모리를 추가 점유 | non-reentrant activation checkpointing, log-prob micro-batch 1, 64-token logit chunk, CPU optimizer snapshot, update 후 gradient 해제 |
| 저장 위치 | 준비 단계가 잘못된 출력 경로를 검사하기 전에 tokenizer 다운로드에 도달할 수 있음 | 다운로드·준비부터 group 경로를 검사. 모델·결과·라이브러리 캐시·임시 파일을 group 하위에 배치하고 외부 경로 및 탈출 symlink 거부 |
| 여러 데이터셋의 상태 | MATH/MBPP에 같은 task 이름이 있어 worker 상태를 잘못 연결할 수 있음 | 데이터셋별 receipt와 로그, active plan/dataset 기록. 모든 queue를 작업 시작 전에 bind |
| 캐시 완료 경합 | 다른 노드가 cache를 완료하는 동안 이전 상태를 보고 작업을 획득할 수 있음 | task lease 획득 후 완료 상태 재검사. 비용 summary가 없는 cache는 export 복구 후 prefix에 전달 |
| GPU 잠금 | 별도 Qwen root가 다른 잠금 디렉터리를 사용할 가능성 | group 공통 GPU UUID 잠금과 기존 OLMo 호환 잠금을 함께 획득. 잠금 파일 삭제로 우회하지 않음 |
| 프로세스 정리 | 범용 `torch.distributed.run` 문자열이 무관한 학습을 SRGC orphan으로 분류할 수 있음 | 명시적 SRGC entry만 정리 대상으로 인정 |
| 노드별 실행 환경 | 같은 queue에 서로 다른 패키지 환경으로 참여할 수 있음 | 고정 패키지 버전 검사 및 Python/PyTorch/CUDA/cuDNN 등의 queue별 signature 비교 |
| 모델 검증 I/O | 큰 weight 파일을 rank/task마다 반복 해시할 수 있음 | 노드별 최초 전체 검증 후 파일 메타데이터 기반 receipt 재사용. 변경 감지 시 전체 재검증 |
| checkpoint 내구성 | 일부 경계 저장에 명시적 fsync가 없음 | 파일 flush/fsync 후 기존 원자적 rename 수행. 실제 저장장치의 장애 내구성은 별도 검증 필요 |
| 시작 검사 | 짧은 smoke가 긴 입력 역전파 메모리를 확인하지 못함 | 준비된 두 데이터셋 중 가장 긴 prompt와 2,048-token 합성 응답의 역전파, rank별 GPU peak 기록. 성공·실패 admission 비용 별도 기록 |

생성 배치 변경은 난수 소비 순서를 바꿀 수 있으므로 새 프로토콜을
**`qwen35-9b-v2`**, 기본 출력 위치를
`$OM_WORK/srgc-rebuttal/qwen35-9b-v2`로 분리했다. 입력·plan·raw cache에
모델 revision, 프로토콜, 생성 배치 및 adapter 정보를 검증한다.
v1 cache/checkpoint와 합치거나 v1 실행을 새 코드로 이어서 수행하지 않는다.

학습의 scientific engine인 `srgc_rebuttal/*.py`는 수정하지 않았다.
SHA-256은 `1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`다.
기존 실행 중인 작업이나 결과 파일은 변경하지 않았다.

## 실행한 검증

- 수정 전 저장 경로 검사 순서와 optimizer snapshot 문제를 실패 테스트로 재현했다.
- 전체 회귀 검사 **247개 통과, 51.197초, skip 없음**:
  `/home/nsh/.venvs/proto-ml/bin/python -m unittest discover -s srgc_rebuttal/tests -v`.
  로컬 로그: `/tmp/qwen-hardening-final-full.log`.
- 작은 실제 Qwen hybrid 모델을 CPU에서 사용해 8응답 생성, dense scoring gradient,
  GRPO update, optimizer 복원, checkpointing 전후 수치 일치를 검사했다.
  checkpointing의 저장 활성값 감소도 확인했으나 이는 9B GPU 메모리 실측이 아니다.
- 서로 다른 GPU UUID를 가진 두 노드를 **별도 CPU 프로세스로 모의 실행**했다.
  실제 Qwen worker/공유 queue로 MATH·MBPP의 cache → prefix → arm **60개 작업**을
  중복 없이 처리했고 동시 작업 수 2와 의존성 순서를 확인했다.
  GPU admission과 실제 학습은 이 테스트에서 대역으로 처리한다.
- cache 완료 경합, 비용 summary 누락, 잘못된 모델 cache, 환경 불일치,
  출력 경로 변경, symlink, 무관한 torchrun 보호를 검사했다.
- 고정 revision의 **실제 Qwen tokenizer**로 MATH/MBPP 10개 bundle 및
  40개 continuation plan을 준비하고 두 데이터셋의 `status`/`results`를 실행했다.
  모든 결과는 미학습 상태이며 reward를 생성하거나 기존 OLMo reward를 복사하지 않았다.
  이 검사는 `/tmp/qwen-srgc-audit-group-p3m7d_dh`를 임시 group 경로로 사용했다.
- shell syntax, Python compile 및 `git diff --check` 통과.

## 아직 검증하지 못한 범위

이 환경에는 사용 가능한 GPU와 실제 `/group-volume`이 없다.
**H100의 9B peak 메모리, FLA/CUDA 실행, NCCL, 실제 노드 간 공유 파일시스템의
잠금·동기화·장애 복구는 아직 실측하지 않았다.** 로컬 두 프로세스 테스트는
네트워크 파일시스템 검증을 대신하지 않는다. 경로 검사는 group 디렉터리 존재와
경로 소속을 확인하며 모든 노드가 동일한 원격 볼륨을 마운트했는지 증명하지 않는다.

실제 실행 전 모든 노드에서 같은 group mount, Qwen root, 코드 및 CUDA 환경을
사용해야 한다. worker의 GPU admission이 통과해야 실험 작업을 받는다.
본 감사에서 원격 GPU 작업을 시작·중단·재시작하지 않았으며, 새로운 학습 결과도 없다.
