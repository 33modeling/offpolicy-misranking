# Qwen3.5-9B 실행 코드 추가 감사

## 2026-10-06 Triton 캐시 파일 누락

**현재 확인한 진행 상태:** `9ac3a6f` 적용 후 사용자가 초기 정책의 응답·보상
캐시 작업이 진행 중이라고 보고했다. 이는 아래 커널 컴파일 캐시와 다른
**실험용 response/reward cache**다. 캐시 완료, prefix/arm 학습 시작·완료,
최종 reward 또는 모든 노드의 장애 해소를 확인한 기록은 아니다.

### 관측과 원인 구분

사용자가 제공한 실제 rank 로그에서 다음 원래 예외를 확인했다.

| 관측 위치 | 직접 확인한 실패 | 경로 범위 |
| --- | --- | --- |
| rank 2 | `FileNotFoundError`: `l2norm_fwd_kernel.cubin`을 열지 못함 | 기존 `qwen-runtime-cache/nodes/<host>/triton/` 아래 생성된 커널 파일 |
| rank 0 | `ImportError: No such file or directory`: `__triton_launcher.so`를 불러오지 못함 | 같은 구조의 노드별 Triton 캐시 아래 생성된 launcher 파일 |
| 바깥 launcher | `srgc_qwen35_smoke.py FAILED`, `ChildFailedError`, `launch_agent` | 자식 rank 실패를 전달하는 요약이며 원래 원인 자체는 아님 |

위 경로의 `<host>`는 노드별 부분을 생략한 표기다. 제공된 로그 조각의
rank 0 `scoring` 시각 `10:30:46`, 오류 시각 `10:31:01`, launcher 요약 시각
`10:31:56`을 단일 실행의 연속 측정으로 합치지 않는다. 조각 사이의 host와
admission/attempt ID가 일치하는지 확인하지 못했으므로, 이 시각으로 실패까지
걸린 시간을 계산하거나 특정 rank가 최초 원인이라고 단정하지 않는다.

직접 확인한 원인은 **실행에 필요한 생성 커널/launcher 파일을 해당 경로에서
읽을 수 없었다는 것**이다. [기존 저장 경로 설정](../scripts/srgc_qwen35_storage.py)은
노드별 경로만 만들었으므로 그 노드의 네 rank가 하나의 Triton 캐시를 공유했다.
여러 rank의 컴파일·파일 접근 경합을 줄이는 방향으로 수정했지만, 실제 파일의
삭제 주체, 정확한 경합 순서, 공유 파일시스템/NFS의 장애 여부는 입증하지 못했다.

앞서 보고된 `NCCLUtils.cpp:77`, NCCL 2.26.2의 CUDA 802 오류와 이번 파일 누락이
같은 원인인지는 확인하지 못했다. 다른 rank가 종료된 뒤의 통신 오류인지,
별개의 CUDA 초기화 문제인지도 이 로그만으로 결정할 수 없다. 따라서
CUDA 802를 모두 컴파일 캐시 문제로 재분류하거나, 반대로 노드/Fabric 문제로
확정하지 않는다.

Qwen과 OLMo의 실행 경로도 구분한다. [Qwen adapter](../scripts/srgc_qwen35.py)는
hybrid/DeltaNet 경로와 `fla-core==0.5.2`를 사용하고,
[모델별 환경 검사](../src/model_matrix.py)의 FLA 요구도 Qwen 분기에 있다.
이번 누락 파일은 그 실행에서 필요한 Triton 생성물이다. OLMo 실행이 진행된다는
사실만으로 Qwen의 FLA/컴파일 경로를 검증할 수 없으며, OLMo가 모든 Triton·
공유 캐시 문제에 영향을 받지 않는다는 뜻도 아니다.

### 대응 순서

| 커밋 | 변경과 확인 범위 |
| --- | --- |
| [`7698a3c`](https://github.com/33modeling/offpolicy-misranking/commit/7698a3c4a5d5eb336f3d7e3b31c5a745b335fe3b) | 해당 smoke 로그에 NCCL과 CUDA 802가 함께 있을 때만 NVLS/cuMem/P2P 설정을 최대 세 단계 추가하며 작은 검사와 모델 검사를 다시 수행. 원인을 확정하거나 성공을 보증한 변경은 아님 |
| [`bea9a73`](https://github.com/33modeling/offpolicy-misranking/commit/bea9a7352bd1ee1a711c5946db91c0585b147686) | 최장 입력·2,048토큰 역전파 시작 조건을 후보 입력·32토큰 readiness 검사로 축소하고 단계 로그 추가. CUDA 802가 과도한 부하 때문이었다는 검증은 아님 |
| [`597bdbd`](https://github.com/33modeling/offpolicy-misranking/commit/597bdbd8eedee0c6dfd9f74ee7691e80cd47c034) | 요약에서 빠지던 protocol/단계, 관련 traceback, NCCL 경고와 `Last error` 후속 줄 보존. 진단 출력 수정이며 CUDA 오류 자체의 치료는 아님 |
| [`9ac3a6f`](https://github.com/33modeling/offpolicy-misranking/commit/9ac3a6f038e1ac4abae9904f5793cd0755495ea0) | 확인된 생성 파일 누락에 대응해 smoke와 실제 cache/train의 컴파일 캐시를 새 노드/rank별 namespace로 분리. 새 entry도 process guard에 등록 |

진단 과정의 부족한 점도 남긴다. 초기에는 바깥 NCCL/launcher 요약에 무게를
두었고, 원래 rank의 파일 누락 예외를 보기 전에 통신 복구와 smoke 부하를
조정했다. 또한 기존 요약 필터가 단계 표식과 여러 줄 CUDA 오류를 버려
새 코드가 실행돼도 같은 요약만 보이게 했다. 이후에는 **원래 rank 예외와
발생 단계부터 확인하고, 그 근거에 맞는 범위만 수정**하는 순서로 바꿨다.
로컬 CPU 회귀 통과를 실제 H100/FLA 컴파일 또는 공유 파일시스템 검증으로
해석하지 않는다.

### 적용 범위와 보존

[rank runtime entry](../scripts/srgc_qwen35_rank_runtime.py)는 원래 저장 경로
검사를 수행한 뒤 Torch/모델 backend를 불러오기 전에 다음 경로를 설정한다.

```text
$OM_WORK/qwen-runtime-cache/nodes/<host>/rank-runtime-v1/rank-<0..3>/
  triton/       TRITON_CACHE_DIR
  inductor/     TORCHINDUCTOR_CACHE_DIR
  extensions/   TORCH_EXTENSIONS_DIR
  cuda/         CUDA_CACHE_PATH
  tmp/          TMPDIR
```

일반 shell의 자동 시작과 명시적 `run` 모두 적용된다. smoke는 짧은 검사 entry
안에서, cache/train은 rank runtime entry를 거쳐 원래 rank 실행기로 들어간다.
같은 노드/rank의 새 경로는 작업 간 재사용하고, 첫 컴파일에는 추가 시간이 들 수
있다. 모델/Hugging Face 캐시는 계속 공유한다. 기존 컴파일 캐시는 삭제하지 않으며
plan·queue·입력·response/reward cache·checkpoint·결과와 adapter/engine hash를
바꾸지 않는다. 실패한 검사를 성공으로 처리하거나 GPU admission을 생략하지 않는다.
기록된 admission 비용에는 실패한 시도도 포함하고, receipt가 없는 시간은
`cost_accounting_complete=false`로 구분한다.

Qwen 회귀 검사 **135개 및 22개 subtest**, 별도 rank-cache/process-guard 검사
**33개**가 통과했다. 두 검사 묶음은 일부 중복되므로 168개의 서로 다른 검사로
합산하지 않는다. rank별 경로 분리, 실제 adapter의 task-command 연결·복원,
살아 있는 새 entry의 부모 보호와 고아 프로세스 구분을 CPU에서 검사했다.
사용자의 후속 캐시 진행 보고와 이 로컬 검사 결과를 별도로 기록하며, 이 감사에서
원격 GPU 작업을 실행·중단하거나 실제 H100/NFS 재발 방지 시험을 수행하지 않았다.

### 다시 실패할 때

```sh
# 마지막 Qwen 모델 검사 로그 조회만 수행한다. 모델 로드·GPU 작업·파일 변경 없음.
sh scripts/run_srgc_qwen35.sh all error
```

`[qwen-rank-cache] rank=... triton=...`으로 새 rank 경로를 확인하고,
`[qwen-smoke]`의 `candidate-readiness-v1` 및 마지막 단계와 원래 rank 예외를
함께 읽는다. `init`, `startup_collective`, `model_load`, `rollout`, `scoring`,
`update`, `final_barrier`는 진행 구분이며, 단계 출력 하나가 전체 성공을 뜻하지 않는다.
요약만으로 부족하면 출력된 `.queue/admission/<worker>/qwen-smoke.log` 또는
receipt가 가리키는 `recovery-XX/qwen-smoke.log` 원본을 보존한다. 작은 NCCL 검사
자체가 실패한 경우에는 해당 오류가 가리키는 `preflight.log`를 먼저 확인한다.
여러 노드의 자료를 비교할 때 host·rank·worker/attempt ID·코드 커밋을 함께
기록한다. 기존 캐시·plan·hash·잠금 파일을 지우거나 초기화하는 절차로 대체하지 않는다.

## 2026-10-06 시작 검사 부하 수정

- 이전 Qwen 검사는 평가용 입력까지 포함해 가장 긴 prompt를 고르고 합성 응답
  8개에 각각 2,048토큰 역전파를 강제했다. OLMo의 작은 NCCL 검사 뒤에 추가된
  이 경로는 실제 작업에 필요하지 않은 최장 평가 입력의 역전파까지 시작 조건으로
  요구했다. 아래 9월의 최장 입력 검사 기록은 현재 동작이 아닌 과거 기록이다.
- 일반 shell 시작의 Qwen smoke만 `srgc_qwen35_smoke.py`로 교체했다.
  현재 plan 첫 seed의 학습 후보 중 짧은 입력 하나를 사용하며, 실제 생성과
  합성 역전파 응답은 각각 최대/고정 32토큰이다. 응답 수 8개와 동일한 모델,
  adapter, scoring/GRPO backend를 유지하고, startup object collective는
  모델 로딩 전에 수행한다. 실패를 성공으로 처리하거나 검사를 생략하지 않는다.
- 평가·validation 및 다른 dataset/seed의 입력으로 부하를 늘리지 않는다.
  실제 실험의 최대 응답 길이 2,048토큰과 입력·학습 설정은 그대로다.
  짧은 시작 검사는 최대 길이의 GPU 메모리 용량을 보증하지 않는다.
- `init`, `startup_collective`, `model_load`, `rollout`, `scoring`, `update`,
  `final_barrier`를 rank별 로그에 남긴다. 새 entry는 smoke 외의 stage를 거부하며
  `bea9a73` 당시에는 cache/train 명령을 수정하지 않았다. 이후 `9ac3a6f`에서
  위의 캐시 분리 entry를 cache/train에도 연결했다. 아래 CUDA 802 복구는 짧은
  검사에도 적용된다.
- adapter/engine hash는 아래 기록과 같으며 기존 plan·queue·checkpoint를
  초기화하지 않는다. 원격 GPU 작업을 시작하거나 중단하지 않았다.

원격 CUDA 802의 직접 원인이 이 부하였다고 확정한 수정은 아니다.
실제 H100에서 새 검사가 통과하는지는 이 CPU 환경에서 검증하지 못했다.
검증: Qwen/NCCL 회귀 검사 197개 및 22개 subtest와 새 workload 검사 7개 통과.
GPU 검사 3개는 skip이며, 기존 multiprocessing fork 경고 5건이 있었다.
모의 smoke에서 입력 분리·32토큰 제한·단계 순서·실패 전파를 확인했고,
당시 실제 학습 및 cache 명령은 기존 entry를 유지하는지 별도로 검사했다.

## 2026-10-06 Qwen NCCL 802 복구

- 작은 NCCL 검사 뒤 실제 Qwen 생성·역전파 검사에서 CUDA 802가 발생하면
  바로 종료하던 경로에 제한된 복구를 추가했다. 현재 smoke 로그에 NCCL과
  CUDA 802가 함께 확인된 경우에만 기존 NVLS/cuMem/P2P 복구 단계를 사용한다.
- 각 단계는 별도 디렉터리에서 작은 검사와 실제 모델 검사를 모두 다시 수행한다.
  최대 세 번 추가 검사하며, 통과 전에는 학습 작업을 배정하지 않는다.
  명시된 NCCL 설정, OOM·다른 오류·사용자 중단은 자동 변경하거나 우회하지 않는다.
- 원본 receipt와 시도별 로그를 보존한다. 성공한 설정은 학습 환경에 전달하고
  실패한 시도를 포함해 기록된 비용을 합산한다. 중단으로 receipt가 없으면
  `cost_accounting_complete=false`로 표시하며 이 합계는 하한이다.
- shell의 자동 시작과 명시적 `run`이 모두 복구 wrapper를 사용한다.
  `error` 명령은 최종 recovery 로그를 표시하며 GPU 작업을 시작하지 않는다.
- 이번 수정은 실행 wrapper에만 적용했다. adapter hash는
  `20e90f784b19afa6602b839ff2f7934eb207f03868a4052896edc9355b7ffd8b`,
  engine hash는
  `12cf5ef830ebfd92fa8a87ea62dc7df734cd9ceab57fbce18fc4b2548385f960`으로
  수정 전과 같다. 기존 plan·queue·입력·checkpoint를 다시 만들지 않는다.
  통신 설정 변경은 실행시간과 부동소수점 합산 순서에 영향을 줄 수 있다.

검증: Qwen 회귀 검사, smoke 복구, NCCL preflight/runtime 검사에서
**194개 및 22개 subtest 통과, GPU 검사 3개 skip**. multiprocessing fork
관련 경고 5건이 있었다. 실제 H100에서 802 해소 여부는 검증하지 못했다.
모든 단계가 실패하면 노드의 드라이버·CUDA 라이브러리·fabric 점검이 필요하다.

## 2026-10-06 재점검

- Pair 입력 후속 오류: importer는 기존 MATH 입력을 `math500`으로 기록하지만
  plan은 `math_train`을 사용한다. Qwen의 단순 문자열 비교를 Pair 출처가 확인된
  기존 이름의 호환 검사로 바꾸고, 다른 데이터셋과 reference 개수 오류는 구분한다.
  실제 Pair importer → Qwen 준비 → queue 검증 경로를 테스트했다. 새 입력은 모두
  검증한 뒤 저장하며, 문제·정답·분할·원본 데이터 이름은 바꾸지 않는다.
  `ded3d11`의 알려진 adapter hash로 준비만 된 MATH plan은 실행 기록·cache가
  없을 때 원본 plan을 보관하고 복구한다. 미확인 hash나 진행 기록이 있으면 거부한다.
  Qwen 관련 회귀 검사 **71개 및 22개 subtest 통과**, 실패·skip 없음.
  로그: `/tmp/qwen-bundle-fix-20261006.log`.
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
