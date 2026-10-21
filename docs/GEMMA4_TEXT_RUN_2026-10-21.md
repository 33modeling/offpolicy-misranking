# Gemma 4 생성·채점 수정과 새 실행

기존 PT v1·text-v2 실험은 보존한다. 현재 text-v3는 같은 `google/gemma-4-12B` 가중치와
같은 문제·seed를 사용하지만 응답 시작을 명시하도록 입력을 고쳤으므로
**새 캐시·새 prefix·새 학습 경로**로 시작한다.
기존 checkpoint를 정상화한 실험의 출발점으로 바꾸거나 기존 보상을 덮어쓰지 않는다.

## 수정 내용

- 모든 후보·순위 검증·최종 평가 입력 끝에 `Solution:`을 넣어 PT 모델이 지시문을
  이어 쓰는 대신 풀이를 시작하도록 한다. MBPP에는 과제와 무관한 identity 함수의
  올바른 들여쓰기 예제와 함수 본문의 네 칸 들여쓰기 지시를 제공한다.
  문제·정답·split·seed·선택 규칙·학습 예산과 기존 verifier는 그대로다.
  이 입력은 모든 arm에 동일하게 적용하며 원시 생성 코드를 사후에 고쳐 채점하지 않는다.
- 공식 PT의 이미지·오디오 종료 토큰 258882·258883 억제를 복원한다. temperature=1,
  top-p=1, top-k=0, 8응답과 원래 문제·학습 예산은 유지한다.
- Scoring·GRPO log probability도 같은 토큰을 제외하고 다시 정규화한다. Native
  soft-capping 뒤에 mask를 적용하며, 기존 v1 rollout의 억제 토큰이 들어오면 거부한다.
- MBPP가 요구하는 여는 fence 다음 줄의 `python`을 언어 표기로 처리한다.
  실제 코드·assertion·기존 자식 프로세스 제한과 부모의 정답 검증은 유지한다.
  다른 모델의 verifier나 기록을 실행 중에 교체하지 않는다.
- 전체 캐시 전에 seed 5의 고정된 첫 8문제 × 8응답을 **실험과 같은 2048-token 상한**으로
  검사한다. 두 과제를 요청하면 각각 검사한다. 원시 토큰·텍스트·보상을 저장하고,
  올바른 응답이 하나도 없으면 전체 캐시·학습을 시작하지 않는다.
  이 통과 조건은 고성능 판정이 아니다. 실제 전체 init은 이후 400문제 × 8응답으로 측정한다.
- 일반 Gemma 시작·상태·결과 명령을 수정본으로 연결한다. 모델 연결 실행도 수정본의
  queue를 확인하며 이전 1% 실험을 재개 대상으로 섞지 않는다.
- 25 updates마다 평가용 checkpoint와 고정 held-out 보상을 보존하는 recording rank,
  원시 학습 보상, 중단·재개, rank 오류 기록과 N12/N13 진단 경로를 새 모델 버전에 연결한다.
  결과는 `gemma4_text-results.json`으로 저장하며 기존 `gemma4-results.json`은 유지한다.

## 실행

추가로 한쪽 plan만 준비된 상태의 중지, 첫 준비 중 즉시 중지, 옵션을 붙인 모델 연결
실행과 도움말 분기를 점검했다. [운영 점검 기록](GEMMA4_OPERATIONAL_AUDIT_2026-10-21.md).
이 운영 수정은 현재 text-v3 실험 조건과 기존 기록을 유지한다.

기존 실험을 실행하던 노드의 같은 저장소·CUDA Python 환경에서 실행한다.
모델 재다운로드는 하지 않는다. 기존 Gemma v1 worker가 해당 GPU를 사용 중이면 먼저
그 worker를 정상 중지해야 하며, 실행 잠금으로 중복 점유를 막는다.

```sh
git pull --ff-only origin master
sh scripts/run_srgc_gemma4.sh all
```

이 시작 명령이 새 입력 준비 → 소량 검사 → **MATH·MBPP 모든 초기 캐시** → MATH prefix·네 arm
→ MBPP prefix·네 arm을 진행한다. [캐시 우선 배정](GEMMA4_CACHE_FIRST_2026-10-21.md).
기존 모델 연결 순서를 유지하며, Gemma만 진행하려면 `SRGC_MODEL_CHAIN=0`을 지정한다.
소량 검사만 실행하고 끝내려면 `all preflight`를 사용할 수 있다. 기본 결과 위치는
`$OM_WORK/srgc-rebuttal/gemma4-12b-text-v3`다. 필요한 경우
`SRGC_GEMMA_TEXT_ROOT`로 **새 전용 경로**를 지정한다. 이전 v1 root를 지정하면
기존 입력·실행이 있어 재사용을 거부한다.

```sh
sh scripts/run_srgc_gemma4.sh status
sh scripts/run_srgc_gemma4.sh results
sh scripts/run_srgc_gemma4.sh stop --now
sh scripts/run_srgc_gemma4.sh all stop --now
SRGC_MODEL_CHAIN=0 sh scripts/run_srgc_gemma4.sh all resume
# 원래 실험을 읽기만 하기
SRGC_GEMMA_LEGACY_PROTOCOL=1 sh scripts/run_srgc_gemma4.sh all status
```

검사 결과는 새 root의 `preflight/<dataset>/summary.json`과 문제별 receipt에 남는다.
검사 로그는 startup/admission 디렉터리의 `gemma-preflight-*.log`다.
입력·설정이 같을 때 완료한 응답은 재사용하며, 손상·누락·다른 프로토콜은 거부한다.

`zero correct responses in the small panel`은 실제 8문제 × 8응답이 모두 오답이라는
검사 결과다. `preflight.py`의 `primary(finish)`와 `distributed.py`의 `raise`는 이 결과를
각 rank에 전달하는 위치다. 이 메시지만으로 CUDA/NCCL 고장이나 채점 오류를 단정하지 않는다.
같은 실패 기록이 있으면 일반 시작 명령이 CPU에서 원시 기록과 summary를 대조하고,
GPU 검사나 전체 캐시·학습을 실행하기 전에 멈춘다. 파일은 삭제하지 않는다.
실행기가 종료되어도 외부 배치 시스템에서 예약한 GPU 할당 시간까지 연장되거나
자동 반환되는 것은 아니다.

다음 기존 명령으로 현재 실험의 작은 검사와 전체 초기 캐시를 구분해서 확인하고,
저장된 응답을 같은 CPU verifier로 다시 채점할 수 있다. JSON 출력은 `--output`으로
실험 폴더 밖의 새 파일에 추가 저장할 수 있다. 원래 응답·보상을 수정하지 않는다.

```sh
sh scripts/run_srgc_gemma4.sh audit --samples 2 --rescore 64
```

일반 실행은 `run_srgc_gemma4.sh` 한 곳에서 처리한다. 별도 실행 파일이나 전환 명령은
사용하지 않는다. 기본 경로는 새 text-v3이므로 이전 캐시·prefix·학습을 이어 쓰지 않는다.

후속 로컬 실제 12B 검사에서는 정답이 6인 기존 MATH 후보에 대해 greedy 생성 상한을
128로 주었지만 4토큰 `\n\nAnswer:` 뒤 EOS로 종료했다. 실제 응답 보상은 0,
같은 verifier에 `Answer: 6`을 넣은 대조 보상은 1이었다. 이는 해당 응답에서 답을
생성하지 않는 증상을 재현한 것이며 전체 초기 정답률이나 정답률 회복의 측정은 아니다.
[원시 토큰·대조 채점](evidence/gemma4-local-preflight-diagnosis-2026-10-21.json)을 보존한다.
기존 진단 모듈의 `--score`는 task verifier 결과를 출력하고 `--stream`은 한 응답의
생성 텍스트를 즉시 표시한다. `--solution-cue`는 별도 진단 출력 안에서만 풀이 시작
문구를 비교하며 실험 입력·프로토콜을 변경하지 않는다.

풀이 시작 문구를 붙인 실제 로컬 12B MATH 응답은 풀이와 정답 6을 생성했고 같은
verifier에서 1로 채점됐다. 원래 입력의 0에서 바뀐 한 문제의 진단 결과이며, 전체 초기
정답률을 측정한 결과는 아니다. MBPP는 단순 들여쓰기 지시만으로는 같은 문제에서
계속 실패했으나, 일반적인 Python 함수 형식 예제를 제공한 입력에서는 올바르게
들여쓴 코드를 생성했고 실제 세 assertion을 통과해 보상 1을 받았다. 배포한 입력
생성 함수와 이 측정에 사용한 입력이 바이트 단위로 같은지 확인했다. 생성 코드를
사후에 수정하지 않았다. 이는 과제별 한 문제의 진단이며, 전체 8×8 preflight나
400×8 초기 정답률을 통과·회복했다고 보고하지 않는다.
[원시 응답·토큰](evidence/gemma4-pt-completion-diagnosis-2026-10-21.json)을 보존한다.

최종 코드의 관련 검사 **323개가 통과**했다. 새 입력 형식·구버전 혼합 거부·일반 시작
명령·중지·모델 연결·결과 출력·25-update checkpoint/보상 기록과 원래 모델 로더를
검사했다. 기존 fork 관련 경고 6건은 남아 있다. 별도 실행 파일과 전환 명령은 제거했다.

## 실제 모델 검사와 한계

공식 revision `023679ed352de9bb66cc873c9009ce3482585c08`의 23,919,549,408-byte
가중치 SHA256을 다시 확인했다. 수정된 로더로 같은 MATH 후보에서 temperature=1,
8응답을 실제 생성했고 이미지·오디오 억제 토큰은 **0개**였다.
[원시 토큰과 응답](evidence/gemma4-text-recovery-short-sampling-2026-10-21.json)을 보존했다.

이 검사는 RTX 3050에서 BF16 CPU weight offload를 사용한 **16-token 짧은 검사**다.
8개 중 빈 응답은 1개였고, 일부 응답은 지시문을 이어 썼다. 이 결과를 정상 init 정답률이나
정답률 회복의 증거로 보고하지 않는다. 원격 H100의 전체 cache 재생성·학습은 아직
실행하지 않았다. 새 runner의 실제 문제 검사와 전체 init을 확인해야 한다.

Qwen·Llama 로더에서는 Gemma처럼 생성 설정 전체를 재생성해 억제를 제거하는 코드는
없었다. 두 모델의 기존 결과·학습본을 바꾸거나 새 점수를 추정하지 않는다.
MBPP fence 문제의 기존 결과에 대한 실제 영향은 원시 응답을 확인해야 한다.
V7 원고에 Gemma 결과를 추가하거나 수치를 바꾸지 않는다.

검증 로그와 원시 로컬 모델 결과는
`/home/kms/reports/gemma-recovery-fix-2026-10-21/`에 보존한다.

## 코드 검증

- Qwen·Llama·Gemma와 공통 실험 코드: **2,007 tests, 325 subtests 통과**.
  최종 CPU suite는 454.38초이며, CPU 배정 fixture에서 CUDA를 숨겼다.
- 마지막 오류 표시 보강을 포함한 복구·모델 프로필 검사: **29개 통과**.
  일반 디스크에서 실제 소형 Gemma 생성·masked log probability/gradient,
  GRPO·AdamW 업데이트 및 정확한 checkpoint 복원, 실제 MBPP 자식 채점,
  캐시·버전 혼합 거부, 두 과제 preflight 연결, JSON·터미널 결과를 확인했다.
  위 전체 suite와 중복되는 검사이므로 합산하지 않는다.
- 25–275 매 25-update 평가용 checkpoint와 학습 보상/held-out 보상의 분리·export를
  제어된 CPU fixture로 검증했다. 이를 원격 실험의 실제 측정값으로 사용하지 않는다.
- 기존 Gemma·Llama·공통 engine의 Python 55개는 수정 전 SHA256과 동일하다.
  Ruff, 변경 Python 파싱, shell 구문, diff 검사를 통과했다.
- 실제 사전학습 12B 검사는 위 짧은 CUDA 생성 검사이며, H100/NCCL 실측과 구분한다.

## 시작·중지 경로 재점검

아래는 앞선 text-v2 운영 경로 재점검 기록이다. 당시 운영 수정은 생성·채점·학습
프로토콜을 바꾸지 않았다. 현재 text-v3 입력 변경에 따른 새 캐시·학습 요건은 서두에
따른다.

- 시작 함수가 재개 배정기 설치보다 먼저 연결되어, 모델 연결 실행에서 MATH를 끝내고도
  아직 실행할 수 없는 MBPP를 기다릴 수 있었다. 공개 명령은 이제 별도 운영 진입점
  `srgc_research.dispatch.gemma_text_run`에서 배정기를 먼저 설치한다. MATH가 완료되고
  MBPP가 남은 조건에서 기존 진입점의 불필요한 GPU 요청을 재현하고, 새 진입점이
  GPU 요청 없이 해당 단계를 종료하는지 검사했다.
- 연결 실행의 사전 점검은 현재 데이터셋만 검사한다. MBPP 단계에서도 첫 번째 MATH
  plan으로 smoke를 실행하거나 두 데이터셋의 후보 검사를 다시 요청하던 경로를 막는다.
  두 과제를 직접 요청한 `all preflight`의 검사 범위는 유지한다.
- 별도 `preflight` 실행도 실제 중지 요청 파일을 확인한다. `--now`를 자식 실행 중
  감시하고, 취소된 검사는 실패와 구분해 저장한다. 이미 통과한 smoke와 시간은 보존한다.
- `status`, `stop --now`, `all stop --now`는 모델용 Python이 없는 조건에서도 시스템
  Python으로 실행한다. 새 진입점의 살아 있는 자식은 고아 프로세스 정리에서 보호한다.

운영 진입점은 frozen adapter 바깥에 있다. 재점검 로그와 과학 코드 58개의 SHA256 보존
확인은 `/home/kms/reports/gemma-recheck-2026-10-21/`에 보관한다. 이 검사는 실행 제어의
회귀 검사이며 원격 H100의 전체 init 정답률을 새로 측정한 결과가 아니다.

최종 코드의 전체 CPU 검사에서 **2,023 tests, 325 subtests가 통과**했다(431.51초).
기존 fork 관련 경고 67건이 남았으며 테스트 실패는 없다. 일반 디스크에서 별도 실행한
운영 경로 검사 11개도 통과했다. 전체 검사와 겹치므로 합산하지 않는다.
사전 점검의 GPU 실행 설정이 실제 작업 환경에 전달되는지, 오류 안내가 백업 정리 뒤
한 번 출력되는지까지 검사했다. 변경한 소스 8개의 해시는 전체 검사 시작·종료 시 같다.
Ruff, shell 구문과 diff 검사도 통과했다.
