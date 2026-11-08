# V8: 한 명령으로 OLMo·Qwen 노드 자동 배정

모든 4×H100 노드에서 같은 명령을 실행한다.

```sh
sh scripts/run_v8_nodes.sh
```

별도 모델명이나 노드별 역할을 지정하지 않는다. 한 worker가 OLMo MATH와 Qwen
MBPP 양쪽의 준비된 작업을 확인하고, 한 번에4GPU 작업 하나를 실행한다. 각 cohort의
원본 Runtime 초기화·노드 검증·CPU 준비·작업 목록 검사는 독립적으로 진행한다. Qwen의 준비가 끝나지 않아도
검증·준비가 끝난 OLMo는 시작할 수 있고, 반대 방향도 같다. 공유 잠금 안에서 실제 점유 중인 작업 lease를
세고 적게 배정된 cohort부터 새 작업을 가져온다. 기존 Qwen 작업10개가 실행 중이면
추가 worker는 준비된 OLMo 작업을 우선한다. 한쪽이 체크포인트를 기다리거나 완료되면
다른 쪽의 준비된 작업을 가져온다. 공유 잠금은 배정 판단·작업 claim 동안만 잡으며,
원본 Runtime의 작업 목록·체크포인트·완료 여부 검사와 모델 로딩·학습·준비 중에는 잡지 않는다. 모델별 고정 노드 할당이나 GPU 분할은
하지 않는다.

## 기존 환경 자동 사용

기존 코드 저장소에서는 위 명령 한 줄을 실행한다. 오프라인 배포 묶음에서는
최상위의 `sh run.sh` 한 줄을 실행한다. 새로 export하거나 경로 인자를 붙이지 않는다.

새 bootstrap은 기존 `scripts/run_srgc.sh`, `scripts/run_srgc_qwen35.sh`의 규칙을
사용한다. 기존 `OM_WORK`/`GROUP_VOLUME`/`OM_USER`를 우선 사용하며, 미지정 기본은
`/group-volume/minsoo3.kim/offpolicy-misranking`이다. 기존 `.venv-cu126/bin/python`을
재사용하고 `PAIR_PYTHON`/`SWITCH_PYTHON` 등 기존 override도 보존한다.

원래 trainer는 기존 `V8_CODE_ROOT`/`OM_CODE_ROOT`, 저장된 locator의 원본 코드 경로,
현재 저장소·상위 저장소 및 기존 user-volume 위치에서 찾는다. 기존 run의 구현 해시와
prefix 해시는 원래 검증 경로에서 확인한다. runtime은 이제 이 코드 저장소의 `v8/experiments`에 함께 포함된다. 그 사본을
우선 사용하며 별도 원고 저장소나 수동 runtime 경로를 요구하지 않는다. source는 paper commit
`3d0c08336950d2697dc66cbe3570a230e449f33b` 그대로다.

공유 결과 경로는 기존 `V8_ROOT`, 또는 같은 작업 폴더의 `v8/`를 쓴다. exporter는
자기 위치를 사용한다. 원래 경로·checkpoint·venv를 옮기거나 다시 만들지 않는다.
명시된 Python이 깨졌거나 기존 입력 신원이 다르면 다른 실험으로 조용히 바꾸지 않고
실패 이유를 출력한다.

선택된 원본 `V8_CODE_ROOT`와 그 `src/`는 캐시의 actor 프로세스와 실제 SRGC 자식의
Python import 경로에 자동으로 들어간다. 저장소 밖에서 실행해도 같은 원본 패키지를
불러오며, 기존 사용자 PYTHONPATH 항목은 뒤에 보존한다.

GitHub 연결이 없는 기존 클러스터에는 소스·최신 실행기·고정 v8 runtime을 포함한
오프라인 묶음을 사용한다. 노드에서 git fetch, package 설치, 모델 재학습용 초기화,
수동 경로 입력은 요구하지 않는다. 모든 노드는 같은 묶음 버전을 사용한다.

## 실행 순서를 반영한 병렬 수

**110노드를 실제 실행의 최대 동시 노드 수나 필요한 노드 수로 안내한 답변은
철회한다.**55+55는 부모 체크포인트가 모두 준비됐다는 정적 학습 그래프의 폭이다.
실제 실행에서는 부모가 진행하는 동안 앞선 분기가 끝나므로 그 수를 노드 필요량으로
바로 환산할 수 없다.

모든 seed25 prefix가 준비된 첫 시점의 작업은 OLMo10개·Qwen10개, 합계20개다.
이후 준비된 작업 수는 부모 checkpoint 생성 속도와 기존 분기의 완료 속도에 따라
변한다. 실행기는 준비된 작업만 노드에 배정한다.

기존 cohort 설정에 기록된 평균GPU시간을 사용하고, 모든 seed가 같은 속도로
진행하며 준비된 SR분기를 즉시 시작한다고 가정한 **주 학습 구간만의 근사 추정**은
다음과 같다. 작업 하나는4GPU다.

| 계산 범위 | 추정 동시4GPU 작업 피크 |
| --- | ---: |
| OLMo의 On trajectory와 SR분기 | 40 |
| Qwen의 On trajectory와 SR분기 | 30 |
| 두 실험을 같은 시각에 시작한 위 구간의 합계 피크 | 55 |

각 cohort의 피크 시각이 달라 합계 피크를40+30으로 계산하지 않는다. 이 계산은
model초기화·I/O·checkpoint 비용과 실제 seed별 속도 차이를 제외한다. Uniform분기,
예산 연장, 최종 평가도 포함하지 않는다. 특히 OLMo 평가 시간은 기존 설정에 없다.
따라서 **55 역시 전체 파이프라인 최대치나 확보할 노드 수가 아니다.** 실제 전체
동시 실행 최대치와 노드별 완료 시간은 아직 측정하지 않았다.

근사 계산의 입력·가정·전체 event timeline은
`research/v8-concurrency-correction-20261108/`에 보존한다. 새로운 GPU 측정이나
속도 향상 결과로 사용하지 않는다.

## 저장된 원본 런타임 선택

기존 saved-prefix 런처의 원본 package 검증을 재사용한다. OLMo의 알려진
보존 런타임 두 버전은 전체 보존 package·engine 바이트를 검증한 뒤 자식이
모델을 만들기 전에 실제 원본 모듈을 불러온다. 현재 엔진을 옛 해시로 바꾸어
표시하거나 다른 체크포인트를 새 결과로 대체하지 않는다.

Qwen은 기존 코드가 기록한 `sha256(engine_digest + adapter_digest)`를 그대로
검증한다. 기본 엔진 해시만 비교하지 않는다. 각 seed의 run.json과
prefix-ready.json도 같은 원본 신원이어야 한다. 모르는 원본 해시나 모델 revision
차이는 그대로 거부하고 원인별로 출력한다. prefix가 없으면 코드 경로 대신
missing_prefix로 표시한다.

수정 전 준비 단계에서 실패했고 아직 작업/입력 고정 기록이 하나도 없는 큐만,
동일한 trainer/config 환경에서 새 검증 코드로 준비를 다시 시도한다. 이전
binding·실패 기록은 setup-history에 보존한다. 학습이 시작된 큐는 이 방식으로
바꾸지 않는다.

## 체크포인트 재사용과 단계

범위는 기존 seeds5–9의 rebuttal Branch Bank다. OLMo MATH는 주 실험, Qwen
MBPP는 다른 모델·과제의 재현 실험이다. 두 cohort로 모델 효과만 분리하지 않는다.
새 seed의 개발·확증 실험과 E8 양방향 전환은 포함하지 않는다.

두 실험 모두 기존 `prefix-ready.json`과 `prefix.pt`를 찾아25-update 공통 상태에서
분기한다. checkpoint 해시·step·trainer 구현을 확인하고 모델·optimizer 상태를
불러온다. 저장된 process RNG가 있으면 복원한다. 같은 v8 작업은 `latest.pt`에서
재개한다. 다른 방법의 기존275-update 결과를 새 v8 작업 완료로 자동 대체하지 않는다.

각 cohort는 독립적으로 준비·입력 검증 → trajectory와 checkpoint가 준비된 분기 →
모든 학습 완료 확인 → 비용만으로 예산 고정 → 예산 연장 → 최종 평가 → 수집·분석
순서를 지킨다. 다른 cohort의 단계와 예산을 섞지 않는다. 한쪽의 GPU 작업 실패는
기록하고 그 cohort의 새 배정을 멈춘다. 정상 cohort의 남은 작업을 마친 뒤 전체
worker는 실패 종료 코드를 반환한다. 환경·입력 사전검사 오류는 가려서 진행하지 않는다.

기존 trainer와 방법·optimizer·seed·난수 설정은 그대로다. 학습 배치 reward와
25-update checkpoint를 보존하고 주기적 held-out 평가는 강제하지 않는다. 최종 평가는
별도이며, 측정하지 않은 중간 held-out 값을 학습 reward로 대신 채우지 않는다.

평가는 seed별 bundle과 실제 선택한 checkpoint의 run/update/hash를 사용한다.
예산 endpoint가 연장 run이나 전환 전 trajectory에 있으면 그 위치에서 평가한다.
선택한 checkpoint가 없거나 달라졌으면 실패로 기록한다.

## 현황과 재개

```sh
sh scripts/run_v8_nodes.sh --status
```

양쪽의 `phase`, `known_task_counts`, `leased_tasks`, `failure`가 함께 출력된다.
worker는60초마다 현황을 출력한다. 자식 출력은 터미널과 각 cohort 로그에 남긴다.
노드마다 같은 GPU 그룹에서 worker 하나를 실행한다.

`--status`의 `dispatch_workers`에는 각 worker가 실제 기록한 제어 코드 해시와
정책 버전이 나온다. 표 형식 `status`는 `dispatch current <hash>`(현재 checkout과
같음), `dispatch different <hash>`(다름), `dispatch unverified`(기록 없음)를 구분한다.
정책 파일이 디스크에 있다는 것만으로 실행 중인 worker에 적용됐다고 판단하지 않는다.
현재 배정 정책은 `source-recovery-verified-retry-budget-v12`이다. 작업 검사는 해당 cohort의
control lease만 잡고 별도로 수행한다. 검사 결과가 준비되면 공유 잠금 안에서 현재
binding·phase·실패 상태·작업 정의·작업 lease를 다시 확인한 뒤 claim한다.
검사와 claim 사이에 다른 노드가 가져간 작업이나 실패한 작업은 가져오지 않는다.
노드 입력 검증도 다른 cohort의 시작을 막지 않으며 검증을 통과한 노드만 claim할 수 있다.
원본 Runtime 초기화에는120초 제한을 두고, 한 cohort가 초기화에 응답하지 않으면
오류를 표시하고 다른 cohort를 계속한다. 학습 실행 시간에는 이 초기화 제한을 적용하지 않는다.

개별 작업 실패는 그 작업에만 남긴다. 새 노드에서 기본 명령을 실행하면, 노드 입력
검증 후 이전 제어 코드에서 실패한 작업을 현재 수정본당 한 번 재개한다. 이전 버전에
제어 코드 기록이 없던 실패도 한 번만 재개한다. 별도 재시도 옵션은 필요 없다.
원래 실패 시도와 체크포인트는 보존하고 복구 사유·제어 코드·worker를 별도로 기록한다.
같은 수정본에서 실패해도 기본 명령이60초 후 자동 재시도한다. 실패 한도 기본값은
검증된 수정본의 복구 시점 이후3회이며 중단75·노드 자체의 실행 실패는 제외한다.
원본 입력·binding을 검증하고 이전 실행기 실패를 복구할 때 `retry_epoch` 경계를 남긴다.
그 전 실패는 모두 보존하되 현재 한도에는 합산하지 않는다. 같은 정책의 새 노드나
status만 달라진 코드 해시는 경계를 새로 만들지 않는다. 현황판도 같은 계산 함수를
사용해 현재 한도와 역사상 실패 합계를 구분한다.60초 대기는 마지막 시도의 종료
시각을 기준으로 한다. 예전 노드의 더 큰 시각이 현재 재시도를 연장하지 않는다. 재시도 대기 중에도
다른 seed와 준비된 분기는 계속 배정한다. 재개는 원래 `latest.pt`가 있으면 그곳에서,
없으면 검증된 부모 checkpoint에서 한다. 실패 시도·optimizer·RNG·학습 보상은 보존한다.
상한에 도달한 실패가 남아 있고 독립적으로 실행할 작업도 없으면 실패로 종료하며
freeze·평가·완료로 넘어가지 않는다. 이전 V8가
개별 `task exit=...` 실패를 cohort 전체 중단으로 저장한 경우, 같은 failed job이
존재할 때만 그 중단을 작업 범위로 옮긴다. 원래 실패는 history에 보존하고 작업·시도·
binding·checkpoint를 고치거나 지우지 않는다. 신원·설정·CPU 준비 오류는 이 방식으로
해제하지 않는다. 입력·설정 검증을 우회하지 않으며 살아 있는 task lease는 건드리지 않는다.
`--retry-failed`는 기존 명시적 복구용 호환 옵션이며 일반 실행에 필요 없다.
자동 복구는 작업 실패에만 적용하며 입력·신원·CPU 준비 오류를 해제하지 않는다.

재시도 상한에 걸린 failed 작업도 원래 실행 잠금과 완료 출력은 먼저 검증한다.
동일 run/config와 endpoint가 정상 완료됐으면 재학습 없이 done으로 인수하며 실패
시도 원문은 보존한다. 원래 실행 잠금이 잡혀 있으면 작업을 중복 실행하지 않고 기다린다.
유효한 완료·진행·재시도 작업을 확인한 경우에만 예전의 집계 실패를 이력으로 옮긴다.
입력·설정 오류를 자동 해제하거나 완료 표지만 보고 검증을 생략하지 않는다.

이미 대기 중인 worker도 다른 검증된 노드가 공유 실패를 복구했는지 다시 확인한다.
기존에는 MATH의 실패를 한 번 본 worker가 그 판단을 영구 보관해, 공유 대기열이
정상으로 돌아와도 MATH를 배정하지 않았다. 복구를 확인하면 같은 binding인지 확인하고
원본 입력을 재검증한 뒤 배정에 복귀한다. 이 확인 자체가 실패를 지우거나 작업을
재시도하지는 않는다. 해당 노드의 원본 코드·입력·권한 오류는 자동 복귀하지 않는다.

기본 다섯 seed는 각각 prefix25에서 On과 SR25를 시작할 수 있으므로 처음에 최대
10개의 독립 MATH 작업이 있다. 어느 On run이50 checkpoint를 보존하면 해당 seed의
SR50도 즉시 배정할 수 있다. 모든 seed나 On run의275 완료까지 기다리지 않는다.
MBPP와 MATH의 실행 개수 차이만으로 모델별 상한을 뜻하지 않는다. 각 대기열의
완료·실패·점유 및 실제 checkpoint 준비 상태를 확인해야 한다.

GPU 자식의 시작·중단·정체 감시는 기존 `srgc_rebuttal.cluster.run_child()`를 사용한다.
옛 child가 strict MATH verifier를 선택하면 Context 구성 뒤 기존 seed 실행의
`srgc_verifier_fallback.py`를 연결한다. 읽히는 gold는 math-verify를 쓰고, 읽히지 않는
gold는 기존 정규화·마지막 `Answer:` 일치 방식으로 판정한다. 빈 gold는 입력 오류로
남기며 사용자 정의·MBPP verifier는 바꾸지 않는다. 적용 시 실제 reward runtime 변경과
verifier identity를 `runtime-repair-rank-N.json`에 기록한다. 기존 원본 코드·입력·이미
기록된 보상은 보존하고, 검증 helper도 실행기와 함께 고정해 이후 git pull의 영향을 막는다.
옛 backend에는 rollout 뒤 캐시 해제와 OOM 시 동일 prompt·응답 수·seed로 한 번 재시도를
연결한다. 기존 보호가 있는 backend는 학습의 rollout에도 같은 보호를 적용한다.
optimizer update 전체를 다시 호출하지 않는다. 계속되는 OOM과 다른 CUDA RuntimeError는
실패로 남긴다. CUDA 오류의 원문·마지막 작업 기록·확인 가능한 allocator 통계는
`cuda-failure-rank-N.json`에 보존한다. 이 처리가 실제 H100 메모리 용량이나 아직 원문이
없는 `RegisterCUDA … SymIntArrayRef` 오류의 해결을 보장하지는 않는다.

V9 캐시 정리만으로는 살아 있는 GPU 텐서를 회수하지 못했다. V10은 SHA256이
`c73ee5e6…`인 보존 backend에서만 두 수명 수정을 적용한다. 문제별 scoring을 마친 뒤
`accumulated`·마지막 `gradient`·`value`·`sequences` 참조를 해제하고, optimizer snapshot은
GPU에서 deepcopy하지 않고 CPU로 직접 복사한다. 보존 파일은 수정하지 않으며 적용한
scorer overlay SHA와 원본 backend SHA를 위 rank receipt의 `dense_memory`에 남긴다.
연산 순서·rank별 문제 배분·한 번의 gather·기존 timing wrapper를 유지한다. 응답 수나
길이·배치·attention을 바꾸지 않는다. CPU 재현과 수치/RNG 일치는 실제 H100 정상화의
증거가 아니며, 실제 오류 위치와 재개 update는 별도로 확인해야 한다.

V11은 OLMo의 decoder activation checkpointing 누락도 보완한다. Qwen 경로는 이미
이 보호를 사용하지만 OLMo는 마지막 logits chunk만 checkpoint하고 decoder의 역전파
활성값은 보관했다. OLMo는 dropout0을 검증하고 logprob forward에서만 training mode와
non-reentrant checkpointing을 켠 뒤 기존 mode를 복원한다. 생성은 기존8응답 묶음과
토큰 한도·seed·attention을 유지하고, 지원하는 모델에서는 마지막 위치 logits만
계산한다. 이때도 생성 microbatch를 나누지 않는다. 기존 Qwen 경로는 변경하지 않는다.
작은 실제 OLMo CPU 모델의 출력·gradient·생성토큰·RNG 일치 및 저장 활성값 감소를
검사했다. 이 값은 실제7B H100의 peak 메모리 측정이 아니다. 실제 OOM 원인 확정과
SR25 update 재개는 별도 확인이 필요하며 다음 CUDA 오류에는 실패 stage도 보존한다.

`answering 8/9`는 가장 최근 기록을 보낸 GPU가 이번 호출에서 맡은 9문제 중 8번째
문제의 답변을 생성 중이라는 뜻이다. 8개 update 완료나 전체 실험의 8/9 완료가 아니다.
문제마다 답변 생성 뒤 보상 계산과 필요한 gradient 계산을 수행하고, 해당 작업의
문제를 다 처리한 뒤 다음 단계로 간다. 다른 노드는 독립적인 준비된 작업을 실행한다.
부모 체크포인트가 아직 없거나 부모가 실패한 분기는 기다리므로, 준비된 작업보다
노드가 많으면 idle이 생길 수 있다. 실패로 인한 대기와 정상 진행을 구분해야 한다.

기본 두 cohort는 기존 OLMo-3 7B/MATH(primary)와 Qwen3.5 9B/MBPP(replication)를
이어받는다. 비교는 각 cohort 안의 학습 방법 사이에서 수행한다. 모델과 데이터가
동시에 다르므로 이 둘의 점수 차이만으로 모델 효과나 데이터 효과를 분리할 수 없다.
OLMo/MBPP·Qwen/MATH 설정도 있지만 기본 자동 배정은 이 교차 조합을 실행하지 않는다.

자식이 점유 중이거나 중단되어 종료75를 기록하면 torchrun의 종료1을 작업 실패로
오인하지 않는다. 실제 rank 예외·치명적 signal이 함께 있으면 실패 기록을 우선한다.
`ChildFailedError` 앞의 원래 traceback은 PyTorch 오류 기록기로 보존한다. 원래 입력
검사의 `SystemExit`도 보존하며, 실패 뒤 `RANK FAILURE`에 원문 오류와 경로를 출력한다.
각 `.multi-node/progress/<attempt>/rank-N-error.json`과 `child-failure.json`에 남기고,
작업 attempt와 `status`에도 실제 rank 오류를 표시한다. 강제 종료 signal에는 Python
traceback이 없으므로 관찰한 signal만 기록한다. 이전 실행의 marker나 오류 파일을
현재 실패에 사용하지 않는다.
모델 준비 때부터 rank별 작업 파일의 실제 변경을 확인하고1800초 동안 변경이 없으면
해당 작업을 timeout으로 기록한다. worker heartbeat와 실제 모델 작업은 구분한다.
아직 rank 작업 기록이 없으면 supervisor는 `ASSIGNED`/`NOT ASSIGNED`/`BLOCKED`로
배정·큐 실패 상태, 모델·seed·방법, 실제 작업 로그와 마지막 launcher 출력을 보여 준다.
기록이 없다는 이유만으로 정상 준비 중이라고 표시하지 않는다. 다른 launch의 작업이나
이름이 비슷한 다른 FQDN 노드의 실행을 이 launch의 GPU 작업으로 사용하지 않는다.

Ctrl-C/SIGTERM은 자식 실행기에 전달된다. 종료 유예는10초이며 강제 종료되면 마지막으로
저장된 체크포인트에서 같은 명령으로 재개한다. 종료 직전 업데이트의 저장은 보장하지 않는다.
수정본을 받은 새 노드는 맨 위의 기본 명령을 그대로 실행한다.
기존 시도·실패를 지우지 않고, 시간 추정만으로 lease를 빼앗지 않는다.

큐·시도·로그는 `$V8_ROOT/<cohort>/.multi-node/`, 학습 결과는 각 `runs/`에 남는다.
각 cohort의 마지막 수집 뒤 기존 history exporter도 실행한다.

## 노드 실행기 점검 결과 (11-08 Claude)

- GPU 점유 잠금은 기존 SRGC worker와 같은 `$OM_WORK/srgc-rebuttal/gpu-node-locks/`의
  GPU UUID 파일(sha256)을 쓴다. 기존 SRGC worker가 같은 GPU를 잡고 있으면 V8 worker는
  시작하지 않고 그 이유를 출력한다. 이전 `/tmp/srgc-v8-gpu-locks`는 다른 사용자가 만든
  폴더나 쓰기 금지 `/tmp`에서 `Permission denied`로 멈췄다.
- 할당된 GPU 중 하나라도 4,000 MiB 넘게 쓰고 있으면 새 GPU 작업을 받지 않고 5분마다
  해당 GPU와 프로세스 번호를 출력한다. nvidia-smi가 4개 GPU를 보여 주지 않는 노드는
  큐를 건드리기 전에 거절한다.
- 한 cohort의 준비 단계나 작업이 실패해도 그 cohort만 멈추고 다른 cohort는 계속한다.
  끝에 실패한 cohort와 이유를 다시 출력하고 종료 코드 1을 반환한다.
- 큐 기록과 코드가 다르면 무엇이 다른지와 할 일을 출력한다. 다른 노드가 큐를 새 코드로
  바꾸면 이전 코드로 돌던 worker는 그 cohort 작업을 더 받지 않는다.
- 다른 묶음으로 시작한 큐는 일반 실행에서 자동으로 바꾸지 않는다. 아직 update가 하나도
  끝나지 않았다면 모든 노드의 worker를 멈춘 뒤 한 노드에서만
  `sh scripts/run_v8_nodes.sh --new-queue`를 한 번 실행한다. 기존 큐는 지우지 않고
  `.multi-node.superseded-<시각>`으로 그대로 보관하고 새 큐를 만든다. 완료된 update가 있거나
  작업 lease가 잡혀 있으면 거부한다. 나머지 노드는 같은 묶음으로 일반 명령을 실행한다.
- cohort별 Python은 기존 실행기와 같은 순서로 고른다. OLMo MATH는 `PAIR_PYTHON`,
  Qwen MBPP는 `QWEN_PYTHON` 다음 `SWITCH_PYTHON`, 없으면 기존 `.venv-cu126`이다.
  OLMo가 MBPP/Qwen 인터프리터를 물려받던 문제를 고쳤다.
- `--status`의 `leased_tasks`에 작업 로그 경로와 `log_written_seconds_ago`(마지막 출력 뒤
  지난 초)가 나온다. audit 단계는 공백이 들어간 경로도 그대로 넘기고, `harvest.sh`는
  worker가 로그를 쓰는 중에도 `tar`의 "file changed" 경고로 실패하지 않는다.

## 검증 범위

기존 단일 cohort 검사와 추가 자동 배정 검사를 함께 실행했다. 실제 v8 CPU
ToyBackend의 작업자3개가 두 cohort를 모두 처리하고, 동일 worker의 GPU 작업이
겹치지 않으며 모든 예산 endpoint를 평가하는지 확인한다. 노드 역할 고정 없이
cohort를 오가고 한 cohort의 대기·작업 실패가 정상 cohort 작업을 막지 않는지도 검사한다.
현재5seed/10SR 분기 설정의55개 준비된 학습 작업 fixture는 정적 그래프 검사이며 실제 시간에 따른 동시 실행 최대치 검증이 아니다.
원래 trainer Python30파일의 해시는 그대로다.

실제 다중 H100 처리량과 네트워크 파일시스템의 노드 간 잠금은 아직 측정하지 않았다.
같은 공유 파일시스템에서 POSIX flock이 노드 사이에 동작해야 한다. 노드 구매·할당이나
GPU 작업의 자동 제출은 수행하지 않는다.

검증 기록: `research/v8-auto-dispatch-20261108/`. 초기 분리 실행과 예산 endpoint
수정 이력은 `research/v8-multinode-20261108/`, `research/v8-budget-endpoint-20261108/`에 보존한다.

## 로딩 표시와 종료

기존 큐를 재개할 때는 `binding.json`의 coordinator/runner 해시와 Python·package 버전을
먼저 읽는다. 현재 코드와 다르면 오프라인 묶음의 `v8/queue-runtimes.zip`에서 원래 코드를
자동 선택한다. 설치나 GitHub 접속은 필요 없으며 큐·학습 결과·control.lock을 바꾸지 않는다.
특정 과거 버전이라고 가정해 배포한 단일 호환 묶음은 이 자동 선택 묶음으로 대체한다.
원본 코드가 묶음에 없거나 기존 Python 환경이 달라졌으면 해당 원인을 출력하고 거부한다.
신원 검사를 끄거나 새 코드에 옛 해시를 붙여서 통과시키지 않는다.
기존 큐에도 현재의 공통 배정기를 적용한다. 각 cohort의 원본 Runtime은 그 큐에 기록된
코드·Python으로 별도 자식 프로세스에서 실행하고, 새 배정기는 두 cohort의 실제 lease를
함께 비교한다. 원래 binding.json을 고치지 않는다. 새 제어 정책과 배정기4파일 해시,
원본 binding은 `$V8_ROOT/.v8-dispatch-policies/`의 별도 불변 기록에 남긴다.
프로세스 간 통신은 상속받은 로컬 socketpair이며 네트워크 listener를 만들지 않는다.
현황 확인 명령은 원본 코드 전환 없이 읽기만 수행한다.

새 제어 정책은 원본 `cluster_child.py`의 해시를 검증한 뒤 별도 `v8_task_entry.py`로
같은 child의 `main()`을 호출한다. 원본 Context가 backend를 만든 뒤 샘플링 인자와
기존 종료 조건을 보존한 always-false 관찰 조건을 추가한다. 5초마다 실제 증가한
디코딩 길이·호출 번호·반환된 decode step을 기록한다. `generation_start`는 진입 기록이고
생성 완료나 optimizer update가 아니다. `generation_finished`도 생성 함수 반환만 뜻한다.
생성 호출 안에서30분간 디코딩 단계가 늘지 않으면 해당 rank가 스택 기록을 시도하고
exit70으로 실패한다. 기존 checkpoint는 남기며, 성공으로 처리하거나 자동 재학습하지 않는다.
이 제한은 전체 학습 시간이나 selection refresh 총시간의 제한이 아니다. 로그 경로가
깨졌거나 콘솔 쓰기가 막혀도 감시 종료가 그 쓰기를 무한 대기하지 않는다.
CPU 작은 무작위 OLMo 모델에서 응답8개·기존 종료조건·출력 바이트·난수 상태가
원래 generate와 같은지 검사했다. 실제 H100 생성 정체의 원인과 해결 여부는 별도 확인이 필요하다.

콘솔은 단일 단계 키 대신 작업 목적과 실제 진행량을 출력한다. 모델·과제, 학습 문항
선택을 위한 후보/기준 문항 점수 계산인지 실제 optimizer update인지, 해당 GPU의
현재/전체 문항 수와 답변 채점·기울기 계산 완료 수를 구분한다. 새 토큰 진행은 입력을
제외한 현재 응답 길이와 최대 생성 길이로 표시하고 마지막 실제 진행 이후 시간을
보여 준다. `optimizer updates completed in this process`는 해당 자식 프로세스에서
실제로 반환된 policy_update 기록의 수이며, 기존 checkpoint의 update나 저장 완료를
대신하지 않는다. 전체 실험 완료율이나 예상 남은 시간을 추정해서 채우지 않는다.

현재 checkout을 선택한 경우도 실행 전 별도의 불변 사본으로 고정한다. 실행 도중
`git pull`이 되어도 이 worker가 이후 띄우는 child/config/helper는 바뀌지 않는다.
이미 사용한 사본도 로컬 캐시에 남겨 후속 pull 뒤 같은 큐를 다시 열 때 재사용한다.
두 cohort의 Python 경로가 달라도 공통 배정기를 사용한다. 각 원본 Runtime의
NCCL 사전검사와 학습은 해당 cohort의 Python에서 실행한다.
한 cohort의 원본/Python/메타데이터 오류는 그 cohort에만 표시하며 유효한 나머지는 계속한다.
일부 실패를 숨기지 않도록 유효한 cohort가 끝난 뒤 전체 종료 코드는 실패로 남긴다.

`git pull`은 이미 실행 중인 worker의 메모리나 고정 실행 사본을 바꾸지 않는다.
수정 이후 시작한 worker부터 공통 배정기를 사용한다. 실행 중인 Qwen 학습을
멈추거나 큐를 초기화할 필요 없이 유휴 노드의 새 worker가 OLMo를 가져올 수 있다.
기존 worker를 나중에 재실행해도 같은 큐와 저장된 checkpoint에서 재개한다.
25-update snapshot은 보존하고 `latest.pt`만 재개용 최신 상태로 갱신한다.

콘솔 출력을 받는 쪽이 멈춰도 학습·종료가 대기하지 않도록 supervisor는 출력을 로컬
임시 폴더의 `v8-console-<uid>/<launch>.log`에 보존하고 별도 thread로 전달한다.
종료 경로는 콘솔 쓰기나 interpreter 최종 flush를 무한 대기하지 않는다. 기존 tag 없는
worker의 종료 중 새로 생긴 rank도 해당 노드·UID·V8_ROOT 범위에서 다시 찾아 정리한다.
상태 화면의 GPU 유휴 기반 경고는 같은 hostname의 작업에만 적용한다. 다른 노드의
GPU 상태를 이 노드의 수치로 추정하지 않으며 FQDN을 생략하지 않는다.

`Loading weights` 출력만으로 학습 정지를 판단하지 않는다. 현재 상태를 바꾸지 않고
GPU 사용과 이 노드의 현재 attempt 기록·로그를 확인할 때:

```sh
sh scripts/v8_probe.sh
```

현재 노드의 해당 V8 실행을 종료할 때:

```sh
sh scripts/v8_stop.sh
```

원래 checkpoint/queue/lock 파일은 지우지 않는다. 종료 신호 뒤10초 동안 기다리고
남아 있는 해당 실행의 프로세스만 강제 정리한다. 다른 UID·root·launch 또는 재사용된
PID는 신호 대상에서 제외한다. kernel의 중단 불가능한 I/O 대기는 남은 PID로 출력하며
GPU가 해제됐다고 성공 처리하지 않는다.

새 기본 실행기는 별도 supervisor를 사용한다. Ctrl-C/TERM/HUP을 처리하고 두 번째
Ctrl-C는 즉시 정리를 요청한다. 자식 출력 pipe의 EOF를 기다리는 무한 join을 사용하지
않는다.15초마다 rank가 실제 기록한 rollout/gradient_scoring/policy_update 단계와
기록 나이를 출력한다. 이 출력은 완료 update나 checkpoint 저장 확인과 구별한다.

종료 supervisor 자체는 wrapper·진단·종료 경로만 바꾸며 기존 큐의 identity를 바꾸지 않는다.
현재 master에는 별도의 child 실행 환경·NCCL 사전 검사·verifier fallback 수정도 포함된다.
따라서 전체 묶음의 child/coordinator가 이전 묶음과 동일하다고 취급하지 않는다.
이미 작업이 생성된 큐는 첫 업데이트 전이어도 runtime identity가 다르면 자동 변경을 거부한다.
로그가 비었거나 작업 lease가 없어도 다른 노드의 idle worker가 없다는 증거가 아니므로,
기존 큐와 control.lock을 옮기거나 바꾸지 않는다. 현재 실행에는 동일한 수치 코드의
호환 묶음으로 supervisor·종료·진단 경로만 추가할 수 있다.
모델 로딩에서 실제로 멈췄는지에 대한 원격 원인은 로그/노드 관측 전에는 미확인이다.
