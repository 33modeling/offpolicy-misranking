# V8: 한 명령으로 OLMo·Qwen 노드 자동 배정

모든 4×H100 노드에서 같은 명령을 실행한다.

```sh
sh scripts/run_v8_nodes.sh
```

별도 모델명이나 노드별 역할을 지정하지 않는다. 기본 순서는
**OLMo MATH → Qwen MBPP → OLMo MBPP → Qwen MATH**다.
한 조합의 학습·ext·최종 평가·결과 수집과 분석까지 모두 끝나 공유 큐가 `complete`가 된 뒤
다음 조합을 시작한다. 같은 조합 안에서는 여러 노드가 준비된 작업을 병렬 처리하며,
각 노드는 한 번에4GPU 작업 하나를 실행한다. 현재 조합의 작업이 모두 실행 중이거나
부모 checkpoint를 기다리면 추가 노드는 대기한다. 빈 노드를 다음 조합에 넘기지 않는다.
실패·준비 오류·부족한 입력은 완료로 취급하지 않는다. 현재 조합을 처리할 수 없는 노드는
원인을 출력하고 종료하며, 해당 조합의 다른 정상 노드는 계속 처리한다.
공유 잠금 안에서 완료 상태와 실제 작업 lease를 다시 확인한 뒤 작업을 claim한다.
공유 잠금은 배정 판단·작업 claim 동안만 잡으며,
원본 Runtime의 작업 목록·체크포인트·완료 여부 검사와 모델 로딩·학습·준비 중에는 잡지 않는다. 모델별 고정 노드 할당이나 GPU 분할은
하지 않는다.

다른 worker가 과거 조합의 공유 큐를 다시 열면 이 worker의 완료 캐시를 무효화하고
입력 신원·현재 단계·준비된 작업을 다시 확인한다. 완료된 출력은 그대로 인수하며
실패 이력과 재시도 상한을 초기화하지 않는다. 실행 중인 lease가 남아 있으면
다음 조합은 기다리고 중지 신호에 응답한다.

OLMo MBPP·Qwen MATH도 기존 seed5–9의25-step prefix와 원래 MBPP 입력·학습 환경을 검증한 뒤
On 궤적·SR 분기·Uniform 비교·예산 연장·최종 평가·수집·분석을 같은 경로로 수행한다.
기존 조합의 큐와 source binding은 그대로 재사용한다. OLMo MBPP의 기본 Python은
기존 MBPP 실행기와 같은 `SWITCH_PYTHON`이며, 기존 `V8_PYTHON_OLMO_MBPP` 지정이
있으면 우선한다. 앞선 조합의 준비나 실행이 실패하면 다음 조합은 시작하지 않는다.
일부 원본 Runtime을 열지 못해도 그 조합을 전체 순서에서 제거하지 않는다.
준비된 작업과 빈 worker가 있으면 계속 배정하며, 모든 작업이 실행 중이거나 부모 checkpoint를
기다리는 경우에는 새 작업을 만들지 않고 기다린다. 한 조합만 선택할 때는
`sh scripts/run_v8_nodes.sh olmo_mbpp`를 사용한다. Qwen MATH만 선택할 때는
`sh scripts/run_v8_nodes.sh qwen_math`를 사용한다. Qwen MATH는 기존
`qwen35-9b-math.json`과 Qwen adapter를 사용하며, `V8_PYTHON_QWEN_MATH`·
`QWEN_PYTHON`·`PAIR_PYTHON` 순서로 기존 Python을 고른다.

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

네 조합의 seed25 prefix가 모두 준비되어도 처음 배정하는 작업은 OLMo MATH의10개다.
나머지 세 조합의 초기 작업30개는 각 조합의 차례까지 대기한다.
이후 준비된 작업 수는 부모 checkpoint 생성 속도와 기존 분기의 완료 속도에 따라
변한다. 실행기는 준비된 작업만 노드에 배정한다.

아래 표는 OLMo MATH와 Qwen MBPP 두 조합만의 예전 추정이며 OLMo MBPP·Qwen MATH는 포함하지 않는다.
네 조합의 실제 필요 노드 수·처리량은 아직 측정하지 않았다.
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

범위는 기존 seeds5–9의 rebuttal Branch Bank다. OLMo와 Qwen 각각 MATH·MBPP를 비교한다. 각 조합의 결과를 따로 기록한다.
새 seed의 개발·확증 실험과 E8 양방향 전환은 포함하지 않는다.

네 조합 모두 기존 `prefix-ready.json`과 `prefix.pt`를 찾아25-update 공통 상태에서
분기한다. checkpoint 해시·step·trainer 구현을 확인하고 모델·optimizer 상태를
불러온다. 저장된 process RNG가 있으면 복원한다. 같은 v8 작업은 `latest.pt`에서
재개한다. 다른 방법의 기존275-update 결과를 새 v8 작업 완료로 자동 대체하지 않는다.

현재 조합 안에서는 **준비된 ext와 최종 eval을 학습과 함께 배정한다.** 다른 조합의
기본 결과를 기다리지 않으며, 관계없는 분기 전체의 학습 완료도 기다리지 않는다.
275 체크포인트가 게시되면 그 최종 eval을 즉시 배정한다. 다른 seed가 실패했거나
비용 예산이 아직 준비되지 않아도 이 평가는 실행할 수 있다. 첫 조기 평가 전에
`.multi-node/budget-rule.json`에 원본 예산 함수 해시·설정·binding·모든 seed·H·입력 신원을
먼저 고정한다. 평가값은 예산 계산에 쓰지 않는다. 수치 예산은 모든 지정 seed의
`srgc_current`가 실제275 endpoint에 도달하고 비용이 기록된 뒤 원래 함수로 계산한다.
부분 seed 평균이나 추정 비용으로 대신하지 않는다. 이미 고정된 예산 파일은 바꾸지 않는다.
ext와 예산 endpoint eval은 이 수치 예산과 각 부모 checkpoint가 준비되면 배정한다.
같은 조합의 다른 분기·seed는 동시에 학습할 수 있다. 다른 모델·데이터 조합은 차례를 기다린다.
준비된 학습·ext·eval은 cohort 안에서도 순환 배정하여 학습 작업이 많다는 이유로
ext/eval이 목록 뒤에 계속 남지 않게 한다. 준비되지 않은 종류는 건너뛰고 다른 작업을 배정한다.
예산 기준 정책의 endpoint·비용이 없으면 ext만 기다리고, 준비된275 eval은 계속한다.
수치 예산보다 먼저 평가한 경우 예산 JSON의 `selection_protocol`과 실제 시간 필드를
함께 기록한다. 원본 계산기의 `rule` 문자열은 호환성을 위해 남기되, 그 문자열에 담긴
예전 실행 순서와 새 순서를 `rule_timing_note`로 구분한다. 사전 기록 없이 평가 결과가
이미 있으면 뒤늦게 사전 고정했다고 기록하지 않고 오류로 남긴다.
작업 실패는 원문을 남기고 즉시 재시작하며, 환경·입력 사전검사 오류는 가려서 진행하지 않는다.

먼저 나온275스텝 결과는 각 cohort의 `analysis/matched-update-275/`에 보존한다.
`.multi-node/primary-results.json`은 보고서·평가 목록·원본 binding·고정 예산의 해시를
기록한다. 이 보고서는 ext 배정의 전역 장벽으로 사용하지 않는다. 기존 evaluation
job·계획·실패 기록은 지우지 않고275스텝 평가는 같은 결과 파일을 재사용한다.
일찍 배정한 작업도 원래 `extension`·`evaluation` job phase와 task key를 유지하며,
claim 직전 고정 예산 또는 조기 평가 사전 기록의 해시와 현재 task 정의·lease를 다시 확인한다. 이미 실행 중인
task lease는 빼앗지 않고 완료된 ext나 평가도 되풀이하지 않는다.
이 순서를 적용하려면 모든 V8 worker를 수정본으로 다시 시작해야 한다. 이전 worker는
고정된 예전 코드를 계속 쓰므로 `git pull`만으로 실행 순서가 바뀌지 않는다. 재시작해도
원래 체크포인트·optimizer·RNG와 학습 기록은 보존한다.

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

네 조합의 `phase`, `known_task_counts`, `leased_tasks`, `failure`가 함께 출력된다.
worker는60초마다 현황을 출력한다. 자식 출력은 터미널과 각 cohort 로그에 남긴다.
노드마다 같은 GPU 그룹에서 worker 하나를 실행한다.

`--status`의 `dispatch_workers`에는 각 worker가 실제 기록한 제어 코드 해시와
정책 버전이 나온다. 표 형식 `status`는 `dispatch current <hash>`(현재 checkout과
같음), `dispatch different <hash>`(다름), `dispatch unverified`(기록 없음)를 구분한다.
정책 파일이 디스크에 있다는 것만으로 실행 중인 worker에 적용됐다고 판단하지 않는다.
현재 배정 정책은 `bounded-json-metadata-errors-v31`이다. 작업 검사는 해당 cohort의
control lease만 잡고 별도로 수행한다. 검사 결과가 준비되면 공유 잠금 안에서 현재
binding·phase·실패 상태·작업 정의·작업 lease를 다시 확인한 뒤 claim한다.
검사와 claim 사이에 다른 노드가 가져간 작업이나 실패한 작업은 가져오지 않는다.
노드 입력 검증도 다른 cohort의 시작을 막지 않으며 검증을 통과한 노드만 claim할 수 있다.
원본 Runtime 초기화에는120초 제한을 두고, 한 cohort가 초기화에 응답하지 않으면
오류를 표시하고 다른 cohort를 계속한다. 학습 실행 시간에는 이 초기화 제한을 적용하지 않는다.

개별 작업 실패는 그 작업에만 남긴다. V20부터 기본 상한은 작업당 누적 실패2회다.
첫 실패 뒤 체크포인트에서 즉시 한 번만 재시도하고, 두 번째 실패 뒤에는 그 작업을
더 배정하지 않는다. 다른 정상 seed와 준비된 ext/eval은 계속 배정한다. 노드 재접속,
새 코드 해시, 정책 변경, `--retry-failed`는 실패 횟수를 초기화하지 않는다.
이전 `retry_epoch` 기록은 보존하지만 상한 계산에서 과거 실패를 빼지 않는다.
이미2회 이상 실패한 작업은 새 노드에서도 자동으로 재개하지 않는다. 원인을 실제로
고친 뒤에만 명시적인 양수 `--max-attempts N`으로 누적 상한을 늘릴 수 있다.
이전 명령에 남은 `--max-attempts 0`도 기본 유한 상한2회로 처리한다. 무제한 모드는 없다.
기본 `--retry-delay 0`은 허용된 한 번의 재시도에 대기시간이 없다는 뜻이다.
강제 노드 종료·의도적 중단은 작업 오류 횟수에 합산하지 않으며 체크포인트에서 재개한다.
원본 실패·로그·optimizer·RNG·학습 보상·체크포인트는 보존한다. 현재 task lease나
원본 실행 잠금이 살아 있으면 건드리지 않으며, 완료 출력은 상한보다 먼저 검증한다.
구형 상태에 무제한 정책이 남았으면 현황판에 `UNBOUNDED legacy retry policy`로 경고한다.
새 V20 정책을 적용한 상태에는 유한 상한과 다음 재시도 또는 격리 사유를 표시한다.
현황판만 새로 읽어서는 이미 실행 중인 구형 worker의 무한 반복이 멈추지 않는다.
모든 worker가 같은 V20 제어 코드를 사용해야 한다. torchrun 자체의
`--max_restarts=0`은 내부 재시작을 끈다는 뜻이며, 바깥 실행기의 상한과 구별한다.
실패 작업만 남으면 실패로 종료하며 freeze·평가·완료로 넘어가지 않는다.
V21은 `torchrun launcher exit -11`(SIGSEGV)을 처음 기록한 시점부터 자동 재실행을
금지한다. torchrun이 이를 exit1로 감싸도 원래 rank signal 기록을 검사한다. 다른
치명적 native signal4/6/7/8도 동일하게 처리하며 새 노드/정책 변경으로 해제하지 않는다.
정상 checkpoint eval과 ext는 계속 배정한다. 이미 큐에 등록된 원본 eval task는
다른 endpoint의 eval 계획에서 빠졌더라도, 원래 고정 예산 또는 해당 H 사전 규칙과
원본 snapshot/spec가 유효하면 그대로 배정한다. ext 계획 오류도 독립 eval을 막지 않는다. 실행할 정상 작업·진행 중 scan/CPU 준비·
살아 있는 task lease/native 실행 잠금이 없는데 실패가 남으면 현재 worker를 exit1로
끝낸다. 다른 cohort가 없는 부모 checkpoint를 기다리는 경우나, 실패 작업이 현재
ext/eval 계획에서 빠진 경우에도 대기 루프에 머무르지 않는다. 작업자별 종료 사유는
`.multi-node/worker-exits/`에 남기며 정상 cohort의 공유 큐를 실패/완료로 바꾸지 않는다.
`9/2` 같은 비율 대신 `BLOCKED`와 보존한 과거 실패 수·상한/치명 signal을 표시한다.
V22는 실패 종료 직전에 원래 planner/ready/done을 다시 검사한다. 첫 scan 직후 다른
노드가 checkpoint를 게시한 경우에는 기존 task를 배정하고 worker를 종료하지 않는다.
현재 phase의 전체 실패 종료와 숨은 실패의 대기 종료 양쪽에 같은 검사를 적용한다.
입력/설정 오류는 해제하지 않으며 재검사에서도 새 작업을 claim하지 않는다. 최종
재검사 잠금 대기는 유한하며 busy loop로 바뀌지 않는다. 종료 receipt와 콘솔에는
`blocked_work` 작업자 종료, 현재 worker의 native crash/OOM/timeout, 이전 worker들의
실패 수를 따로 남긴다. exit1만으로 새로운 모델 crash라고 단정하지 않는다. 실패
코드·checkpoint·원문 로그는 그대로 보존하고 미완성 실험을 성공 처리하지 않는다.
V23은 사용자의 eval 복구 지시에 따라 평가 작업의 SIGSEGV에 한정해1회의 제한된
재실행을 허용한다. 이미 막힌 eval도 기본 실행에서 입력 검증 후 자동으로 재개한다.
원래 snapshot/spec/question set/responses/seed와 task key를 유지한다. 완료된 평가
파일은 원래 validator로 확인해 재사용하며, 미완료 평가는 같은 checkpoint에서 다시
실행한다. 과거 실패가9회여도 그 이력을 보존하고 명시적인 eval recovery receipt 뒤의
재실행 오류만1회까지 허용한다. 고정 recovery policy는 새 node/코드 hash/재시작으로
갱신되지 않는다. 그1회도 실패하면 더 배정하지 않는다. 입력·binding 오류, 학습의
native fault, 다른 치명 signal을 이 방식으로 해제하지 않는다. 이는 SIGSEGV 원인이
고쳐졌다는 기록이 아니라 사용자가 요청한 유한 평가 복구 정책이다.
별도 기록 오류로, 고정 예산이 추가 update를 허용하지 않는 ext도 원래 fork 상태의
끝점 snapshot을 요구하는데 saved runner는 latest.pt가 있을 때만 그 시작점을 저장했다.
원본 SHA f14d92ee...에만 적용하는 기록 overlay가 budget fork의 retained 시작점을
저장한다. 원래 예산·optimizer step·샘플링을 바꾸지 않고 model/optimizer/RNG 상태를
그대로 보존한다. 원본 runner 파일은 수정하지 않으며 overlay SHA를 rank receipt에 남긴다.
여러 runtime actor가 읽는 per-launch plan은 UUID가 포함된 경로에 원자적으로 게시한다.
actor 시작 중 같은 JSON을 truncate/write해 빈 파일이나 반쪽 문자열을 읽는 경합을 막는다.
이는 native SIGSEGV와 별도의 초기화 오류이며 scientific task/config 값은 바꾸지 않는다.
SIGSEGV를 차단한 것은 native 오류 원인을 고쳤다는 뜻이 아니다. 원래 오류와 checkpoint는
보존하고, 원인 수리 없이 상한만 올려 native crash를 재실행하지 않는다.
 이전 V8가
개별 `task exit=...` 실패를 cohort 전체 중단으로 저장한 경우, 같은 failed job이
존재할 때만 그 중단을 작업 범위로 옮긴다. 원래 실패는 history에 보존하고 작업·시도·
binding·checkpoint를 고치거나 지우지 않는다. 신원·설정·CPU 준비 오류는 이 방식으로
해제하지 않는다. 입력·설정 검증을 우회하지 않으며 살아 있는 task lease는 건드리지 않는다.
`--retry-failed`는 기존 명시적 복구용 호환 옵션이며 일반 실행에 필요 없다.
자동 복구는 작업 실패에만 적용하며 입력·신원·CPU 준비 오류를 해제하지 않는다.
별도 제어 코드 예외로, 구형 예산 함수에 `expected_seeds`를 넘겨 생긴 정확한
`freeze_budget() got an unexpected keyword argument 'expected_seeds'` 중단은 입력·원본
binding을 검증한 새 worker가 이력을 보존하고 재개한다. 구형 함수에는 지원하지 않는
인자를 넘기지 않으며, 모든 seed의 선택된 endpoint와 비용 준비 검사는 제어 코드에서 유지한다.
하나의 예산 endpoint eval을 계획하다 오류가 나도 다른 정상 checkpoint eval과 준비된 ext는
배정한다. 해당 오류는 해결됐다고 처리하지 않고 `planning_errors`와 실패 이력에 남긴다.
이 오류로 예전 제어 코드가 조합 전체를 멈췄다면 입력 검증 후 독립 작업만 재개하며,
원래 필수 계획이 완성되기 전에는 전체 실험을 완료로 처리하지 않는다.
준비된 새 seed·분기를 실패한 원 궤적의 반복 재시작보다 먼저 배정하여 작업 독점을 막는다.
다른 cohort의 메타데이터 잠금이 사용 중이어도 준비된 cohort는 즉시 claim할 수 있다.
활성 lease 수로 덜 바쁜 cohort를 우선하되 다른 cohort의 잠금을 기다리느라 GPU를 비우지 않는다.
GPU busy 관측은1초 후 다시 확인하여 이미 종료된 프로세스의 오래된 캐시 때문에30초 기다리지 않는다.
실제 GPU 점유·task/native 실행 잠금과 부모 체크포인트 의존성 검사는 그대로 적용한다.

`exit -11`은 SIGSEGV이며 CUDA OOM과 구분한다. SIGSEGV는 실제 작업 실패로 남기고
원래 오류·종료 코드·실패 횟수를 보존한다. 원인 확인 없이 정상 중단이나 성공으로 바꾸지 않는다.
V14는 MBPP 채점 subprocess의 Python `preexec_fn`을 제거하고 exec 뒤에 같은
주소 공간·CPU·파일 크기 제한을 적용한다. 원본 verifier·code_check 파일과 채점 규칙·
프로토콜 버전은 바꾸지 않으며, 알려진 두 원본 verifier 해시에만 별도 실행 overlay를 적용한다.
제한 설정에 실패하면 학습 오류로 전달하고 보상0으로 바꾸지 않는다. 제한 설정 뒤 같은
PID에서 원래 `python -I -B code_check.py ...`를 exec해 `__main__`·loader·argv도 보존한다.
이 수정이 실제 H100 SIGSEGV 원인과 일치하는지는 rank 로그로 확인해야 한다.

원래 child 실행 전부터 `faulthandler`를 켜고 `.multi-node/progress/<attempt>/`에
`native-fault-rank-N-<invocation>.log`와 해당 JSON 메타데이터를 남긴다. torchrun의
일반 ChildFailedError 뒤에서도 실제 rank·SIGSEGV·호출 스택과 현재 PID의 마지막 진행
기록을 `child-failure.json`에 연결한다. Python 버전에 따라 C 스택은 지원되지 않을 수 있다.
다른 rank의 Python 예외와 fatal signal이 동시에 있으면 둘 다 남기고, 이전 invocation의
기록은 현재 실패로 사용하지 않는다. 체크포인트·optimizer·RNG와 학습 보상은 보존한다.

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

기본 네 cohort는 기존 OLMo-3 7B/MATH·MBPP와 Qwen3.5 9B/MATH·MBPP를
이어받는다. 비교는 각 cohort 안의 학습 방법 사이에서 수행한다. 서로 다른 데이터의
점수 차이만으로 모델 효과를 분리하지 않는다.

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

터미널 연결만 끊겨 supervisor에 SIGHUP이 오면 현재 학습은 계속한다. 닫힌 출력 연결은
학습 자식과 분리하며, 출력은 기존 console 로그에 계속 저장한다. 연결 종료는
`$V8_ROOT/.node-progress/<host>/<launch>/session-disconnect.jsonl`에 기록한다.
명시적인 Ctrl-C·SIGTERM·`stop` 명령은 그대로 중단한다. 노드 자체가 강제 종료되면
공유 볼륨의 실제 task lease와 native 실행 잠금이 풀린 뒤 다른 빈 worker가 자동 인수한다.
원래 시도에는 lease 상실 관측과 `node_lost`를 남기고 새 시도에 `resumed_from_attempt`를
연결한다. 실제 종료 시각·종료 코드는 관측하지 못했다면 만들지 않는다.

프로세스의 SIGKILL·SIGTERM·SIGINT·SIGHUP 종료 증거만 있고 원래 Python 오류가 없으면
학습 오류 한도를 소비하지 않는 중단으로 기록하고 기본 즉시 재개한다. SIGKILL이
노드 퇴거·수동 종료·호스트 메모리 부족 중 무엇 때문인지는 단정하지 않는다.
기록된 Python/CUDA 오류와 SIGSEGV는 일반 작업 실패로 보존하며 기본 무제한 즉시 재시작을 적용한다.
원래 `latest.pt`의 모델·optimizer·RNG에서 이어가고 retained checkpoint와 학습 보상은
지우지 않는다. 재실행 전에 살아 있는 자식의 잠금이 남아 있으면 기다린다.
이는 이미 할당되어 실행 중이거나 새로 참여한 worker가 작업을 이어받는 기능이며,
클라우드에서 새 물리 노드를 구매하거나 할당하는 기능은 아니다.

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
cohort를 오가는 예전 균등 배정은 내부 회귀 검사로 보존한다. 기본 실행의 순차 배정 검사는
`tests/test_srgc_v8_sequential_cohorts.py`에 있으며,6개 동시 worker가2개 작업을 채운 뒤에도
다음 조합을 시작하지 않고 최종 평가·수집 후에만 넘어가는지 확인한다.
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
코드·Python으로 별도 자식 프로세스에서 실행하고, 새 배정기는 조합 순서·공유 완료 상태·실제 lease를
확인한다. 원래 binding.json을 고치지 않는다. 새 제어 정책과 전체 조합 순서·배정기 해시,
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
네 cohort의 Python 경로가 달라도 공통 배정기를 사용한다. 각 원본 Runtime의
NCCL 사전검사와 학습은 해당 cohort의 Python에서 실행한다.
한 cohort의 원본/Python/메타데이터 오류는 해당 조합의 차례에 표시하고 다음 조합은 대기한다.
이 노드가 현재 조합을 처리할 수 없으면 실패로 종료하며 다른 조합을 대신 시작하지 않는다.

`git pull`은 이미 실행 중인 worker의 메모리나 고정 실행 사본을 바꾸지 않는다.
수정 이후 시작한 worker부터 새 순서를 사용한다. 전체 노드에 같은 순서를 적용하려면
기존 worker를 안전하게 중지하고 `git pull` 후 원래 명령으로 다시 시작한다. 큐 초기화는 하지 않는다.
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

### 반복되는 native 충돌과 수정본 적용

`node-native-fault-quarantine-v24`는 작업별 재시도 제한에 노드별 제한을 추가한다.
같은 GPU 묶음·cohort·실행기 코드에서 native 충돌이 두 번 기록되면 그 GPU 묶음에
해당 cohort의 새 작업을 배정하지 않는다. 작업을 바꾸거나 worker PID를 바꿔도
제한을 초기화하지 않는다. 다른 GPU 묶음은 현재 cohort의 정상 작업을 계속 처리한다.
다음 cohort는 현재 cohort의 완료까지 기다린다.
기존 명령은 같은 GPU 묶음에서 native 충돌 이력이 있는 미완료 작업을 발견하면,
작업을 claim하기 전에 원본4GPU NCCL 검사를 새로 수행한다. 통과한 영수증을 보존한 뒤
해당 묶음의 재참여를 허용한다. 작업별 누적 실패 상한은 유지하며 실패·checkpoint를 지우지 않는다.
검사 실패·중단 후 재실행 시 검사 기회는 같은 GPU 묶음·실행기에서 총2회다.
검사 통과 후 다시 native 충돌이 나면 반복 복구를 멈춘다. Ctrl+C나 PID 변경으로 한도가 늘지 않는다.
공유 큐 전체를 실패로 바꾸거나 기존 실패·checkpoint·완료 평가를 삭제하지 않는다.
기록은 각 cohort의 `.multi-node/node-native-faults/`에 남는다.

eval의 고정 1회 복구 예산은 그대로다. 이 제한과 노드 격리는 SIGSEGV 원인 수정의
증거가 아니다. 실제 H100에서 같은 task가 정상 진행하는지는 별도로 확인해야 한다.
원래 child의 진단 초기화 후에도 attempt 전용 fatal-signal FD를 복원하므로,
유효한 rank 덤프가 있으면 기본 작업 로그의 `NATIVE STACK`에 원문을 출력한다.

이미 실행 중인 worker에 수정본을 적용할 때는 각 노드에서 다음 한 줄을 실행한다.

```sh
git pull && sh scripts/run_v8_nodes.sh stop && sh scripts/run_v8_nodes.sh
```

`stop`은 해당 노드의 기존 V8 실행만 종료한다. 새 명령은 기존 큐와 checkpoint를
사용하며, 실패 숫자를 지워서 성공으로 보이게 처리하지 않는다.

### native 충돌 작업은 다른 GPU 묶음에서 이어 받음 (v25, 2026-11-10)

`native-crash-retry-other-host-v25`. 사용자 지시(11-10): "failed로 된거 복구 되는거지? 느린건지 복구가 안된건지
모르겠어 디버깅 계속 해보고 수정해". v21~v24 에서는 SIGSEGV(-11) 학습 작업은 첫 충돌부터 다시 배정되지 않았고,
평가 작업도 노드가 새로 합류할 때만 1회 재생 예산이 붙었다. 그래서 failed 로 남은 작업은 느린 것이 아니라 복구되지 않는 상태였다.

- native 충돌(SIGSEGV·SIGABRT·SIGBUS·SIGILL·SIGFPE)로 끝난 작업은 노드 합류를 기다리지 않고 곧바로 다시 배정된다.
  마지막 저장 checkpoint 부터 이어 가며, 충돌한 GPU 묶음(호스트+GPU 집합)에는 다시 가지 않는다. 같은 호스트의
  다른 GPU 묶음은 받을 수 있다.
- 서로 다른 GPU 묶음에서 두 번 충돌하면 작업 자체 문제일 가능성이 커서 자동 재시도를 멈춘다(`TASK_NATIVE_HOST_LIMIT = 2`).
  status 의 next 줄에 `BLOCKED: SIGSEGV (exit -11) on 2 different nodes (...)` 로 나온다.
- v25 이전 기록에서 한 노드가 같은 작업을 여러 번 다시 띄워 쌓인 충돌(V8-136·V8-144)은 노드당 1회로 센다.
  그 노드 말고 다른 곳에서 1회 더 시도한다. v25 이후 시도는 attempt 마다 `host`·`gpus` 를 기록하고 하나씩 센다.
- 충돌한 GPU 묶음의 worker 는 그 작업을 "나중에 다른 노드가 받을 일"로 세므로 큐 전체 실패를 쓰지 않는다.
  다른 일이 없고 아무것도 돌지 않으면 기존처럼 그 worker 만 끝난다(`blocked_work`).
- 노드별 격리(같은 GPU 묶음에서 native 충돌 2번이면 그 cohort 일을 받지 않음), 실패 상한(작업당 2), 평가 재생 예산
  기록은 그대로다. 이 규칙은 SIGSEGV 원인 수정이 아니다. 원인은 다음 충돌의 status why 줄(충돌 함수·파일·줄)로 찾는다.
