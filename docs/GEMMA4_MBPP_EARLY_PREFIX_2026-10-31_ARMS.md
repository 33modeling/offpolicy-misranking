# Gemma MBPP arm도 MATH가 남아 있을 때 배정 — 2026-10-31

사용자 보고(2026-10-31): MBPP prefix까지 끝났는데 다음 단계가 시작되지 않는다.

원인: MBPP arm(on_policy, switch, sr, random)은 MATH 작업이 모두 끝나야 배정됐다
(`ReadyClaims.task_allowed`의 MATH 장벽). MATH switch 하나가 끝나지 않으면 MBPP prefix가
끝나 있어도 남는 GPU가 MBPP arm을 잡지 못하고 쉬었다.

바뀐 배정(Gemma IT, `SRGC_GEMMA_MBPP_EARLY`가 0이 아닐 때):

- MATH 단계 worker가 고르는 순서 맨 끝에 MBPP arm을 붙였다(`late` 범위). 새 MBPP
  cache·prefix → 실패한 MBPP cache·prefix 재시도 → MATH ERROR·재개 작업 → MATH
  cache·prefix·arm → MBPP arm. 잡을 MATH 작업이 있으면 MATH가 먼저다.
- 체인 child의 실행 가능 판정(`runnable`)도 MBPP arm을 본다. 그래서 MATH가 다른 노드에서
  돌고 있을 뿐인 노드는 양보하고 쉬지 않고 MBPP arm을 잡는다.
- MBPP arm이 실패해도 MATH 단계 child는 끝나지 않고 ERROR로 남겨 다시 시도한다.
- 그대로인 것: arm은 같은 seed의 MBPP prefix가 끝난 뒤에만, 저장된 MBPP 고정 패널이
  PASS일 때만 배정된다. 입력, plan, seed, sampling, 보상, checkpoint, 학습 코드와
  adapter·engine digest 파일은 바꾸지 않았다. MBPP arm과 MATH 사이에는 데이터 의존이 없다
  (`TaskQueue.dependency`는 같은 데이터셋의 cache → prefix → arm뿐).

검사: `test_gemma_mbpp_early.py` 49건 통과(새 검사 7건), 배정 관련 검사 파일 14개 전부
통과. CPU 검사이며 클러스터에서 MBPP arm이 시작된 기록은 아직 없다.

적용: `git pull --ff-only`. 체인 coordinator가 다음에 띄우는 Gemma child(10초 간격
재시작)부터 적용된다.
