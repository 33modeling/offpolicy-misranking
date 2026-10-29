# 논문 코드 버그 현황판

게시 주소: https://77modeling.pages.dev/paper-code-bugs/

Gemma 초기 1%대 사용자 보고부터 모델 선택·생성·실행·기록·status 문제를 상세
사건으로 남긴다. 과거 수정커밋과 감사 시점의 전체 Git 이력도 별도 목록으로 보존한다.
실제 측정·사용자 보고·코드 테스트를 구분하며, 원격 실행 근거가 없는 결과는 미확인이다.

- `incidents.json`: 사건의 원래 설명과 추가되는 이벤트. 기존 사건·이벤트를 덮거나 지우지 않는다.
- `data.json`, `commits.json`: 검증된 게시용 내보내기. 최초 보관본은 `archive/`에 둔다.
- `scripts/record_paper_bugs.py`: 이전 사건과 이벤트 prefix, 기존 커밋 및 Git에 보관한
  최초 archive를 검사한다. 수정·정정·재발·검증은 새 이벤트로 추가한다.
- 현재 감사 기준은 data.json의 coverage.source_commit이다. 이후 현황판 자체의
  커밋이 생겨도 같은 감사 기준을 재생할 수 있도록 history snapshot을 보존한다.
- 화면에는 삭제·초기화 기능이 없다. 수정된 사건도 기본 목록과 전체 내려받기에 포함한다.

```sh
python3 scripts/record_paper_bugs.py --through <감사할-코드-커밋>
python3 scripts/record_paper_bugs.py --through <동일한-코드-커밋> --check
python3 -m pytest -q srgc_research/tests/test_paper_bug_retention.py
```

사이트 소스 동기화는 33modeling-site/update-site.sh의 paper-code-bugs 대상으로 등록한다.
화면 파일(index.html·app.js·styles.css)은 허브의 `/assets/status-kit/`(현황판 공통 모양·위쪽 현황판 바로가기)을
불러오므로 허브 사본(`/paper-code-bugs/`)에서 확인한다. 2026-10-29 허브에만 있던 status-kit 판을 이 폴더로 옮겼다.
77의 기존 일반 bug-board/D1 기록은 별도로 보존한다. 이 화면의 커밋 이력을 실제
학습 성공 기록으로 사용하지 않는다.
