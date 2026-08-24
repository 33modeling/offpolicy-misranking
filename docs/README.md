# 문서 체계

이 디렉터리는 **현재 절차**, **변경 가능한 계획**, **수정하지 않는 감사 기록**,
**수신 결과의 provenance**를 분리한다. 같은 주제가 여러 파일에 보이면 아래 우선순위를
따른다.

## 1. 현재 정본

| 목적 | 정본 | 갱신 규칙 |
|---|---|---|
| 실행·재개·복구·수확 | [`USAGE.md`](USAGE.md) | 실제 진입점과 기본값이 바뀔 때 함께 갱신 |
| 설계와 데이터 흐름 | [`ARCHITECTURE.md`](ARCHITECTURE.md) | 구현 제약과 알려진 불일치를 포함 |
| 모듈·스크립트 역할 | [`CODE.md`](CODE.md) | 줄 번호 대신 책임과 계약을 기록 |
| 장애 원인과 재발 방지 | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | 해결된 사건도 재발 방지 기록으로 유지 |
| 현재 작업·제출 게이트 | [`BACKLOG.md`](BACKLOG.md) | 완료/폐기/진행 상태를 한곳에서 관리 |

복구 절차는 `USAGE.md` 7절에 통합했다. 별도 `RECOVERY.md`는 2026-08-12의
`go_full.sh`/v2 경로를 계속 현재 절차처럼 보이게 해 제거했으며, 과거 내용은 Git
이력에서 확인한다.

## 2. 감사 기록

다음 파일은 당시 코드와 원고를 판정한 **시점별 스냅샷**이다. 현재 실행법이나
현재 주장을 결정하지 않으며, 과거 판정이 바뀌어도 내용을 사후 수정하지 않는다.

| 파일 | 역할 |
|---|---|
| [`PAPER_REVIEW_2026-08-19.md`](PAPER_REVIEW_2026-08-19.md) | 최초 논문·수식·코드 전수 점검 |
| [`FULL_AUDIT_2026-08-20.md`](FULL_AUDIT_2026-08-20.md) | P0 교정 뒤 통합 감사 |
| [`RESULTS_0820_ANALYSIS.md`](RESULTS_0820_ANALYSIS.md) | 8월 20일 수신 번들 판독 |

현재 상태는 각 기록의 문장보다 `BACKLOG.md`, 현재 코드, private paper 레포의
`EXPERIMENT_PLAN.md`를 우선한다.

## 3. 결과 provenance

`results/YYYY-MM-DD/`는 외부에서 받은 파일을 그대로 보존하는 디렉터리다. 각 폴더의
`README.md`가 원본 archive hash, 완주율, 해석 가능 범위를 정한다. 생성 보고서를
서로 병합하거나 최신 결과로 덮어쓰지 않는다.

- [`results/2026-08-24/README.md`](results/2026-08-24/README.md): 15/20 partial v4
  bundle. 제출용 동결 결과가 아님.

## 4. 레포 경계

| 레포 | 담당 |
|---|---|
| `offpolicy-misranking` | 실행 코드, 계약 검사, 결과 provenance |
| `offpolicy-misranking-paper` | 현재 LaTeX 원고, 정본 실험 계획, 제출 감사 |
| `new-paper-ideas/68-one-sided-offpolicy-misranking` | 아이디어 형성 이력과 Q&A |
| `obsidian` | 읽기용 색인과 쉬운 해설. 정본을 복제하지 않음 |

논문 주장과 현재 실험 설계는 private paper 레포를, 실행 명령은 이 레포의
`USAGE.md`를 따른다. 로컬의 `offpolicy-misranking-audit`와
`offpolicy-misranking-codecheck-0820`은 같은 원격의 과거 감사 브랜치이므로 현재
문서 정본으로 사용하지 않는다.
