# Repository Instructions

- Use `master` for code, experiment runners, tests and current execution guides.
  Do not create or switch to a separate work branch unless the user explicitly
  requests one. Verify changes and publish them to `origin/master`.
- Preserve historical commit pins and frozen runtime snapshots used to reproduce
  existing experiments. Publishing current code on master does not authorize
  changing the code, inputs or results of an in-flight or archived experiment.
- Keep executable experiment code in this repository, not the manuscript repository.
- Keep experiment-code bug reports, fixes, failures and verification in the permanent
  `research-bug-board` ledger published at `77modeling.pages.dev/paper-code-bugs/`.
  Preserve existing incidents and events; append corrections and status changes.
  Include original user reports and old commits. Never equate CPU tests with actual
  remote model performance or completion. Verify retention with
  `scripts/record_paper_bugs.py --through <audited-code-commit> --check` before publication.
