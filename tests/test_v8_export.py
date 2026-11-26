"""sh scripts/v8_export.sh: one command packs results, recount and harvest into one archive, read-only on V8_ROOT."""
import os, shutil, subprocess, tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def make_checkout(tmp, recount_body):
    co = tmp / "co"
    (co / "scripts").mkdir(parents=True)
    (co / "v8" / "experiments").mkdir(parents=True)
    for n in ("v8_export.sh", "v8_results.sh"):
        shutil.copyfile(REPO / "scripts" / n, co / "scripts" / n)
    shutil.copyfile(REPO / "v8/experiments/harvest.sh", co / "v8/experiments/harvest.sh")
    (co / "scripts/v8_recount.sh").write_text(recount_body)
    return co


def fake_root(tmp):
    c = tmp / "work/v8/olmo_math"
    (c / "analysis").mkdir(parents=True)
    (c / "runs/r1").mkdir(parents=True)
    (c / "analysis/report.md").write_text("# report olmo_math\n")
    (c / "runs/r1/events.jsonl").write_text('{"x":1}\n')
    return tmp / "work"


RECOUNT = ('mkdir -p "$V8_RECOUNT_OUT/olmo_math/analysis"\n'
           'ln -s "$V8_ROOT/olmo_math/runs" "$V8_RECOUNT_OUT/olmo_math/runs"\n'
           'echo diff > "$V8_RECOUNT_OUT/olmo_math/analysis/diff.txt"\n'
           'echo recount > "$V8_RECOUNT_TEXT"; echo explain > "$V8_EXPLAIN_TEXT"\n')


def run(co, work, tmp):
    env = dict(os.environ, OM_WORK=str(work), HOME=str(tmp / "home"))
    env.pop("V8_ROOT", None)
    return subprocess.run(["sh", str(co / "scripts/v8_export.sh")], env=env, capture_output=True, text=True, timeout=120)


def members(path):
    with tarfile.open(path) as t:
        return {m.name.split("/", 1)[1] if "/" in m.name else "" : m for m in t.getmembers()}


def test_one_archive_holds_every_output_and_leaves_the_root_alone(tmp_path):
    work = fake_root(tmp_path)
    before = sorted(str(p.relative_to(work)) for p in (work / "v8").rglob("*"))
    co = make_checkout(tmp_path, RECOUNT)
    r = run(co, work, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    finals = list((work / "v8-export").glob("v8-export-*.tar.gz"))
    assert len(finals) == 1 and Path(str(finals[0]) + ".sha256").is_file()
    m = members(finals[0])
    for name in ("v8-results.txt", "v8-recount.txt", "v8-explain.txt", "steps.txt",
                 "recount-analysis/olmo_math/diff.txt", "logs/harvest.log"):
        assert name in m, name
    assert any(n.startswith("v8-harvest-") and n.endswith(".tar.gz") for n in m)
    assert not any(v.issym() for v in m.values())          # recount mirror symlinks are not packed
    assert sorted(str(p.relative_to(work)) for p in (work / "v8").rglob("*")
                  if "harvest" not in p.parts) == before     # source tree unchanged
    assert not list((work / "v8/harvest").glob("*.tar.gz"))  # inner harvest moved into the export
    assert "all steps ok" in r.stdout


def test_a_failed_step_is_reported_and_the_rest_still_packed(tmp_path):
    work = fake_root(tmp_path)
    co = make_checkout(tmp_path, "echo boom >&2; exit 3\n")
    r = run(co, work, tmp_path)
    assert r.returncode == 1
    assert "recount FAILED (exit 3)" in r.stdout and "failed steps: recount" in r.stdout
    m = members(next((work / "v8-export").glob("v8-export-*.tar.gz")))
    assert "v8-results.txt" in m and any(n.startswith("v8-harvest-") for n in m)
