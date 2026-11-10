import re

import pytest

from harness import matrix
from harness.matrix import build_matrix, render_markdown, write_reports
from harness.storage import Store
from tests.conftest import make_results, make_run


def runs(*specs: tuple[str, str]):
    return [(make_run(f"r-{m}", model=m, suite="fake"), make_results(spec)) for m, spec in specs]


SPECS = (
    ("strong", "A1:a+ A2:a+ A3:a+ T1:t+ R1:r+ R2:r+"),
    ("mid", "A1:a+ A2:a- A3:a+ T1:t+ R1:r+ R2:r-"),
    ("weak", "A1:a- A2:a- A3:a+ T1:t- R1:r+ R2:r-"),
)


def test_matrix_numbers_by_model_and_category():
    d = build_matrix("fake", runs(*SPECS), generated="2026-11-10 12:00")
    by = {s.model: s for s in d.models}
    assert (by["strong"].passed, by["strong"].n) == (6, 6)
    assert (by["mid"].passed, by["mid"].n) == (4, 6)
    assert by["weak"].by_cat == {"answer": (1, 3), "tool": (0, 1), "refuse": (1, 2)}
    assert d.categories == ["answer", "refuse", "tool"] and d.case_ids == ["A1", "A2", "A3", "T1", "R1", "R2"]
    assert len(d.pairs) == 3  # strong-mid, strong-weak, mid-weak
    assert by["strong"].ci[1] == pytest.approx(1.0) and by["weak"].ci[0] < 0.5 < by["weak"].ci[1]


def test_matrix_uses_only_cases_every_model_ran_and_says_so():
    a = runs(("one", "A1:a+ A2:a+ A3:a+"), ("two", "A1:a+ A2:a-"))
    d = build_matrix("fake", a)
    assert d.case_ids == ["A1", "A2"] and all(s.n == 2 for s in d.models)
    assert any("different case-id sets" in n for n in d.notes)


def test_matrix_flags_incomplete_runs():
    r = runs(("one", "A1:a+ A2:ae"))
    r[0][0].status = "partial"
    d = build_matrix("fake", r)
    assert d.models[0].errors == 1 and any("partial" in n for n in d.notes)


def test_markdown_contains_the_required_columns():
    md = render_markdown(build_matrix("fake", runs(*SPECS), generated="2026-11-10 12:00"))
    for needle in (
        "# Model matrix: fake",
        "## Accuracy",
        "answer",
        "refuse",
        "tool",
        "## Latency, tokens, cost",
        "p50",
        "p95",
        "prompt tok/case",
        "cost/case",
        "INR 0.00",
        "## Are the differences real?",
        "McNemar p",
        "bootstrap 95%",
        "## Runs",
    ):
        assert needle in md, needle
    assert "| strong | 6/6 (100%) |" in md and "| weak | 2/6 (33%) |" in md
    assert "SYNTHETIC" not in md


def test_html_matrix_is_self_contained_and_complete(tmp_path):
    d = build_matrix("fake", runs(*SPECS))
    md_path, html_path = write_reports(d, tmp_path / "reports")
    html = html_path.read_text(encoding="utf-8")
    assert md_path.name == "matrix.md" and html_path.name == "matrix.html"
    assert html.count("<svg") >= 4  # accuracy, category, latency, case grid
    for title in (
        "Overall accuracy",
        "Accuracy by category",
        "Latency",
        "Tokens and cost",
        "Every case, every model",
        "Are the differences real?",
    ):
        assert title in html
    assert not re.search(r"<script|<link|<img|@import|url\(|https?://", html)
    assert "prefers-color-scheme: dark" in html and 'name="viewport"' in html
    assert html.count("<title>") >= 6  # svg hover titles and the page title
    for m in ("strong", "mid", "weak"):
        assert m in html


def test_synthetic_runs_are_bannered_in_both_reports(tmp_path):
    r = runs(*SPECS)
    for run, _ in r:
        run.config["synthetic"] = True
    d = build_matrix("fake", r)
    assert d.synthetic and "SYNTHETIC" in render_markdown(d)
    _, html = write_reports(d, tmp_path)
    assert "SYNTHETIC" in html.read_text(encoding="utf-8")


def test_a_single_model_matrix_still_renders(tmp_path):
    d = build_matrix("fake", runs(SPECS[0]))
    assert d.pairs == []
    _, html = write_reports(d, tmp_path)
    assert "<svg" in html.read_text(encoding="utf-8")


def test_matrix_cli_runs_a_subprocess_per_model_and_resume_reuses_finished_runs(tmp_path, capsys):
    db, out = tmp_path / "m.db", tmp_path / "out"
    args = [
        "--suite",
        "fake",
        "--models",
        "fake-strong,fake-weak",
        "--db",
        str(db),
        "--out",
        str(out),
        "--limit",
        "9",
        "--no-preflight",
    ]
    assert matrix.main(args) == 0
    with Store(db) as s:
        first = [r.run_id for r in s.list_runs()]
    assert len(first) == 2
    md = (out / "matrix.md").read_text(encoding="utf-8")
    assert "fake-strong" in md and "fake-weak" in md and "SYNTHETIC" in md
    assert matrix.main([*args, "--resume"]) == 0  # nothing changed: both runs are reused, none repeated
    with Store(db) as s:
        assert [r.run_id for r in s.list_runs()] == first
    assert matrix.main([*args, "--report-only"]) == 0
    capsys.readouterr()


def test_matrix_cli_reports_a_failing_model_and_still_writes_the_rest(tmp_path):
    db, out = tmp_path / "m.db", tmp_path / "out"
    code = matrix.main(
        [
            "--suite",
            "fake",
            "--models",
            "fake-strong,fake-mid",
            "--db",
            str(db),
            "--out",
            str(out),
            "--limit",
            "5",
            "--no-preflight",
        ]
    )
    assert code == 0
    # a suite that does not exist makes every subprocess exit 2: nothing to report, exit 2
    assert (
        matrix.main(
            ["--suite", "nope", "--models", "x", "--db", str(tmp_path / "n.db"), "--out", str(out / "n")]
        )
        == 2
    )


def test_matrix_cli_rejects_a_bad_price(tmp_path):
    assert (
        matrix.main(["--suite", "fake", "--models", "m", "--price", "oops", "--db", str(tmp_path / "x.db")])
        == 2
    )


def test_paid_model_prices_flow_into_cost(tmp_path):
    db, out = tmp_path / "p.db", tmp_path / "out"
    assert (
        matrix.main(
            [
                "--suite",
                "fake",
                "--models",
                "paid-model",
                "--db",
                str(db),
                "--out",
                str(out),
                "--limit",
                "4",
                "--no-preflight",
                "--price",
                "paid-model=3,15",
                "--usd-to-inr",
                "80",
            ]
        )
        == 0
    )
    md = (out / "matrix.md").read_text(encoding="utf-8")
    assert "INR 0.00" not in md.split("## Latency, tokens, cost")[1].split("##")[0]


def test_matrix_warns_when_models_ran_against_different_feature_versions():
    r = runs(*SPECS)
    r[0][0].prompt_hash, r[1][0].prompt_hash, r[2][0].prompt_hash = "p1", "p1", "p2"
    d = build_matrix("fake", r)
    assert any("different prompt (p1, p2)" in n and "Rerun them on one version" in n for n in d.notes)
    same = runs(*SPECS)
    for run, _ in same:
        run.prompt_hash = "p1"
    assert not any("different prompt" in n for n in build_matrix("fake", same).notes)


# ---- report-only picks full-suite runs; pins; absolute db; clock and orphan notes ----


def stored_run(store, model, results, *, limit=None, ids=None, total=None, status="complete", run_id=None):
    from harness.models import Run

    total = total if total is not None else len(make_results(results))
    run = Run(
        run_id=run_id or f"r-{model}-{len(store.list_runs()) + 1}",
        suite="fake",
        model=model,
        status=status,
        started_at=f"2026-11-10T10:{len(store.list_runs()):02d}:00",
        n_cases=len(make_results(results)),
        config={"limit": limit, "ids": ids, "n_cases_total": total},
    )
    store.create_run(run)
    for r in make_results(results):
        store.save_result(run.run_id, r)
    return run


def test_report_only_skips_smoke_and_partial_runs(store, tmp_path):
    full = stored_run(store, "m", "A1:a+ A2:a+ A3:a+ T1:t+", run_id="full")
    stored_run(
        store, "m", "A1:a+ A2:a+", limit=2, total=4, run_id="smoke-limit"
    )  # newer, complete, but limited
    stored_run(store, "m", "A1:a+ A2:a+", ids=["A1", "A2"], total=4, run_id="smoke-ids")
    stored_run(store, "m", "A1:a+ A2:a+", total=4, run_id="short")  # n_cases 2 of a 4-case file
    assert matrix.latest_complete(store, "fake", "m").run_id == "full"
    assert matrix.latest_complete(store, "fake", "m", full_suite=False).run_id == "short"
    assert full.is_full_suite
    out = tmp_path / "out"
    code = matrix.main(
        ["--suite", "fake", "--models", "m", "--db", str(store.path), "--out", str(out), "--report-only"]
    )
    assert code == 0 and "full" in (out / "matrix.md").read_text(encoding="utf-8")
    only_smoke = tmp_path / "smoke.db"
    from harness.storage import Store

    with Store(only_smoke) as s2:
        stored_run(s2, "m", "A1:a+ A2:a+", limit=2, total=4, run_id="only-a-smoke")
    code = matrix.main(
        ["--suite", "fake", "--models", "m", "--db", str(only_smoke), "--out", str(out), "--report-only"]
    )
    assert code == 2  # nothing but a smoke run exists: no matrix rather than a wrong one


def test_a_pinned_run_overrides_the_latest_one(store, tmp_path):
    stored_run(store, "m", "A1:a+ A2:a+", run_id="older")
    stored_run(store, "m", "A1:a- A2:a-", run_id="newer")  # the noise-floor rerun, say
    out = tmp_path / "out"
    base = ["--suite", "fake", "--models", "m", "--db", str(store.path), "--out", str(out), "--report-only"]
    assert matrix.main(base) == 0 and "newer" in (out / "matrix.md").read_text(encoding="utf-8")
    assert matrix.main([*base, "--pin", "m=older"]) == 0
    text = (out / "matrix.md").read_text(encoding="utf-8")
    assert "older" in text and "newer" not in text
    assert (
        matrix.main([*base, "--pin", "other-model=older"]) == 2
    )  # a pin for a model not in --models is a typo
    assert matrix.main([*base, "--pin", "oops"]) == 2


def test_a_pin_must_belong_to_that_model(store, tmp_path):
    stored_run(store, "m1", "A1:a+", run_id="r-m1")
    stored_run(store, "m2", "A1:a+", run_id="r-m2")
    out = tmp_path / "out"
    code = matrix.main(
        [
            "--suite",
            "fake",
            "--models",
            "m1",
            "--db",
            str(store.path),
            "--out",
            str(out),
            "--report-only",
            "--pin",
            "m1=r-m2",
        ]
    )
    assert code == 2


def test_a_relative_db_path_reaches_the_children_in_the_same_place(tmp_path, monkeypatch):
    """The children run in the harness folder; a relative --db must still mean the caller's folder."""
    monkeypatch.chdir(tmp_path)
    code = matrix.main(
        [
            "--suite",
            "fake",
            "--models",
            "fake-strong",
            "--db",
            "rel.db",
            "--out",
            "out",
            "--limit",
            "4",
            "--no-preflight",
        ]
    )
    assert code == 0
    assert (tmp_path / "rel.db").exists() and (tmp_path / "out" / "matrix.md").exists()
    assert not (matrix.ROOT / "rel.db").exists()


def test_matrix_notes_different_clocks_and_orphan_flags():
    r = runs(*SPECS)
    r[0][0].config["clock_now"] = "2026-11-10T09:00:00+05:30"
    from harness.run import flag_result

    flag_result(r[1][1][0], "orphan_running")
    d = build_matrix("fake", r)
    assert any("different frozen clocks" in n and "wall clock" in n for n in d.notes)
    assert any("mid" in n and "1 case(s) ran while a timed-out case was still working" in n for n in d.notes)
    assert not any("different frozen clocks" in n for n in build_matrix("fake", runs(*SPECS)).notes)
