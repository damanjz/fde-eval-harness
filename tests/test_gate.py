import json
import subprocess
import sys

import pytest

from harness import gate
from harness.gate import evaluate_gate
from harness.storage import write_run_file
from tests.conftest import FIXTURES, ROOT, full_run, make_results


def pair(a_spec: str, b_spec: str, a_kw: dict | None = None, b_kw: dict | None = None):
    """Two full-suite runs with the hashes a gate needs. a_kw and b_kw override the run fields."""
    a, b = make_results(a_spec), make_results(b_spec)
    return (full_run("base", a, **(a_kw or {})), a), (full_run("cand", b, **(b_kw or {})), b)


def many(n_a_pass: int, total: int = 20, cat: str = "a") -> str:
    return " ".join(f"A{i}:{cat}{'+' if i <= n_a_pass else '-'}" for i in range(1, total + 1))


def test_pass_when_candidate_is_equal_or_better():
    base, cand = pair(many(15), many(15))
    assert evaluate_gate(base, cand).exit_code == 0
    base, cand = pair(many(15), many(18))
    assert evaluate_gate(base, cand).exit_code == 0


def test_overall_drop_beyond_max_drop_blocks_and_at_the_limit_passes():
    base, cand = pair(many(15), many(12))  # drop of 3
    assert evaluate_gate(base, cand, max_drop=2).exit_code == 1
    assert evaluate_gate(base, cand, max_drop=3).exit_code == 0
    assert evaluate_gate(base, cand).exit_code == 1  # the default allows a drop of 2, not 3
    assert evaluate_gate(*pair(many(15), many(13))).exit_code == 0
    res = evaluate_gate(base, cand, max_drop=2)
    assert "drop of 3, allowed 2" in res.checks[0].detail and not res.checks[0].ok


def test_must_pass_category_losing_one_case_blocks_even_inside_the_overall_tolerance():
    base = "A1:a+ A2:a+ A3:a+ A4:a+ R1:r+ R2:r+ R3:r+"
    cand = "A1:a+ A2:a+ A3:a+ A4:a+ R1:r+ R2:r+ R3:r-"
    b, c = pair(base, cand)
    res = evaluate_gate(b, c, max_drop=2, must_pass=["refuse"])
    assert res.exit_code == 1
    assert any(not k.ok and "lost: R3" in k.detail for k in res.checks)
    assert evaluate_gate(b, c, max_drop=2).exit_code == 0  # same pair, no must-pass: tolerated


def test_must_pass_is_per_case_so_a_fix_cannot_offset_a_break():
    base = "R1:r+ R2:r- A1:a+"
    cand = "R1:r- R2:r+ A1:a+"  # same count in refuse, but R1 now fails
    b, c = pair(base, cand)
    assert evaluate_gate(b, c, must_pass=["refuse"]).exit_code == 1
    assert evaluate_gate(b, c, must_pass=["refuse"], count_only=True).exit_code == 0


def test_count_only_blocks_when_the_count_drops():
    b, c = pair("R1:r+ R2:r+ A1:a+", "R1:r+ R2:r- A1:a+")
    assert evaluate_gate(b, c, must_pass=["refuse"], count_only=True).exit_code == 1


def test_a_must_pass_category_that_only_improves_passes():
    b, c = pair("R1:r- R2:r+ A1:a+", "R1:r+ R2:r+ A1:a+")
    assert evaluate_gate(b, c, must_pass=["refuse"]).exit_code == 0


def test_errored_and_timed_out_cases_count_as_failures_never_skipped():
    b, c = pair("A1:a+ A2:a+ A3:a+ A4:a+", "A1:a+ A2:a+ A3:ae A4:a+")
    res = evaluate_gate(b, c, max_drop=0)
    assert (
        res.exit_code == 1 and res.comparison.b_pass == 3
    )  # the errored case is a failure, not a skipped case
    assert any("count as failures" in n for n in res.notes)
    assert (
        evaluate_gate(b, c, max_drop=1).exit_code == 0
    )  # and it spends the --max-drop budget like any failure
    b, c = pair("R1:r+ A1:a+", "R1:re A1:a+")
    assert evaluate_gate(b, c, must_pass=["refuse"]).exit_code == 1


# ---- exit 2: the runs cannot be compared ----


@pytest.mark.parametrize(
    "a_kw,b_kw,needle",
    [
        ({"suite": "x"}, {"suite": "y"}, "different suites"),
        ({"case_set_hash": "h1"}, {"case_set_hash": "h2"}, "different case files"),
        ({"config": {"scorer_hash": "s1"}}, {"config": {"scorer_hash": "s2"}}, "different scoring code"),
        (
            {"config": {"clock_now": "2026-11-10T09:00:00+05:30"}},
            {"config": {"clock_now": "2026-11-12T09:00:00+05:30"}},
            "different clocks",
        ),
        (
            {"config": {"clock_now": "2026-11-10T09:00:00+05:30"}},
            {},
            "different clocks",
        ),  # frozen vs wall clock
    ],
)
def test_not_comparable_exit_2(a_kw, b_kw, needle):
    b, c = pair(many(10), many(10), a_kw, b_kw)
    res = evaluate_gate(b, c)
    assert res.exit_code == 2 and any(needle in p for p in res.problems) and res.checks == []


def test_the_same_frozen_clock_is_comparable():
    clock = {"config": {"clock_now": "2026-11-10T09:00:00+05:30"}}
    assert evaluate_gate(*pair(many(10), many(10), clock, clock)).exit_code == 0


def test_a_limit_run_cannot_pass_the_gate():
    full = many(18, 20)
    limited = " ".join(full.split()[:10])  # the candidate ran only the first 10 cases and passed them all
    b, c = pair(full, limited)
    res = evaluate_gate(b, c, max_drop=100)
    assert res.exit_code == 2 and any("different case-id sets" in p for p in res.problems)
    b, c = pair(limited, full)
    assert evaluate_gate(b, c).exit_code == 2  # and the other way round


def test_two_partial_runs_with_the_same_ids_cannot_pass_either():
    """Same ids, same hashes: only the recorded size of the case file shows they are partial."""
    first10 = " ".join(many(10, 20).split()[:10])
    a, b = make_results(first10), make_results(first10)
    base = full_run("base", a, config={"n_cases_total": 20, "limit": 10})
    cand = full_run("cand", b, config={"n_cases_total": 20, "limit": 10})
    res = evaluate_gate((base, a), (cand, b))
    assert res.exit_code == 2
    assert any("covers 10 of 20 cases" in p for p in res.problems) and any(
        "--limit or --ids" in p for p in res.problems
    )
    # even if a run claims no limit, covering 10 of 20 cases is caught
    base = full_run("base", a, config={"n_cases_total": 20})
    cand = full_run("cand", b, config={"n_cases_total": 20})
    assert evaluate_gate((base, a), (cand, b)).exit_code == 2


def test_a_run_with_ids_selected_is_not_gate_grade():
    b, c = pair(many(10), many(10), None, {"config": {"ids": ["A1", "A2"]}})
    res = evaluate_gate(b, c)
    assert res.exit_code == 2 and any("--limit or --ids" in p for p in res.problems)


def test_zero_cases_and_missing_hashes_are_exit_2_not_a_pass():
    base, _ = pair(many(10), many(10))
    empty = full_run("empty", [])
    assert evaluate_gate(base, (empty, [])).exit_code == 2
    assert any("no results" in p for p in evaluate_gate(base, (empty, [])).problems)
    no_hash, res_ = pair(many(10), many(10), None, {"case_set_hash": None})
    r = evaluate_gate(no_hash, res_)
    assert r.exit_code == 2 and any("no case-file hash" in p for p in r.problems)
    no_scorer = pair(many(10), many(10), None, {"config": {"scorer_hash": None}})
    r = evaluate_gate(*no_scorer)
    assert r.exit_code == 2 and any("no scoring-code hash" in p for p in r.problems)
    # a run that records neither the case-file size nor whether it was limited cannot be shown to be complete
    cand_res = make_results(many(10))
    from tests.conftest import make_run

    vague = make_run("vague", config={"scorer_hash": "s1"}, case_set_hash="h1")
    r = evaluate_gate(pair(many(10), many(10))[0], (vague, cand_res))
    assert r.exit_code == 2 and any("does not record how many cases" in p for p in r.problems)


def test_unknown_must_pass_category_and_bad_max_drop_are_exit_2():
    b, c = pair(many(5), many(5))
    res = evaluate_gate(b, c, must_pass=["refuse"])
    assert res.exit_code == 2 and "must-pass category not in the runs" in res.problems[0]
    assert evaluate_gate(b, c, max_drop=-1).exit_code == 2


def test_mcnemar_note_is_context_only():
    b, c = pair(many(15), many(14))
    res = evaluate_gate(b, c, max_drop=1)
    assert res.exit_code == 0 and any("Context only" in n and "within noise" in n for n in res.notes)


# ---- --rerun-lost: a lost must-pass case is rerun before it blocks ----

LOST = ("R1:r+ R2:r+ A1:a+ A2:a+", "R1:r- R2:r+ A1:a+ A2:a+")


def test_a_lost_case_that_passes_most_reruns_is_cleared_as_noise():
    b, c = pair(*LOST)
    calls = []

    def rerun(case_id):
        calls.append(case_id)
        return [True, False, True]  # 2 of 3 pass: strict majority

    res = evaluate_gate(b, c, must_pass=["refuse"], rerun=rerun)
    assert res.exit_code == 0 and calls == ["R1"]
    assert "R1 (reran 3x: 2 passed, cleared as noise)" in res.checks[1].detail


def test_a_lost_case_that_fails_most_reruns_is_a_confirmed_loss():
    b, c = pair(*LOST)
    res = evaluate_gate(b, c, must_pass=["refuse"], rerun=lambda cid: [False, True, False])
    assert res.exit_code == 1 and "confirmed lost" in res.checks[1].detail
    # a tie is not a majority: 1 of 2 passing does not clear it
    assert evaluate_gate(b, c, must_pass=["refuse"], rerun=lambda cid: [True, False]).exit_code == 1
    assert evaluate_gate(b, c, must_pass=["refuse"], rerun=lambda cid: [True, True]).exit_code == 0


def test_reruns_happen_only_for_lost_must_pass_cases_in_per_case_mode():
    b, c = pair(*LOST)
    calls = []

    def rerun(cid):
        calls.append(cid)
        return [True]

    assert evaluate_gate(b, c, rerun=rerun).exit_code == 0 and calls == []  # no must-pass: nothing to rerun
    assert (
        evaluate_gate(b, c, must_pass=["refuse"], count_only=True, rerun=rerun).exit_code == 1 and calls == []
    )
    ok_b, ok_c = pair("R1:r+ A1:a+", "R1:r+ A1:a+")
    assert evaluate_gate(ok_b, ok_c, must_pass=["refuse"], rerun=rerun).exit_code == 0 and calls == []


def test_rerun_lost_cli_refuses_when_the_feature_changed_since_the_candidate_run(tmp_path, capsys):
    from tests.conftest import ScriptedAdapter

    ad = ScriptedAdapter(n=4)
    ra, rb = make_results("C1:r+ C2:a+ C3:r+ C4:a+"), make_results("C1:r- C2:a+ C3:r+ C4:a+")
    base = full_run("base", ra, suite="scripted", model="m")
    cand = full_run("cand", rb, suite="scripted", model="m")  # fingerprint does not match the live adapter
    fa = write_run_file(tmp_path / "a.run.json", base, ra)
    fb = write_run_file(tmp_path / "b.run.json", cand, rb)
    code = gate.main(
        [
            "--baseline",
            str(fa),
            "--candidate",
            str(fb),
            "--must-pass",
            "refuse",
            "--rerun-lost",
            "3",
            "--db",
            str(tmp_path / "none.db"),
        ],
        adapter=ad,
    )
    assert code == 2 and "changed since the candidate run" in capsys.readouterr().err and ad.calls == []


def test_rerun_lost_cli_reruns_on_the_candidates_model(tmp_path, capsys):
    from harness.run import fingerprint_of
    from tests.conftest import ScriptedAdapter

    ad = ScriptedAdapter(n=4)
    fp = fingerprint_of(ad.provenance(), ad.cases())
    cfg = {"scorer_hash": fp[4]}
    ra, rb = make_results("C1:r+ C2:a+ C3:r+ C4:a+"), make_results("C1:r- C2:a+ C3:r+ C4:a+")
    kw = {
        "suite": "scripted",
        "model": "m",
        "git_hash": fp[0],
        "prompt_hash": fp[1],
        "code_hash": fp[2],
        "case_set_hash": fp[3],
    }
    base = full_run("base", ra, config=cfg, **kw)
    cand = full_run("cand", rb, config=cfg, **kw)
    fa = write_run_file(tmp_path / "a.run.json", base, ra)
    fb = write_run_file(tmp_path / "b.run.json", cand, rb)
    args = [
        "--baseline",
        str(fa),
        "--candidate",
        str(fb),
        "--must-pass",
        "refuse",
        "--db",
        str(tmp_path / "n.db"),
    ]
    assert gate.main(args) == 1  # no rerun: the loss blocks
    # C1 is category "answer" in the scripted adapter (odd id) and "refuse" in the results above; rerun it live
    code = gate.main([*args, "--rerun-lost", "3", "--no-preflight"], adapter=ad)
    assert ad.calls == ["C1", "C1", "C1"]  # the scripted adapter passes C1 every time, so the loss is cleared
    assert code == 0
    assert "cleared as noise" in capsys.readouterr().out


# ---- malformed input is exit 2, and a crash is never exit 1 ----


def test_cli_exit_codes_and_markdown_summary(tmp_path):
    ra, rb, rc = make_results(many(15)), make_results(many(10)), make_results(many(15))
    rd = make_results(many(15, total=10))
    fa = write_run_file(tmp_path / "a.run.json", full_run("a", ra), ra)
    fb = write_run_file(tmp_path / "b.run.json", full_run("b", rb), rb)
    fc = write_run_file(tmp_path / "c.run.json", full_run("c", rc), rc)
    fd = write_run_file(tmp_path / "d.run.json", full_run("d", rd), rd)
    summary = tmp_path / "summary.md"
    base = ["--db", str(tmp_path / "none.db")]
    assert gate.main(["--baseline", str(fa), "--candidate", str(fc), *base, "--md", str(summary)]) == 0
    assert (
        gate.main(
            ["--baseline", str(fa), "--candidate", str(fb), "--max-drop", "2", *base, "--md", str(summary)]
        )
        == 1
    )
    assert gate.main(["--baseline", str(fa), "--candidate", str(fd), *base, "--md", str(summary)]) == 2
    assert gate.main(["--baseline", str(fa), "--candidate", "missing-run", *base]) == 2
    text = summary.read_text(encoding="utf-8")
    assert text.count("# Gate:") == 3 and "PASS" in text and "BLOCKED" in text and "NOT COMPARABLE" in text


def test_malformed_run_files_exit_2_never_1(tmp_path, capsys):
    good_results = make_results(many(10))
    good = write_run_file(tmp_path / "good.run.json", full_run("g", good_results), good_results)
    data = json.loads(good.read_text(encoding="utf-8"))
    cases = {
        "not-json": "this is not json",
        "list": "[1, 2, 3]",
        "wrong-format": json.dumps({"format": "other"}),
        "no-run": json.dumps({"format": data["format"], "results": data["results"]}),
        "no-results": json.dumps({"format": data["format"], "run": data["run"]}),
        "result-missing-field": json.dumps({**data, "results": [{"case_id": "A1"}]}),
        "result-not-object": json.dumps({**data, "results": ["A1"]}),
        "duplicate-ids": json.dumps({**data, "results": [data["results"][0], data["results"][0]]}),
        "empty-id": json.dumps({**data, "results": [{**data["results"][0], "case_id": ""}]}),
        "run-not-object": json.dumps({**data, "run": 7}),
    }
    base = ["--db", str(tmp_path / "none.db")]
    for name, body in cases.items():
        bad = tmp_path / f"{name}.run.json"
        bad.write_text(body, encoding="utf-8")
        for args in (
            ["--baseline", str(good), "--candidate", str(bad)],
            ["--baseline", str(bad), "--candidate", str(good)],
        ):
            assert gate.main([*args, *base]) == 2, name
    capsys.readouterr()


def test_an_unexpected_crash_is_exit_2_not_a_regression(tmp_path, monkeypatch, capsys):
    results = make_results(many(10))
    f = write_run_file(tmp_path / "a.run.json", full_run("a", results), results)

    def boom(*a, **k):
        raise RuntimeError("bug in the gate")

    monkeypatch.setattr(gate, "evaluate_gate", boom)
    code = gate.main(["--baseline", str(f), "--candidate", str(f), "--db", str(tmp_path / "none.db")])
    assert code == 2 and "internal error (RuntimeError" in capsys.readouterr().err


def test_a_stored_pass_on_an_errored_case_is_not_a_pass(tmp_path):
    """A hand-edited or buggy file that says passed=true for an errored case must not help the candidate."""
    ra, rb = make_results(many(10)), make_results(many(10))
    fa = write_run_file(tmp_path / "a.run.json", full_run("a", ra), ra)
    fb = write_run_file(tmp_path / "b.run.json", full_run("b", rb), rb)
    data = json.loads(fb.read_text(encoding="utf-8"))
    data["results"][0]["status"] = "error"
    data["results"][0]["passed"] = True
    fb.write_text(json.dumps(data), encoding="utf-8")
    code = gate.main(
        ["--baseline", str(fa), "--candidate", str(fb), "--max-drop", "0", "--db", str(tmp_path / "n.db")]
    )
    assert code == 1


# ---- the committed fixture runs: real recorded project 1 runs, exit codes 0, 1 and 2 ----

RUN2 = FIXTURES / "support_agent-p1-run2-agent.run.json"
RUN3 = FIXTURES / "support_agent-p1-run3-agent.run.json"
RUN4 = FIXTURES / "support_agent-p1-run4-agent.run.json"
RUN4_NONE = FIXTURES / "support_agent-p1-run4-no-retrieval.run.json"
RUN4_FIRST20 = FIXTURES / "support_agent-p1-run4-agent-first20.run.json"


@pytest.fixture
def cli(tmp_path):
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "harness.gate", *args, "--db", str(tmp_path / "unused.db")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    return run


def test_fixture_gate_passes_an_improvement_that_is_within_noise(cli):
    r = cli("--baseline", str(RUN2), "--candidate", str(RUN3), "--max-drop", "2", "--must-pass", "refuse")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_fixture_gate_blocks_a_refusal_regression_that_the_overall_tolerance_would_allow(cli):
    args = ("--baseline", str(RUN3), "--candidate", str(RUN4), "--max-drop", "2")
    blocked = cli(*args, "--must-pass", "refuse")
    assert blocked.returncode == 1 and "lost: R10" in blocked.stdout
    assert cli(*args).returncode == 0  # one case down overall is inside --max-drop 2


def test_fixture_gate_blocks_a_large_overall_drop(cli):
    r = cli(
        "--baseline", str(RUN4), "--candidate", str(RUN4_NONE), "--max-drop", "2", "--must-pass", "refuse"
    )
    assert r.returncode == 1 and "BLOCK" in r.stdout
    assert (
        "variant agent" in r.stdout and "variant no_retrieval" in r.stdout
    )  # the header says what each run was


def test_fixture_gate_refuses_to_compare_a_limit_run(cli):
    r = cli("--baseline", str(RUN4), "--candidate", str(RUN4_FIRST20), "--max-drop", "2")
    assert r.returncode == 2 and "NOT COMPARABLE" in r.stdout and "different case-id sets" in r.stdout
    assert "covers 20 of 60 cases" in r.stdout
    both = cli("--baseline", str(RUN4_FIRST20), "--candidate", str(RUN4_FIRST20), "--max-drop", "2")
    assert both.returncode == 2  # partial against itself cannot pass


def test_fixture_runs_are_complete_real_runs():
    from harness.storage import read_run_file

    for f in (RUN2, RUN3, RUN4, RUN4_NONE):
        run, results = read_run_file(f)
        assert len(results) == 60 and run.suite == "support_agent" and run.case_set_hash
        assert {r.category for r in results} == {"answer", "tool", "refuse"}
        assert run.config["imported"] is True and "evals/results" in run.config["source"]
        assert run.is_full_suite and run.n_cases_total == 60 and run.scorer_hash and run.clock_now is None
    first20, res20 = read_run_file(RUN4_FIRST20)
    assert len(res20) == 20 and not first20.is_full_suite and first20.config["limit"] == 20
