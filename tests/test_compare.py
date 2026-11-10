import os
import re

import pytest

from harness import compare as cmpmod
from harness.compare import (
    compare_runs,
    explain,
    noise_floor_lines,
    not_comparable,
    render_markdown,
)
from harness.report import compare_html
from harness.stats import mcnemar_exact
from tests.conftest import ROOT as ROOT_FOR_TESTS
from tests.conftest import make_results, make_run, res


def build(a_spec: str, b_spec: str, **kw):
    a, b = make_results(a_spec), make_results(b_spec)
    return compare_runs(make_run("A", model="ma"), a, make_run("B", model="mb"), b, n_boot=2000, **kw)


def test_flip_counts_and_mcnemar_come_from_the_paired_outcomes():
    # A passes 1-4, B passes 3-6: A1,A2 regressed; A5,A6 fixed; A3,A4 stay passed; A7 fails both
    a = " ".join(f"A{i}:a{'+' if i <= 4 else '-'}" for i in range(1, 8))
    b = " ".join(f"A{i}:a{'+' if 3 <= i <= 6 else '-'}" for i in range(1, 8))
    c = build(a, b)
    assert c.n == 7 and c.a_pass == 4 and c.b_pass == 4
    assert (c.mcnemar.only_a, c.mcnemar.only_b) == (2, 2) and c.mcnemar.p_value == 1.0
    assert sorted((f.case_id, f.direction) for f in c.flips) == [
        ("A1", "regressed"),
        ("A2", "regressed"),
        ("A5", "fixed"),
        ("A6", "fixed"),
    ]
    assert c.n_same == 3 and c.verdict == "noise"


def test_clear_regression_is_flagged_by_both_tests():
    a = " ".join(f"A{i}:a+" for i in range(1, 21))
    b = " ".join(f"A{i}:a{'-' if i <= 10 else '+'}" for i in range(1, 21))
    c = build(a, b)
    assert c.mcnemar.p_value == pytest.approx(mcnemar_exact(10, 0).p_value) and c.mcnemar.significant
    assert c.boot.excludes_zero and c.boot.diff == pytest.approx(-0.5)
    assert c.verdict == "worse" and c.net_cases == -10
    assert "WORSE" in render_markdown(c)


def test_identical_runs():
    spec = "A1:a+ A2:a- T1:t+"
    c = build(spec, spec)
    assert c.verdict == "identical" and c.flips == [] and c.mcnemar.p_value == 1.0
    assert "Nothing flipped" in " ".join(explain(c))


def test_per_category_deltas_and_flags():
    a = "A1:a+ A2:a+ A3:a+ T1:t+ T2:t- R1:r+"
    b = "A1:a- A2:a- A3:a- T1:t+ T2:t+ R1:r+"
    c = build(a, b)
    by = {d.category: d for d in c.categories}
    assert (by["answer"].a_pass, by["answer"].b_pass, by["answer"].delta) == (3, 0, -3)
    assert (by["tool"].a_pass, by["tool"].b_pass, by["tool"].delta) == (1, 2, 1)
    assert by["refuse"].delta == 0 and by["refuse"].mcnemar.flips == 0
    assert by["answer"].mcnemar.p_value == pytest.approx(0.25)
    md = render_markdown(c)
    assert "weak: within noise" in md and "no flips" in md


def test_errors_count_as_failures_and_are_reported():
    c = build("A1:a+ A2:a+ A3:a+", "A1:a+ A2:a+ A3:ae")
    assert c.b_pass == 2 and c.b_errors == 1 and c.mcnemar.only_a == 1
    assert any("Errored or timed-out" in n for n in c.notes)


def test_cases_only_in_one_run_are_ignored_but_reported():
    c = build("A1:a+ A2:a+ A3:a-", "A1:a+ A2:a- A4:a+")
    assert c.n == 2 and c.only_in_a == ["A3"] and c.only_in_b == ["A4"]
    assert any("only in A" in n for n in c.notes) and any("only in B" in n for n in c.notes)


def test_latency_tokens_and_cost_deltas_use_completed_cases_only():
    a = [
        res("A1", "answer", True, latency_ms=1000, pt=100, ct=10),
        res("A2", "answer", True, latency_ms=3000, pt=300, ct=30),
    ]
    b = [
        res("A1", "answer", True, latency_ms=2000, pt=200, ct=20),
        res("A2", "answer", True, latency_ms=60000, status="timeout"),
    ]
    a[0].cost_inr, a[1].cost_inr = 0.5, 1.5
    c = compare_runs(make_run("A"), a, make_run("B"), b, n_boot=200)
    assert c.a_perf.p50_ms == 1000 and c.a_perf.p95_ms == 3000
    assert c.b_perf.completed == 1 and c.b_perf.p95_ms == 2000  # the timeout does not skew p95
    assert c.a_perf.mean_prompt_tokens == 200 and c.b_perf.mean_prompt_tokens == 200
    assert c.a_perf.total_cost_inr == 2.0 and c.a_perf.mean_cost_inr == 1.0


def test_not_comparable_reasons():
    a_run, b_run = make_run("A", suite="x", case_set_hash="h1"), make_run("B", suite="y", case_set_hash="h2")
    a, b = make_results("A1:a+ A2:a+"), make_results("A1:a+ A3:a+")
    problems = not_comparable(a_run, a, b_run, b)
    assert len(problems) == 3
    assert "different suites" in problems[0] and "A2" in problems[1] and "A3" in problems[1]
    assert "case files" in problems[2]
    same = make_run("C", case_set_hash="h1")
    assert not_comparable(same, a, make_run("D", case_set_hash="h1"), make_results("A2:a+ A1:a-")) == []
    # a missing hash on either side (imported or older run) is not a reason to refuse
    assert not_comparable(make_run("E"), a, make_run("F", case_set_hash="h"), a) == []


def test_provenance_notes_tell_you_what_changed():
    ra = make_run("A", model="m", prompt_hash="p1", code_hash="c1")
    rb = make_run("B", model="m", prompt_hash="p2", code_hash="c1")
    c = compare_runs(ra, make_results("A1:a+"), rb, make_results("A1:a+"), n_boot=100)
    assert any("prompt (p1 vs p2)" in n for n in c.notes)
    same = compare_runs(
        ra,
        make_results("A1:a+"),
        make_run("B", model="m", prompt_hash="p1", code_hash="c1"),
        make_results("A1:a+"),
        n_boot=100,
    )
    assert any("pure run-to-run noise" in n for n in same.notes)


def test_explanations_say_how_many_flips_are_noise():
    c = build(
        " ".join(f"A{i}:a{'+' if i <= 6 else '-'}" for i in range(1, 13)),
        " ".join(f"A{i}:a{'+' if i in (1, 2, 3, 4, 7, 8) else '-'}" for i in range(1, 13)),
    )
    # flips: A5, A6 regressed; A7, A8 fixed -> 4 flips, 2 cancel pairs
    text = " ".join(explain(c))
    assert "4 cases flipped" in text
    assert "split about evenly (2 each way expected); the net 0 is how far this split is from even" in text
    assert (
        "cancel" not in text
    )  # a split that is even is what noise looks like; flips do not 'cancel' it away
    assert "every possible resample of the 12 cases was enumerated exactly (no random draws)" in text
    assert "McNemar exact test, p = 1.000" in text
    assert "no split can reach p < 0.05" in text  # only 4 flips: nothing could be significant
    c2 = build(
        " ".join(f"A{i}:a+" for i in range(1, 13)),
        " ".join(f"A{i}:a{'-' if i <= 9 else '+'}" for i in range(1, 13)),
    )
    assert "has to be at least 7 case(s)" in " ".join(explain(c2))  # 9 flips need a net of 7


def test_noise_floor_section():
    spec_a = " ".join(f"A{i}:a+" for i in range(1, 31))
    spec_b = " ".join(f"A{i}:a{'-' if i in (3, 9, 20) else '+'}" for i in range(1, 31))
    ra, rb = make_run("A", model="m"), make_run("B", model="m")
    c = compare_runs(ra, make_results(spec_a), rb, make_results(spec_b), n_boot=300)
    lines = noise_floor_lines(c)
    assert (
        "3 of 30 cases flipped (3 pass to fail, 0 fail to pass)" in lines[0]
        and "all of it is noise" in lines[0]
    )
    # seen from a gate the first run is the baseline: the second run lost 3 cases
    assert (
        "overall drop was 3 case(s)" in lines[1]
        and "--max-drop of 3 or more would have let this pair through" in lines[1]
    )
    assert "with 3 flips a net change of up to 3 case(s) cannot be told from noise" in lines[2]
    assert "larger of 3 and 3 (3)" in lines[2] and "one pair of runs" in lines[2]
    assert "Smallest --max-drop" not in " ".join(lines)  # the old, loosely worded claim
    other = compare_runs(
        make_run("A", model="m1"),
        make_results(spec_a),
        make_run("B", model="m2"),
        make_results(spec_b),
        n_boot=100,
    )
    assert "WARNING" in noise_floor_lines(other)[0]
    assert "## Noise floor" in render_markdown(c, noise_floor=True)
    assert "## Noise floor" not in render_markdown(c)


def test_markdown_has_every_section():
    c = build("A1:a+ A2:a- T1:t+", "A1:a- A2:a+ T1:t+")
    md = render_markdown(c)
    for section in (
        "## Verdict",
        "## In plain words",
        "## By category",
        "## Flipped cases",
        "## Latency, tokens, cost",
    ):
        assert section in md
    assert "| A1 | answer | pass -> fail |" in md and "| A2 | answer | fail -> pass |" in md


def test_html_report_is_self_contained():
    c = build("A1:a+ A2:a- T1:t+ R1:r+", "A1:a- A2:a+ T1:t+ R1:r-")
    html = compare_html(c)
    assert html.startswith("<!doctype html>") and "<svg" in html and "Flipped cases" in html
    assert "<script" not in html and "http://" not in html and "https://" not in html
    assert not re.search(r"(src|href)=", html)
    assert "prefers-color-scheme: dark" in html
    assert "SYNTHETIC" not in html
    synth = compare_runs(
        make_run("A", config={"synthetic": True}),
        make_results("A1:a+"),
        make_run("B"),
        make_results("A1:a+"),
        n_boot=100,
    )
    assert "SYNTHETIC" in compare_html(synth) and "SYNTHETIC" in render_markdown(synth)


def test_cli_compare_writes_files_and_handles_missing_runs(tmp_path, capsys):
    from harness.storage import write_run_file

    ra, rb = make_run("A"), make_run("B")
    fa = write_run_file(tmp_path / "a.run.json", ra, make_results("A1:a+ A2:a-"))
    fb = write_run_file(tmp_path / "b.run.json", rb, make_results("A1:a+ A2:a+"))
    md, html = tmp_path / "c.md", tmp_path / "c.html"
    assert cmpmod.main([str(fa), str(fb), "--md", str(md), "--html", str(html), "--boot", "200"]) == 0
    assert md.read_text(encoding="utf-8").startswith("# Compare") and "<svg" in html.read_text(
        encoding="utf-8"
    )
    assert cmpmod.main([str(fa), "nope", "--db", str(tmp_path / "none.db")]) == 2
    capsys.readouterr()


def test_cli_survives_characters_the_console_cannot_print(tmp_path):
    import subprocess
    import sys

    from harness.storage import write_run_file

    a = make_results("A1:a+")
    b = make_results("A1:a-")
    b[0].reasons = ["price shown as ₹ 99 instead of INR 99"]
    fa = write_run_file(tmp_path / "a.run.json", make_run("A"), a)
    fb = write_run_file(tmp_path / "b.run.json", make_run("B"), b)
    p = subprocess.run(
        [sys.executable, "-m", "harness.compare", str(fa), str(fb), "--boot", "100"],
        cwd=ROOT_FOR_TESTS,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
    )
    assert p.returncode == 0, p.stderr.decode("cp1252", "replace")
    assert b"price shown as" in p.stdout


def test_noise_floor_reports_the_per_category_flip_rate_and_the_false_block_cost():
    spec_a = " ".join([f"A{i}:a+" for i in range(1, 21)] + [f"R{i}:r+" for i in range(1, 11)])
    # the rerun lost 2 of 10 refuse cases and 1 of 20 answer cases
    spec_b = " ".join(
        [f"A{i}:a{'-' if i == 1 else '+'}" for i in range(1, 21)]
        + [f"R{i}:r{'-' if i <= 2 else '+'}" for i in range(1, 11)]
    )
    c = compare_runs(
        make_run("A", model="m"),
        make_results(spec_a),
        make_run("B", model="m"),
        make_results(spec_b),
        n_boot=100,
    )
    text = "\n".join(noise_floor_lines(c))
    assert (
        "refuse: 2 of 10 cases flipped; 2 of 10 cases that passed in the first run failed in the second (20%)"
        in text
    )
    # 1 - (1 - 0.2)^10 = 89%: a strict per-case gate on this category would block about nine reruns in ten
    assert "about 89% of reruns of an unchanged configuration would be blocked" in text
    assert "answer: 1 of 20 cases flipped" in text and "(5%)" in text
    assert "--rerun-lost" in text


def test_noise_floor_with_no_loss_only_bounds_the_rate():
    spec = " ".join(f"R{i}:r+" for i in range(1, 11))
    c = compare_runs(
        make_run("A", model="m"), make_results(spec), make_run("B", model="m"), make_results(spec), n_boot=100
    )
    text = "\n".join(noise_floor_lines(c))
    assert "none of the 10 cases that passed in the first run failed in the second" in text
    assert "below about 30% (rule of three)" in text and "bounded, not measured" in text


def test_strict_comparability_for_the_gate_but_not_for_a_plain_compare():
    from tests.conftest import full_run

    res_a, res_b = make_results("A1:a+ A2:a+"), make_results("A1:a+ A2:a+")
    bare_a, bare_b = make_run("A"), make_run("B")  # no hashes, no coverage record
    assert not_comparable(bare_a, res_a, bare_b, res_b) == []  # a plain compare accepts it
    strict = not_comparable(bare_a, res_a, bare_b, res_b, strict=True)
    assert any("does not record how many cases" in p for p in strict)
    assert any("no case-file hash" in p for p in strict) and any("no scoring-code hash" in p for p in strict)
    ok_a, ok_b = full_run("A", res_a), full_run("B", res_b)
    assert not_comparable(ok_a, res_a, ok_b, res_b, strict=True) == []
    clocked = full_run("C", res_b, config={"clock_now": "2026-11-10T09:00:00+05:30"})
    assert not_comparable(ok_a, res_a, clocked, res_b) == []  # plain compare only notes it
    assert any("different clocks" in p for p in not_comparable(ok_a, res_a, clocked, res_b, strict=True))
    assert any(
        "scoring code" in n and "none vs s2" not in n
        for n in compare_runs(
            ok_a, res_a, full_run("D", res_b, config={"scorer_hash": "s2"}), res_b, n_boot=50
        ).notes
    )


def test_provenance_notes_name_the_variant_scorer_and_clock_and_the_header_shows_them():
    ra = make_run("A", model="m", config={"variant": "agent", "scorer_hash": "s1"})
    rb = make_run(
        "B",
        model="m",
        config={"variant": "no_retrieval", "scorer_hash": "s2", "clock_now": "2026-11-10T09:00:00+05:30"},
    )
    c = compare_runs(ra, make_results("A1:a+"), rb, make_results("A1:a+"), n_boot=50)
    note = next(n for n in c.notes if n.startswith("Changed between"))
    assert (
        "variant (agent vs no_retrieval)" in note
        and "scoring code (s1 vs s2)" in note
        and "frozen clock (none vs 2026" in note
    )
    md = render_markdown(c)
    assert (
        "variant agent" in md
        and "variant no_retrieval" in md
        and "scorer s2" in md
        and "clock 2026-11-10T09:00:00+05:30" in md
    )
