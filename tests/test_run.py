import time

import pytest

from harness import run as runmod
from harness.models import COMPLETE, ERROR, OK, PARTIAL, TIMEOUT, Run
from harness.run import (
    PreflightError,
    Prices,
    ResumeMismatch,
    call_with_timeout,
    execute_run,
    run_one,
    validate_cases,
)
from tests.conftest import ScriptedAdapter

QUIET = {"log": lambda _m: None, "sleep": lambda _s: None}


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def test_full_run_is_stored_with_provenance(store):
    ad = ScriptedAdapter(n=4, fail={"C2"})
    s = execute_run(ad, "m1", store, **QUIET)
    assert s.run.status == COMPLETE and s.passed == 3 and s.errors == 0 and s.ran == 4
    stored = store.get_run(s.run.run_id)
    assert (stored.git_hash, stored.prompt_hash, stored.code_hash) == ("abc1234", "p1", "c1")
    assert stored.model == "m1" and stored.suite == "scripted" and stored.n_cases == 4
    assert stored.config["synthetic"] is True and stored.config["retries"] == 1
    assert stored.case_set_hash and stored.finished_at
    rows = {r.case_id: r for r in store.results(stored.run_id)}
    assert rows["C2"].passed is False and rows["C2"].reasons == ["bad"] and rows["C1"].passed
    assert rows["C1"].prompt_tokens == 1000 and rows["C1"].completion_tokens == 100
    assert ad.configured == runmod.DEFAULT_TIMEOUT_S


def test_latency_is_measured_by_the_harness_clock(store):
    clock = FakeClock()
    ad = ScriptedAdapter(n=2)
    orig = ad.run_case

    def slow(case, model):
        clock.advance(2.5)
        return orig(case, model)

    ad.run_case = slow
    s = execute_run(ad, "m", store, clock=clock, **QUIET)
    assert [r.latency_ms for r in s.results] == [2500.0, 2500.0]


def test_cost_uses_prices_and_defaults_to_zero(store):
    s = execute_run(ScriptedAdapter(n=1), "m", store, **QUIET)
    assert s.results[0].cost_inr == 0.0
    prices = Prices(usd_in_per_mtok=3.0, usd_out_per_mtok=15.0, usd_to_inr=80.0)
    s2 = execute_run(ScriptedAdapter(n=1), "paid", store, prices=prices, **QUIET)
    # (1000 * 3 + 100 * 15) / 1e6 USD = 0.0045 USD = INR 0.36
    assert s2.results[0].cost_inr == pytest.approx(0.36)
    assert store.get_run(s2.run.run_id).config["prices"]["usd_to_inr"] == 80.0


def test_one_bad_case_never_crashes_the_run(store):
    ad = ScriptedAdapter(n=4, raises={"C2": RuntimeError("boom")}, score_raises={"C3"})
    s = execute_run(ad, "m", store, **QUIET)
    by = {r.case_id: r for r in s.results}
    assert (
        by["C2"].status == ERROR and "RuntimeError: boom" in by["C2"].reasons[0] and by["C2"].passed is False
    )
    assert by["C3"].status == ERROR and "scoring error" in by["C3"].reasons[0]
    assert by["C1"].status == OK and by["C4"].status == OK
    assert s.run.status == PARTIAL and s.errors == 2 and len(s.results) == 4


def test_exceptions_are_retried_then_succeed(store):
    ad = ScriptedAdapter(n=1, raises={"C1": [ConnectionError("blip")]})
    r = run_one(ad, ad.cases()[0], "m", retries=1, sleep=lambda _s: None)
    assert r.status == OK and r.attempts == 2 and ad.calls == ["C1", "C1"]


def test_retries_are_bounded(store):
    ad = ScriptedAdapter(n=1, raises={"C1": RuntimeError("down")})
    r = run_one(ad, ad.cases()[0], "m", retries=2, sleep=lambda _s: None)
    assert r.status == ERROR and r.attempts == 3 and len(ad.calls) == 3


def test_timeouts_are_recorded_and_not_retried(store):
    ad = ScriptedAdapter(n=2)
    orig = ad.run_case

    def sometimes_slow(case, model):
        if case["id"] == "C1":
            time.sleep(0.6)
        return orig(case, model)

    ad.run_case = sometimes_slow
    s = execute_run(ad, "m", store, timeout_s=0.1, retries=3, **QUIET)
    by = {r.case_id: r for r in s.results}
    assert by["C1"].status == TIMEOUT and by["C1"].passed is False and by["C1"].attempts == 1
    assert by["C2"].status == OK
    assert s.run.status == PARTIAL


def test_adapter_raised_timeout_is_a_timeout(store):
    ad = ScriptedAdapter(n=1, raises={"C1": TimeoutError("turn took too long")})
    assert run_one(ad, ad.cases()[0], "m", retries=2, sleep=lambda _s: None).status == TIMEOUT


def test_call_with_timeout_passes_values_and_errors():
    assert call_with_timeout(lambda: 7, 1) == 7
    assert call_with_timeout(lambda: 8, None) == 8
    with pytest.raises(ZeroDivisionError):
        call_with_timeout(lambda: 1 / 0, 1)
    with pytest.raises(TimeoutError):
        call_with_timeout(lambda: time.sleep(0.5), 0.05)


def test_non_dict_output_is_an_error(store):
    ad = ScriptedAdapter(n=1)
    ad.run_case = lambda case, model: "text"
    assert run_one(ad, ad.cases()[0], "m", sleep=lambda _s: None).status == ERROR


def test_resume_runs_only_the_remaining_cases(store):
    ad = ScriptedAdapter(n=5)
    ad.interrupt_at = "C3"
    with pytest.raises(KeyboardInterrupt):
        execute_run(ad, "m", store, **QUIET)
    runs = store.list_runs()
    assert len(runs) == 1 and runs[0].status == PARTIAL
    assert [r.case_id for r in store.results(runs[0].run_id)] == ["C1", "C2"]

    ad2 = ScriptedAdapter(n=5)
    s = execute_run(ad2, "m", store, resume=True, **QUIET)
    assert s.run.run_id == runs[0].run_id and s.resumed
    assert ad2.calls == ["C3", "C4", "C5"] and s.skipped == 2
    assert s.run.status == COMPLETE and len(store.list_runs()) == 1
    assert len(store.results(s.run.run_id)) == 5


def test_resume_retries_errored_cases_but_keeps_passes(store):
    ad = ScriptedAdapter(n=3, raises={"C2": RuntimeError("flaky")})
    first = execute_run(ad, "m", store, retries=0, **QUIET)
    assert first.run.status == PARTIAL
    ad2 = ScriptedAdapter(n=3)
    s = execute_run(ad2, "m", store, resume=True, **QUIET)
    assert ad2.calls == ["C2"] and s.run.status == COMPLETE and s.passed == 3


def test_resume_refuses_a_run_made_with_a_different_prompt(store):
    ad = ScriptedAdapter(n=3)
    ad.interrupt_at = "C2"
    with pytest.raises(KeyboardInterrupt):
        execute_run(ad, "m", store, **QUIET)
    changed = ScriptedAdapter(n=3, prompt_hash="p2")
    s = execute_run(changed, "m", store, resume=True, **QUIET)
    assert len(store.list_runs()) == 2 and not s.resumed
    assert any("changed since" in n for n in s.notes)
    old = store.list_runs()[0]
    with pytest.raises(ResumeMismatch):
        execute_run(
            ScriptedAdapter(n=3, prompt_hash="p2"), "m", store, resume=True, run_id=old.run_id, **QUIET
        )


def test_resume_with_explicit_run_id_checks_suite_and_model(store):
    ad = ScriptedAdapter(n=2)
    ad.interrupt_at = "C2"
    with pytest.raises(KeyboardInterrupt):
        execute_run(ad, "m", store, **QUIET)
    rid = store.list_runs()[0].run_id
    with pytest.raises(ResumeMismatch):
        execute_run(ScriptedAdapter(n=2), "other-model", store, resume=True, run_id=rid, **QUIET)
    s = execute_run(ScriptedAdapter(n=2), "m", store, resume=True, run_id=rid, **QUIET)
    assert s.run.run_id == rid and s.run.status == COMPLETE


def test_resume_with_nothing_to_resume_starts_fresh(store):
    s = execute_run(ScriptedAdapter(n=2), "m", store, resume=True, **QUIET)
    assert not s.resumed and s.run.status == COMPLETE and "nothing to resume" in s.notes[0]


def test_reuse_complete_runs_nothing(store):
    first = execute_run(ScriptedAdapter(n=3), "m", store, **QUIET)
    ad = ScriptedAdapter(n=3)
    s = execute_run(ad, "m", store, reuse_complete=True, **QUIET)
    assert s.run.run_id == first.run.run_id and ad.calls == [] and ad.warmups == []
    ad_changed = ScriptedAdapter(n=3, prompt_hash="new")
    s2 = execute_run(ad_changed, "m", store, reuse_complete=True, **QUIET)
    assert s2.run.run_id != first.run.run_id and len(ad_changed.calls) == 3


def test_limit_and_ids_select_cases_and_limit_runs_share_the_case_file_hash(store):
    full = execute_run(ScriptedAdapter(n=5), "m", store, **QUIET)
    lim = execute_run(ScriptedAdapter(n=5), "m2", store, limit=2, **QUIET)
    ids = execute_run(ScriptedAdapter(n=5), "m3", store, ids=["C4", "C5"], **QUIET)
    assert [r.case_id for r in lim.results] == ["C1", "C2"] and lim.run.n_cases == 2
    assert [r.case_id for r in ids.results] == ["C4", "C5"]
    assert lim.run.case_set_hash == full.run.case_set_hash  # same case file, so only the id set differs


def test_preflight_failure_stops_before_any_run_is_created(store):
    ad = ScriptedAdapter(n=2)
    ad.preflight_problems = ["Ollama is not reachable"]
    with pytest.raises(PreflightError, match="Ollama"):
        execute_run(ad, "m", store, **QUIET)
    assert store.list_runs() == [] and ad.calls == []
    s = execute_run(ad, "m", store, preflight=False, **QUIET)  # --no-preflight
    assert s.run.status == COMPLETE


def test_warmup_runs_once_before_the_first_case_and_is_untimed(store):
    clock = FakeClock()
    ad = ScriptedAdapter(n=2)
    order: list[str] = []
    ad.warmup = lambda model: (order.append("warmup"), clock.advance(120))  # a slow model load
    orig = ad.run_case
    ad.run_case = lambda case, model: (order.append(case["id"]), orig(case, model))[1]
    s = execute_run(ad, "m", store, clock=clock, **QUIET)
    assert order == ["warmup", "C1", "C2"]
    assert all(r.latency_ms == 0.0 for r in s.results)  # the 120 s load is not in any case


def test_warmup_failure_is_a_preflight_error_and_can_be_skipped(store):
    ad = ScriptedAdapter(n=1)

    def bad(model):
        raise RuntimeError("model will not load")

    ad.warmup = bad
    with pytest.raises(PreflightError, match="warm-up"):
        execute_run(ad, "m", store, **QUIET)
    assert execute_run(ad, "m", store, warmup=False, **QUIET).run.status == COMPLETE


def test_no_warmup_or_preflight_when_nothing_is_pending(store):
    first = ScriptedAdapter(n=2, raises={"C2": RuntimeError("x")})
    execute_run(first, "m", store, retries=0, **QUIET)  # C2 errored: the run is partial
    ad = ScriptedAdapter(n=2)
    ad.preflight_problems = ["down"]
    s = execute_run(ad, "m", store, resume=True, retry_errors=False, **QUIET)  # keep the error, run nothing
    assert ad.calls == [] and ad.warmups == [] and s.resumed and s.run.status == PARTIAL


def test_validate_cases():
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases([{"id": "A"}, {"id": "A"}])
    with pytest.raises(ValueError, match="without an id"):
        validate_cases([{"category": "x"}])


def test_empty_selection_is_an_error(store):
    with pytest.raises(ValueError, match="no cases"):
        execute_run(ScriptedAdapter(n=2), "m", store, ids=["nope"], **QUIET)


def test_cli_main_end_to_end_with_the_fake_suite(tmp_path, capsys):
    db = tmp_path / "r.db"
    code = runmod.main(
        [
            "--suite",
            "fake",
            "--model",
            "fake-strong",
            "--db",
            str(db),
            "--limit",
            "6",
            "--json-out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "6/6" in out or "passed" in out
    import json

    info = json.loads((tmp_path / "out.json").read_text())
    assert info["status"] == COMPLETE and info["cases"] == 6
    assert runmod.main(["--suite", "nope", "--model", "x", "--db", str(db)]) == 2
    assert runmod.main(["--model", "x"]) == 2


def test_cli_exit_codes_for_partial_and_preflight(tmp_path):
    ad = ScriptedAdapter(n=2, raises={"C1": RuntimeError("x")})
    db = str(tmp_path / "r.db")
    assert runmod.main(["--suite", "scripted", "--model", "m", "--db", db, "--retries", "0"], adapter=ad) == 1
    ad2 = ScriptedAdapter(n=2)
    ad2.preflight_problems = ["down"]
    assert runmod.main(["--suite", "scripted", "--model", "m2", "--db", db], adapter=ad2) == 2
    assert ad2.closed


def test_run_dataclass_fingerprint_includes_the_scorer_and_the_clock():
    r = Run(run_id="x", suite="s", model="m", git_hash="g", prompt_hash="p", code_hash="c", case_set_hash="h")
    assert r.fingerprint() == ("g", "p", "c", "h", None, None)
    r.config.update(scorer_hash="s1", clock_now="2026-11-10T09:00:00+05:30")
    assert r.fingerprint() == ("g", "p", "c", "h", "s1", "2026-11-10T09:00:00+05:30")
    assert r.scorer_hash == "s1" and r.clock_now.startswith("2026-11-10")


def test_cli_reports_adapter_errors_as_exit_2(tmp_path, capsys):
    from harness.adapters.base import AdapterError

    ad = ScriptedAdapter(n=1)

    def broken():
        raise AdapterError("feature repo not found")

    ad.cases = broken
    code = runmod.main(["--suite", "scripted", "--model", "m", "--db", str(tmp_path / "r.db")], adapter=ad)
    assert code == 2 and "feature repo not found" in capsys.readouterr().err


def test_adapter_config_extras_are_stored_with_the_run(store):
    ad = ScriptedAdapter(n=1)
    ad.provenance = lambda: {"prompt_hash": "p", "config": {"judge": None, "variant": "agent"}}
    run = execute_run(ad, "m", store, **QUIET).run
    assert (
        store.get_run(run.run_id).config["judge"] is None
        and store.get_run(run.run_id).config["variant"] == "agent"
    )


def test_a_run_stops_after_repeated_errors_and_can_be_resumed(store):
    down = {f"C{i}": RuntimeError("Ollama is down") for i in range(1, 7)}
    ad = ScriptedAdapter(n=10, raises=down)
    s = execute_run(ad, "m", store, retries=0, abort_after=5, **QUIET)
    assert s.aborted and s.run.status == PARTIAL and ad.calls == ["C1", "C2", "C3", "C4", "C5"]
    ad2 = ScriptedAdapter(n=10)
    s2 = execute_run(ad2, "m", store, resume=True, **QUIET)  # the service is back
    assert s2.run.status == COMPLETE and not s2.aborted and ad2.calls == [f"C{i}" for i in range(1, 11)]
    # one good case in between resets the streak
    flaky = {
        "C1": RuntimeError("x"),
        "C2": RuntimeError("x"),
        "C4": RuntimeError("x"),
        "C5": RuntimeError("x"),
    }
    s3 = execute_run(ScriptedAdapter(n=6, raises=flaky), "m2", store, retries=0, abort_after=3, **QUIET)
    assert not s3.aborted and s3.errors == 4
    # 0 turns the circuit breaker off
    s4 = execute_run(ScriptedAdapter(n=8, raises=down), "m3", store, retries=0, abort_after=0, **QUIET)
    assert not s4.aborted and s4.errors == 6


def test_a_run_records_its_coverage_scorer_and_clock(store):
    ad = ScriptedAdapter(n=5)
    ad.provenance = lambda: {
        "prompt_hash": "p",
        "scorer_hash": "sc1",
        "config": {"clock_now": "2026-11-10T09:00:00+05:30"},
    }
    full = execute_run(ad, "m", store, **QUIET).run
    part = execute_run(ad, "m2", store, limit=2, **QUIET).run
    f, p = store.get_run(full.run_id), store.get_run(part.run_id)
    assert f.config["n_cases_total"] == 5 and f.n_cases_total == 5 and f.is_full_suite
    assert p.config["n_cases_total"] == 5 and p.config["limit"] == 2 and not p.is_full_suite
    assert f.scorer_hash == "sc1" and f.clock_now == "2026-11-10T09:00:00+05:30"
    ids = execute_run(ad, "m3", store, ids=["C1", "C2", "C3", "C4", "C5"], **QUIET).run
    assert not store.get_run(ids.run_id).is_full_suite  # --ids is never gate-grade, even with every id


def test_resume_refuses_a_run_scored_by_different_code_or_under_a_different_clock(store):
    def adapter(scorer, clock):
        a = ScriptedAdapter(n=3)
        a.provenance = lambda: {"prompt_hash": "p", "scorer_hash": scorer, "config": {"clock_now": clock}}
        return a

    first = adapter("s1", "2026-11-10T09:00:00+05:30")
    first.interrupt_at = "C2"
    with pytest.raises(KeyboardInterrupt):
        execute_run(first, "m", store, **QUIET)
    for other in (adapter("s2", "2026-11-10T09:00:00+05:30"), adapter("s1", None)):
        s = execute_run(other, "m", store, resume=True, **QUIET)
        assert not s.resumed and any("changed since" in n for n in s.notes)


def test_older_runs_without_the_new_config_keys_still_load_and_say_what_they_know(store):
    old = Run(run_id="old", suite="s", model="m", n_cases=60, config={"limit": None, "ids": None})
    assert old.is_full_suite and old.n_cases_total == 60 and old.scorer_hash is None and old.clock_now is None
    limited = Run(run_id="lim", suite="s", model="m", n_cases=10, config={"limit": 10, "ids": None})
    assert not limited.is_full_suite and limited.n_cases_total is None
    unknown = Run(run_id="unk", suite="s", model="m", n_cases=10, config={})
    assert unknown.n_cases_total is None and not unknown.is_full_suite


# ---- a timed-out case leaves a thread running: wait for it (bounded) or flag the next case ----


def test_the_runner_waits_for_a_timed_out_cases_thread_before_the_next_case(store):
    import threading

    from harness.run import has_flag

    release = threading.Event()
    seen: dict = {}
    ad = ScriptedAdapter(n=3)
    orig = ad.run_case

    def run_case(case, model):
        if case["id"] == "C1":
            seen["c1"] = threading.current_thread()
            release.wait(5)  # the abandoned worker: still running after the harness gave up on it
        if case["id"] == "C2":
            seen["c1_alive_at_c2"] = seen["c1"].is_alive()
        return orig(case, model)

    ad.run_case = run_case
    threading.Timer(0.5, release.set).start()  # the orphan finishes 0.5 s after the timeout
    try:
        s = execute_run(ad, "m", store, timeout_s=0.1, orphan_wait_s=3.0, **QUIET)
    finally:
        release.set()
    by = {r.case_id: r for r in s.results}
    assert by["C1"].status == TIMEOUT
    assert seen["c1_alive_at_c2"] is False  # the runner waited until the orphan was gone
    assert not has_flag(by["C2"], "orphan_running") and by["C2"].status == OK


def test_a_case_that_runs_while_an_orphan_is_still_working_is_flagged(store):
    import threading

    from harness.run import has_flag

    release = threading.Event()
    ad = ScriptedAdapter(n=3)
    orig = ad.run_case

    def run_case(case, model):
        if case["id"] == "C1":
            release.wait(5)
        return orig(case, model)

    ad.run_case = run_case
    try:
        s = execute_run(
            ad, "m", store, timeout_s=0.1, orphan_wait_s=0.2, **QUIET
        )  # waits 0.2 s, then gives up
    finally:
        release.set()
    by = {r.case_id: r for r in s.results}
    assert by["C1"].status == TIMEOUT
    assert has_flag(by["C2"], "orphan_running") and by["C2"].output["_harness"] == {"orphan_running": True}
    assert by["C2"].output["ok"] is True  # the adapter's own output is kept next to the flag
    stored = {r.case_id: r for r in store.results(s.run.run_id)}
    assert has_flag(stored["C2"], "orphan_running") and has_flag(
        stored["C3"], "orphan_running"
    )  # still running


def test_adapter_orphan_threads_are_waited_for_too(store):
    import threading

    gate = threading.Event()
    worker = threading.Thread(target=lambda: gate.wait(5), daemon=True)
    worker.start()
    ad = ScriptedAdapter(n=2)
    ad.orphan_threads = lambda: [worker] if worker.is_alive() else []
    threading.Timer(0.3, gate.set).start()
    s = execute_run(ad, "m", store, orphan_wait_s=3.0, **QUIET)
    assert all(
        not __import__("harness.run", fromlist=["has_flag"]).has_flag(r, "orphan_running") for r in s.results
    )
    assert not worker.is_alive()


def test_wait_for_orphans_is_bounded():
    import threading
    import time

    from harness.run import wait_for_orphans

    stop = threading.Event()
    t = threading.Thread(target=lambda: stop.wait(5), daemon=True)
    t.start()
    threads = [t]
    t0 = time.monotonic()
    assert wait_for_orphans(threads, 0.2) is False and time.monotonic() - t0 < 1.0 and threads == [t]
    stop.set()
    assert wait_for_orphans(threads, 1.0) is True and threads == []
