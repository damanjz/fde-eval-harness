import json

import pytest

from harness.models import COMPLETE, PARTIAL, RUNNING, Run
from harness.storage import (
    RunNotFound,
    import_run,
    load_run,
    read_run_file,
    write_run_file,
)
from tests.conftest import make_results, make_run, res


def test_roundtrip_run_and_results(store):
    run = make_run(
        "20261110-100000-s-m",
        config={"timeout_s": 5, "prices": {"usd_in_per_mtok": 0.0}},
        git_hash="abc",
        prompt_hash="p",
        code_hash="c",
        case_set_hash="h",
        n_cases=2,
    )
    store.create_run(run)
    r = res("A1", "answer", True, latency_ms=1234.5, pt=11, ct=3)
    r.checks = {"facts": True}
    r.output = {"reply": {"text": "café"}, "usage": {"prompt_tokens": 11}}
    store.save_result(run.run_id, r)
    store.save_result(run.run_id, res("A2", "answer", False, reasons=["missing fact"]))
    back = store.get_run(run.run_id)
    assert back.config == run.config and back.git_hash == "abc" and back.n_cases == 2
    got = store.results(run.run_id)
    assert [x.case_id for x in got] == ["A1", "A2"]
    assert got[0].passed is True and got[0].checks == {"facts": True}
    assert got[0].output["reply"]["text"] == "café"
    assert got[1].reasons == ["missing fact"] and got[1].passed is False


def test_save_result_replaces_the_same_case(store):
    store.create_run(make_run("r1"))
    store.save_result("r1", res("A1", "answer", False, status="error"))
    store.save_result("r1", res("A1", "answer", True))
    got = store.results("r1")
    assert len(got) == 1 and got[0].passed is True and got[0].status == "ok"


def test_get_run_by_unique_prefix_and_errors(store):
    store.create_run(make_run("20261110-100000-support-agent-a"))
    store.create_run(make_run("20261110-110000-support-agent-b"))
    assert store.get_run("20261110-1000").run_id.endswith("-a")
    with pytest.raises(RunNotFound, match="matches 2"):
        store.get_run("20261110")
    with pytest.raises(RunNotFound, match="no run"):
        store.get_run("nope")
    with pytest.raises(RunNotFound):
        store.get_run("%")  # LIKE wildcards in the reference are literal


def test_new_run_id_is_unique_and_sortable(store):
    a = store.new_run_id("support_agent", "fusion27b:iq3", "2026-11-10T10:00:00")
    assert a == "20261110-100000-support-agent-fusion27b-iq3"
    store.create_run(make_run(a))
    assert store.new_run_id("support_agent", "fusion27b:iq3", "2026-11-10T10:00:00") == a + "-2"


def test_find_resumable_and_complete_use_the_fingerprint(store):
    fp = ("g", "p", "c", "h", "s", "2026-11-10T09:00:00+05:30")
    run = make_run(
        "r1",
        git_hash="g",
        prompt_hash="p",
        code_hash="c",
        case_set_hash="h",
        status=RUNNING,
        config={"scorer_hash": "s", "clock_now": "2026-11-10T09:00:00+05:30"},
    )
    store.create_run(run)
    assert run.fingerprint() == fp
    assert store.find_resumable("s", "m", fp).run_id == "r1"
    assert store.find_resumable("s", "m", ("g", "OTHER", "c", "h", "s", fp[5])) is None
    assert store.find_resumable("s", "m", (*fp[:4], "other-scorer", fp[5])) is None  # scoring changed
    assert store.find_resumable("s", "m", (*fp[:5], "2026-11-12T09:00:00+05:30")) is None  # clock changed
    assert store.find_complete("s", "m", fp) is None
    store.update_run("r1", status=PARTIAL)
    assert store.find_resumable("s", "m", fp).run_id == "r1"
    store.update_run("r1", status=COMPLETE)
    assert store.find_resumable("s", "m", fp) is None
    assert store.find_complete("s", "m", fp).run_id == "r1"
    assert store.find_complete("s", "other-model", fp) is None


def test_list_runs_filters(store):
    store.create_run(make_run("r1", suite="a", model="x"))
    store.create_run(make_run("r2", suite="b", model="x"))
    store.create_run(make_run("r3", suite="a", model="y"))
    assert [r.run_id for r in store.list_runs("a")] == ["r1", "r3"]
    assert [r.run_id for r in store.list_runs(model="x")] == ["r1", "r2"]


def test_delete_run_cascades(store):
    store.create_run(make_run("r1"))
    store.save_result("r1", res("A1", "answer", True))
    store.delete_run("r1")
    assert store.results("r1") == []


def test_run_file_roundtrip_and_load_run(tmp_path, store):
    run = make_run("fx1", config={"synthetic": True})
    results = make_results("A1:a+ A2:a- T1:t+")
    path = write_run_file(tmp_path / "x.run.json", run, results, with_output=False)
    back_run, back_results = read_run_file(path)
    assert back_run.run_id == "fx1" and back_run.config == {"synthetic": True}
    assert [(r.case_id, r.passed) for r in back_results] == [("A1", True), ("A2", False), ("T1", True)]
    run2, res2 = load_run(str(path), tmp_path / "missing.db")  # a file path needs no database
    assert run2.run_id == "fx1" and len(res2) == 3
    import_run(store, back_run, back_results)
    assert load_run("fx1", store.path)[0].run_id == "fx1"


def test_run_file_rejects_other_formats(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a harness run file"):
        read_run_file(bad)


def test_load_run_without_database(tmp_path):
    with pytest.raises(RunNotFound, match="does not exist"):
        load_run("some-run", tmp_path / "nope.db")


def test_import_duplicate_run_id_fails(store):
    run = make_run("dup")
    import_run(store, run, [])
    with pytest.raises(Exception):  # noqa: B017 - sqlite3.IntegrityError
        import_run(store, Run.from_dict(run.to_dict()), [])


def test_replacing_a_result_keeps_its_position(store):
    store.create_run(make_run("r1"))
    for cid in ("A1", "A2", "A3"):
        store.save_result("r1", res(cid, "answer", True, status="error" if cid == "A2" else "ok"))
    store.save_result("r1", res("A2", "answer", True))  # the retry after --resume
    assert [r.case_id for r in store.results("r1")] == ["A1", "A2", "A3"]


# ---- run files are untrusted input ----


def _good_file(tmp_path):
    results = make_results("A1:a+ A2:a-")
    return write_run_file(tmp_path / "good.run.json", make_run("g"), results)


def test_an_errored_result_in_a_file_is_never_a_pass(tmp_path):
    import json

    f = _good_file(tmp_path)
    data = json.loads(f.read_text(encoding="utf-8"))
    for status in ("error", "timeout"):
        data["results"][0]["status"], data["results"][0]["passed"] = status, True
        f.write_text(json.dumps(data), encoding="utf-8")
        _, results = read_run_file(f)
        assert results[0].passed is False and results[0].status == status
    from harness.models import CaseResult

    assert CaseResult("A", "answer", True, status="timeout").passed is False  # enforced in the type itself


def test_duplicate_case_ids_are_rejected(tmp_path):
    import json

    f = _good_file(tmp_path)
    data = json.loads(f.read_text(encoding="utf-8"))
    data["results"].append(dict(data["results"][0]))
    f.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case id A1"):
        read_run_file(f)


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda d: d.pop("run"), "malformed run file"),
        (lambda d: d.pop("results"), "malformed run file"),
        (lambda d: d.update(run=[1]), "malformed run file"),
        (lambda d: d["results"].append({"case_id": "X"}), "malformed run file"),
        (lambda d: d["results"].append("nope"), "malformed run file"),
        (lambda d: d["results"][0].update(case_id=""), "no usable case_id"),
        (lambda d: d["run"].update(config="not an object"), "config is not an object"),
    ],
)
def test_malformed_run_files_raise_value_error_not_key_or_type_error(tmp_path, mutate, needle):
    import json

    f = _good_file(tmp_path)
    data = json.loads(f.read_text(encoding="utf-8"))
    mutate(data)
    f.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=needle):
        read_run_file(f)


def test_files_that_are_not_json_or_not_utf8_raise_value_error(tmp_path):
    bad = tmp_path / "bad.run.json"
    bad.write_text("{ nope", encoding="utf-8")
    with pytest.raises(ValueError):
        read_run_file(bad)
    bad.write_bytes(b"\xff\xfe\x00\x00 not utf-8")
    with pytest.raises(ValueError):
        read_run_file(bad)
