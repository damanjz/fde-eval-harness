from harness import runs
from harness.storage import Store, read_run_file
from tests.conftest import P1_SNAPSHOT, make_results, make_run


def seed(db):
    with Store(db) as s:
        s.create_run(make_run("run-a", suite="fake", model="m1", config={"synthetic": True}))
        for r in make_results("A1:a+ A2:a-"):
            s.save_result("run-a", r)


def test_list_show_export_import_roundtrip(tmp_path, capsys):
    db = tmp_path / "r.db"
    seed(db)
    assert runs.main(["--db", str(db), "list"]) == 0
    out = capsys.readouterr().out
    assert "run-a" in out and "1/2 passed" in out and "(synthetic)" in out
    assert runs.main(["--db", str(db), "show", "run-a"]) == 0
    assert "passed 1/2 (50%)" in capsys.readouterr().out
    f = tmp_path / "x.run.json"
    assert runs.main(["--db", str(db), "export", "run-a", str(f)]) == 0
    run, res = read_run_file(f)
    assert run.run_id == "run-a" and len(res) == 2
    db2 = tmp_path / "other.db"
    assert runs.main(["--db", str(db2), "import", str(f)]) == 0
    assert runs.main(["--db", str(db2), "import", str(f)]) == 2  # same run id twice is refused
    capsys.readouterr()


def test_unknown_run_and_empty_db(tmp_path, capsys):
    assert runs.main(["--db", str(tmp_path / "e.db"), "list"]) == 0
    assert "no runs" in capsys.readouterr().out
    assert runs.main(["--db", str(tmp_path / "e.db"), "show", "nope"]) == 2
    capsys.readouterr()


def test_import_p1_writes_a_run_file(tmp_path, capsys):
    out = tmp_path / "p1.run.json"
    code = runs.main(
        [
            "--db",
            str(tmp_path / "r.db"),
            "import-p1",
            str(P1_SNAPSHOT / "agent-results.jsonl"),
            "--model",
            "fusion27b:iq3",
            "--cases",
            str(P1_SNAPSHOT / "evals" / "cases.jsonl"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    run, res = read_run_file(out)
    assert len(res) == 60 and run.config["imported"] and all(r.output is None for r in res)
    capsys.readouterr()
