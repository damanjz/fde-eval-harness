"""The workflow is thin: every step calls scripts/ci.sh, which the tests below exercise locally."""

import re

import yaml

from tests.conftest import ROOT

WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SCRIPT = (ROOT / "scripts" / "ci.sh").read_text(encoding="utf-8")


def load():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_triggers_and_python_version():
    wf = load()
    triggers = wf.get("on", wf.get(True))  # YAML reads the bare key `on` as True
    assert "push" in triggers and "pull_request" in triggers
    steps = wf["jobs"]["checks"]["steps"]
    setup = next(s for s in steps if str(s.get("uses", "")).startswith("actions/setup-python"))
    assert setup["with"]["python-version"] == "3.13"


def test_every_ci_step_is_defined_in_the_script_and_the_gate_runs():
    steps = load()["jobs"]["checks"]["steps"]
    called = [m.group(1) for s in steps if (m := re.search(r"bash scripts/ci\.sh (\w+)", s.get("run", "")))]
    assert called == ["lint", "tests", "gate", "smoke"]
    for name in called:
        assert re.search(rf"^\s*{name}\) {name} ;;", SCRIPT, re.MULTILINE), name


def test_the_gate_step_asserts_all_three_exit_codes_on_committed_fixtures():
    assert re.search(r"expect 0 .*harness\.gate", SCRIPT)
    assert re.search(r"expect 1 .*harness\.gate", SCRIPT)
    assert re.search(r"expect 2 .*harness\.gate", SCRIPT)
    for name in re.findall(r'"\$FX/([\w.\-]+\.run\.json)"', SCRIPT):
        assert (ROOT / "fixtures" / "runs" / name).is_file(), name


def test_pytest_is_part_of_ci_and_requirements_cover_the_test_imports():
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("pytest", "ruff", "pyyaml"):
        assert pkg in reqs


def test_the_improvement_step_is_labelled_for_what_it_is_an_improvement_within_noise():
    """The fixture pair passes the gate, but the compare says its p is above 0.05: do not call it a genuine gain."""
    from harness.compare import compare_runs
    from harness.storage import read_run_file

    assert "an improvement within noise passes (exit 0)" in SCRIPT and "genuine improvement" not in SCRIPT
    fx = ROOT / "fixtures" / "runs"
    a_run, a_res = read_run_file(fx / "support_agent-p1-run2-agent.run.json")
    b_run, b_res = read_run_file(fx / "support_agent-p1-run3-agent.run.json")
    c = compare_runs(a_run, a_res, b_run, b_res)
    assert c.net_cases == 5 and c.mcnemar.p_value > 0.05 and c.verdict == "noise"
