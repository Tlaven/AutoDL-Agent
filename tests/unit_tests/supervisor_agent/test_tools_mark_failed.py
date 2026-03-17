import json

from src.supervisor_agent.tools import _mark_plan_steps_failed


def test_mark_plan_steps_failed_with_list_plan() -> None:
    plan = [
        {"step_id": "s1", "status": "pending"},
        {"step_id": "s2", "status": "running"},
        {"step_id": "s3", "status": "completed"},
        {"step_id": "s4"},
    ]

    result = _mark_plan_steps_failed(json.dumps(plan, ensure_ascii=False), "RuntimeError: boom")
    parsed = json.loads(result)

    assert parsed[0]["status"] == "failed"
    assert parsed[1]["status"] == "failed"
    assert parsed[2]["status"] == "completed"
    assert parsed[3]["status"] == "failed"
    assert "RuntimeError: boom" in parsed[0]["failure_reason"]


def test_mark_plan_steps_failed_with_steps_object() -> None:
    plan = {
        "steps": [
            {"step_id": "s1", "status": "pending"},
            {"step_id": "s2", "status": "failed"},
        ]
    }

    result = _mark_plan_steps_failed(json.dumps(plan, ensure_ascii=False), "ValueError: bad")
    parsed = json.loads(result)

    assert parsed["steps"][0]["status"] == "failed"
    assert "ValueError: bad" in parsed["steps"][0]["failure_reason"]
    assert parsed["steps"][1]["status"] == "failed"


def test_mark_plan_steps_failed_returns_original_on_invalid_json() -> None:
    raw = "not json"
    assert _mark_plan_steps_failed(raw, "err") == raw


def test_mark_plan_steps_failed_returns_original_on_blank_input() -> None:
    assert _mark_plan_steps_failed("", "err") == ""
