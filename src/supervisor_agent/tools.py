"""AutoDL-Agent 主循环工具 - 永远只绑定两个工具"""

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Callable, List, Annotated, cast

from langchain_core.messages import BaseMessage
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool

from src.supervisor_agent.state import State
from src.executor_agent.graph import run_executor
from src.planner_agent.graph import run_planner

logger = logging.getLogger(__name__)


@tool
async def generate_plan(state: Annotated[State, InjectedState]) -> str:
    """生成或更新结构化的深度学习执行计划。
    调用独立的 Planner Agent 完成规划，返回干净的 JSON 计划字符串。
    当用户提出新需求、或需要调整计划时调用此工具。
    """
    if state.planner_session is None:
        session_id = f"plan_{uuid.uuid4().hex[:8]}"
    else:
        session_id = state.planner_session.session_id

    # 若已有带执行状态的 plan，将其拼入消息末尾让 Planner 感知
    messages = cast(List[BaseMessage], state.messages)
    if state.planner_session and state.planner_session.plan_json:
        from langchain_core.messages import HumanMessage
        plan_context = (
            "\n\n[当前计划执行状态]\n"
            "以下是上一次执行后的 Plan（含各步骤执行状态），请在此基础上修订：\n"
            f"{state.planner_session.plan_json}"
        )
        messages = list(messages) + [HumanMessage(content=plan_context)]

    plan_json = await run_planner(
        messages=messages,
        thread_id=session_id,
    )

    logger.info("Planner 生成计划，session_id=%s，长度=%d", session_id, len(plan_json))
    return plan_json


@tool
async def execute_plan(state: Annotated[State, InjectedState]) -> str:
    """按当前计划执行深度学习任务。
    从 State 中读取最新的 JSON 计划，交给 Executor Agent 执行。
    调用前必须已经通过 generate_plan 生成了计划。
    """
    if state.planner_session is None:
        return "错误：尚未生成计划，请先调用 generate_plan。"
    if not state.planner_session.plan_json:
        return "错误：计划内容为空，请先调用 generate_plan 生成有效计划。"

    plan_json = state.planner_session.plan_json
    executor_session_id = f"exec_{uuid.uuid4().hex[:8]}"

    logger.info(
        "Executor 开始执行，executor_session_id=%s，planner_session_id=%s",
        executor_session_id,
        state.planner_session.session_id,
    )

    try:
        executor_result = await run_executor(plan_json)
        status = executor_result.status
        summary = executor_result.summary
        updated_plan_json = executor_result.updated_plan_json
        error_detail: str | None = None
        logger.info("Executor 执行完成，status=%s，executor_session_id=%s", status, executor_session_id)
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}"
        full_tb = traceback.format_exc()
        summary = f"Executor 执行过程中发生异常：\n{error_detail}\n\n{full_tb[:800]}"
        status = "failed"
        # 把异常原因标注到 plan_json 所有 pending/running 步骤的 failure_reason，
        # 确保 Planner 重规划时能看到失败信息
        updated_plan_json = _mark_plan_steps_failed(plan_json, error_detail)
        logger.error("Executor 执行失败，executor_session_id=%s，错误：%s", executor_session_id, error_detail)

    # 结构化返回，供 dynamic_tools_node 解析 updated_plan_json 写回 State
    # 格式约定：[EXECUTOR_RESULT] 标记行后接 JSON
    meta = {
        "executor_session_id": executor_session_id,
        "planner_session_id": state.planner_session.session_id,
        "status": status,
        "error_detail": error_detail,
        "started_at": datetime.now(UTC).isoformat(),
        "updated_plan_json": updated_plan_json,
    }
    meta_line = f"[EXECUTOR_RESULT] {json.dumps(meta, ensure_ascii=False)}"

    return f"{summary}\n\n{meta_line}"


def _mark_plan_steps_failed(plan_json: str, error_detail: str) -> str:
    """将 plan_json 中所有 pending/running 步骤标记为 failed，并写入 failure_reason。

    当 Executor 因异常崩溃，无法返回带状态的 updated_plan 时调用。
    保证 Planner 重规划时能看到哪些步骤未完成及失败原因。
    """
    if not plan_json or not plan_json.strip():
        return plan_json
    try:
        data = json.loads(plan_json)
    except json.JSONDecodeError:
        return plan_json

    steps = data if isinstance(data, list) else data.get("steps", [])
    for step in steps:
        if isinstance(step, dict) and step.get("status") in ("pending", "running", None):
            step["status"] = "failed"
            step["failure_reason"] = f"Executor 异常中断：{error_detail}"

    return json.dumps(data, ensure_ascii=False, indent=2)


async def get_tools() -> List[Callable[..., Any]]:
    """主 ReAct 循环永远只返回这两个工具。"""
    return [generate_plan, execute_plan]
