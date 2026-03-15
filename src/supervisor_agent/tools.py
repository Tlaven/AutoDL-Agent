"""AutoDL-Agent 主循环工具 - 永远只绑定两个工具"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Callable, List, Annotated

from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool

from src.supervisor_agent.state import State, PlannerSession, ExecutorRef
from src.executor_agent.graph import run_executor
from src.planner_agent.graph import run_planner

logger = logging.getLogger(__name__)


@tool
async def generate_plan(state: Annotated[State, InjectedState]) -> str:
    """生成或更新结构化的深度学习执行计划。
    调用独立的 Planner Agent 完成规划，返回干净的 JSON 计划字符串。
    当用户提出新需求、或需要调整计划时调用此工具。
    """
    # 复用已有 session，或新建一个
    if state.planner_session is None:
        session_id = f"plan_{uuid.uuid4().hex[:8]}"
        # 注意：直接修改 state 字段在 LangGraph 中仅用于工具内传递，
        # 实际状态更新通过返回值（ToolMessage）让 LangGraph reducer 处理。
        # 这里记录 session_id 供 run_planner 使用即可，
        # PlannerSession 的持久化由 generate_plan 的调用方（ToolNode）通过消息历史保存。
        session_id_to_use = session_id
    else:
        session_id_to_use = state.planner_session.session_id

    plan_json = await run_planner(
        messages=state.messages,
        thread_id=session_id_to_use,
    )

    logger.info("Planner 生成计划，session_id=%s，长度=%d", session_id_to_use, len(plan_json))
    return plan_json


@tool
async def execute_plan(state: Annotated[State, InjectedState]) -> str:
    """按当前计划执行深度学习任务。
    从 State 中读取最新的 JSON 计划，交给 Executor Agent 执行。
    调用前必须已经通过 generate_plan 生成了计划。
    """
    # 前置检查：必须存在 planner_session 且有 plan_json
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
        executor_message = await run_executor(plan_json)
        result = executor_message.content
        status = "completed"
        logger.info("Executor 执行完成，executor_session_id=%s", executor_session_id)
    except Exception as e:
        import traceback
        result = f"Executor 执行过程中发生异常：\n{str(e)}\n{traceback.format_exc()[:800]}"
        status = "failed"
        logger.error("Executor 执行失败，executor_session_id=%s，错误：%s", executor_session_id, str(e))

    # 构造 ExecutorRef 供调用方写入 State（通过返回值中携带，主循环 LLM 可读取）
    # 实际 State 更新需要在 graph 层做，这里在返回内容中附带结构化信息
    exec_ref_info = (
        f"\n\n[执行记录] executor_session_id={executor_session_id} "
        f"planner_session_id={state.planner_session.session_id} "
        f"status={status} "
        f"started_at={datetime.now(UTC).isoformat()}"
    )

    return result + exec_ref_info


async def get_tools() -> List[Callable[..., Any]]:
    """主 ReAct 循环永远只返回这两个工具。"""
    return [generate_plan, execute_plan]
