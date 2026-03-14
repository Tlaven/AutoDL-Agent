"""AutoDL-Agent 主循环工具 - 永远只绑定 Planner 两个工具"""

import re
import logging
import json
from typing import Any, Callable, List, Annotated

from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from common.context import Context
from common.prompts import SYSTEM_PROMPT
from src.supervisor_agent.state import InputState, State
from src.executor_agent.graph import run_executor
from src.planner_agent.graph import run_planner

logger = logging.getLogger(__name__)


@tool
async def generate_plan(
    state: Annotated[State, InjectedState] = None
) -> str:
    """
    生成结构化的深度学习执行计划。
    会自动调用独立的 planner_agent 来完成规划。
    返回的是干净的 JSON 计划。
    """
    if state is None:
        return '{"error": "无法获取对话历史"}'

    messages = state.messages

    try:
        # ← 这里就是 supervisor 让 planner 跑起来的关键一行！
        plan_json = await run_planner(messages)
        
        return plan_json                     # 直接返回 JSON，超级干净

    except Exception as e:
        return json.dumps({
            "error": "planner_agent 执行失败",
            "detail": str(e)
        })

@tool
async def execute_plan(plan: str) -> str:
    """当规划阶段你和用户认为计划已经足够完整时调用此工具。
    将结构化的计划交给独立的 Executor Agent 执行。
    Executor 将拥有独立的上下文和 ReAct 循环。
    
    参数：
    plan: 字符串，包含完整的训练计划
    """

    try:
        result = await run_executor(plan)

        return result
    except Exception as e:
        import traceback
        return f"Executor 执行过程中发生异常：\n{str(e)}\n{traceback.format_exc()[:800]}"


async def get_tools() -> List[Callable[..., Any]]:
    """主 ReAct 循环永远只返回这两个工具（符合用户要求）。"""
    return [generate_plan, execute_plan]