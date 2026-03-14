"""AutoDL-Agent 主循环工具 - 永远只绑定 Planner 两个工具"""

import re
import logging
from typing import Any, Callable, List, Annotated

from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

import json
from common.context import Context
from src.common.prompts import SYSTEM_PROMPT
from src.executor_agent.graph import run_executor
from src.planner_agent.graph import generate_planner, run_planner

logger = logging.getLogger(__name__)


@tool
async def generate_plan(state: Annotated[Any, InjectedState] = None) -> str:
    """
    根据当前全部对话历史生成结构化的任务执行计划。
    无需输入，输出计划。
    """
    if state is None:
        return json.dumps({"error": "无法获取 graph state"})

    try:
        agent = generate_planner()
        result = await run_planner(agent, [SystemMessage(content=SYSTEM_PROMPT), *state.messages[:-1]])

        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        code_blocks = re.findall(code_block_pattern, result, re.DOTALL)
        return code_blocks
    except Exception as e:
        import traceback
        return f"Executor 执行过程中发生异常：\n{str(e)}\n{traceback.format_exc()[:800]}"

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