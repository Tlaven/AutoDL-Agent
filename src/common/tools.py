"""AutoDL-Agent 主循环工具 - 永远只绑定 Planner 两个工具"""

import logging
from datetime import UTC, datetime
from typing import Any, Callable, List

from langchain_core.tools import tool

from common.prompts import PLANNER_RULES

logger = logging.getLogger(__name__)


@tool
async def activate_planner_mode() -> str:
    """激活 Planner 模式。

    使用场景：面对复杂任务时，应当制定Plan。
    调用后会向消息历史一次性追加完整的 Planner 工作守则。
    """
    rules = PLANNER_RULES.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    logger.info("Planner 模式已激活 - 守则已追加到历史上下文")
    return rules


@tool
async def execute_plan() -> str:
    """执行已制定的 Plan（当前占位）。

    使用场景：Planner 模式下认为计划已成熟时调用。
    后续会切换到独立的 Executor 子 Agent（不同上下文 + 更多工具）。
    """
    logger.info("execute_plan 被调用（占位模式）")
    return "执行模式已激活（占位）。计划执行逻辑开发中，请继续在 Planner 模式下完善计划。"


async def get_tools() -> List[Callable[..., Any]]:
    """主 ReAct 循环永远只返回这两个工具（符合用户要求）。"""
    return [activate_planner_mode, execute_plan]