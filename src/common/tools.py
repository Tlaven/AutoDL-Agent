"""AutoDL-Agent 主循环工具 - 永远只绑定 Planner 两个工具"""

import logging
from datetime import UTC, datetime
from typing import Any, Callable, List

from langchain_core.tools import tool
import json
from common.prompts import PLANNER_RULES
from src.executor_agent.graph import run_executor

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
async def execute_plan(plan_json: str) -> str:
    """当规划阶段认为计划已经足够完整时调用此工具。
    将结构化的计划交给独立的 Executor Agent 执行。
    Executor 将拥有独立的上下文和 ReAct 循环。
    
    参数：
    plan_json: 必须是合法的 JSON 字符串，包含完整的训练计划
    """
    try:
        plan = json.loads(plan_json)
        plan_str = json.dumps(plan, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"计划 JSON 格式错误，无法启动 Executor：{str(e)}"

    # 准备给 executor 的初始消息
    initial_messages = [
        ("system", "立即开始执行以下计划，不要询问用户，直接使用工具推进。"),
        ("human", f"执行计划：\n```json\n{plan_str}\n```")
    ]

    try:
        result = await run_executor(initial_messages)
        
        # 简单提取最终输出（可以根据需要更精细地解析）
        messages = result["messages"]
        final_msg = messages[-1]
        
        if hasattr(final_msg, "content") and final_msg.content:
            summary = final_msg.content[:1200] + "..." if len(final_msg.content) > 1200 else final_msg.content
        else:
            summary = "Executor 完成，但未返回可读文本总结。"

        return (
            f"Executor Agent 已完成执行。\n"
            f"最终输出摘要：\n{summary}\n\n"
            f"完整消息历史长度：{len(messages)}"
        )
    except Exception as e:
        import traceback
        return f"Executor 执行过程中发生异常：\n{str(e)}\n{traceback.format_exc()[:800]}"


async def get_tools() -> List[Callable[..., Any]]:
    """主 ReAct 循环永远只返回这两个工具（符合用户要求）。"""
    return [activate_planner_mode, execute_plan]