# planner_agent/graph.py
"""Planner Agent - 使用自定义 StateGraph 实现（高度灵活版）

核心特点：
1. PLANNER_RULES 严格放在消息列表最底部（符合你的要求）
2. 强制输出干净 JSON（无任何多余文字）
3. 单次调用即可完成规划（后续可轻松改成多轮）
4. 与项目整体风格完全一致（StateGraph + async）
"""

from datetime import UTC, datetime
from typing import List, Dict, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from common.utils import load_chat_model
from common.prompts import SYSTEM_PROMPT
from .state import PlannerState
from .prompts import PLANNER_SYSTEM_PROMPT


async def call_planner(state: PlannerState) -> Dict[str, List[BaseMessage]]:
    """Planner 核心节点：把规则放在最后 + 强制 JSON 输出"""
    model = load_chat_model("siliconflow:Pro/deepseek-ai/DeepSeek-V3.2")

    messages = state.messages.copy()

    if PLANNER_SYSTEM_PROMPT.strip():
        messages.insert(0, SystemMessage(content=PLANNER_SYSTEM_PROMPT))

    messages.append(
        SystemMessage(
            content=SYSTEM_PROMPT
        )
    )

    # 4. 最后一条用户指令（强制 JSON）
    messages.append(
        HumanMessage(
            content=(
                "根据以上全部对话历史，立即生成完整的深度学习任务执行计划。\n"
                "要求：\n"
                "1. 严格只输出合法的 JSON 对象，不要输出任何其他文字、markdown、代码块、思考过程、解释、致歉。\n"
                "2. JSON 必须包含以下字段（不允许缺少）：\n"
                '   "title", "objective", "requirements", "steps", "success_criteria", "risks_and_fallbacks"\n'
                "现在就开始输出 JSON！"
            )
        )
    )

    # 调用模型
    response = await model.ainvoke(messages)

    # 返回新消息（保持状态更新）
    return {
        "messages": [AIMessage(content=response.content.strip())]
    }


# ==================== 构建 Graph ====================
builder = StateGraph(PlannerState)

builder.add_node("call_planner", call_planner)

builder.add_edge(START, "call_planner")
builder.add_edge("call_planner", END)

# 编译成可执行的 graph
planner_graph = builder.compile(name="Planner Agent")


# ==================== 对外暴露的运行函数 ====================
async def run_planner(messages: List[BaseMessage]) -> str:
    """
    推荐使用的入口函数。
    输入：任意历史消息列表
    输出：干净的 JSON 字符串（计划）
    """
    result = await planner_graph.ainvoke({"messages": messages})
    
    final_message = result["messages"][-1]
    content = final_message.content.strip()

    # 额外清理（防止模型偶尔还是加了 ```json）
    if content.startswith("```json"):
        content = content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
    elif content.startswith("```"):
        content = content.split("```", 2)[1].strip()

    return content


# ==================== 测试用同步接口 ====================
def plan_simple(task_description: str) -> str:
    """快速测试用"""
    import asyncio
    from langchain_core.messages import HumanMessage
    
    messages = [HumanMessage(content=task_description)]
    return asyncio.run(run_planner(messages))