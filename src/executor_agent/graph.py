# executor_agent/graph.py

import logging
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from common.utils import load_chat_model
from .prompts import EXECUTOR_SYSTEM_PROMPT
from .tools import get_executor_tools

logger = logging.getLogger(__name__)

# ReAct agent（LangGraph prebuilt）
executor_agent = create_react_agent(
    model=load_chat_model("siliconflow:Pro/deepseek-ai/DeepSeek-V3.2"),
    tools=get_executor_tools(),
    prompt=EXECUTOR_SYSTEM_PROMPT,
)


async def run_executor(plan_json: str) -> AIMessage:
    """运行 Executor Agent，按 JSON 计划执行任务。

    Args:
        plan_json: 由 Planner 生成的 JSON 计划字符串。

    Returns:
        AIMessage: Executor 最终输出，name="executor"，方便调用方辨认来源。

    Raises:
        ValueError: plan_json 为空时抛出。
        RuntimeError: 当 Executor 未产生任何 AIMessage 输出时抛出。
    """
    if not plan_json or not plan_json.strip():
        raise ValueError("plan_json 不能为空")

    initial_messages = [HumanMessage(content=f"请严格按照以下计划执行：\n\n{plan_json}")]

    result = await executor_agent.ainvoke({"messages": initial_messages})
    messages = result.get("messages", [])

    # 取最后一条 AIMessage（最终结论）
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            # 用 name 标注来源，方便 Supervisor 和 Trace 辨认
            return AIMessage(content=message.content, name="executor")

    logger.error("Executor 未产生任何 AIMessage，消息列表长度=%d", len(messages))
    raise RuntimeError("Executor 执行完毕但未产生任何输出，请检查工具调用链。")
