# executor_agent/graph.py

from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable
from common.utils import load_chat_model
from .prompts import EXECUTOR_SYSTEM_PROMPT
from .tools import get_executor_tools


# 快速创建一个独立的 ReAct agent
executor_agent = create_react_agent(
    model=load_chat_model("siliconflow:Pro/deepseek-ai/DeepSeek-V3.2"), 
    tools=get_executor_tools(),
    prompt=EXECUTOR_SYSTEM_PROMPT
)

# 为了方便外部调用，暴露一个 async run 方法
async def run_executor(initial_messages):
    result = await executor_agent.ainvoke({"messages": initial_messages})
    messages = result['messages']
    # 检查类型并提取内容
    for message in messages:
        if isinstance(message, AIMessage):
            return message.content