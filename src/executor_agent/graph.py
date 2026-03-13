# executor_agent/graph.py

from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable
from common.utils import load_chat_model  # 假设你有这个工具函数
from .prompts import EXECUTOR_SYSTEM_PROMPT
from .tools import search_huggingface_hub, propose_training_code, save_final_report

def get_executor_tools():
    return [search_huggingface_hub, propose_training_code, save_final_report]

# 快速创建一个独立的 ReAct agent
executor_agent = create_react_agent(
    model=load_chat_model("siliconflow:deepseek-ai/DeepSeek-V3"),  # 或从 .env 读取
    tools=get_executor_tools(),
    prompt=EXECUTOR_SYSTEM_PROMPT,
    # 如果你想更精细控制循环，可以不用 create_react_agent，自己写节点
)

# 为了方便外部调用，暴露一个 async run 方法
async def run_executor(initial_messages):
    result = await executor_agent.ainvoke({"messages": initial_messages})
    return result