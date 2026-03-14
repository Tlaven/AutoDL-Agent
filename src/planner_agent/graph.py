# planner_agent/graph.py

from langchain_core.messages import AIMessage, BaseMessage
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

from common.utils import load_chat_model
from .prompts import PLANNER_RULES


def generate_planner():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    model=load_chat_model('siliconflow:Pro/deepseek-ai/DeepSeek-V3.2')
    agent = initialize_agent(
        tools=[],
        llm=model,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    return agent

async def run_planner(agent, messages: list[BaseMessage], thread_id="default"):
    """
    运行 planner agent - 直接接受 BaseMessage 对象
    
    Args:
        agent: create_react_agent 返回的 agent
        checkpointer: MemorySaver 实例
        messages: BaseMessage 对象列表 (HumanMessage, AIMessage, SystemMessage)
        thread_id: 会话 ID
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # 直接使用传入的 messages，不需要转换
    result = await agent.ainvoke(
        messages,
        config=config
    )
    
    # 获取最新的 AI 回复
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    
    return str(result)