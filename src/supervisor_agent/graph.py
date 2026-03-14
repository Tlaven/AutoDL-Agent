"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from src.common.context import Context
from src.common.utils import load_chat_model
from src.supervisor_agent.tools import get_tools
from src.supervisor_agent.state import InputState, State


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """调用LLM支持Agent。
    此函数负责准备提示、初始化模型并处理响应。
    参数：
        状态(State)：对话的当前状态。
        config (RunnableConfig)：模型运行的配置。
    返回：
        dict:一个包含模型响应消息的字典。
    """
    available_tools = await get_tools()
    model = load_chat_model(runtime.context.model).bind_tools(available_tools)
    system_message = runtime.context.system_prompt
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


async def dynamic_tools_node(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[ToolMessage]]:
    """根据配置动态执行工具。
    此函数根据当前配置获取可用工具，并执行上一条消息中请求的工具调用。
    """
    # Get available tools based on configuration
    available_tools = await get_tools()

    # Create a ToolNode with the available tools
    tool_node = ToolNode(available_tools)

    # Execute the tool node
    result = await tool_node.ainvoke(state)

    return cast(Dict[str, List[ToolMessage]], result)


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", dynamic_tools_node)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
