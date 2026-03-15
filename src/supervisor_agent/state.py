"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """为智能体定义输入状态，代表与外界交互的一个更窄的接口。

    这个类用于定义传入数据的初始状态和结构。
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    跟踪代理主要执行状态的消息。

    通常积累的模式如下：1. HumanMessage - 用户输入 2. 带有 .tool_calls 的 AIMessage - 代理选择用于收集信息的工具 
    3. ToolMessage(s) - 执行工具后的响应（或错误） 4. 不带 .tool_calls 的 AIMessage - 代理以非结构化格式对用户进行响应 
    5. HumanMessage - 用户在下一个对话回合中做出响应

    步骤2至5可根据需要重复执行。

    `add_messages` 注解确保新消息与现有消息合并，除非提供了具有相同ID的消息，否则将按ID进行更新，以保持“仅可追加”状态。
    """

@dataclass
class PlannerSession:
    """表示一次活跃的 planner 会话的上下文"""
    session_id: str
    plan_json: Optional[str] = None          # 最新的规划 JSON 字符串
    # 未来可以轻松加字段
    # status: str = "active"                 # active / completed / failed / needs_refine
    # version: int = 1
    # created_at: str = field(default_factory=utc_now_iso)
    # last_updated_at: Optional[str] = None

@dataclass
class ExecutorRef:
    """一条 Executor 的引用"""
    executor_session_id: str
    planner_session_id: str              # 这个 Executor 使用的 Planner 会话
    plan_json: str                       # 启动时拿到的计划
    status: str = "running"              # running / paused / completed / failed
    experiment_name: str = ""            # 可选：人类可读的名字，如 "resnet50_bs32"
    started_at: str = ""                 # ISO 时间
    # 后续可加 metrics_summary, checkpoint_path 等

@dataclass
class State(InputState):
    """表示智能体的完整状态，在InputState的基础上扩展了其他属性。

    这个类可用于存储代理生命周期中所需的任何信息。
    """
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
    planner_session: Optional[PlannerSession] = None
    executors: dict[str, ExecutorRef] = field(default_factory=dict)
    is_last_step: IsLastStep = field(default=False)

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
