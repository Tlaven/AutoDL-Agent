# CLAUDE.md

> 本文件是给 AI 助手读的工程文档，记录项目真实现状、架构设计意图和已确定的决策。
> **每次对话开始前必须先读此文件。**

---

## 项目定位

**AutoDL_Agent** 是一个端到端的 AutoML/AutoDL 智能体系统，目标是自动化完成深度学习实验全流程：
- 理解用户需求 → 制定执行计划 → 搜索模型/数据集 → 生成训练代码 → 本地执行训练 → 输出报告

"AutoDL" 指「自动化深度学习」，不是 autodl.com 平台。当前阶段以**本地训练**为主，后续可扩展到云平台。

---

## 架构概览（三层 Multi-Agent）

```
用户
 │
 ▼
Supervisor Agent（主循环）          src/supervisor_agent/
  - 框架：自定义 StateGraph（ReAct 模式）
  - 模型：qwen:qwen-flash（可通过 Context 配置）
  - 工具：永远只有 generate_plan 和 execute_plan 两个
  - 职责：理解用户意图，协调 Planner 和 Executor，汇报最终结果
  - 系统提示：src/common/prompts.py（SYSTEM_PROMPT，已迁移至此）
  │
  ├── generate_plan ──▶ Planner Agent    src/planner_agent/
  │                       - 框架：自定义 StateGraph（单节点，单次调用）
  │                       - 模型：siliconflow:Pro/deepseek-ai/DeepSeek-V3.2
  │                       - 职责：把用户需求转化为结构化 JSON 执行计划
  │                       - 记忆：MemorySaver（thread_id = PlannerSession.session_id）
  │                       - 特性：SYSTEM_PROMPT 注入消息列表开头，PLANNER_SYSTEM_PROMPT 严格放在最后
  │
  └── execute_plan ───▶ Executor Agent   src/executor_agent/
                          - 框架：create_react_agent（LangGraph prebuilt）
                          - 模型：siliconflow:Pro/deepseek-ai/DeepSeek-V3.2
                          - 职责：按 JSON 计划逐步调用工具，完成实际操作
```

### 入口
`langgraph.json` 注册的唯一图：`src/supervisor_agent/graph.py:graph`

---

## 已确定的关键设计决策

### 1. execute_plan 使用 InjectedState（重要）

`execute_plan` 工具**不接受 LLM 传入参数**，改用 `InjectedState` 注入 State，自己从 `state.planner_session.plan_json` 取计划。

```python
@tool
async def execute_plan(state: Annotated[State, InjectedState]) -> str:
    plan = state.planner_session.plan_json  # 直接从 State 取，不依赖 LLM 填参
```

`generate_plan` 同样使用 `InjectedState`，从 `state.messages` 取完整历史传给 Planner。

原因：避免 LLM 错误传参，保证计划来源可信。

### 2. ExecutorRef 记录执行状态（部分实现）

`State` 已定义 `executors: dict[str, ExecutorRef]` 字段，结构已就绪：

```
执行开始 → ExecutorRef(status="running")
执行完成 → ExecutorRef(status="completed")
执行失败 → ExecutorRef(status="failed")
```

**当前实现现状**：`execute_plan` 工具目前通过 ToolMessage 的文本内容携带执行状态信息（`[执行记录] executor_session_id=... status=...`），尚未将 `ExecutorRef` 真正写入 `state.executors` dict。这是待完成的 TODO 项。

### 3. 执行结果路由（LLM 自主决策）

Executor 执行完后，结果返回给 Supervisor，**由 LLM 自主判断**走哪条路：

```
Supervisor 收到 Executor 结果
  ├── 成功(completed) → 汇报用户，结束
  └── 失败(failed)
        ├── 可调整 → 调 generate_plan 重规划（携带完整历史 + 失败原因）→ 再 execute_plan
        └── 无法解决（多次失败仍无法推进）→ 告知用户，搁置（当前阶段先不处理）
```

路由方式选 **LLM 自主判断**（而非硬编码路由），原因：Supervisor 的 `state.messages` 保留完整历史上下文，LLM 可以据此做更智能的决策；重规划时 Planner 也能看到失败记录。Supervisor 提示词中明确限制**重规划不超过 3 次**，避免无效循环。

### 4. 单线程执行

当前阶段明确为**单线程**，不做并发 Executor。

### 5. Executor 工具现状（占位，后续重做）

`src/executor_agent/tools.py` 中的三个工具均为**占位实现（mock）**，后续会删掉重新接入真实 API：

| 工具 | 状态 | 说明 |
|---|---|---|
| `search_huggingface_hub` | mock | 返回硬编码假数据 |
| `propose_training_code` | mock | 返回固定模板代码 |
| `save_final_report` | mock | 返回固定格式字符串 |

**当前不需要关注这些工具的实现细节。**

### 6. Planner 工具现状（定义但未绑定）

`src/planner_agent/tools.py` 中的四个工具已定义但**未绑定到 graph**，当前 Planner 是纯 LLM 调用（无工具）。这是有意为之，后续迭代再绑定。

### 7. Planner 提示词结构

Planner 调用时的消息顺序（已实现）：
1. `SystemMessage(SYSTEM_PROMPT)` —— 全局背景（放开头）
2. `state.messages[:-1]` —— 历史消息（除最后一条）
3. `SystemMessage(PLANNER_SYSTEM_PROMPT)` —— Planner 规则和 JSON 格式要求（严格放最后）

`PLANNER_SYSTEM_PROMPT` 要求输出 Plan 之外有一点思考，JSON 放在 ` ```json ``` ` 代码块中，`run_planner()` 会自动提取代码块内容。

### 8. dynamic_tools_node 同步 PlannerSession

`dynamic_tools_node` 执行 `generate_plan` 工具后，自动提取 ToolMessage 内容，通过 `_extract_generate_plan_output()` 函数写入 `state.planner_session`，供后续 `execute_plan` 使用。逻辑：复用已有 `session_id` 或新建。

---

## 后续计划

### 近期（当前迭代）

- [x] 重构 `execute_plan` 为 InjectedState 方式（不再依赖 LLM 传参）
- [x] 修复 `run_executor` 入参 bug（字符串→HumanMessage）
- [x] `dynamic_tools_node` 执行完工具后同步更新 `PlannerSession.plan_json` 到 State
- [x] 优化 Executor 和 Supervisor 提示词（重规划次数限制、工作流程明确化）
- [x] `generate_plan` 同样改为 InjectedState，直接从 `state.messages` 取历史
- [x] Planner 提示词结构调整（SYSTEM_PROMPT 开头，PLANNER_RULES 最后）
- [ ] ExecutorRef 完整写入 `state.executors`（当前仅通过 ToolMessage 文本携带）

### 中期

- [ ] Executor 工具接入真实 HuggingFace Hub API
- [ ] `propose_training_code` 生成真实可运行的训练代码文件（写入本地 .py 文件）
- [ ] 添加本地执行训练代码的工具（`execute_training_code`，用 subprocess）
- [ ] 补充测试用例（tests/ 目前全为空）

### 远期（进阶功能）

- [ ] **Executor 中途中断机制**：`execute_plan` 改用 `astream_events` 逐步迭代，每轮 ReAct 结束后检查 State 中断信号，主循环可注入 `HumanMessage` 强制 Executor 停下
- [ ] Docker / Kubernetes 部署支持（见 ROADMAP.md v0.3.0）

---

## 模块速查

| 文件 | 职责 |
|---|---|
| `src/supervisor_agent/graph.py` | 主循环图定义，`call_model` + `dynamic_tools_node` + `_extract_generate_plan_output` |
| `src/supervisor_agent/state.py` | `State`、`InputState`、`PlannerSession`、`ExecutorRef` |
| `src/supervisor_agent/tools.py` | `generate_plan`、`execute_plan` 两个主循环工具（均用 InjectedState） |
| `src/planner_agent/graph.py` | Planner 图，`call_planner` 节点，`run_planner()` 对外接口（含 JSON 提取） |
| `src/planner_agent/prompts.py` | `PLANNER_SYSTEM_PROMPT`，含 JSON 计划格式要求，放消息列表最后 |
| `src/planner_agent/tools.py` | 4 个规划工具（已定义，未绑定到 graph） |
| `src/executor_agent/graph.py` | Executor ReAct agent，`run_executor()` 对外接口 |
| `src/executor_agent/prompts.py` | `EXECUTOR_SYSTEM_PROMPT`，执行原则和输出格式 |
| `src/executor_agent/tools.py` | 3 个执行工具（均为占位 mock） |
| `src/common/context.py` | 运行时配置，支持环境变量覆盖 |
| `src/common/prompts.py` | `SYSTEM_PROMPT`，Supervisor 全局系统提示（注入 Planner 消息开头 + Supervisor 主循环） |
| `src/common/utils.py` | `load_chat_model("provider:model")` 统一入口 |
| `src/common/models/qwen.py` | Qwen/QwQ/QvQ 模型，支持国内/国际端点 |
| `src/common/models/siliconflow.py` | SiliconFlow 模型，支持国内/国际端点 |
| `src/common/mcp.py` | MCP 客户端管理（DeepWiki 动态工具加载） |

---

## 环境配置

```bash
# 必须
SILICONFLOW_API_KEY=sk-...      # Planner/Executor 使用 DeepSeek-V3.2
DASHSCOPE_API_KEY=sk-...        # Supervisor 使用 Qwen

# 可选
REGION=prc                      # prc/cn 或 international/en
MODEL=qwen:qwen-flash           # Supervisor 默认模型
ENABLE_DEEPWIKI=true            # 启用 DeepWiki MCP 工具
LANGCHAIN_TRACING_V2=true       # LangSmith 追踪
LANGCHAIN_API_KEY=lsv2_sk_...
LANGCHAIN_PROJECT=...
```

## 常用命令

```bash
make dev          # 启动 LangGraph 开发服务器
make dev_ui       # 启动 LangGraph Studio（有 UI）
make lint         # ruff + mypy 检查
make format       # ruff 自动格式化
uv sync --dev     # 安装所有依赖
```
