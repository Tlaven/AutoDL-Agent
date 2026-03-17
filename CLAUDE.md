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
                          - 框架：自定义 StateGraph（ReAct 模式，含 ExecutorState）
                          - 模型：siliconflow:Pro/deepseek-ai/DeepSeek-V3.2
                          - 职责：按意图层 JSON 计划自主选工具执行，完成后返回带步骤状态的 updated_plan
                          - 返回值：ExecutorResult(status, updated_plan_json, summary)
```

### 入口
`langgraph.json` 注册的唯一图：`src/supervisor_agent/graph.py:graph`

---

## 已确定的关键设计决策

### 1. execute_plan 使用 InjectedState（重要）

`execute_plan` 工具**不接受 LLM 传入参数**，改用 `InjectedState` 注入 State，自己从 `state.planner_session.plan_json` 取计划。

`generate_plan` 同样使用 `InjectedState`，从 `state.messages` 取完整历史传给 Planner。

原因：避免 LLM 错误传参，保证计划来源可信。

### 2. Plan 是"意图层"，不包含工具名（重要）

Planner **不知道 Executor 有哪些工具**，Plan 的每个 step 只描述**意图和期望产出**，不指定工具名称。Executor 自主根据 intent 选择合适的工具。

好处：Planner 与 Executor 工具集完全解耦，更换工具无需修改 Planner 提示词。

Plan JSON schema（每个 step）：
```json
{
  "step_id": "step_1",
  "intent": "意图描述（不含工具名）",
  "expected_output": "完成验收标准",
  "status": "pending / completed / failed / skipped",
  "result_summary": "成功时的摘要，初始为 null",
  "failure_reason": "失败时的原因，初始为 null"
}
```

### 3. Executor 上报执行状态，重规划走 Supervisor

Executor 遇到无法继续的情况时**直接停止**，把带执行状态的 updated_plan 返回给 Supervisor，**不在 Executor 内部主动重规划**。

重规划决策权在 Supervisor：
```
Supervisor 收到 Executor 结果
  ├── status=completed → 汇报用户，结束
  └── status=failed
        ├── 可调整 → 调 generate_plan（Planner 能看到带状态的 plan）→ 再 execute_plan（新 Executor，跳过已完成步骤）
        └── 多次失败无法推进 → 告知用户
```

### 4. ExecutorResult 结构化返回值

`run_executor()` 不再返回 `AIMessage`，改为返回 `ExecutorResult`：
```python
@dataclass
class ExecutorResult:
    status: Literal["completed", "failed"]
    updated_plan_json: str   # 带步骤执行状态的完整 plan JSON
    summary: str             # 给 Supervisor LLM 读的摘要
```

`execute_plan` 工具把 `updated_plan_json` 嵌入返回文本（`[EXECUTOR_RESULT] {...}`），`dynamic_tools_node` 解析后写回 `planner_session.plan_json`。

**失败处理保障（已实现）**：
- 正常失败（Executor LLM 主动停止）：`updated_plan_json` 由 Executor 自行填写各步骤 `status/failure_reason`
- 异常崩溃（Python Exception）：`execute_plan` 捕获异常，调用 `_mark_plan_steps_failed()` 把所有 `pending` 步骤标记为 `failed` 并写入 `failure_reason`，确保 `updated_plan_json` 永不为空
- `dynamic_tools_node` 解析 `[EXECUTOR_RESULT]` 时同时提取 `status` 和 `error_detail`，写入 `PlannerSession.last_executor_status / last_executor_error`

### 5. generate_plan 传入带执行状态的 plan（修订场景）

重规划时，`generate_plan` 工具会把 `planner_session.plan_json`（已含执行状态）拼入消息末尾，作为 `HumanMessage` 传给 Planner，让 Planner 在修订时能看到哪些步骤已完成、哪步失败及原因。

### 6. dynamic_tools_node 同步 PlannerSession（双向）

- `generate_plan` 执行后：将新 plan_json 写入 `planner_session`
- `execute_plan` 执行后：将 `updated_plan_json`（带执行状态）写回 `planner_session`

这样 `planner_session.plan_json` 始终是**最新版本的 plan**（含执行进度）。

### 7. 单线程执行

当前阶段明确为**单线程**，不做并发 Executor。

### 8. Executor 工具现状

`src/executor_agent/tools.py` 已接入以下工具：

- `write_file(path, content, overwrite=True)`：将文本内容写入本地文件，自动递归创建目录；可控制是否覆盖；返回结构化结果（ok/path/overwritten/bytes/error）。
- `run_local_command(command, cwd=None, timeout=600)`：本地命令执行能力（当前过渡方案）。

后续将迁移为 **Sandbox 执行**：Executor 默认在隔离环境中运行代码，逐步替换直接本地命令执行路径。

### 9. Planner 工具现状（定义但未绑定）

`src/planner_agent/tools.py` 中的四个工具已定义但**未绑定到 graph**，当前 Planner 是纯 LLM 调用（无工具）。这是有意为之，后续迭代再绑定。

### 10. Planner 提示词结构与消息过滤

Planner 调用时的消息处理逻辑（已实现）：
1. 过滤掉所有带 `tool_calls` 的 `AIMessage`（对 Planner 无意义，可能造成误解）
2. 若消息列表中尚未包含 `SYSTEM_PROMPT`，则插入 `SystemMessage(SYSTEM_PROMPT)` 到开头
3. 追加 `HumanMessage(PLANNER_SYSTEM_PROMPT)` 到最后（注意：必须用 HumanMessage，不能用 SystemMessage；DeepSeek/SiliconFlow API 要求 system 消息只能出现在列表第一条，放到末尾会报 400）

`PLANNER_SYSTEM_PROMPT` 要求输出 Plan 之外有一点思考，JSON 放在 ` ```json ``` ` 代码块中，`run_planner()` 会自动提取。

---

## 后续计划

### 中期

- [ ] Executor 工具接入真实 HuggingFace Hub API（`search_huggingface_hub`）
- [ ] `propose_training_code` 生成真实可运行的训练代码文件（写入本地 .py 文件）
- [ ] 添加 Sandbox 执行训练代码能力（`execute_training_code` 基于隔离环境运行，替代直接 subprocess 本地执行）
- [ ] `save_final_report` 写入本地文件
- [ ] 补充测试用例（tests/ 目前全为空）

### 远期（进阶功能）

- [ ] **Executor 中途中断机制**：`execute_plan` 改用 `astream_events` 逐步迭代，每轮 ReAct 结束后检查 State 中断信号，主循环可注入 `HumanMessage` 强制 Executor 停下
- [ ] Docker / Kubernetes 部署支持（见 ROADMAP.md v0.3.0）

---

## 模块速查

| 文件 | 职责 |
|---|---|
| `src/supervisor_agent/graph.py` | 主循环图定义，`call_model` + `dynamic_tools_node` + `_build_id_to_name` + `_extract_updated_plan_from_executor` + `_extract_executor_status` |
| `src/supervisor_agent/state.py` | `State`、`InputState`、`PlannerSession`（含 `last_executor_status/last_executor_error`）、`ExecutorRef` |
| `src/supervisor_agent/tools.py` | `generate_plan`、`execute_plan`、`_mark_plan_steps_failed` |
| `src/planner_agent/graph.py` | Planner 图，`call_planner` 节点，`run_planner()` 对外接口（含 JSON 提取） |
| `src/planner_agent/prompts.py` | `PLANNER_SYSTEM_PROMPT`，含意图层 Plan JSON 格式要求（含步骤状态字段），放消息列表最后 |
| `src/planner_agent/tools.py` | 4 个规划工具（已定义，未绑定到 graph） |
| `src/executor_agent/graph.py` | Executor 自定义 StateGraph，`ExecutorState`、`ExecutorResult`、`call_executor` + `tools_node` + `route_executor_output`、`_parse_executor_output`、`run_executor()` 对外接口 |
| `src/executor_agent/prompts.py` | `EXECUTOR_SYSTEM_PROMPT`，按 intent 自主选工具，遇阻停止，输出 updated_plan |
| `src/executor_agent/tools.py` | Executor 工具集合（含 `write_file`、`run_local_command`） |
| `src/common/context.py` | 运行时配置，支持环境变量覆盖 |
| `src/common/prompts.py` | `SYSTEM_PROMPT`，Supervisor 全局系统提示 |
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
