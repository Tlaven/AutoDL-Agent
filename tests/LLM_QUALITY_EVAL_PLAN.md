# 节点级 LLM 质量评测测试计划（Monkeypatch 隔离 + 纯逻辑节点直调）

## 1. 目标

为 graph 中「有 LLM 参与」的单节点建立可重复、可回归的质量评测测试方案，验证：
- 节点输入消息组装是否正确
- 节点与模型交互契约是否正确（消息顺序、System Prompt、工具绑定）
- 节点输出结构是否符合预期（含异常路径）

本计划仅做**节点级评测**，不覆盖全链路图执行。

---

## 2. 评测范围

| Agent | 节点函数 | 文件 | 是否绑定工具 |
|---|---|---|---|
| Supervisor | `call_model` | `src/supervisor_agent/graph.py` | 是 |
| Planner | `call_planner` | `src/planner_agent/graph.py` | 否 |
| Executor | `call_executor` | `src/executor_agent/graph.py` | 是 |

---

## 3. 核心策略

### 3.1 Monkeypatch 隔离
使用 `monkeypatch` 替换外部依赖，确保测试只验证节点逻辑：
- `load_chat_model` -> `FakeModel`
- `get_tools` / `get_executor_tools` -> 假工具列表

不连接真实模型、不访问外部服务。

### 3.2 纯逻辑节点直调
直接调用异步节点函数，而非运行完整图：
- `await call_model(state, runtime)`
- `await call_planner(state)`
- `await call_executor(state)`

### 3.3 质量断言方式
`FakeModel` 记录 `ainvoke(messages)` 收到的入参并返回受控 `AIMessage`：
- 对入参做断言（消息过滤、提示词注入、顺序、数量）
- 对返回值做断言（字段、角色名、JSON 文本结构、收敛分支行为）

---

## 4. 通用测试夹具设计

## 4.1 FakeModel 设计要点
- 提供 `bind_tools(tools)`：记录绑定工具并返回 `self`
- 提供 `ainvoke(messages)`：
  - 保存 `messages` 到可检查变量
  - 返回预设 `AIMessage`

### 4.2 断言维度（统一）
- **输入维度**：节点传入模型的 messages 是否符合预期
- **调用维度**：是否进行了工具绑定（仅 Supervisor/Executor）
- **输出维度**：节点输出字典结构是否稳定
- **异常维度**：空输出、最后一步强制收敛等边界路径

---

## 5. 节点评测用例计划

## 5.1 Supervisor `call_model`

### Case S1：正常调用 + 工具绑定
- **意图**：验证工具绑定发生，且返回结构为 `{"messages": [AIMessage]}`
- **输入**：常规 `state.messages`，`is_last_step=False`
- **Mock 输出**：`AIMessage(content="ok", tool_calls=[])`
- **断言**：
  - `bind_tools` 被调用且工具列表非空
  - `ainvoke` 收到的消息包含系统提示 + 用户上下文
  - 返回中仅追加 1 条 `AIMessage`

### Case S2：最后一步且模型仍请求工具（强制收敛）
- **意图**：验证 `is_last_step=True` 时的防扩散逻辑
- **输入**：`is_last_step=True`
- **Mock 输出**：带 `tool_calls` 的 `AIMessage`
- **断言**：
  - 节点返回替代性收敛消息（无继续工具调用）
  - 输出结构稳定，不抛异常

### Case S3：最后一步且模型不请求工具
- **意图**：验证不触发收敛分支时的原样返回
- **Mock 输出**：普通文本 `AIMessage`
- **断言**：直接返回模型消息

---

## 5.2 Planner `call_planner`

### Case P1：过滤 `AIMessage(tool_calls)`
- **意图**：验证 planner 入模前会过滤掉带 `tool_calls` 的 AI 消息
- **输入**：混合消息序列（含 `AIMessage(tool_calls=[...])`）
- **Mock 输出**：合法 Plan JSON 文本
- **断言**：
  - `ainvoke` 收到的消息不含该类 AI 消息
  - 其余消息顺序保持稳定

### Case P2：System Prompt 注入顺序
- **意图**：验证 SYSTEM_PROMPT 和 PLANNER_SYSTEM_PROMPT 注入位置
- **输入**：无 system message 的普通会话
- **断言**：
  - 开头存在基础 `SystemMessage(SYSTEM_PROMPT)`
  - 末尾追加 `SystemMessage(PLANNER_SYSTEM_PROMPT)`

### Case P3：输出包装规范
- **意图**：验证节点返回 `AIMessage(name="planner")`
- **Mock 输出**：合法 Plan JSON
- **断言**：
  - 返回 `{"messages": [AIMessage]}`
  - `AIMessage.name == "planner"`
  - `content` 非空且可被 JSON 解析

### Case P4：空输出防御
- **意图**：验证空内容时抛出 `RuntimeError`
- **Mock 输出**：`AIMessage(content="")`
- **断言**：抛出 `RuntimeError`

---

## 5.3 Executor `call_executor`

### Case E1：正常调用 + 工具绑定
- **意图**：验证工具绑定与基础返回结构
- **输入**：包含 plan 和用户请求的消息
- **Mock 输出**：含 ```json 代码块``` 的执行结果文本
- **断言**：
  - `bind_tools` 被调用
  - `ainvoke` 入参完整
  - 返回 `{"messages": [AIMessage]}`

### Case E2：最后一步且模型仍请求工具（强制收敛）
- **意图**：验证 `is_last_step=True` 的收敛逻辑
- **Mock 输出**：带 `tool_calls` 的 `AIMessage`
- **断言**：
  - 节点返回收敛消息
  - 不继续外部工具路径

### Case E3：输出契约可解析性（与执行结果协议对齐）
- **意图**：验证模型返回符合 ExecutorResult 协议
- **Mock 输出**：
  - `status` in {`completed`, `failed`}
  - `summary` 非空
  - `updated_plan` 为完整 plan schema
- **断言**：
  - 从 `AIMessage.content` 提取 JSON 成功
  - 关键字段完整且类型正确

---

## 6. 用例落地组织建议

建议新增测试文件：
- `tests/unit_tests/supervisor_agent/test_node_llm_quality.py`
- `tests/unit_tests/planner_agent/test_node_llm_quality.py`
- `tests/unit_tests/executor_agent/test_node_llm_quality.py`

命名建议：
- `test_call_model_*`
- `test_call_planner_*`
- `test_call_executor_*`

---

## 7. 验收标准（Definition of Done）

- 三个节点的核心正向/异常路径均有用例覆盖
- 所有断言均基于节点入模参数 + 节点输出结构
- 不依赖真实 LLM / 网络
- 本地可稳定重复执行
- 用例失败时可直接定位到具体节点行为（而非图级联噪声）

---

## 8. 执行顺序

1. 先实现 Planner 节点用例（消息加工逻辑最丰富）
2. 再实现 Supervisor 节点用例（收敛分支已存在参考）
3. 最后实现 Executor 节点用例（工具绑定 + 协议内容断言）
4. 回归运行 unit tests，补齐 flaky 风险点

---

## 9. 风险与对策

- **风险**：模型消息对象字段在上游库升级后变化
  - **对策**：断言聚焦契约字段（`content`、`name`、`tool_calls`），减少对内部实现耦合

- **风险**：提示词文本微调导致“全文等值断言”脆弱
  - **对策**：断言“存在性 + 位置关系”（开头/结尾），避免整段硬编码

- **风险**：测试污染（monkeypatch 未回收）
  - **对策**：严格使用 `monkeypatch` fixture，避免全局状态残留
