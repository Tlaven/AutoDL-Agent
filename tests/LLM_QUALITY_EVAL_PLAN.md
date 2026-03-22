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
  - 末尾追加 `HumanMessage(PLANNER_SYSTEM_PROMPT)`

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

### Case E4：`write_file` 参数透传准确性（tools_node）
- **意图**：验证 LLM 产生的 tool_calls 参数会被准确传给工具实现
- **输入**：`AIMessage.tool_calls` 含 `write_file(path, content, overwrite)`
- **断言**：
  - 工具函数实收 `path/content/overwrite` 与 tool_calls.args 一致
  - 返回消息类型为 `ToolMessage`

### Case E5：`write_file` 参数合理性（live_llm）
- **意图**：验证真实 LLM 调用 `write_file` 时参数具备基础安全与可执行性
- **输入**：明确要求调用一次 `write_file`（相对路径、`.txt` 后缀、可控内容）
- **断言**：
  - 产生 `tool_calls`
  - `name == "write_file"`
  - `path` 为非空相对路径，且不含 `..`
  - `content` 非空且长度在限制内
  - `overwrite` 若提供则为布尔值

---


## 6. 用例落地组织建议

已新增测试文件：
- `tests/unit_tests/supervisor_agent/test_supervisor_node_llm_quality.py`
- `tests/unit_tests/planner_agent/test_planner_node_llm_quality.py`
- `tests/unit_tests/executor_agent/test_executor_node_llm_quality.py`


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

---

## 10. 已落地进展（2026-03-18）

### 10.1 用例实现状态
- [x] Planner 节点用例（P1/P2/P3/P4）
- [x] Supervisor 节点用例（S1/S2/S3）
- [x] Executor 节点用例（E1/E2/E3）

### 10.2 已实现文件
- `tests/unit_tests/planner_agent/test_planner_node_llm_quality.py`
- `tests/unit_tests/supervisor_agent/test_supervisor_node_llm_quality.py`
- `tests/unit_tests/executor_agent/test_executor_node_llm_quality.py`

### 10.3 本地回归结果
- 新增 3 个测试文件定向执行：`10 passed`
- 相关目录回归执行：
  - `tests/unit_tests/supervisor_agent`
  - `tests/unit_tests/planner_agent`
  - `tests/unit_tests/executor_agent`
- 合计结果：`32 passed`

---

## 11. `test_live_llm_nodes.py` 实战经验（2026-03-18）

### 11.1 分层定位
- 该文件定位为**真实 API 连通性 + 节点输出契约校验**，不替代 unit mock 测试。
- 覆盖 3 个节点：
  - `call_planner`
  - `call_executor`
  - `call_model`

### 11.2 必要工程约束
- 使用 `pytest.mark.live_llm` 将在线测试与默认测试集隔离。
- 使用 `skipif(not SILICONFLOW_API_KEY)`，避免无密钥环境报错。
- 在 `pyproject.toml` 注册 marker，避免 `PytestUnknownMarkWarning`。

### 11.3 断言策略（抗波动）
- 不做“逐字文案”断言，只做契约断言：
  - Planner：可提取 JSON、关键字段完整（`goal/steps/overall_expected_output`）。
  - Executor/Supervisor：文本非空，命中约束回复或命中“最后一步收敛提示”。
- 对 Planner 输出采用“代码块优先提取”：先提取 ` ```json ... ``` `，再 `json.loads`。

### 11.4 踩坑与修复
- 坑：JSON 提取正则写成了转义版 `\\s` / `\\S`，导致无法从代码块提取内容。
- 修复：改为原始正则中的 `\s` / `\S`（`r"```(?:json)?\s*([\s\S]*?)```"`）。
- 现象：模型会在 JSON 前附加“思考说明”，所以必须做代码块提取，不能直接 `json.loads(全文)`。

### 11.5 如何查看真实 LLM 输出文本
- 默认 `pytest` 会捕获 `print`，所以看不到模型原文。
- 在测试中加入 `_show_output(tag, content)` 打印，并使用 `-s` 运行。
- 推荐命令（自动读取 `.env`）：

```bash
cd /d c:/Projects/Agents/AutoDL_Agent && uv run python -c "from dotenv import load_dotenv; load_dotenv(); import pytest; raise SystemExit(pytest.main(['tests/integration_tests/live_llm/test_live_llm_nodes.py','-m','live_llm','-q','-s']))"
```

### 11.6 当前基线结果
- `tests/integration_tests/live_llm/test_live_llm_nodes.py`
- 最近回归：`3 passed`（真实 API + 输出可见）

### 11.7 工具调用质量补充（2026-03-22）
- 新增 `call_executor` 的 `write_file` 参数质量 live 用例（路径、内容、overwrite 类型约束）。
- 新增 `tools_node` 参数透传用例，验证 `tool_calls.args -> 工具函数入参` 一致性。
- `run_local_command` 增加“适度安全限制”（仅拦截高风险命令）与 LLM 使用提示文案。

### 11.8 工具安全边界与回归结果（2026-03-22）

#### 11.8.1 代码变更
- `write_file` 新增入参校验：
  - `path` 必须为相对路径
  - 拒绝 `..` 父目录跳转
  - `content` 字节大小上限（`MAX_WRITE_FILE_BYTES`）
- `run_local_command` 新增入参与安全校验：
  - 空命令/超长命令拦截
  - `timeout` 上限（`MAX_LOCAL_COMMAND_TIMEOUT`）
  - 高风险命令拦截（关机/重启/格式化/危险删除）
  - `cwd` 必须存在且为目录
- `get_executor_capabilities_docs()` 为 `run_local_command` 追加 LLM 安全使用提示。

#### 11.8.2 测试补充
- `tests/unit_tests/executor_agent/test_tools.py`
  - 新增 `write_file` 非法路径与超大内容拒绝用例
  - 新增 `run_local_command` 空命令/危险命令/无效 `cwd` 拒绝用例
- `tests/unit_tests/executor_agent/test_executor_node_llm_quality.py`
  - 新增 `tools_node` 对 `write_file` 参数透传一致性用例（`ToolMessage` 返回）
- `tests/integration_tests/live_llm/test_live_llm_nodes.py`
  - 新增 `call_executor` 的 `write_file` 参数合理性 live 用例

#### 11.8.3 最近回归结果
- 命令：
  - `uv run pytest tests/unit_tests/executor_agent/test_tools.py tests/unit_tests/executor_agent/test_executor_node_llm_quality.py tests/integration_tests/live_llm/test_live_llm_nodes.py -q`
- 结果：`18 passed, 4 skipped`







