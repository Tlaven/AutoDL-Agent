# 测试构建计划（MVP）

本计划用于 `supervisor_agent` 的测试落地，目标是：**少而关键、避免重复、稳定可回归**。

## 1. 测试目标

- 保证主流程可用：状态流转、工具路由、失败兜底。
- 优先覆盖高风险行为：`planner_session` 回填、`[EXECUTOR_RESULT]` 解析、失败后收敛。
- 不追求表面覆盖率，优先“对稳定性有价值”的断言。

## 2. 当前测试版图（按层，不按时间）

### Unit

覆盖模块：
- `src/supervisor_agent/graph.py`
  - `_extract_updated_plan_from_executor`
  - `_extract_executor_status`
  - `route_model_output`
  - `call_model`
  - `dynamic_tools_node`
- `src/supervisor_agent/tools.py`
  - `_mark_plan_steps_failed`

已落地文件：
- `tests/unit_tests/supervisor_agent/test_graph_parsing_and_routing.py`
- `tests/unit_tests/supervisor_agent/test_tools_mark_failed.py`
- `tests/unit_tests/supervisor_agent/test_graph_nodes.py`

关键场景：
- 正常解析/路由。
- 脏输出或缺字段时的容错。
- `execute_plan` 元数据回填到 `planner_session`。
- **Executor 异常抛出路径**（已覆盖）：模拟 `run_executor` 抛异常，校验 `_mark_plan_steps_failed` 与结构化回填。

### Integration（mock 下游，验证主循环契约）

已落地文件：
- `tests/integration_tests/supervisor_agent/test_supervisor_smoke.py`
- `tests/integration_tests/supervisor_agent/test_supervisor_internal_trajectory.py`
- `tests/integration_tests/supervisor_agent/test_supervisor_multiturn_state_evolution.py`
- `tests/integration_tests/supervisor_agent/test_supervisor_executor_result_contract.py`

关键场景：
- `happy path`：`generate_plan -> execute_plan -> end`。
- `failure path`：失败状态与错误信息回填。
- 消息轨迹可机检（Human/AI/Tool 顺序、`tool_call_id` 对齐）。
- `[EXECUTOR_RESULT]` 契约脏数据兜底（invalid json、缺 marker、缺 `updated_plan_json`）。
- 失败后重规划再执行成功。
- **连续失败后收敛**（已覆盖）：两次失败后再次重规划，第三次执行成功。

### E2E Smoke

已落地文件：
- `tests/e2e_tests/supervisor_agent/test_supervisor_e2e_smoke.py`

关键场景：
- 完整 supervisor graph 单轮收敛。
- 最终状态、计划更新、结束条件验证。

## 3. 防重复清单（已完成项）

以下能力已具备，后续不要重复造同类用例：

- 轨迹顺序与 `tool_call_id` 一致性校验。
- `planner_session.last_executor_status / last_executor_error / plan_json` 回填校验。
- `completed / failed` 双路径覆盖。
- 失败后重规划收敛覆盖。
- `[EXECUTOR_RESULT]` 脏输出契约兜底。
- Executor 抛异常时 `_mark_plan_steps_failed` 兜底。
- 多次连续失败后的状态演进一致性。
- Executor 工具安全边界回归（已覆盖）：
  - `write_file` 相对路径/拒绝 `..`/内容大小上限
  - `run_local_command` 空命令、危险命令、无效 `cwd`、超时约束
  - `call_executor` 的 `write_file` 参数合理性（live_llm）


## 4. 运行命令

全量（supervisor 相关）：

```bash
uv run pytest tests/e2e_tests/supervisor_agent tests/integration_tests/supervisor_agent tests/unit_tests/supervisor_agent -q
```

关键门禁：

```bash
make test_supervisor_key
```

增量定向（本次新增场景）：

```bash
uv run pytest tests/unit_tests/supervisor_agent/test_graph_nodes.py tests/integration_tests/supervisor_agent/test_supervisor_multiturn_state_evolution.py -q
```

## 5. 新增测试触发条件（避免过度测试）

仅在以下情况新增：
- 修改 `supervisor_agent` 状态字段或回写逻辑。
- 修改 `[EXECUTOR_RESULT]` 结构或解析规则。
- 调整路由分支（`tools` / `__end__`）或收敛策略。
- 修复真实回归问题（线上/实际使用）。

不因文案、日志、注释变更新增测试。

## 6. 暂不纳入主门禁

- LLM 文风/措辞质量评审。
- 长链路多轮语义评分。

可作为手动评估，不阻塞日常合并。