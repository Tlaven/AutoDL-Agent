# 测试构建计划（MVP）

本计划用于 `supervisor_agent` 的分层测试落地，目标是**少而关键**，先保稳定再扩覆盖。

## 1. 目标与原则

- 先保证主流程可用：状态流转、异常兜底、路由正确。
- 测试分层：`unit` 为主，补少量 `integration/e2e smoke`。
- 避免过度细节断言，优先断言对业务稳定性有影响的结果。

## 2. 分层范围

### A. Unit（已完成）

覆盖函数：
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

### B. Integration Smoke（已完成 2 条）

已落地文件：
- `tests/integration_tests/supervisor_agent/test_supervisor_smoke.py`

覆盖场景：
1. `happy path`：`generate_plan -> execute_plan -> 结束`
2. `failure path`：`execute_plan` 回传 `failed` 后的状态回填

说明：Integration 层采用 mock 下游调用，保证稳定与可重复。

### C. E2E Smoke（已完成 1 条）

已落地文件：
- `tests/e2e_tests/supervisor_agent/test_supervisor_e2e_smoke.py`

覆盖场景：
- 完整启动 supervisor graph，单轮对话触发 `generate_plan -> execute_plan -> end`
- 验证最终收敛、状态回填与计划更新

## 3. 里程碑状态

- M1（已完成）：核心 unit（解析 + 异常兜底 + 路由）
- M2（已完成）：关键节点 unit（`call_model`、`dynamic_tools_node`）
- M3（已完成）：2 条 integration smoke
- M4（已完成）：1 条 e2e smoke

## 4. 当前结果

- `uv run pytest tests/e2e_tests/supervisor_agent tests/integration_tests/supervisor_agent tests/unit_tests/supervisor_agent -q` → `22 passed`
- integration/e2e/unit 组合用例连续执行 3 次：
  - Run 1: `22 passed in 0.78s`
  - Run 2: `22 passed in 0.76s`
  - Run 3: `22 passed in 0.76s`

## 5. 通过标准（MVP）

- `tests/unit_tests/supervisor_agent` 全部通过（已满足）
- integration smoke 稳定通过（连续 3 次）（已满足）
- 流程失败时能看到明确错误状态与回填信息（已满足）


## 6. 执行命令

```bash
uv run pytest tests/e2e_tests/supervisor_agent tests/integration_tests/supervisor_agent tests/unit_tests/supervisor_agent -q
```

> 注：若后续加入“LLM 质量评审”，建议作为 nightly/手动评估，不作为主 CI 阻塞条件。

## 7. 新增测试触发条件（避免过度测试）

仅在以下情况新增用例：
- 修改了 `supervisor_agent` 的状态字段或状态回写逻辑
- 修改了 `[EXECUTOR_RESULT]` 的结构或解析规则
- 调整了路由分支（`tools` / `__end__`）或步数收敛策略
- 修复线上/真实使用中出现过的回归问题

不因纯文案调整、日志调整而新增测试。

## 8. 暂不纳入主门禁的范围

- LLM 输出文风/措辞质量评审
- 长链路、多轮复杂任务的语义评分

以上可作为评估脚本或 nightly 任务，不阻塞日常开发合并。

## 9. 阶段记录（自动化内部运行验证）

### 目前已完成

- 新增内部轨迹自动化测试：
  - `tests/integration_tests/supervisor_agent/test_supervisor_internal_trajectory.py`
- 新增命令入口：
  - `make test_supervisor_trace`
- 已实现的自动断言能力：
  - 自动检查消息轨迹顺序（`Human -> AI(tool) -> Tool -> AI(tool) -> Tool -> AI(final)`）
  - 自动检查 `tool_call_id` 与 `tool_calls.id` 的一致性
  - 自动检查 `planner_session` 的 `last_executor_status`、`last_executor_error`、`plan_json` 回填
  - 覆盖 `completed` / `failed` 两条执行路径
- 当前回归结果（含新增轨迹测试）：
  - `uv run pytest tests/integration_tests/supervisor_agent/test_supervisor_internal_trajectory.py tests/integration_tests/supervisor_agent/test_supervisor_smoke.py tests/e2e_tests/supervisor_agent/test_supervisor_e2e_smoke.py tests/unit_tests/supervisor_agent -q` → `24 passed`

### 未来打算做什么（下一阶段）

- 新增“多轮状态演进”自动化测试：
  - 验证多轮输入下 `planner_session.plan_json` 与 `last_executor_*` 的阶段性变化和收敛
- 新增“异常/脏输出契约”回归测试：
  - 重点覆盖 `[EXECUTOR_RESULT]` 缺字段、脏 JSON、混合文本等场景
  - 确保解析失败时仍能稳定兜底，不破坏主流程
- 增加统一测试入口：
  - 提供一条命令聚合 smoke + trajectory + contract 关键检查，降低人工验证成本

