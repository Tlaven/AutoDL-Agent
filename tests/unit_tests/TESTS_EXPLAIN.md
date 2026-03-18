# Supervisor Agent 测试说明（Unit + Smoke）

本文记录 `supervisor_agent` 已落地测试的覆盖范围与目的，遵循“少而关键”的稳定性策略。

## 1) Unit: `test_graph_parsing_and_routing.py`

### 覆盖目标
- `src/supervisor_agent/graph.py` 纯逻辑函数：
  - `_extract_updated_plan_from_executor`
  - `_extract_executor_status`
  - `_build_id_to_name`
  - `route_model_output`

### 关键用例
- `updated_plan_json` 提取：正常、缺标记、非法 JSON
- `status/error_detail` 提取：正常、缺标记、非法 JSON
- `tool_call_id -> tool_name` 映射：正常映射与空映射
- 路由：有 `tool_calls` 到 `"tools"`；无 `tool_calls` 到 `"__end__"`；非 `AIMessage` 抛 `ValueError`

---

## 2) Unit: `test_tools_mark_failed.py`

### 覆盖目标
- `src/supervisor_agent/tools.py::_mark_plan_steps_failed`

### 关键用例
- `list` 结构计划：`pending/running/缺失 status` 改为 `failed`
- `{ "steps": [...] }` 结构计划：待执行步骤改 `failed` 并写入 `failure_reason`
- 非法 JSON：原样返回
- 空字符串：原样返回

---

## 3) Unit: `test_graph_nodes.py`

### 覆盖目标
- `src/supervisor_agent/graph.py` 核心节点：
  - `call_model`
  - `dynamic_tools_node`

### 关键用例
- `call_model`：
  - `is_last_step=True` 且模型仍返回 `tool_calls` 时触发强制收敛
  - 非最后一步时正常透传模型输出
- `dynamic_tools_node`：
  - `execute_plan` 工具消息回写 `planner_session.plan_json`
  - 同步回写 `last_executor_status / last_executor_error`
  - `generate_plan` 工具消息写入新的 `planner_session`

---

## 4) Integration Smoke: `test_supervisor_smoke.py`

### 覆盖目标
在 mock 下游依赖的前提下验证 Supervisor 主流程可用性：
- `happy path`：`generate_plan -> execute_plan -> end`
- `failure path`：`execute_plan` 回传失败后的状态回填

### 核心断言
- `planner_session.plan_json` 被更新
- `planner_session.last_executor_status` 正确
- `planner_session.last_executor_error` 正确
- 最终消息收敛为无 `tool_calls` 的 `AIMessage`

---

## 5) E2E Smoke: `test_supervisor_e2e_smoke.py`

### 覆盖目标
- 以完整 supervisor graph 为入口，验证单轮对话可走完主流程。

### 核心断言
- 对话中触发规划与执行
- `planner_session` 最终存在且状态为 `completed`
- `plan_json` 被更新为执行后状态
- 最终输出收敛为无 `tool_calls` 的 `AIMessage`

---

## 6) `tests/conftest.py` 说明

`tests/conftest.py` 将项目根目录加入 `sys.path`，解决测试导入 `src.*` 时的路径问题。

---

## 7) 执行方式与当前结果

```bash
uv run pytest tests/e2e_tests/supervisor_agent tests/integration_tests/supervisor_agent tests/unit_tests/supervisor_agent -q
```

当前结果：`22 passed`。
