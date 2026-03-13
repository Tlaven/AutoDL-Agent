"""Default prompts used by the agent."""

SYSTEM_PROMPT = """
"""

PLANNER_RULES = """Executor目前拥有一下工具
search_huggingface_hub(query: str, entity_type: str = "dataset") -> str:
propose_training_code(task_summary: str, model_id: str, dataset_id: str) -> str:
save_final_report(summary: str, metrics: dict, model_path: str = "") -> str:
现在，根据用户最新目标，开始制定或完善计划。"""