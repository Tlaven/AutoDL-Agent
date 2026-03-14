# executor_agent/tools.py

from langchain_core.tools import tool
import json

@tool
async def search_huggingface_hub(query: str, entity_type: str = "dataset") -> str:
    """在 Hugging Face Hub 搜索数据集或模型。
    entity_type 可选：dataset, model, space
    返回前 5 个最相关结果的简要信息。
    """
    # 这里只是占位，实际应调用 huggingface_hub API
    return json.dumps({
        "query": query,
        "entity_type": entity_type,
        "results": [
            {"id": f"{entity_type}-example-1", "downloads": 12000, "likes": 450},
            {"id": f"{entity_type}-example-2", "downloads": 8900, "likes": 320},
        ]
    }, ensure_ascii=False, indent=2)


@tool
async def propose_training_code(task_summary: str, model_id: str, dataset_id: str) -> str:
    """根据任务、模型、数据集，生成 PyTorch Lightning 风格的训练代码框架。
    返回代码字符串（包含 LightningModule、DataModule、Trainer 等）。
    """
    # 占位实现
    code = f"""# 自动生成的训练代码框架
import lightning as L
from transformers import AutoModelForImageClassification, AutoImageProcessor

class LitClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained("{model_id}", num_labels=2)
        self.processor = AutoImageProcessor.from_pretrained("{model_id}")

    # ... 其他方法占位

# 建议后续步骤：实现 train_dataloader, training_step 等
"""
    return code


@tool
async def save_final_report(summary: str, metrics: dict, model_path: str = "") -> str:
    """保存最终报告（占位）。实际项目中可以写入文件或上传到 wandb/huggingface 等。"""
    report = {
        "summary": summary,
        "final_metrics": metrics,
        "model_path": model_path or "local/checkpoint-last",
        "timestamp": "2025-03-12T23:55:00Z"
    }
    return f"报告已生成：\n{json.dumps(report, indent=2, ensure_ascii=False)}"


def get_executor_tools():
    return [search_huggingface_hub, propose_training_code, save_final_report]