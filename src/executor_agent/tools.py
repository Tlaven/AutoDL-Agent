# executor_agent/tools.py
import os
import subprocess
from typing import TypedDict

from langchain_core.tools import tool


class WriteFileResult(TypedDict):
    ok: bool
    path: str
    overwritten: bool
    bytes: int
    error: str | None


class LocalCommandResult(TypedDict):
    ok: bool
    command: str
    cwd: str
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    error: str | None


@tool
def write_file(path: str, content: str, overwrite: bool = True) -> WriteFileResult:
    """写入文本文件并返回结构化确认信息。"""
    abs_path = os.path.abspath(path)
    parent_dir = os.path.dirname(abs_path)

    try:
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if not overwrite and os.path.exists(abs_path):
            return {
                "ok": False,
                "path": abs_path,
                "overwritten": False,
                "bytes": 0,
                "error": "目标文件已存在且 overwrite=False",
            }

        existed_before = os.path.exists(abs_path)
        encoded = content.encode("utf-8")
        with open(abs_path, "w", encoding="utf-8") as f:
            _ = f.write(content)

        return {
            "ok": True,
            "path": abs_path,
            "overwritten": existed_before,
            "bytes": len(encoded),
            "error": None,
        }
    except OSError as e:
        return {
            "ok": False,
            "path": abs_path,
            "overwritten": False,
            "bytes": 0,
            "error": str(e),
        }


@tool
def run_local_command(command: str, cwd: str | None = None, timeout: int = 600) -> LocalCommandResult:
    """在本地执行命令并返回执行结果。"""
    if timeout <= 0:
        return {
            "ok": False,
            "command": command,
            "cwd": os.path.abspath(cwd) if cwd else os.getcwd(),
            "returncode": None,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
            "error": "timeout 必须为正整数秒",
        }

    exec_cwd = os.path.abspath(cwd) if cwd else os.getcwd()

    try:
        completed = subprocess.run(
            command,
            cwd=exec_cwd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "command": command,
            "cwd": exec_cwd,
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "error": None,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "command": command,
            "cwd": exec_cwd,
            "returncode": None,
            "timed_out": True,
            "stdout": e.stdout if isinstance(e.stdout, str) else "",
            "stderr": e.stderr if isinstance(e.stderr, str) else "",
            "error": f"命令执行超时（>{timeout}s）",
        }
    except OSError as e:
        return {
            "ok": False,
            "command": command,
            "cwd": exec_cwd,
            "returncode": None,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
            "error": str(e),
        }


def get_executor_tools() -> list[object]:
    """返回 Executor 可用的工具列表。"""
    return [write_file, run_local_command]


def get_executor_capabilities_docs() -> str:
    """返回供 Planner/Executor 共享的能力描述文案。"""
    capabilities: list[str] = []
    for idx, tool_obj in enumerate(get_executor_tools(), start=1):
        description = str(getattr(tool_obj, "description", "") or "").strip()
        if description:
            first_line = description.splitlines()[0].strip()
            capabilities.append(f"- {first_line}")
        else:
            capabilities.append(f"- 工具 {idx}")

    return "\n".join(capabilities) if capabilities else "- （当前无可用工具）"

