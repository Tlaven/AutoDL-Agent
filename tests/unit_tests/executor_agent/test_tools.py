import builtins
import subprocess
from pathlib import Path

import pytest

import src.executor_agent.tools as tools_module


def test_write_file_creates_parent_and_reports_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    rel_path = "artifacts/nested/result.txt"
    target = (tmp_path / rel_path).resolve()

    result = tools_module.write_file.func(rel_path, "你好")

    assert result["ok"] is True
    assert result["path"] == str(target)
    assert result["overwritten"] is False
    assert result["bytes"] == len("你好".encode("utf-8"))
    assert result["error"] is None
    assert target.read_text(encoding="utf-8") == "你好"


def test_write_file_respects_overwrite_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    rel_path = "artifacts/exists.txt"
    target = tmp_path / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old", encoding="utf-8")

    result = tools_module.write_file.func(rel_path, "new", overwrite=False)

    assert result["ok"] is False
    assert result["overwritten"] is False
    assert result["bytes"] == 0
    assert "overwrite=False" in str(result["error"])
    assert target.read_text(encoding="utf-8") == "old"


def test_write_file_handles_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    rel_path = "artifacts/fail.txt"

    def fake_open(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(builtins, "open", fake_open)

    result = tools_module.write_file.func(rel_path, "x")

    assert result["ok"] is False
    assert result["bytes"] == 0
    assert result["overwritten"] is False
    assert "disk full" in str(result["error"])


def test_write_file_rejects_absolute_path(tmp_path: Path) -> None:
    target = (tmp_path / "abs.txt").resolve()

    result = tools_module.write_file.func(str(target), "x")

    assert result["ok"] is False
    assert result["bytes"] == 0
    assert "相对路径" in str(result["error"])


def test_write_file_rejects_parent_dir_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    result = tools_module.write_file.func("../escape.txt", "x")

    assert result["ok"] is False
    assert result["bytes"] == 0
    assert ".." in str(result["error"])


def test_write_file_rejects_too_large_content(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    too_large_content = "a" * (tools_module.MAX_WRITE_FILE_BYTES + 1)

    result = tools_module.write_file.func("artifacts/big.txt", too_large_content)

    assert result["ok"] is False
    assert result["bytes"] == 0
    assert "content 过大" in str(result["error"])


def test_run_local_command_rejects_non_positive_timeout() -> None:
    result = tools_module.run_local_command.func("echo hi", timeout=0)

    assert result["ok"] is False
    assert result["returncode"] is None
    assert result["timed_out"] is False
    assert "timeout" in str(result["error"])


def test_run_local_command_rejects_empty_command() -> None:
    result = tools_module.run_local_command.func("   ")

    assert result["ok"] is False
    assert result["returncode"] is None
    assert "command 不能为空" in str(result["error"])


def test_run_local_command_rejects_dangerous_command() -> None:
    result = tools_module.run_local_command.func("shutdown /s /t 0")

    assert result["ok"] is False
    assert result["returncode"] is None
    assert "禁止执行" in str(result["error"])


def test_run_local_command_rejects_nonexistent_cwd(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    result = tools_module.run_local_command.func("echo hi", cwd=str(missing_dir))

    assert result["ok"] is False
    assert result["returncode"] is None
    assert "cwd 不存在" in str(result["error"])


def test_run_local_command_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wd_path = (tmp_path / "wd").resolve()
    wd_path.mkdir(parents=True, exist_ok=True)
    expected_cwd = str(wd_path)

    def fake_run(*_args, **kwargs):

        assert kwargs["cwd"] == expected_cwd
        return subprocess.CompletedProcess(args="echo hi", returncode=0, stdout="hello\n", stderr="")

    monkeypatch.setattr(tools_module.subprocess, "run", fake_run)

    result = tools_module.run_local_command.func("echo hi", cwd=expected_cwd, timeout=10)

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert result["stdout"] == "hello\n"
    assert result["stderr"] == ""
    assert result["error"] is None


def test_run_local_command_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="sleep", timeout=1, output="partial out", stderr="partial err")

    monkeypatch.setattr(tools_module.subprocess, "run", fake_run)

    result = tools_module.run_local_command.func("sleep", timeout=1)

    assert result["ok"] is False
    assert result["returncode"] is None
    assert result["timed_out"] is True
    assert result["stdout"] == "partial out"
    assert result["stderr"] == "partial err"
    assert "超时" in str(result["error"])


def test_run_local_command_handles_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(tools_module.subprocess, "run", fake_run)

    result = tools_module.run_local_command.func("badcmd")

    assert result["ok"] is False
    assert result["returncode"] is None
    assert result["timed_out"] is False
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert "spawn failed" in str(result["error"])


def test_get_executor_capabilities_docs_uses_first_line(monkeypatch: pytest.MonkeyPatch) -> None:
    class T1:
        description = "第一行\n第二行"
        name = "write_file"

    class T2:
        description = ""
        name = "run_local_command"

    monkeypatch.setattr(tools_module, "get_executor_tools", lambda: [T1(), T2()])

    docs = tools_module.get_executor_capabilities_docs()

    assert "- 第一行" in docs
    assert "- 工具 2" in docs
    assert "run_local_command 使用提示" in docs

