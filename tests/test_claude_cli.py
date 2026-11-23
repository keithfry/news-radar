import json
import os
import subprocess
from pathlib import Path

import pytest

from newsradar import claude_cli, llm
from newsradar.claude_cli import ISOLATION_FLAGS, ClaudeCallError, ClaudeResult, run_claude


class _FakeRun:
    def __init__(self):
        self.calls = []
        self.stdout = json.dumps({"is_error": False, "subtype": "success", "result": "ok"})
        self.returncode = 0

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs.get("cwd")))
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, "")


@pytest.fixture
def fake_run(monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    return fake


def test_command_has_isolation_flags_and_options(fake_run):
    run_claude("hi", "claude-opus-5-5", system_prompt="sys", json_schema={"type": "object"}, effort="medium")
    cmd, cwd = fake_run.calls[0]
    for flag in ISOLATION_FLAGS:
        assert flag in cmd
    assert "--bare" not in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    assert cmd[cmd.index("--effort") + 1] == "medium"
    assert cmd[cmd.index("--system-prompt") + 1] == "sys"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == {"type": "object"}
    assert cmd[cmd.index("--output-format") + 1] == "json"


def test_runs_in_an_empty_temp_dir(fake_run):
    run_claude("hi", "m")
    _, cwd = fake_run.calls[0]
    assert cwd != os.getcwd() and Path(cwd).name.startswith("newsradar-claude-")


def test_parses_structured_output_usage_and_cost(fake_run):
    fake_run.stdout = json.dumps({"is_error": False, "subtype": "success", "result": "{}",
                                  "structured_output": {"is_ad": True}, "usage": {"input_tokens": 5},
                                  "total_cost_usd": 0.01})
    r = run_claude("hi", "m")
    assert r.structured == {"is_ad": True} and r.usage == {"input_tokens": 5} and r.cost_usd == 0.01


@pytest.mark.parametrize("stdout,code", [('{"is_error": true, "subtype": "error", "result": "rate limited"}', 0), ("", 1), ("not json", 0)])
def test_failures_raise_claude_call_error(fake_run, stdout, code):
    fake_run.stdout, fake_run.returncode = stdout, code
    with pytest.raises(ClaudeCallError):
        run_claude("hi", "m")


def test_chat_anthropic_uses_isolated_call(monkeypatch):
    seen = {}
    monkeypatch.setattr(llm, "run_claude", lambda p, m, **kw: seen.update(p=p, m=m) or ClaudeResult("ok", None, {}, 0.0))
    assert llm._chat_anthropic("prompt", "claude-haiku-5-5") == "ok"
    assert seen == {"p": "prompt", "m": "claude-haiku-5-5"}
