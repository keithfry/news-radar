"""Run the local `claude` CLI without the user's personal context.

The flags in ISOLATION_FLAGS stop the call from loading CLAUDE.md, hooks,
skills, MCP servers and tools. Never add --bare: it ignores the subscription
login. Each call runs in a fresh empty temporary directory.
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

ISOLATION_FLAGS = [
    "--setting-sources", "",
    "--strict-mcp-config",
    "--mcp-config", '{"mcpServers":{}}',
    "--disable-slash-commands",
    "--tools", "",
    "--no-session-persistence",
]


class ClaudeCallError(RuntimeError):
    """The claude CLI failed, returned unparseable output, or reported an error."""


@dataclass
class ClaudeResult:
    text: str
    structured: dict | None
    usage: dict
    cost_usd: float


def run_claude(
    prompt: str,
    model: str,
    *,
    system_prompt: str | None = None,
    json_schema: dict | None = None,
    effort: str | None = None,
    timeout: int = 900,
) -> ClaudeResult:
    claude_bin = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    cmd = [claude_bin, "-p", prompt, "--model", model, "--output-format", "json", *ISOLATION_FLAGS]
    if system_prompt is not None:
        cmd += ["--system-prompt", system_prompt]
    if json_schema is not None:
        cmd += ["--json-schema", json.dumps(json_schema)]
    if effort is not None:
        cmd += ["--effort", effort]

    with tempfile.TemporaryDirectory(prefix="newsradar-claude-") as cwd:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeCallError(f"claude timed out after {timeout}s") from e

    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError) as e:
        raise ClaudeCallError(
            f"claude exited {proc.returncode} with unparseable output: "
            f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
        ) from e
    if not isinstance(data, dict):
        raise ClaudeCallError(f"claude returned non-object JSON: {proc.stdout[:500]}")

    if proc.returncode != 0 or data.get("is_error"):
        detail = data.get("result") or (proc.stderr or "").strip()
        raise ClaudeCallError(f"claude failed (exit {proc.returncode}): {str(detail)[:500]}")

    return ClaudeResult(
        text=data.get("result") or "",
        structured=data.get("structured_output"),
        usage=data.get("usage") or {},
        cost_usd=float(data.get("total_cost_usd") or 0.0),
    )
