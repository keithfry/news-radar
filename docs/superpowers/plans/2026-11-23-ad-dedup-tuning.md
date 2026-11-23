# Daily ad and dedup tuning implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After each daily scan, Claude Opus 5.5 judges the ad gate's and duplicate check's decisions, and a script applies the fixes that pass a regression check.

**Architecture:** The scan writes an evidence file of every ad and dedup decision. A new `newsradar tune` command sends that evidence to an isolated `claude -p` judge, builds a candidate `ad-detector` model and dedup example block from the judge's proposals, scores old against new, and applies only what passes. Learned data lives in news-radar's ignored `data/` folder. The tuner never commits.

**Tech Stack:** Python 3.13, uv, pytest, Ollama (qwen3.5:9b base), the `claude` CLI on the user's subscription.

**Spec:** `docs/superpowers/specs/2026-11-23-ad-dedup-tuning-design.md`

## Global Constraints

- Judge: `--model claude-opus-5-5 --effort medium`.
- Every `claude` call uses these flags and runs in a fresh empty temporary directory: `--setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --disable-slash-commands --tools "" --no-session-persistence`. Never `--bare` (it ignores the subscription login).
- Data lives under `[tuning] data_dir`, default `<news-radar repo>/data`, which `.gitignore` excludes. The tuner never runs `git`.
- The tuner sends the ad model exactly what the scan sends: `classify_ad(title, text[:1000], model)`.
- `AdDetectorModelfile` uses `PARAMETER num_ctx 8192`.
- Learned example summaries are at most 200 characters, and links are cut to `scheme://host/`.
- Evidence older than 30 days is deleted.
- A tuner failure never changes `run-agent.sh`'s exit status.
- Never run a scan with the real config while testing: even `--dry-run` writes pages into `techradar/techradar/`, which the 08:00 job deploys.
- Tests that need Ollama or `claude` are skipped when unavailable, like the existing `ollama_has_model` checks in `tests/conftest.py`. Unit tests replace `run_claude` and `classify_ad` with fakes.

## Review Focus

1. The judge returns malformed or partial JSON, or skips record ids. Expected: the run applies nothing and the report names the batch. Test in Task 6 and Task 8.
2. `claude` exits non-zero partway through (rate limit, expired login). Expected: nothing applied, report says why, exit status 0 from `run-agent.sh`. Test in Task 8.
3. `ollama create` fails while applying. Expected: the previous learned files are restored and the live model rebuilt from them. Test in Task 8.
4. An evidence file has a truncated last line because the scan was killed. Expected: bad lines are skipped and counted in the report. Test in Task 3.
5. The scan runs twice on the same date (a manual rerun). Expected: the loader keeps one record per item key, the last one written. Test in Task 3.

---

## File structure

news-radar:
- Create `src/newsradar/claude_cli.py`: one isolated `claude -p` call and its result.
- Modify `src/newsradar/llm.py`: `_chat_anthropic` uses `claude_cli`; dedup prompt reads a learned-examples block.
- Modify `src/newsradar/config.py`: `TuningConfig`.
- Create `src/newsradar/tuning/__init__.py` (empty), and:
  - `evidence.py`: record and load day evidence.
  - `store.py`: learned data files, snapshots, rollback.
  - `judge.py`: judge prompts, schemas and calls.
  - `gate.py`: scoring and the accept rules.
  - `tuner.py`: the daily run, rebuild, rollback, report.
  - `prompts/judge_ads.md`, `prompts/judge_dedup.md`, `prompts/second_opinion.md`: judge system prompts.
- Modify `src/newsradar/ad_detector/update_modelfile.py`: split formatting from sampling; deterministic build from base plus learned.
- Modify `src/newsradar/ad_detector/AdDetectorModelfile`: `num_ctx 8192`, learned-rules markers.
- Modify `src/newsradar/cli.py`: evidence hooks, learned dedup block at start-up, `tune` subcommand.
- Modify `.gitignore`, `tests/test_data/labeled_examples.json`, `src/newsradar/ad_detector/AD_DETECTION.md`.
- Tests: `tests/test_claude_cli.py`, `tests/test_tuning_config.py`, `tests/test_evidence.py`, `tests/test_tuning_store.py`, `tests/test_modelfile_build.py`, `tests/test_dedup_learned.py`, `tests/test_judge.py`, `tests/test_gate.py`, `tests/test_tuner.py`.

techradar:
- Modify `run-agent.sh`, `config/config.toml`.

---

### Task 1: Isolated `claude` calls

**Files:**
- Create: `src/newsradar/claude_cli.py`
- Modify: `src/newsradar/llm.py:124-150` (`_chat_anthropic`)
- Test: `tests/test_claude_cli.py`

**Interfaces:**
- Produces:
  - `ISOLATION_FLAGS: list[str]`
  - `@dataclass ClaudeResult: text: str; structured: dict | None; usage: dict; cost_usd: float`
  - `class ClaudeCallError(RuntimeError)`
  - `run_claude(prompt: str, model: str, *, system_prompt: str | None = None, json_schema: dict | None = None, effort: str | None = None, timeout: int = 900) -> ClaudeResult`

- [ ] **Step 1: Write the failing tests**

```python
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
```

`fake_run` is a fixture in the test file that monkeypatches `subprocess.run` in `newsradar.claude_cli`, records `(cmd, cwd)`, and returns `stdout` (default: a success JSON with `result` "ok") and `returncode` (default 0).

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_claude_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newsradar.claude_cli'`

- [ ] **Step 3: Implement `run_claude` in `src/newsradar/claude_cli.py`**

Find the binary the way `_chat_anthropic` does today (`shutil.which("claude")`, falling back to `~/.local/bin/claude`). Run in `tempfile.TemporaryDirectory(prefix="newsradar-claude-")`. Read `is_error`, `result`, `structured_output`, `usage`, `total_cost_usd` from stdout JSON. Raise `ClaudeCallError` on non-zero exit, unparseable stdout, or `is_error: true`, including `result` or stderr in the message. Keep the `[llm:N]` log line and call counters by having `_chat_anthropic` keep its timing code and call `run_claude` in place of `subprocess.run`, returning `result.text`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_claude_cli.py tests/test_dedup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/newsradar/claude_cli.py src/newsradar/llm.py tests/test_claude_cli.py
git commit -m "Run claude CLI calls without personal context"
```

---

### Task 2: Tuning config and ignored data folder

**Files:**
- Modify: `src/newsradar/config.py`, `.gitignore`
- Test: `tests/test_tuning_config.py`

**Interfaces:**
- Produces:
  - `REPO_DATA_DIR: Path` (`Path(__file__).resolve().parents[2] / "data"`)
  - `@dataclass TuningConfig: enabled: bool = False; data_dir: Path = REPO_DATA_DIR; judge_model: str = "claude-opus-5-5"; judge_effort: str = "medium"; max_items_per_run: int = 300; max_new_examples_per_day: int = 8; learned_examples_per_class: int = 12; max_rules_chars: int = 1500; evidence_days: int = 30`
  - `Config.tuning: TuningConfig` (field with default factory)

- [ ] **Step 1: Write the failing tests**

```python
def test_tuning_defaults_off_with_repo_data_dir(tmp_path):
    cfg = load_config(_write_config(tmp_path, ""))
    assert cfg.tuning.enabled is False
    assert cfg.tuning.data_dir == Path(newsradar.__file__).resolve().parents[2] / "data"
    assert cfg.tuning.judge_model == "claude-opus-5-5" and cfg.tuning.judge_effort == "medium"

def test_tuning_table_overrides_and_resolves_relative_data_dir(tmp_path):
    cfg = load_config(_write_config(tmp_path, '[tuning]\nenabled = true\ndata_dir = "d"\nmax_items_per_run = 50\n'))
    assert cfg.tuning.enabled and cfg.tuning.data_dir == tmp_path / "d" and cfg.tuning.max_items_per_run == 50

def test_data_dir_is_git_ignored():
    out = subprocess.run(["git", "check-ignore", "data/x"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0
```

`_write_config` writes a minimal config with one topic file, as in `tests/test_dedup.py`.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_tuning_config.py -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'tuning'`

- [ ] **Step 3: Add `TuningConfig`, read `[tuning]` in `load_config`, add `data/` to `.gitignore`**

Resolve `data_dir` with `_resolve_path(base, ...)` when set.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_tuning_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/newsradar/config.py .gitignore tests/test_tuning_config.py
git commit -m "Add [tuning] config and ignore the data folder"
```

---

### Task 3: Evidence capture during the scan

**Files:**
- Create: `src/newsradar/tuning/__init__.py`, `src/newsradar/tuning/evidence.py`
- Modify: `src/newsradar/cli.py` (`_process_one` around lines 150-163, the `deduplicate` call at line 495, `main`)
- Test: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `TuningConfig` (Task 2)
- Produces:
  - `@dataclass AdRecord: key: str; date: str; topic: str; title: str; link: str; source: str; source_type: str; text_sent: str; text_full: str; verdict: str; reason: str; model: str` where `verdict` is `"ad"`, `"not_ad"` or `"heuristic"`, and `key` is a sha1 of `title + "\n" + text_sent`.
  - `@dataclass DedupRecord: key: str; date: str; topic: str; titles: list[str]; dropped: list[int]` (indices into `titles`; arXiv items excluded, matching `deduplicate`)
  - `class EvidenceRecorder: __init__(self, data_dir: Path, date: str); record_ad(self, *, topic, item: dict, title: str, text: str, source_type: str, verdict: str, reason: str, model: str) -> None; record_dedup(self, *, topic, titles: list[str], dropped: list[int]) -> None`
  - `load_day(data_dir: Path, date: str) -> tuple[list[AdRecord], list[DedupRecord], int]` (third value: bad lines skipped)
  - `prune_evidence(data_dir: Path, today: str, keep_days: int) -> list[str]` (dates removed)
  - Files: `data_dir/evidence/YYYY-MM-DD/<topic>.jsonl`, one JSON object per line with `"kind": "ad"` or `"kind": "dedup"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_records_ad_once_per_run_across_topics(tmp_path):
    rec = EvidenceRecorder(tmp_path, "2026-11-23")
    for topic in ("ai", "robotics"):
        rec.record_ad(topic=topic, item={"link": "https://x/1", "source": "S"}, title="T", text="body " * 500,
                      source_type="email", verdict="ad", reason="r", model="ad-detector")
    ads, dedups, bad = load_day(tmp_path, "2026-11-23")
    assert len(ads) == 1 and ads[0].topic == "ai"
    assert ads[0].text_sent == ("body " * 500)[:1000] and len(ads[0].text_full) <= 4000

def test_records_dedup_batch(tmp_path):
    rec = EvidenceRecorder(tmp_path, "2026-11-23")
    rec.record_dedup(topic="ai", titles=["A", "A again", "B"], dropped=[1])
    _, dedups, _ = load_day(tmp_path, "2026-11-23")
    assert dedups[0].titles == ["A", "A again", "B"] and dedups[0].dropped == [1]

def test_truncated_last_line_is_skipped_and_counted(tmp_path):
    rec = EvidenceRecorder(tmp_path, "2026-11-23")
    rec.record_ad(topic="ai", item={}, title="T", text="x", source_type="feed", verdict="not_ad", reason="", model="m")
    f = tmp_path / "evidence" / "2026-11-23" / "ai.jsonl"
    f.write_text(f.read_text() + '{"kind": "ad", "title": "cut')
    ads, _, bad = load_day(tmp_path, "2026-11-23")
    assert len(ads) == 1 and bad == 1

def test_second_scan_same_day_keeps_last_record_per_key(tmp_path):
    for verdict in ("ad", "not_ad"):
        EvidenceRecorder(tmp_path, "2026-11-23").record_ad(topic="ai", item={}, title="T", text="x",
                                                           source_type="feed", verdict=verdict, reason="", model="m")
    ads, _, _ = load_day(tmp_path, "2026-11-23")
    assert [a.verdict for a in ads] == ["not_ad"]

def test_prune_removes_days_older_than_keep(tmp_path):
    for d in ("2026-10-20", "2026-10-25", "2026-11-23"):
        (tmp_path / "evidence" / d).mkdir(parents=True)
    assert prune_evidence(tmp_path, "2026-11-23", 30) == ["2026-10-20"]

def test_scan_records_ad_decisions_when_tuning_enabled(monkeypatch, tmp_path):
    # _process_one with a fake classify_ad returning (True, "pitch"); recorder set via cli._evidence
    ...
    assert [a.verdict for a in load_day(tmp_path, DATE)[0]] == ["ad"]
```

For the last test, build the `Topic`/`Config` the way `tests/test_removed.py` does, set `cli._evidence = EvidenceRecorder(tmp_path, DATE)`, and call `cli._process_one`.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_evidence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newsradar.tuning'`

- [ ] **Step 3: Implement `evidence.py`, and wire it into `cli.py`**

The recorder holds a lock and a seen-set of keys, and appends one line per new key. In `cli.py`, add module-level `_evidence: EvidenceRecorder | None = None`. `main` sets it when `config.tuning.enabled` and the run is a normal scan (not `--podcast-only`, `--transcript-only` or `--refresh-token`). In `_process_one`, record a `"heuristic"` verdict where `_is_advertisement` removes the item, and the gate's verdict right after `_classify_ad_once`. `source_type` is `"email"` for emails, `"linked_article"` when `item.get("_from_email_link")`, else `"feed"`. Around the `deduplicate` call, record the titles of the non-arXiv items sent and the indices of those missing from the result.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_evidence.py tests/test_removed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/newsradar/tuning tests/test_evidence.py src/newsradar/cli.py
git commit -m "Record ad and dedup decisions as day evidence when tuning is on"
```

---

### Task 4: Learned data store and Modelfile build

**Files:**
- Create: `src/newsradar/tuning/store.py`
- Modify: `src/newsradar/ad_detector/update_modelfile.py`, `src/newsradar/ad_detector/AdDetectorModelfile`
- Test: `tests/test_tuning_store.py`, `tests/test_modelfile_build.py`

**Interfaces:**
- Consumes: `TuningConfig` (Task 2)
- Produces, in `update_modelfile.py`:
  - `format_training_block(examples: list[dict]) -> str` (today's per-example format, interleaving ads and non-ads)
  - `replace_training_block(content: str, block: str) -> str`
  - `LEARNED_RULES_START = "LEARNED RULES (added by the tuner):\n"`, `LEARNED_RULES_END = "END LEARNED RULES\n"`
  - `build_live_modelfile(base_text: str, base_examples: list[dict], learned_examples: list[dict], learned_rules: str, per_class: int) -> str`
  - `generate_training_block` and `update_modelfile` keep working as before.
- Produces, in `store.py`:
  - `class TuningStore: __init__(self, data_dir: Path)`
  - `ad_examples() -> list[dict]`, `save_ad_examples(examples: list[dict]) -> None` (`tuning/learned_ad_examples.json`, each with an `added` date)
  - `ad_rules() -> str`, `save_ad_rules(text: str) -> None` (`tuning/learned_ad_rules.txt`)
  - `dedup_cases() -> list[dict]`, `save_dedup_cases(cases: list[dict]) -> None` (`tuning/learned_dedup_cases.json`)
  - `dedup_examples_block() -> str` (formatted from the newest cases, empty string when none)
  - `snapshot(date: str, built_modelfile: str) -> Path` (`tuning/snapshots/YYYY-MM-DD/`; a second snapshot the same day gets a `-2` suffix)
  - `restore(snapshot_dir: Path) -> str` (restores the learned files, returns that snapshot's Modelfile text)
  - `latest_snapshot(date: str) -> Path | None`
  - `report_path(date: str) -> Path` (`tuning/reports/YYYY-MM-DD.md`)
  - `domain_only(url: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
def test_live_build_includes_all_base_and_newest_learned_per_class():
    learned = [ex(f"n{i}", is_ad=False, added=f"2026-11-{i:02d}") for i in range(1, 20)]
    text = build_live_modelfile(BASE, BASE_EXAMPLES, learned, "", per_class=12)
    assert all(repr(e["title"]) in text for e in BASE_EXAMPLES)
    assert "'n19'" in text and "'n8'" in text and "'n7'" not in text

def test_live_build_is_deterministic():
    assert build_live_modelfile(BASE, BASE_EXAMPLES, LEARNED, "r", 12) == build_live_modelfile(BASE, BASE_EXAMPLES, LEARNED, "r", 12)

def test_rules_go_between_markers_and_hand_patterns_are_untouched():
    text = build_live_modelfile(BASE, BASE_EXAMPLES, [], "Judge the whole item.", 12)
    start, end = text.index(LEARNED_RULES_START), text.index(LEARNED_RULES_END)
    assert "Judge the whole item." in text[start:end]
    assert text.split(LEARNED_RULES_START)[0] == BASE.split(LEARNED_RULES_START)[0]

def test_packaged_modelfile_has_ctx_8192_and_markers():
    text = (PKG / "AdDetectorModelfile").read_text()
    assert "PARAMETER num_ctx 8192" in text and LEARNED_RULES_START in text and LEARNED_RULES_END in text

def test_update_modelfile_still_regenerates_examples(tmp_path): ...  # existing behavior, one example in, its title out

def test_store_round_trip_snapshot_and_restore(tmp_path):
    s = TuningStore(tmp_path)
    s.save_ad_examples([ex("a", True)]); s.save_ad_rules("r1")
    snap = s.snapshot("2026-11-23", "MODELFILE-1")
    s.save_ad_examples([]); s.save_ad_rules("r2")
    assert s.restore(snap) == "MODELFILE-1"
    assert [e["title"] for e in s.ad_examples()] == ["a"] and s.ad_rules() == "r1"

def test_domain_only():
    assert domain_only("https://thehustle.co/join?ref=6693394292") == "https://thehustle.co/"
    assert domain_only("") == ""
```

`BASE` is the packaged Modelfile text and `BASE_EXAMPLES` the packaged labeled examples. `ex()` builds an example dict.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_modelfile_build.py tests/test_tuning_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_live_modelfile'`

- [ ] **Step 3: Implement, and edit the packaged Modelfile**

In `AdDetectorModelfile`, set `PARAMETER num_ctx 8192` and add an empty `LEARNED_RULES_START` + `LEARNED_RULES_END` pair right before `NOT ADVERTISEMENTS (output is_ad: false):`. `build_live_modelfile` fills the rules block and replaces the training block with base examples plus, per class, the `per_class` learned examples with the latest `added` dates, sorted by `added` then `id`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_modelfile_build.py tests/test_tuning_store.py tests/test_ad_detector.py -v`
Expected: PASS (`test_ad_detector.py` Ollama tests skip if the model is missing)

- [ ] **Step 5: Commit**

```bash
git add src/newsradar/tuning/store.py src/newsradar/ad_detector tests/test_modelfile_build.py tests/test_tuning_store.py
git commit -m "Build the ad-detector Modelfile from base plus learned data"
```

---

### Task 5: Dedup prompt reads learned examples

**Files:**
- Modify: `src/newsradar/llm.py` (`_dedup_batch` prompt, near line 438), `src/newsradar/cli.py` (`main`)
- Test: `tests/test_dedup_learned.py`

**Interfaces:**
- Consumes: `TuningStore.dedup_examples_block()` (Task 4)
- Produces: `llm.set_learned_dedup_examples(block: str) -> None`, `llm.learned_dedup_examples() -> str`

- [ ] **Step 1: Write the failing tests**

```python
def test_learned_block_appears_in_dedup_prompt(monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "_chat_meta", lambda p, m, **kw: prompts.append(p) or llm.ChatResult('{"groups": []}'))
    llm.set_learned_dedup_examples("LEARNED EXAMPLES:\nX and Y are different stories.")
    try:
        llm._dedup_batch([(0, {"title": "A"}), (1, {"title": "B"})], "m")
    finally:
        llm.set_learned_dedup_examples("")
    assert "X and Y are different stories." in prompts[0]
    assert prompts[0].index("LEARNED EXAMPLES") < prompts[0].index("Articles:")

def test_no_block_leaves_prompt_unchanged(monkeypatch): ...  # prompt has no "LEARNED EXAMPLES"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_dedup_learned.py -v`
Expected: FAIL with `AttributeError: module 'newsradar.llm' has no attribute 'set_learned_dedup_examples'`

- [ ] **Step 3: Implement**

A module-level string, set by `set_learned_dedup_examples`, inserted before `"Articles:"` when non-empty. In `cli.main`, when `config.tuning.enabled`, call `llm.set_learned_dedup_examples(TuningStore(config.tuning.data_dir).dedup_examples_block())` before topics run.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_dedup_learned.py tests/test_dedup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/newsradar/llm.py src/newsradar/cli.py tests/test_dedup_learned.py
git commit -m "Let the dedup prompt include learned examples"
```

---

### Task 6: The judge

**Files:**
- Create: `src/newsradar/tuning/judge.py`, `src/newsradar/tuning/prompts/judge_ads.md`, `src/newsradar/tuning/prompts/judge_dedup.md`, `src/newsradar/tuning/prompts/second_opinion.md`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `run_claude`, `ClaudeCallError` (Task 1); `AdRecord`, `DedupRecord` (Task 3); `domain_only` (Task 4)
- Produces:
  - `class JudgeError(RuntimeError)`
  - `@dataclass AdVerdict: key: str; correct: bool; is_ad: bool; ad_type: str; reason: str; visible_in_excerpt: bool; example: dict | None`
  - `@dataclass ProposedRule: text: str; leaning: str` (`"keep"` when it makes the gate keep more, `"remove"` when it makes it remove more)
  - `@dataclass AdJudgement: verdicts: list[AdVerdict]; proposed_rules: list[ProposedRule]; cost_usd: float`
  - `@dataclass DedupCase: titles: list[str]; expected_groups: list[list[int]]; kind: str; note: str` (`kind` is `"wrong_merge"` or `"missed_duplicate"`)
  - `judge_ads(records: list[AdRecord], tuning: TuningConfig, batch_size: int = 20) -> AdJudgement`
  - `judge_dedup(records: list[DedupRecord], tuning: TuningConfig) -> tuple[list[DedupCase], float]`
  - `second_opinion_ad(record: AdRecord, tuning: TuningConfig) -> bool` (returns the second judge's `is_ad`)
  - `second_opinion_dedup(case: DedupCase, tuning: TuningConfig) -> bool` (True when it agrees the expected groups are right)

The ads prompt tells the judge what the gate is for (remove pitches, keep news, per the spec's "judge the whole item" rule), shows `text_sent` and `text_full` separately, and asks per record: was the verdict right, the correct label, `ad_type` from `referral_program | lead_gen_offer | newsletter_cta | subscription_pitch | sponsor_content | product_feature_disguised | not_ad`, a reason, whether `text_sent` alone shows the right answer, and for mistakes an example `{"id", "title", "summary" (≤200 chars, starting from text_sent), "source", "link", "is_ad", "ad_type", "notes"}`. Rules are proposed only when examples can't express the pattern. The second-opinion prompt sees only the record, not the first verdict.

- [ ] **Step 1: Write the failing tests**

```python
def test_judge_ads_calls_isolated_opus_medium_with_schema(fake_claude):
    judge_ads([rec("k1")], TUNING)
    call = fake_claude.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["effort"] == "medium"
    assert call["json_schema"]["type"] == "object" and call["system_prompt"]

def test_judge_ads_batches_and_maps_verdicts_by_key(fake_claude):
    fake_claude.reply = lambda prompt: {"verdicts": [verdict(k) for k in keys_in(prompt)], "proposed_rules": []}
    out = judge_ads([rec(f"k{i}") for i in range(45)], TUNING, batch_size=20)
    assert len(fake_claude.calls) == 3 and [v.key for v in out.verdicts] == [f"k{i}" for i in range(45)]

def test_missing_key_raises_judge_error(fake_claude):
    fake_claude.reply = lambda prompt: {"verdicts": [], "proposed_rules": []}
    with pytest.raises(JudgeError, match="k1"):
        judge_ads([rec("k1")], TUNING)

def test_example_summary_capped_and_link_cut_to_domain(fake_claude):
    fake_claude.reply = lambda p: {"verdicts": [verdict("k1", correct=False, example={**EX, "summary": "x" * 300,
                                   "link": "https://a.com/p?ref=1"})], "proposed_rules": []}
    ex = judge_ads([rec("k1")], TUNING).verdicts[0].example
    assert len(ex["summary"]) == 200 and ex["link"] == "https://a.com/"

def test_claude_failure_becomes_judge_error(fake_claude):
    fake_claude.error = ClaudeCallError("rate limited")
    with pytest.raises(JudgeError):
        judge_ads([rec("k1")], TUNING)

def test_second_opinion_prompt_omits_first_verdict(fake_claude):
    second_opinion_ad(rec("k1", verdict="ad", reason="FIRST-REASON"), TUNING)
    assert "FIRST-REASON" not in fake_claude.calls[0]["prompt"]

def test_judge_dedup_returns_cases_with_valid_indices(fake_claude): ...  # out-of-range indices raise JudgeError
```

`fake_claude` monkeypatches `newsradar.tuning.judge.run_claude`, records keyword arguments plus the prompt, and returns `ClaudeResult` with `structured=fake_claude.reply(prompt)`.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newsradar.tuning.judge'`

- [ ] **Step 3: Implement `judge.py` and the three prompt files**

Load prompts with `importlib.resources`. Records go into the prompt as JSON with their `key`. Validate every reply against the batch: each key exactly once, indices in range. Any `ClaudeCallError` or validation failure raises `JudgeError` naming the batch keys.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_judge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/newsradar/tuning/judge.py src/newsradar/tuning/prompts tests/test_judge.py
git commit -m "Add the Opus judge for ad and dedup decisions"
```

---

### Task 7: Scoring and the accept rules

**Files:**
- Create: `src/newsradar/tuning/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `classify_ad` (`llm.py`), `llm.set_learned_dedup_examples`, `llm._dedup_voted` (Task 5); `DedupCase` (Task 6)
- Produces:
  - `@dataclass Case: id: str; title: str; text: str; expected: bool; today_mistake: bool; countable: bool` (`countable` false when the judge said the excerpt doesn't show the answer)
  - `@dataclass GateResult: accepted: bool; reason: str; current_errors: int; candidate_errors: int; fixed: list[str]; broken: list[str]`
  - `predict_ads(model: str, cases: list[Case]) -> list[bool]` (calls `classify_ad(c.title, c.text[:1000], model)`)
  - `stabilize(current: list[bool], candidate: list[bool], rerun: Callable[[int], tuple[bool, bool]]) -> list[bool]` (returns the candidate list with unconfirmed differences reset to the current value)
  - `decide(cases: list[Case], current_ok: list[bool], candidate_ok: list[bool]) -> GateResult`
  - `dedup_case_ok(case: DedupCase, dropped: set[int]) -> bool`
  - `predict_dedup(cases: list[DedupCase], model: str, examples_block: str) -> list[set[int]]` (sets the block, runs `_dedup_voted` per case, restores the previous block in `finally`)

`decide` accepts only when all three spec rules hold: `broken` is empty, at least one `fixed` id belongs to a case with `today_mistake and countable`, and `candidate_errors < current_errors`.

- [ ] **Step 1: Write the failing tests**

```python
def test_accepts_when_a_today_mistake_is_fixed_and_nothing_breaks():
    r = decide(CASES, current_ok=[False, True, True], candidate_ok=[True, True, True])
    assert r.accepted and r.fixed == ["today-1"] and r.broken == []

def test_rejects_any_regression_even_with_net_gain():
    r = decide(CASES4, current_ok=[False, False, True, True], candidate_ok=[True, True, False, True])
    assert not r.accepted and r.broken == ["c3"]

def test_rejects_when_only_uncountable_mistakes_are_fixed():
    r = decide([case("t", today_mistake=True, countable=False)], [False], [True])
    assert not r.accepted and "no fix" in r.reason

def test_rejects_when_nothing_changes():
    assert not decide(CASES, [True, True, True], [True, True, True]).accepted

def test_stabilize_keeps_only_differences_confirmed_on_rerun():
    current, candidate = [True, True], [False, False]
    reruns = {0: (True, False), 1: (True, True)}
    assert stabilize(current, candidate, lambda i: reruns[i]) == [False, True]

def test_dedup_case_ok():
    c = DedupCase(titles=["A", "A2", "B"], expected_groups=[[0, 1]], kind="missed_duplicate", note="")
    assert dedup_case_ok(c, {1}) and not dedup_case_ok(c, set()) and not dedup_case_ok(c, {1, 2})

def test_predict_dedup_restores_previous_block(monkeypatch): ...
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newsradar.tuning.gate'`

- [ ] **Step 3: Implement `gate.py`**

`dedup_case_ok` is true when, for each expected group, all but one member is dropped, and nothing outside the expected groups is dropped.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/newsradar/tuning/gate.py tests/test_gate.py
git commit -m "Add the regression gate for ad and dedup changes"
```

---

### Task 8: The daily run, rebuild, rollback and report

**Files:**
- Create: `src/newsradar/tuning/tuner.py`
- Modify: `src/newsradar/cli.py` (`_build_parser`, `main`)
- Test: `tests/test_tuner.py`

**Interfaces:**
- Consumes: everything from Tasks 2 to 7; `install_ad_detector_model(modelfile_path, model_name)`.
- Produces:
  - `@dataclass TuneOutcome: ads_applied: bool; dedup_applied: bool; summary: str; report_path: Path`
  - `run_tune(config: Config, date: str, log: Callable[[str], None] = print) -> TuneOutcome`
  - `rebuild_live_model(config: Config, model_name: str = "ad-detector") -> str` (returns the built Modelfile text)
  - `rollback(config: Config, date: str) -> None`
  - CLI: `newsradar tune --config C [--date YYYY-MM-DD] [--rebuild] [--rollback YYYY-MM-DD]`. `--date` defaults to today in ET.

`run_tune` follows spec section 4 steps 1 to 10:
1. `load_day`; with no records, write a one-line report and return.
2. Cap ad records at `max_items_per_run`, ad-gate removals first (`verdict == "ad"`), then items that passed. Skip `"heuristic"` records.
3. `judge_ads`, then `judge_dedup`. On `JudgeError`, write the report with the error, apply nothing, return.
4. Remove-more proposals (examples with `is_ad: true`, rules with `leaning == "remove"`, `missed_duplicate` cases) go through `second_opinion_*` and are kept only on agreement.
5. New examples are capped at `max_new_examples_per_day`, mistakes first, with one correctly handled opposite-label example from the same day. Each gets `added = date`. Rules are appended to the learned rules if the total stays within `max_rules_chars`, otherwise dropped and reported.
6. Build the candidate with `build_live_modelfile` and install it as `ad-detector-candidate`.
7. Cases are every base example, every learned example, and every judged record of the day (with `today_mistake` and `countable` from the verdicts). `predict_ads` on both models, `stabilize`, `decide`. Same for dedup with `predict_dedup` against the scan's `dedup_model`.
8. For each accepted part: `snapshot` first, then save the learned files. Rebuild `ad-detector` with `rebuild_live_model`. If the rebuild fails, `restore` the snapshot, rebuild from it, and report.
9. Write the report to `report_path(date)`: mistakes (title, source, gate verdict, judge reason, second-opinion result), proposals, both scores, accept or reject and why, the summed `cost_usd`, and the `--rebuild` and `--rollback` commands. Log one summary line.
10. Delete `ad-detector-candidate` with `ollama rm` and run `prune_evidence`.

- [ ] **Step 1: Write the failing tests**

```python
def test_no_evidence_writes_short_report_and_applies_nothing(cfg):
    out = run_tune(cfg, "2026-11-23")
    assert not out.ads_applied and "no evidence" in out.report_path.read_text()

def test_judge_failure_applies_nothing_and_reports(cfg, evidence, fakes):
    fakes.judge_error = JudgeError("rate limited")
    out = run_tune(cfg, "2026-11-23")
    assert not out.ads_applied and "rate limited" in out.report_path.read_text()
    assert TuningStore(cfg.tuning.data_dir).ad_examples() == []

def test_accepted_change_snapshots_then_saves_and_rebuilds(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake()
    out = run_tune(cfg, "2026-11-23")
    store = TuningStore(cfg.tuning.data_dir)
    assert out.ads_applied and store.latest_snapshot("2026-11-23") is not None
    assert len(store.ad_examples()) >= 1 and fakes.installed[-1] == "ad-detector"

def test_rejected_change_leaves_learned_data_alone(cfg, evidence, fakes):
    fakes.candidate_breaks_a_base_example()
    out = run_tune(cfg, "2026-11-23")
    assert not out.ads_applied and TuningStore(cfg.tuning.data_dir).ad_examples() == []

def test_remove_more_example_needs_second_opinion(cfg, evidence, fakes):
    fakes.judge_proposes_ad_example(); fakes.second_opinion_ad = False
    run_tune(cfg, "2026-11-23")
    assert all(not e["is_ad"] for e in TuningStore(cfg.tuning.data_dir).ad_examples())

def test_rebuild_failure_restores_previous_learned_data(cfg, evidence, fakes):
    fakes.candidate_fixes_today_mistake(); fakes.fail_install_of = "ad-detector"
    out = run_tune(cfg, "2026-11-23")
    assert not out.ads_applied and "restored" in out.report_path.read_text()

def test_candidate_model_removed_and_old_evidence_pruned(cfg, evidence, fakes): ...

def test_cli_tune_subcommand_parses(): ...  # --date, --rebuild, --rollback
```

`fakes` monkeypatches `judge_ads`, `judge_dedup`, `second_opinion_*`, `predict_ads`, `predict_dedup`, `install_ad_detector_model` and the `ollama rm` call in `newsradar.tuning.tuner`.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_tuner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newsradar.tuning.tuner'`

- [ ] **Step 3: Implement `tuner.py` and the `tune` subcommand**

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (Ollama-dependent tests may skip)

- [ ] **Step 5: Commit**

```bash
git add src/newsradar/tuning/tuner.py src/newsradar/cli.py tests/test_tuner.py
git commit -m "Add newsradar tune: judge, gate, apply and report"
```

---

### Task 9: One-time setup and docs

**Files:**
- Modify: `tests/test_data/labeled_examples.json`, `src/newsradar/ad_detector/AD_DETECTION.md`
- Test: `tests/test_ad_detector.py` (add one test)

- [ ] **Step 1: Write the failing test**

```python
def test_labeled_example_links_are_domain_only():
    for e in _load_examples():
        assert e["link"] in ("", domain_only(e["link"])), e["id"]
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/test_ad_detector.py::test_labeled_example_links_are_domain_only -v`
Expected: FAIL naming `hustle-referral-...` or another example with a full link

- [ ] **Step 3: Cut every link in `labeled_examples.json` to its domain**

- [ ] **Step 4: Update `AD_DETECTION.md`**

Replace the `ollama rm ad-detector && ollama create ...` steps with `uv run newsradar tune --config <config> --rebuild`, add a section on the daily tuner (where data lives, the report, `--rollback`), and note that a bare `ollama create` drops learned data.

- [ ] **Step 5: Seed the learned data (not committed)**

Write `data/tuning/learned_ad_examples.json` with the four non-ad examples below, each with `"added": "2026-11-23"`, and `data/tuning/learned_ad_rules.txt` with the rule text below. Then, from `/Users/keithfry/projects/techradar`, run `uv run run.py tune --config config/config.toml --rebuild`.

Examples (`ad_type` `not_ad`):
- `⚡️Trump and China eye AI guardrails | The AI Report`, source `theaireport.ai`, link `https://www.theaireport.ai/`: "Your competitors are reading this every morning. 400,000+ business leaders start their day with The AI Report. ADVERTISE | Presented by Alumni Ventures. 2. Trump and China eye AI guardrails". Notes: web copy of a news newsletter issue; subscribe banner and sponsor slots wrap real news.
- `⚡️Trump and China eye AI guardrails`, source `The AI Report`, no link: "[Read Online] [ADVERTISE] | [SEE WHERE YOU STAND] Hey there, Here's today's news, what's actually working with AI, where you stand against your peers. Presented by Alumni Ventures. Trump and China". Notes: news newsletter issue; the advertise link and sponsor block don't make it an ad.
- `SignalPulse:`, source `The Tech Buzz`, no link: "WHAT'S INSIDE Top Signals: Wednesday PCE, Anthropic's $83 IPO price, Nvidia succession. Mega Rounds: Isomorphic Labs $5B. IN PARTNERSHIP WITH ATLASSIAN State of Product reveals what AI still". Notes: funding and markets digest; one sponsor block doesn't make it an ad.
- `AI coding agents ace isolated tests but complete fewer than one in ten tasks in a simulated workplace`, source `The Decoder`, link `https://the-decoder.com/`: "A new experiment puts AI coding agents to work inside a simulated software company. THE DECODER+ Continue reading with THE DECODER+. First month $4.99, then $9.99 a month. Subscribe monthly". Notes: news article cut off by a paywall offer.

Rule text:

```
Newsletters and articles often wrap real news in promotional parts: a subscribe banner, an "Advertise" link, a "Presented by" or "In partnership with" sponsor block, a referral program, a paid-tier or free-trial offer, or a paywall ("Continue reading with ... Subscribe"). These parts do NOT make the item an ad.
If the item reports news, analysis or research for the reader, output is_ad: false even when such parts appear in the text, including at the very top.
Output is_ad: true only when the item as a whole exists to sell or promote, so that removing the pitch would leave nothing.
```

- [ ] **Step 6: Remove the earlier test models**

Run: `ollama rm ad-detector-test ad-detector-test2`
Expected: both deleted (or "not found" if already gone)

- [ ] **Step 7: Run the suite and commit**

Run: `uv run pytest -q`
Expected: PASS

```bash
git add tests/test_data/labeled_examples.json tests/test_ad_detector.py src/newsradar/ad_detector/AD_DETECTION.md
git commit -m "Cut labeled example links to domains and document the tuner"
```

---

### Task 10: Run the tuner after each scan (techradar)

**Files:**
- Modify: `/Users/keithfry/projects/techradar/run-agent.sh`, `/Users/keithfry/projects/techradar/config/config.toml`

- [ ] **Step 1: Turn on tuning in `config/config.toml`**

Add:

```toml
[tuning]
# After each scan, Claude Opus 5.5 judges the ad and dedup decisions and
# applies fixes that pass a regression check. Data lives in news-radar's
# ignored data/ folder. See news-radar docs/superpowers/specs/2026-11-23-ad-dedup-tuning-design.md.
enabled = true
```

- [ ] **Step 2: Change `run-agent.sh` to run the scan, then the tuner**

Replace the final `exec` line with: run `run.py` with the same arguments into the log and keep its exit status; then run `/opt/homebrew/bin/uv run run.py tune --config config/config.toml --date "$LOG_DATE"` appending to the same log, with `|| echo "tuner failed (exit $?)"` so its failure is logged and ignored; then `exit` with the scan's status. `set -e` must not stop the script when the scan fails, so capture the scan's status with `|| status=$?`.

- [ ] **Step 3: Check the script**

Run: `bash -n run-agent.sh && grep -n "tune" run-agent.sh`
Expected: no syntax errors, the tune line shown

- [ ] **Step 4: Run the techradar tests**

Run: `uv run --with pytest python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add run-agent.sh config/config.toml
git commit -m "Run the ad and dedup tuner after each daily scan"
```

---

### Task 11: End-to-end check on 2026-11-23's items

No commit. This uses the subscription for a handful of Opus calls.

- [ ] **Step 1: Build a test evidence day**

Write a one-off script in the scratchpad (not the repo) that turns the 10 replayed 2026-11-23 items in `scratchpad/adreview/today_full.json` into `AdRecord` lines (verdict `ad` for all ten, as the scan decided) under a temporary `data_dir`. The file is `/private/tmp/claude-501/-Users-keithfry-projects-techradar/42ed2ca5-05e2-49a0-b8c0-416722733ed9/scratchpad/adreview/today_full.json`, plus five correctly passed items from today's AI page JSON as `not_ad`. If the scratchpad file is gone, rebuild it by re-fetching the seven links listed in the 2026-11-23 AI page's removed-items section.

- [ ] **Step 2: Run the tuner against it**

Run, from techradar: `uv run run.py tune --config <temp config> --date 2026-11-23`, where the temp config is a copy of `config/config.toml` in the scratchpad with `[tuning] data_dir` set to the temp dir and `[paths] output_root` set to a temp folder
Expected: the report lists the 8 news items as wrong removals and the AI Report advertise page and Alumni Ventures page as correct. With the seed already applied in Task 9, most of the 8 are already fixed, so expect either a small accepted change or "no fix" rejections.

- [ ] **Step 3: Confirm the live model is untouched by the temp run, and that the scan still works**

Run: `ollama list | grep ad-detector`, then from techradar `uv run run.py --config <temp config> --dry-run --no-email --no-podcast --topic research`. Use the temp config from Step 2: a `--dry-run` scan still writes pages under `output_root`, and with the real config those would land in `techradar/techradar/` and be deployed by the next 08:00 sync.
Expected: `ad-detector` present, no `ad-detector-candidate`; the dry run finishes, writes pages only to the temp folder, and writes evidence under the temp `data_dir`.
