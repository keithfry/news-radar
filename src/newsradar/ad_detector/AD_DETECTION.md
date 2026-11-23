# Ad Detection — How It Works and How to Improve It

## Pipeline Position

```
item received
  → _is_advertisement()     heuristic (ad phrase in the title) — zero LLM cost
  → classify_ad()           LLM gate (ad-detector model) — catches contextual ads
  → classify_ai()           relevance check (only for non-keyword items)
  → summarize()             only runs if item passed both gates
```

Set `AD_GATE_ENABLED=0` in `.env` to disable the LLM gate (e.g. for a speed test).
Set `AD_DETECTOR_MODEL=qwen3.5:9b` in `.env` to use the base model directly (no system prompt).

The packaged `ad-detector` model carries the rules and examples in its Modelfile
`SYSTEM` prompt, so `classify_ad()` sends it only `Title: ...\nSummary: ...`. A
model without a `SYSTEM` prompt (such as base `qwen3.5:9b`) gets the full
instructions inline. Sending both used to double prompt time and overflow the
model's 4096-token context on long items.

arXiv items skip both checks. Each verdict is remembered for the run, so emails
and linked articles are checked once, not once per topic.

## Ad Categories the Gate Catches

| Category | Example signals |
|---|---|
| Referral program | "share your link", "earn rewards", "prizes available" |
| Lead-gen offer | "free guide", "download now", "150+ prompts", "free stack" |
| Newsletter CTA | "still interested?", "monitoring subscriber activity", "high email costs" |
| Advertising pitch | "advertise in", "reach X million", "20.1x ROI" |
| Subscription product | "one agent per city", "apply now", "$X/year", "weekly leads" |
| Product feature pitch | imperative verbs targeting reader ("start shopping", "try it") |

## Labeled Dataset

`test_data/labeled_examples.json` — ground truth for all detector evaluations.

Schema per example:

```json
{
  "id": "source-slug-YYYY-MM-DD",
  "title": "...",
  "summary": "...",
  "source": "sender or domain",
  "link": "...",
  "is_ad": true,
  "ad_type": "referral_program | lead_gen_offer | newsletter_cta | subscription_pitch | sponsor_content | product_feature_disguised | not_ad",
  "notes": "why this is or isn't an ad"
}
```

## Daily Tuner

`newsradar tune --config <config> [--date YYYY-MM-DD] [--rebuild] [--rollback YYYY-MM-DD]`
reviews one day of ad-gate and dedup decisions, asks a judge model (Claude) to find
mistakes, and applies a change only if it passes a regression gate. It runs only when
`[tuning] enabled = true` in the config. A tuner failure never changes the exit status
of `run-agent.sh`.

### Where the data lives

Everything is under `[tuning] data_dir`, which defaults to `data/` in the news-radar
repo. `.gitignore` excludes it, so learned data is never committed.

| Path | Contents |
|---|---|
| `data/evidence/DATE/` | The day's ad-gate and dedup decisions, written by the scan |
| `data/tuning/learned_ad_examples.json` | Learned ad and not-ad examples (a JSON list) |
| `data/tuning/learned_ad_rules.txt` | Learned rule text added to the system prompt |
| `data/tuning/learned_dedup_cases.json` | Learned dedup cases |
| `data/tuning/snapshots/` | A copy of the learned files and built Modelfile from before each applied change |
| `data/tuning/reports/DATE.md` | The report for that day |
| `data/tuning/build/` | Scratch space for candidate builds, deleted at the end of each run |

Learned example links are cut to `scheme://host/` and summaries to 200 characters,
so nothing sensitive from a URL is stored. Evidence, snapshots and reports older than
`evidence_days` (30) are deleted. The newest snapshot is always kept.

### The report

`data/tuning/reports/DATE.md` lists the evidence counted, each gate mistake with the
judge's verdict and second opinion, the proposed example, rule and dedup changes, the
regression gate results, what was applied, the judge cost, and clean-up. It ends with
the exact `--rebuild` and `--rollback` commands for that config.

### Rebuild and rollback

- `--rebuild` rebuilds the live `ad-detector` model from `AdDetectorModelfile` plus the
  learned files. Run it after any manual edit to the learned files.
- `--rollback DATE` restores the latest snapshot taken on DATE (learned files and live
  model) and rebuilds the live model from it.

The tuner builds a `*-candidate` model to test a change and removes it at the end
of the run. A bare `ollama create` drops learned data, as noted above.

## Ongoing Workflow

### When new ads slip through the digest

1. **Extract candidates from the offending HTML file:**
   ```bash
   uv run extract_examples.py ../../techradar/AI/ai-radar-YYYY-MM-DD.html
   ```
   Output is newline-delimited JSON to stdout. Each item has `"is_ad": null`.

2. **Find the ad items**, copy them into `test_data/labeled_examples.json`, set `is_ad: true` and `ad_type`.

3. **Add a matching non-ad example** from the same file to keep the dataset balanced.

4. **Run the test harness to measure current performance:**
   ```bash
   uv run test_ad_detector.py --verbose
   ```
   Check recall — any false negatives listed are the ads still slipping through.

5. **Rebuild the custom model** with the new examples baked in:
   ```bash
   uv run update_modelfile.py
   uv run newsradar tune --config <config> --rebuild
   ```
   Do not run a bare `ollama create ad-detector -f AdDetectorModelfile`. It builds
   from the base Modelfile only and drops everything the daily tuner learned.
   `--rebuild` builds from the base Modelfile plus the learned data.

6. **Re-run the harness** to confirm improvement:
   ```bash
   uv run test_ad_detector.py --detector custom
   ```

7. **Dry-run the pipeline** to confirm no false positives on real content:
   ```bash
   uv run main.py --dry-run --no-email --date YYYY-MM-DD
   grep "ad gate\|advertisement" ../../logs/ai-techradar-agent-YYYY-MM-DD.log
   ```

### Test harness usage

```bash
uv run test_ad_detector.py                    # all three detectors
uv run test_ad_detector.py --detector heuristic
uv run test_ad_detector.py --detector llm      # llama3.2:3b, no system prompt
uv run test_ad_detector.py --detector custom   # ad-detector (qwen3.5:9b + system prompt)
uv run test_ad_detector.py --verbose           # per-example pass/fail table
```

## Current Benchmark (May 2026)

24 labeled examples — 15 ads, 9 not-ads.

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| heuristic | 0.00 | 0.00 | 0.00 |
| llm (llama3.2:3b) | 0.67 | 0.13 | 0.22 |
| custom (ad-detector) | 1.00 | 1.00 | 1.00 |

The 3B model was too small — it identified ad patterns in its reasoning but output `false` anyway.
`qwen3.5:9b` as the Modelfile base resolved this.

## Model Files

| File | Purpose |
|---|---|
| `AdDetectorModelfile` | Ollama model definition — `FROM qwen3.5:9b` + system prompt + training examples |
| `update_modelfile.py` | Regenerates the training examples section from `labeled_examples.json` |

`AdDetectorModelfile` is **generated** — don't edit the training examples section by hand.
Edit `labeled_examples.json`, then run `update_modelfile.py`.

The pattern/keyword definitions at the top of the Modelfile ARE hand-maintained — edit those
when you need to add a new ad category.
