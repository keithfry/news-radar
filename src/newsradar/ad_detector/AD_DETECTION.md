# Ad Detection — How It Works and How to Improve It

## Pipeline Position

```
item received
  → _is_advertisement()     heuristic (price patterns, phrase list) — zero LLM cost
  → classify_ad()           LLM gate (ad-detector model) — catches contextual ads
  → classify_ai()           relevance check (only for non-keyword items)
  → summarize()             only runs if item passed both gates
```

Set `AD_GATE_ENABLED=0` in `.env` to disable the LLM gate (e.g. for a speed test).
Set `AD_DETECTOR_MODEL=qwen3.5:9b` in `.env` to use the base model directly (no system prompt).

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
   ollama rm ad-detector && ollama create ad-detector -f AdDetectorModelfile
   ```

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
