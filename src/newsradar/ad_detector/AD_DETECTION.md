# Ad Detection — How It Works and How to Improve It

## Pipeline Position

```
item received
  → classify_ad()           LLM gate (ad-detector model)
  → classify_ai()           relevance check (only for non-keyword items)
  → summarize()             only runs if item passed both checks
```

Set `AD_GATE_ENABLED=0` in `.env` to disable the LLM gate (e.g. for a speed test).
Set `AD_DETECTOR_MODEL=qwen3.5:9b` in `.env` to use the base model directly (no system prompt).

The packaged `ad-detector` model carries the rules and examples in its Modelfile
`SYSTEM` prompt, so `classify_ad()` sends it only `Title: ...\nSummary: ...`. A
model without a `SYSTEM` prompt (such as base `qwen3.5:9b`) gets the full
instructions inline. Sending both used to double prompt time and overflow the
model's 4096-token context on long items.

Every item except arXiv papers goes to the ad model. There is no title-phrase
pre-check and no webinar check on the summary: across 54 daily logs they removed
5 items in two months, 2 of them news, and nothing judged their decisions. The
daily tuner now judges the ad gate's removals, up to `[tuning] max_items_per_run`
items a day. Removals past that cap are not judged.

arXiv items skip the ad gate. Each verdict is remembered for the run, so emails
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
| Vendor webinar or event | a vendor's own webinar, event or recorded talk promoting its product ("watch", "register", speakers and a topic); coverage by a third party is news |

## Labeled Dataset

`test_data/labeled_examples.json` — ground truth for all detector evaluations.

Schema per example:

```json
{
  "id": "source-slug-YYYY-MM-DD",
  "title": "...",
  "summary": "...",
  "source": "sender or domain",
  "link": "https://host/",
  "is_ad": true,
  "ad_type": "referral_program | lead_gen_offer | newsletter_cta | subscription_pitch | sponsor_content | product_feature_disguised | not_ad",
  "notes": "why this is or isn't an ad",
  "added": "YYYY-MM-DD"
}
```

## Daily Tuner

`newsradar tune --config <config> [--date YYYY-MM-DD] [--include-seed DIR] [--rebuild] [--rollback YYYY-MM-DD]`
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
| `data/tuning/snapshots/` | A copy of the learned files from before each applied change, the parts it applied (`applied.json`), and, when it changed the ads, the live model's Modelfile from `ollama show --modelfile` |
| `data/tuning/reports/DATE.md` | The report for that day |
| `data/tuning/build/` | Scratch space for candidate builds, deleted at the end of each run |

Learned example links are cut to `scheme://host/` and summaries to 200 characters,
so nothing sensitive from a URL is stored. Evidence, snapshots and reports older than
`evidence_days` (30) are deleted. The newest snapshot is always kept.

### Kept ads

The ads gate normally rejects a change that breaks any case the live model gets right.
One kind of break may pass: one of today's items labeled an ad, which the live model
removes and the candidate keeps. A today item's label is the judge's verdict, except
when the second opinion disputed the judge's call on it. That case is settled, and its
label is the gate's own verdict. The change still passes only if every
broken case is of that kind, it fixes at least 5 cases per kept ad
(`KEPT_AD_FIX_RATIO` in `tuning/gate.py`), it has fewer errors in total, and it fixes
a countable mistake from today. Fixed seed examples don't count toward that ratio,
because the seed is in the candidate's own training block and fixing it proves
little. News the candidate now removes, and any break in a base, learned or seed
example, always blocks. The report lists each kept ad by id and title.

### Gate cases

The ads gate scores the live and candidate models on these cases:

- every base example in `tests/test_data/labeled_examples.json`;
- every learned example, including older ones the Modelfile no longer holds. They are
  the regression set for news the gate once removed by mistake;
- every seed example;
- of today's records, every mistake, every removal (verdict "ad"), every record whose
  label a second opinion settled, and every record behind a new example;
- records that passed the gate, up to `[tuning] max_gate_today_cases` (60) of today's
  cases in all. The records above fill the cap first and are scored even past it.
  When more passed records exist than the cap leaves room for, the gate picks them
  by a SHA-256 hash of the date and record key, so every run on the same day picks
  the same ones.

The report's gate results say how many passed records were sampled out. Today's
records are the part of the case set that grows with the day's volume, so they carry
the cap. A run with 89 cases took 10 minutes, and 2026-11-24 had 141 records. A news
record the candidate starts removing outside the sample still shows up the next day,
because the judge reviews every removal first.

### Learned examples in the Modelfile

The Modelfile holds every base example plus the newest
`[tuning] learned_examples_per_class` (24) learned examples of each label, by `added`
date. The default was 12. On 2026-11-24 all 12 learned not-ad examples were from the
day before, so a run's 4 new examples pushed 4 of them out of the candidate, the
candidate then called one of those news items an ad, and the gate rightly blocked the
change. At 12, each new day pushes out examples the model still needs.

Measured with Ollama on 2026-11-24, 38 examples in the Modelfile used 4454 prompt
tokens, about 95 tokens per example. The scan sends the gate the title and the first
1,000 characters of the item text, about 270 tokens. Before installing a build, the
tuner estimates the SYSTEM block, rules and examples included, at 4 characters per
token. If that is over 70% of the Modelfile's `num_ctx`, it raises
`ModelfileTooLargeError`, the report says "The candidate Modelfile is too large for
num_ctx", and the run applies nothing.

`AdDetectorModelfile` sets `num_ctx` to 12288, up from 8192, so the full 24 of each
label fit. At 8192 the limit was 5734 tokens and only about 26 learned examples fit.
At 12288 the limit is 8601 tokens. The base build alone estimates at 3138 tokens.
With 48 learned examples of the size the tuner saves (a title, a 200-character summary
and a short note), it estimates at 7255. The estimate comes out about 5% under
Ollama's count, so the real figure is near 7600 tokens. That leaves about 4,700 tokens
of `num_ctx` for the item and the reply.

### The report

`data/tuning/reports/DATE.md` lists the evidence counted, each gate mistake with the
judge's verdict and second opinion, the proposed example, rule and dedup changes, the
regression gate results, what was applied, the judge cost, and clean-up. It ends with
the exact `--rebuild` and `--rollback` commands for that config.

### Seeding learned data

`--include-seed DIR` adds the `learned_ad_examples.json` and `learned_ad_rules.txt`
in DIR to that day's proposals. Seed examples get the same checks as the judge's
(forbidden Modelfile text, 200-character summaries, domain-only links) but not the
per-day cap, and keep their own `added` date. Every seed rule needs the rule
second opinion's approval. The seed is saved only when the ads gate accepts the
change, and the gate accepts only on a day where the candidate also fixes at least
one of that day's countable mistakes. On a day with no such fix, the seed is not
saved; run it again on another day. The pending seed lives in `data/tuning/seed/`.

### Rebuild and rollback

- `--rebuild` rebuilds the live `ad-detector` model from `AdDetectorModelfile` plus the
  learned files. Run it after any manual edit to the learned files.
- `--rollback DATE` restores the latest snapshot taken on DATE. It puts back the learned
  files for the parts that run applied. If that run changed the ads, it also reinstalls
  the live model from the Modelfile the snapshot read from Ollama, which is the model
  that was live before the change. A dedup-only day leaves the ad model alone. If the
  snapshot's `applied.json` doesn't parse, or an ads snapshot has no Modelfile, the
  rollback prints one line saying so, exits with status 1 and changes nothing.

The tuner builds a `*-candidate` model to test a change and removes it at the end
of the run. A bare `ollama create` drops learned data, as the Ongoing Workflow section explains.

The tuner and `newsradar ad-detector install` look for the `ollama` command on PATH,
then at `/usr/local/bin/ollama`, then at `/opt/homebrew/bin/ollama`, so a launchd job
with a short PATH still finds it. Before the first judge call, each run checks that
the command resolves and that `ollama show --modelfile` can read the live model. If
either check fails, the report says why, the run applies nothing, and no Claude call
is made. A day with no evidence skips the check, since it makes no paid calls.

## Ongoing Workflow

### When new ads slip through the digest

The daily tuner (above) handles most of this. To add examples by hand:

1. **Find the ad items** in the offending digest page or in the day's evidence under
   `data/evidence/DATE/`. Copy the title, summary and source.

2. **Add them to `tests/test_data/labeled_examples.json`** with `is_ad: true` and an
   `ad_type`. Cut `link` to `scheme://host/` (a test enforces this) and leave it empty if
   the item has no link.

3. **Add a matching non-ad example** from the same page to keep the dataset balanced.

4. **Run the tests** to check the dataset:
   ```bash
   uv run pytest tests/test_ad_detector.py
   ```

5. **Regenerate the Modelfile examples and rebuild the live model:**
   ```bash
   uv run -m newsradar.ad_detector.update_modelfile
   uv run newsradar tune --config <config> --rebuild
   ```
   Do not run a bare `ollama create ad-detector -f AdDetectorModelfile`. It builds
   from the base Modelfile only and drops everything the daily tuner learned.
   `--rebuild` builds from the base Modelfile plus the learned data. To install the
   base model with no learned data, for example on a new machine, run
   `uv run newsradar ad-detector install`.

6. **Re-run the tests.** `test_llm_ad_gate_detector_metrics` runs only when the
   `ad-detector` model is installed in Ollama, and requires F1 of at least 0.8:
   ```bash
   uv run pytest tests/test_ad_detector.py
   ```

7. **Check real content with a scratch config, never the real one.** A scan with the
   real config writes pages into the site folder that the 08:00 job deploys, even with
   `--dry-run`, and writes evidence into the real `data/`. Copy the whole `config/`
   folder, since the config names its topic and feed files by relative path. In the
   copy's `config.toml`, point `[paths] output_root` at a temporary folder, delete the
   `[publish]` section, and set `[tuning] enabled = false` (or point `[tuning] data_dir`
   at a temporary folder) so the scan's evidence stays out of the real `data/`. Then
   scan with the copy:
   ```bash
   cp -R config /tmp/scratch-config
   # edit /tmp/scratch-config/config.toml as above
   uv run newsradar --config /tmp/scratch-config/config.toml --dry-run --no-email --date YYYY-MM-DD
   ```
   Then read the scan log for "ad gate" lines.

### Tests

```bash
uv run pytest tests/test_ad_detector.py
```

The file checks that the dataset loads, that every example link is domain-only, and (when
the model is installed) that the `ad-detector` model scores F1 of at least 0.8.

## Current Benchmark (May 2026)

Measured in May 2026 on 24 of the examples. The file now has 26, 16 ads and 10 not-ads.

| Detector | Precision | Recall | F1 |
|---|---|---|---|
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
