# Daily tuning of the ad gate and duplicate check

Status: design, 2026-11-23

## Intent

The ad gate (`ad-detector`, a qwen3.5:9b Ollama model with a few-shot
Modelfile) and the duplicate check (Haiku through the `claude` CLI) make
mistakes nobody sees unless Keith reads the removed-items list. On
2026-11-23, 8 of the 10 ad-gate removals on the AI page were news: The AI
Report issues, a Tech Buzz digest and a paywalled Decoder article. In
November only 4 of 27 AI Report emails reached a page.

Keith wants a job that runs after each daily scan, has Claude Opus 5.5
review the day's ad and duplicate decisions, and applies fixes on its own.
In his words, the agent should make "nearly all the decisions, but [do]
them correctly". The safety comes from checks in code, not from asking him.

## Goals

- Every day, find the ad gate's wrong removals and missed ads, and the
  duplicate check's wrong merges and missed duplicates.
- Turn those findings into labeled examples, and into rule text where
  examples aren't enough.
- Apply a change automatically when checks show it fixes something and
  breaks nothing that was right before.
- Leave a short daily report of what was found, what changed, and the
  scores.

## Non-goals

- Tuning the title-phrase heuristic (`_AD_PHRASES`), the topic classifier,
  the ranker or the summarizer.
- Changing the base model of `ad-detector`, or the length of text the scan
  sends it.
- Re-running or republishing the day's pages after a fix. Fixes apply from
  the next scan on.
- Committing anything to git. The tuner changes only private data and the
  local Ollama model.

## Anti-goals

These would count as failures even if the job technically works:

- A bad change that quietly removes real news for days.
- Claude editing code. It returns verdicts and proposals as data, and a
  script decides what happens to them.
- Keith's personal context (CLAUDE.md, hooks, skills, MCP servers, memory)
  going into these calls.
- Email text, tracking links or day evidence reaching git (both
  `keithfry/news-radar` and `keithfry/techradar` are public) or the
  deployed site (`projects/techradar/techradar/`).

## Constraints

- Runs on Keith's Claude subscription through `claude -p`. `--bare` can't
  be used, because it requires an API key and ignores the OAuth login.
- The judge is Opus 5.5 at medium effort: `--model claude-opus-5-5
  --effort medium`.
- A rebuilt `ad-detector` model is used as soon as `ollama create`
  finishes, so applied changes go live on the next scan.
- The live ad prompt is about 3,300 tokens of a 4,096-token context. More
  examples or rules need `num_ctx` raised to 8,192. Without that, a long
  item makes the call fail with "prompt is longer than the context length".
- qwen3.5:9b at temperature 0 gave different verdicts on the same item when
  the input length changed (1,000 vs 2,500 characters), so a single test
  run is noisy.
- The scan sends the ad model the first 1,000 characters of an item. The
  tuner tests with exactly that input.

## Approaches considered

- **A. Claude runs the whole loop.** `claude -p` gets the evidence and
  edit access to news-radar, judges, edits, tests and commits. The most
  flexible, but the safety check is whatever Claude decides that day, and
  it can change any code. Rejected for that reason.
- **B. A script runs the loop and Claude only judges.** Chosen. A Python
  step sends Claude the evidence and gets fixed-format JSON back: verdicts
  and proposed examples and rule text. The script builds a candidate,
  scores it and applies it only if it passes. The safety check is plain
  code.
- **C. B for ads, report-only for duplicates** until enough duplicate
  verdicts build up. Safer for duplicates but slower to fix them. Rejected
  in favor of gating duplicate changes the same way as ad changes.

## Decided by Keith

- Approach B, applying changes automatically when they pass the checks.
- Changes that make the gate remove more (from missed ads) and changes that
  make it keep more (from wrong removals) both apply automatically.
- Judge model: Opus 5.5, medium effort.
- Remove personal context from the judge's calls and from the existing
  duplicate check's `claude` calls.
- The tuner's private data lives in a `data/` folder in the news-radar
  project folder, which git ignores, so it is never pushed and can never
  reach the website.

## Design

### 1. Isolated `claude` calls

`_chat_claude` in `llm.py` (used by the duplicate check today) and the new
judge both call the CLI with:

```
--setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}'
--disable-slash-commands --tools "" --no-session-persistence
```

and run with an empty working directory created for the call, so no
project memory or CLAUDE.md is found. Checked on 2026-11-23: a session with
these flags saw no CLAUDE.md, hooks, skills or MCP tools. The account email
remains, because it comes from the login.

The judge also passes `--system-prompt-file` for its instructions and
`--json-schema` for its output, plus `--output-format json` so the script
can read the result and the usage.

### 2. Private data location

A new `[tuning] data_dir` setting, default `data/` in the news-radar
project folder. `data/` is added to news-radar's `.gitignore`, so nothing
in it is committed or pushed. It holds:

- `evidence/YYYY-MM-DD/<topic>.jsonl`, the scan's record of each decision
  (section 3). Deleted after 30 days.
- `tuning/learned_ad_examples.json` and
  `tuning/learned_ad_rules.txt`, what the tuner has learned for the ad gate.
- `tuning/learned_dedup_cases.json`, labeled duplicate test cases, and the
  dedup examples derived from them.
- `tuning/snapshots/YYYY-MM-DD/`, a copy of the learned files before each
  applied change, the parts it applied, and, when it changes the ads, the
  live model's Modelfile from `ollama show --modelfile`, for rollback.
- `tuning/reports/YYYY-MM-DD.md`, the daily report.

The committed `AdDetectorModelfile` and `labeled_examples.json` stay the
hand-written base. The live model is built from base plus learned data.

### 3. Evidence capture during the scan

Today the text each check judged lives only in memory, and a removal record
keeps only title, link and source. When `[tuning] enabled` is true, the scan
writes one evidence file per topic per day.

Per ad-gate decision (both verdicts, so items that passed are covered too):
date, topic, title, link, source, source type (email, feed, linked
article), the exact text sent to the model, up to 4,000 characters of the
item for the judge, verdict, reason and model name. Items removed by the
title-phrase heuristic are recorded with that reason and are out of scope
for tuning. Each verdict is cached per run across topics, so each item is
recorded once per run, not once per topic.

Per duplicate-check batch: the titles and ids sent, the groups returned,
and the ids dropped. The exact-title pre-pass is recorded but out of scope.

### 4. The tuner

A new subcommand, `newsradar tune --config <config> --date YYYY-MM-DD`.
Steps:

1. **Load** the day's evidence. Exit quietly if there is none.
2. **Judge.** Send the ad decisions to Opus in batches. For each, the judge
   returns whether the verdict was right, the correct label, an `ad_type`
   from the existing labeled-data schema, a reason, and whether the
   1,000-character excerpt alone shows the right answer. For each mistake
   it returns a labeled example in the `labeled_examples.json` format, with
   a summary of at most 200 characters (the Modelfile shows only the first
   200) and the link cut down to its domain. Then send the duplicate
   batches. For each batch the judge returns wrong merges and missed
   duplicates, each as a small test case: the titles involved plus a few
   unrelated titles from the same batch, and the expected groups.
3. **Second opinion on remove-more changes.** Every proposed change that
   would make the gate remove more (a new ad example or ad-leaning rule
   text), and every proposed missed-duplicate case, goes to a second,
   separate judge call that sees the item without the first judge's
   verdict. It's included only if both agree. Keep-more changes skip this
   step.
4. **Propose.** The script assembles a candidate from the learned files
   plus today's agreed findings: new ad examples, plus one correctly
   handled example of the opposite label from the same day to keep the
   classes balanced (as `AD_DETECTION.md` asks), and any rule text the
   judge proposed. Rule text goes into a `LEARNED RULES` block of the
   system prompt, capped in length, below the hand-written patterns, which
   stay untouched. For duplicates, the new cases join the labeled duplicate
   set, and a `LEARNED EXAMPLES` block for the dedup prompt is regenerated
   from it.
5. **Build.** Write the candidate Modelfile to a work directory and run
   `ollama create ad-detector-candidate`.
6. **Gate (ads).** Score both `ad-detector` and the candidate on every base
   and learned labeled example plus every one of today's judged items,
   including the ones that passed the gate correctly. Send each item exactly
   as the scan does. Accept only if:
   - no item the current model gets right is wrong under the candidate;
   - the candidate fixes at least one of today's mistakes;
   - the candidate's total errors are lower.
   Any item whose result differs between the two models is re-run once,
   and the difference counts only if both runs agree.
   Mistakes the judge marked as not visible in the 1,000-character excerpt
   don't count toward "fixes at least one". They're listed in the report
   as needing a longer input.
   Later change (2026-11-24): `max_gate_today_cases` (default 300, equal
   to `max_items_per_run`) caps today's cases, so a normal day still
   scores every judged item. Only a config that judges more items than
   that samples the items that passed. Every mistake and removal is still
   scored, and at least 20 passed items are. `AD_DETECTION.md`, Gate
   cases, has the rule.
   Keith later relaxed the first rule for one kind of break (D5): one of
   today's items labeled an ad that the candidate now keeps. That break
   doesn't block when every broken item is of that kind and the candidate
   fixes at least 5 items per kept ad. The other two rules still apply.
   An item the candidate now removes, and any break in a base, learned or
   seed example, still blocks. The gate checks this after the second
   opinions on blockers, and again when it re-gates an edited change. The
   report names each kept ad.
7. **Gate (duplicates).** Run the current and candidate dedup prompts on
   every labeled duplicate case, using the scan's own model and voting.
   The same three rules apply: no new wrong merge or new miss, at least one
   fix, fewer errors in total.
8. **Apply.** For each part that passed: snapshot the current learned files
   and the live Modelfile as `ollama show --modelfile` prints it, write the
   new learned files, and rebuild `ad-detector` from base plus learned data.
   If `ollama show` fails, apply nothing. If the rebuild fails, the old model
   is still installed, so restore only the learned files. Nothing is
   committed to git.
9. **Report.** Write the daily report: the mistakes found (title, source,
   what the gate did, what each judge said), what was proposed, the scores
   of both versions, whether each change was applied or rejected and why,
   and the subscription usage the run reported. Add one summary line to
   the scan log.
10. **Clean up.** Remove `ad-detector-candidate` and delete evidence older
    than 30 days.

### 5. The dedup prompt reads learned examples

`_dedup_batch` adds the `LEARNED EXAMPLES` block from
`tuning/` to its prompt when the file exists, so dedup changes take effect
without code edits or commits.

### 6. Rebuilding by hand

`newsradar tune --rebuild` builds `ad-detector` from the committed base
plus the private learned data. `AD_DETECTION.md` changes to point to this
command instead of a bare `ollama create`, which would drop the learned
examples. `newsradar tune --rollback YYYY-MM-DD` restores the parts that
day's snapshot applied and, if it changed the ads, reinstalls the model
that was live before it. The report prints both commands.

### 7. Limits

- **Examples shown to the model.** The prompt shows all base examples plus
  the newest learned examples up to a cap per class, and stays within
  `num_ctx`. Older learned examples drop out of the prompt but stay in the
  test set, so the gate still checks the model against them. This replaces
  the random sampling in `update_modelfile.py` for the live build.
- **Usage.** One judge pass per day, with a cap on items judged per run.
  If any judge or gate call fails or hits a rate limit, the run applies
  nothing and the report says why. The next day's run starts fresh.

### 8. Scheduling

`run-agent.sh` runs the tuner after `run.py` finishes. Today it `exec`s
`run.py`, so that changes to running `run.py` and then the tuner. The tuner
runs whether or not the scan succeeded, logs to the same day's log, and a
tuner failure never changes the scan's exit status. Publishing has already
happened by then, so the tuner can't delay the site.

`[tuning] enabled = true` in techradar's config turns on both evidence
capture and the tuner. It's off by default in news-radar.

### 9. One-time setup before the first run

- Add `data/` to news-radar's `.gitignore`.
- Raise `num_ctx` in `AdDetectorModelfile` to 8,192, and add the
  `LEARNED RULES` marker.
- Seed the private learned data with the 2026-11-23 findings: the "judge
  the whole item" rule text and four non-ad examples (AI Report issue as
  web copy and as email, Tech Buzz digest, Decoder paywalled article). On
  2026-11-23, at the scan's 1,000-character input, this got 5 of 6 unseen
  items right (4 AI Report issues and 2 real ads). The live model got 2 of
  6. No existing labeled example changed.
- Cut the links in the committed `labeled_examples.json` down to their
  domains, removing a Hustle referral ID and HubSpot tracking tokens. They
  stay in git history.

## Calls made without asking Keith

- Second judge call for remove-more changes (section 4, step 3).
- No git commits by the tuner. It writes only to the ignored `data/`
  folder and the local Ollama model, so it never changes tracked files in
  the news-radar checkout Keith works in.
- Newest-first example cap instead of random sampling (section 7).
- Apply nothing on a partial or rate-limited run (section 7).
- Scrub links in the committed labeled examples (section 9).

## Left to the builder

- Evidence file layout and field names, within section 3.
- Judge prompt wording, batch sizes and the JSON schemas.
- The caps: rule text length, new examples per day, examples per class in
  the prompt, items judged per run.
- How labeled duplicate cases are stored.
- Test structure, following the existing news-radar and techradar tests.
