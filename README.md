# news-radar

Config-driven RSS + email news digest pipeline: fetch feeds and (optionally)
Gmail newsletters, classify and summarize with local LLMs via
[Ollama](https://ollama.com), deduplicate, rank, and publish a styled HTML
digest plus podcast audio (with chapters, transcript, and an episode cover
image) — all defined by a single TOML config file and CSV feed list. Nothing
about your identity, output paths, or topics is hardcoded: everything lives
in config you write and commit yourself.

Originally extracted from a personal AI/Robotics news aggregator; the topic
system is fully generic — configure any number of topics (not just AI and
robotics) by writing a `classifier_prompt`, feed category, and output
directory for each.

## Quickstart

```bash
uv sync

cp examples/config.example.toml config.toml
cp examples/feeds.example.csv feeds.csv
# edit config.toml: [site] base_url/author, [[topics]] to taste

newsradar --config config.toml --dry-run
```

`--dry-run` runs the full pipeline (fetch → classify → summarize → dedupe →
rank → generate HTML/podcast) and writes output under `[paths] output_root`,
but skips the publish hook — good for a first end-to-end smoke test. Drop
`--dry-run` once you're ready to publish (see below), or add `--no-email` to
skip Gmail entirely and just try it against RSS feeds.

Before your first real run, read **[docs/SETUP.md](docs/SETUP.md)** — it
covers Gmail OAuth credentials, installing Ollama and the packaged
ad-detector model, Kokoro TTS for podcast audio, and what's gitignored (and
why).

## Publishing your own output

`newsradar.output_writer.write_outputs` hands the list of freshly-written
output paths to an optional **publish hook** — a plain
`(paths: list[Path], config: Config) -> None` function you point at via
`[publish] hook = "module:function"` in `config.toml` (or `--publish-hook` on
the CLI). The package itself has zero git/deploy opinions; publishing is
explicitly not core behavior.

`examples/hooks/git_publish.py` is a ready-to-use reference implementation
that `git add`s the output paths, commits, and pushes — copy it, adjust it,
or write your own (e.g. to sync to S3, trigger a static site rebuild, etc.).
Scheduling templates for running this on a timer are in `examples/launchd/`
(macOS) and `examples/cron/` (Linux).

## Ad detection

Before classification, every item passes through a cheap heuristic filter and
then an LLM-based ad/spam gate (the packaged `ad-detector` Ollama model, or
any model you configure via `[models] ad_detector_model`). See
[`src/newsradar/ad_detector/AD_DETECTION.md`](src/newsradar/ad_detector/AD_DETECTION.md)
for how the gate works, how to disable it, and how to retrain it on your own
labeled examples.

## License

MIT — see [LICENSE](LICENSE).
