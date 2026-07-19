#!/usr/bin/env python3
"""news-radar — config-driven RSS + email digest generator.

Usage:
    newsradar --config config.toml                          # run all configured topics
    newsradar --config config.toml --topic ai                # single topic
    newsradar --config config.toml --hours 48                # override lookback window
    newsradar --config config.toml --date 2026-04-13          # use a specific date
    newsradar --config config.toml --time 08:00                # cut off at 08:00 ET today
    newsradar --config config.toml --dry-run                   # generate output only, skip publish hook
    newsradar --config config.toml --no-email                  # skip Gmail (RSS only)
    newsradar --config config.toml --refresh-token              # re-authenticate with Gmail, then exit
    newsradar ad-detector install                                # install the ad-detector Ollama model
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

from .article_fetcher import enrich_with_full_text, fetch_article_text, source_name_from_url
from .config import Config, load_config
from .email_fetcher import _get_credentials, fetch_emails
from .enricher import enrich, write_enriched_json
from .feed_fetcher import fetch_all_feeds, is_arxiv
from .html_generator import generate_html
from .llm import (
    classify_ad,
    classify_topic,
    deduplicate,
    llm_stats,
    summarize,
    tag,
    unload_all_models,
)
from .output_writer import save_html, save_json, write_outputs
from .podcast_generator import generate_podcast
from .podcast_rss import generate_podcast_rss
from .topics import Topic

MAX_LINKS_PER_EMAIL = 5

_log_lock = threading.Lock()
_log_file: io.TextIOWrapper | None = None


def _open_log_file(log_dir: Path, date: datetime) -> io.TextIOWrapper:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"newsradar-{date.strftime('%Y-%m-%d')}.log"
    return open(log_path, "a", buffering=1)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line, flush=True)
        if _log_file:
            print(line, file=_log_file, flush=True)


# ---------------------------------------------------------------------------
# Single-item processor — runs in a thread pool worker
# ---------------------------------------------------------------------------

import re as _re

_PRICE_PATTERN = _re.compile(r"\$\d+(\.\d{2})?")
_AD_PHRASES = {
    "shop now", "buy now", "add to cart", "free shipping", "order now",
    "gift idea", "gift guide", "on sale", "discount code",
    "promo code", "coupon", "limited time", "shop our", "browse our",
    "new arrivals", "bestseller", "best seller",
    "register now", "register for free", "register today", "save your seat",
    "save my seat", "reserve your spot", "claim your spot",
    "join us live", "join us for a", "join our webinar", "join our live",
    "free webinar", "live webinar", "upcoming webinar",
}

_WEBINAR_PHRASES = {
    "webinar", "web seminar", "virtual event", "online event",
    "register to attend", "register for this", "register at",
    "upcoming event", "live session", "live demo",
}


def _is_advertisement(title: str, text: str) -> bool:
    combined = (title + " " + text).lower()
    if len(_PRICE_PATTERN.findall(combined)) >= 3:
        return True
    if any(phrase in combined for phrase in _AD_PHRASES):
        return True
    return False


def _is_webinar_summary(summary: str) -> bool:
    lowered = summary.lower()
    return any(phrase in lowered for phrase in _WEBINAR_PHRASES)


def _process_one(idx: int, total: int, item: dict, source_type: str, topic: Topic, config: Config) -> dict | None:
    title = item.get("title", "").strip()
    existing_text = item.get("body") or item.get("summary", "")
    label = item.get("source", "unknown")

    if source_type == "email" and not title:
        title = existing_text[:70].strip() or "(untitled)"

    if not title:
        log(f"  [{idx}/{total}] skip: no title")
        return None
    log(f"  [{idx}/{total}] {label}: {title[:70]}")

    if _is_advertisement(title, existing_text):
        log(f"    [{idx}] → skip (advertisement)")
        return None

    if config.models.ad_gate_enabled:
        is_ad, ad_reason = classify_ad(title, existing_text[:1000], config.models.ad_detector_model)
        if is_ad:
            log(f"    [{idx}] → skip (ad gate: {ad_reason})")
            return None

    lowered = (title + " " + existing_text).lower()

    if not any(kw in lowered for kw in topic.keywords):
        log(f"    [{idx}] → classifying with LLM (no {topic.display_name} keywords matched)...")
        if not classify_topic(title, existing_text[:500], config.models.summarize_model, topic):
            log(f"    [{idx}] → skip (not {topic.display_name}-related)")
            return None
        log(f"    [{idx}] → classified as {topic.display_name}-related")

    log(f"    [{idx}] → summarizing...")
    summary_text = summarize(title, existing_text, config.models.summarize_model)

    if _is_webinar_summary(summary_text):
        log(f"    [{idx}] → skip (webinar/promotional event)")
        return None

    log(f"    [{idx}] → tagging...")
    tags = tag(title, summary_text, config.models.summarize_model)
    log(f"    [{idx}] → done  tags={tags}")

    return {
        "title": title,
        "link": item.get("link", ""),
        "source": item.get("source", ""),
        "summary": summary_text,
        "tags": tags,
        "published": item.get("published"),
        "_source_type": source_type,
        "_is_arxiv": is_arxiv(item.get("source", "")),
    }


def _process_items(raw_items: list[dict], source_type: str, topic: Topic, config: Config) -> list[dict]:
    if not raw_items:
        return []

    total = len(raw_items)
    results: dict[int, dict | None] = {}

    with ThreadPoolExecutor(max_workers=config.pipeline.llm_workers) as executor:
        futures = {
            executor.submit(_process_one, idx, total, item, source_type, topic, config): idx
            for idx, item in enumerate(raw_items, 1)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return [results[i] for i in sorted(results) if results[i] is not None]


def _fetch_links_parallel(email_items: list[dict], config: Config) -> list[dict]:
    work = []
    for email in email_items:
        links = email.get("links", [])[:MAX_LINKS_PER_EMAIL]
        for url in links:
            work.append((email["source"], url))

    if not work:
        return []

    log(f"  Fetching {len(work)} links across {len(email_items)} emails ({config.pipeline.url_workers} workers)...")

    fetched: list[dict] = []

    def _fetch(email_source: str, url: str) -> dict | None:
        log(f"    → fetching: {url[:80]}")
        text = fetch_article_text(url)
        if not text:
            log(f"    → skip (could not fetch): {url[:60]}")
            return None
        source = source_name_from_url(url)
        log(f"    → fetched {len(text):,} chars from {source}")
        return {
            "title": "",
            "link": url,
            "source": source,
            "summary": text[: config.pipeline.article_body_char_cap],
            "body": text[: config.pipeline.article_body_char_cap],
            "_from_email_link": True,
        }

    with ThreadPoolExecutor(max_workers=config.pipeline.url_workers) as executor:
        futures = {executor.submit(_fetch, src, url): url for src, url in work}
        for future in as_completed(futures):
            result = future.result()
            if result:
                fetched.append(result)

    return fetched


def _stop_models(log_fn) -> None:
    log_fn("")
    log_fn("── Stopping Ollama models ──")
    try:
        unloaded = unload_all_models()
        for name in unloaded:
            log_fn(f"  stopped: {name}")
        if not unloaded:
            log_fn("  none loaded")
    except Exception as e:
        log_fn(f"  error unloading models: {e}")


def _resolve_publish_hook(dotted_path: str | None):
    if not dotted_path:
        return None
    module_name, _, func_name = dotted_path.partition(":")
    if not func_name:
        raise ValueError(f"publish hook must be 'module:function', got {dotted_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


# ---------------------------------------------------------------------------
# Per-topic run
# ---------------------------------------------------------------------------

def _run_topic(
    args: argparse.Namespace,
    config: Config,
    as_of: datetime,
    topic: Topic,
    email_items: list[dict] | None = None,
    linked_articles: list[dict] | None = None,
) -> list[Path]:
    email_items = email_items or []
    linked_articles = linked_articles or []

    base_dir = config.output_root / topic.output_dir
    output_dir = base_dir / as_of.strftime("%Y-%m")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = topic.file_prefix

    log(f"=== news-radar — {topic.display_name} ===")
    log(f"As-of:           {as_of.strftime('%Y-%m-%d %H:%M ET')}")
    log(f"Lookback:        {args.hours}h")
    log(f"Topic:           {topic.name}")
    log(f"Output dir:      {output_dir}")
    log(f"Summarize model: {config.models.summarize_model}")
    log(f"Rank model:      {config.models.rank_model}")
    log(f"Dedup model:     {config.models.dedup_model}")
    log(f"LLM workers:     {config.pipeline.llm_workers}  (set OLLAMA_NUM_PARALLEL={config.pipeline.llm_workers} to match)")
    log(f"URL workers:     {config.pipeline.url_workers}")
    log(f"Dry run:         {args.dry_run}")
    log("")

    as_of_utc = as_of.astimezone(timezone.utc)

    # --- Step 1: Fetch RSS feeds ---
    log(f"── Step 1: Fetching {topic.display_name} RSS feeds ──")
    rss_articles, rss_errors = fetch_all_feeds(config, topic, hours=args.hours, as_of=as_of_utc)
    log(f"  {len(rss_articles)} articles fetched, {len(rss_errors)} feed errors")
    for e in rss_errors:
        log(f"  ERROR {e['source']}: {e['error']}")
    enrich_with_full_text(rss_articles, config.pipeline.url_workers, config.pipeline.article_body_char_cap, log)
    log("")

    # --- Step 2: Process emails ---
    log(f"── Step 2: Processing {len(email_items)} emails ({config.pipeline.llm_workers} workers) [{topic.display_name} filter] ──")
    processed_emails = _process_items(email_items, "email", topic, config)
    log(f"  {len(processed_emails)}/{len(email_items)} emails kept")
    log("")

    # --- Step 3: Process linked articles ---
    log(f"── Step 3: Processing {len(linked_articles)} linked articles ({config.pipeline.llm_workers} workers) [{topic.display_name} filter] ──")
    processed_links = _process_items(linked_articles, "rss", topic, config)
    log(f"  {len(processed_links)}/{len(linked_articles)} linked articles kept")
    log("")

    # --- Step 4: Process RSS articles ---
    log(f"── Step 4: Processing {len(rss_articles)} RSS articles ({config.pipeline.llm_workers} workers) [{topic.display_name} filter] ──")
    processed_rss = _process_items(rss_articles, "rss", topic, config)
    log(f"  {len(processed_rss)}/{len(rss_articles)} articles kept")
    log("")

    all_items = processed_emails + processed_links + processed_rss

    _stop_models(log)
    log("")

    # --- Step 5: Deduplicate ---
    log(f"── Step 5: Deduplicating {len(all_items)} items "
        f"({len(processed_emails)} newsletters + {len(processed_links)} linked + {len(processed_rss)} RSS) ──")
    all_items = deduplicate(all_items, config.models.dedup_model)
    log(f"  {len(all_items)} items after deduplication")
    log("")

    # --- Step 6: Enrich ---
    log(f"── Step 6: Enriching {len(all_items)} items ──")
    json_path = output_dir / f"{file_prefix}-{as_of.strftime('%Y-%m-%d')}.json"
    enriched_data = enrich(
        all_items,
        as_of,
        json_path,
        summarize_model=config.models.summarize_model,
        rank_model=config.models.rank_model,
        topic=topic,
        llm_workers=config.pipeline.llm_workers,
        log=log,
    )
    podcast_count = len([i for i in enriched_data["items"] if i.get("include_in_podcast")])
    log(f"  Enrichment complete — {podcast_count} podcast items")
    log("")

    # --- Steps 7a + 7b: Generate HTML and podcast in parallel ---
    newsletters = [i for i in all_items if i["_source_type"] == "email"]
    papers = [i for i in all_items if i.get("_is_arxiv")]
    articles = [i for i in all_items if i["_source_type"] == "rss" and not i.get("_is_arxiv")]

    html_result: list = []
    html_error: list = []
    podcast_result: list = []
    podcast_error: list = []

    def _gen_html():
        try:
            log("── Step 7a: Generating HTML ──")
            log(f"  Newsletters: {len(newsletters)}, Articles: {len(articles)}, Papers: {len(papers)}, Errors: {len(rss_errors)}")
            ym_dir = as_of.strftime("%Y-%m")
            date_str = as_of.strftime("%Y-%m-%d")
            _url_base = f"{config.site.base_url}/{config.site.public_path_prefix}".rstrip("/") if config.site.public_path_prefix else config.site.base_url
            _mp3_url = None if args.no_podcast else f"{_url_base}/{topic.output_dir}/{ym_dir}/{file_prefix}-{date_str}.mp3"
            _rss_url = None if args.no_podcast else f"{_url_base}/{topic.output_dir}/podcast.rss"
            _og_url = None if args.no_podcast else f"{_url_base}/{topic.output_dir}/{ym_dir}/{file_prefix}-{date_str}.og.jpg"
            html = generate_html(
                newsletters=newsletters, articles=articles,
                papers=papers, errors=rss_errors, date=as_of,
                topic_label=topic.display_name,
                mp3_url=_mp3_url, podcast_rss_url=_rss_url, og_image_url=_og_url,
            )
            html_result.append(html)
            log(f"  HTML generated ({len(html):,} chars)")
        except Exception as e:
            html_error.append(e)

    def _gen_podcast():
        if args.no_podcast:
            log("── Step 7b: Podcast skipped (--no-podcast) ──")
            return
        try:
            log("── Step 7b: Generating podcast audio ──")
            mp3, chap_json, transcript, cover, og = generate_podcast(
                enriched_data, as_of, output_dir, topic=topic,
                tts_workers=config.pipeline.tts_workers, log=log,
            )
            podcast_result.append((mp3, chap_json, transcript, cover, og))
            log(f"  Podcast generated: {mp3.name}")
        except Exception as e:
            podcast_error.append(e)
            log(f"  WARNING: podcast generation failed: {e}")

    t_html = threading.Thread(target=_gen_html)
    t_pod = threading.Thread(target=_gen_podcast)
    t_html.start()
    t_pod.start()
    t_html.join()
    t_pod.join()

    if podcast_result:
        write_enriched_json(enriched_data, json_path)
        log("  Updated JSON with actual chapter times")

    if html_error:
        raise html_error[0]
    if not html_result:
        raise RuntimeError("HTML generation thread exited without producing output")

    html = html_result[0]

    # --- Step 8: Save ---
    log("── Step 8: Saving ──")
    html_path = save_html(html, as_of, output_dir=output_dir, prefix=file_prefix)
    log(f"  Saved HTML: {html_path}")

    out_paths = [html_path, json_path]
    if podcast_result:
        mp3_path, chap_path, transcript_path, cover_path, og_path = podcast_result[0]
        out_paths.extend([p for p in [mp3_path, chap_path, transcript_path, cover_path, og_path] if p])

    # --- Step 8b: Generate podcast RSS feed ---
    log("── Step 8b: Generating podcast RSS ──")
    _rss_output_dir_rel = f"{config.site.public_path_prefix}/{topic.output_dir}" if config.site.public_path_prefix else topic.output_dir
    rss_path = generate_podcast_rss(
        base_dir, topic, base_url=config.site.base_url,
        author_name=config.site.author_name, output_dir_rel=_rss_output_dir_rel, log=log,
    )
    out_paths.append(rss_path)

    call_count, total_duration = llm_stats()
    log(f"LLM calls: {call_count}  total time: {total_duration:.3f}s")

    if args.dry_run:
        log("")
        log("Dry run complete — skipping publish hook.")
        log(f"Preview: open {html_path}")
        _stop_models(log)
        return []

    _stop_models(log)
    log(f"Done — {topic.display_name} digest complete.")
    return out_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a config-driven news digest.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the digest pipeline (default)")
    _add_run_args(run_parser)

    ad_parser = subparsers.add_parser("ad-detector", help="Ad-detector model tooling")
    ad_sub = ad_parser.add_subparsers(dest="ad_command", required=True)
    install_parser = ad_sub.add_parser("install", help="Install the ad-detector model into Ollama")
    install_parser.add_argument("--model-name", default="ad-detector")
    install_parser.add_argument("--modelfile", type=str, default=None)

    _add_run_args(parser)  # allow bare `newsradar --config ...` (no subcommand) to run the pipeline
    return parser


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=False, help="Path to the news-radar TOML config file")
    parser.add_argument("--topic", type=str, default=None, help="Which topic(s) to generate, comma-separated (default: all configured topics)")
    parser.add_argument("--hours", type=int, default=None, help="Lookback window in hours (default: config's lookback_hours)")
    parser.add_argument("--date", type=str, default=None, help="Reference date YYYY-MM-DD in ET (default: today)")
    parser.add_argument("--time", type=str, default=None, help="Reference time HH:MM in ET (default: current time)")
    parser.add_argument("--dry-run", action="store_true", help="Generate output only, skip the publish hook")
    parser.add_argument("--no-email", action="store_true", help="Skip Gmail — fetch RSS feeds only")
    parser.add_argument("--no-podcast", action="store_true", help="Skip podcast audio generation")
    parser.add_argument("--publish-hook", type=str, default=None, help="Override the configured publish hook: 'module:function'")
    parser.add_argument("--refresh-token", action="store_true", help="Delete the Gmail token and re-authenticate, then exit")


def main() -> None:
    global _log_file

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "ad-detector":
        if args.ad_command == "install":
            from .ad_detector.install import install_ad_detector_model
            modelfile = Path(args.modelfile) if args.modelfile else None
            install_ad_detector_model(modelfile_path=modelfile, model_name=args.model_name)
            print(f"Installed ad-detector model: {args.model_name}")
        return

    if not args.config:
        parser.error("--config is required")

    config = load_config(args.config)

    if args.topic:
        requested = [t.strip() for t in args.topic.split(",")]
        topics = [config.topic(t) for t in requested]
    else:
        topics = list(config.topics.values())

    if args.refresh_token:
        if config.gmail.token_path.exists():
            config.gmail.token_path.unlink()
            print(f"Deleted {config.gmail.token_path}")
        print("Starting Gmail OAuth flow...")
        _get_credentials(config.gmail)
        print("Token refreshed. Exiting.")
        return

    now = datetime.now(ET)
    if args.date or args.time:
        ref_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else now.date()
        ref_time_str = args.time or now.strftime("%H:%M")
        ref_hour, ref_minute = (int(p) for p in ref_time_str.split(":"))
        as_of = datetime(ref_date.year, ref_date.month, ref_date.day, ref_hour, ref_minute, tzinfo=ET)
    else:
        as_of = now

    if args.hours is None:
        args.hours = config.pipeline.lookback_hours

    publish_hook = _resolve_publish_hook(args.publish_hook or config.publish_hook)

    log_dir = config.repo_root / "logs"
    _log_file = _open_log_file(log_dir, as_of)
    try:
        as_of_utc = as_of.astimezone(timezone.utc)

        email_items: list[dict] = []
        linked_articles: list[dict] = []
        if not args.no_email:
            log("── Fetching Gmail (shared across topics) ──")
            try:
                email_items = fetch_emails(config, hours=args.hours, as_of=as_of_utc)
                log(f"  {len(email_items)} emails fetched")
                for i, e in enumerate(email_items, 1):
                    log(f"  {i}. [{e['source']}] {e['title'][:70]}")
                log("")
                log(f"── Fetching email links ({config.pipeline.url_workers} workers) ──")
                linked_articles = _fetch_links_parallel(email_items, config)
                log(f"  {len(linked_articles)} linked articles fetched")
            except FileNotFoundError as e:
                log(f"  WARNING: {e}")
                log("  Continuing without email. Run with --no-email to suppress.")
            except Exception as e:
                from google.auth.exceptions import RefreshError
                if isinstance(e, RefreshError):
                    log("  Gmail token is invalid or expired.")
                    log("  Run: newsradar --config ... --refresh-token")
                    return
                raise
        else:
            log("── Gmail skipped (--no-email) ──")
        log("")

        all_out_paths: list[Path] = []
        for topic in topics:
            log(f"\n{'='*60}")
            log(f"  TOPIC: {topic.display_name}")
            log(f"{'='*60}")
            out_paths = _run_topic(args, config, as_of, topic, email_items=email_items, linked_articles=linked_articles)
            if out_paths:
                all_out_paths.extend(out_paths)

        if not args.dry_run and all_out_paths:
            log(f"\n{'='*60}")
            log("── Running publish hook ──")
            write_outputs(all_out_paths, config, publish_hook=publish_hook)

    finally:
        _log_file.close()
        _log_file = None


if __name__ == "__main__":
    main()
