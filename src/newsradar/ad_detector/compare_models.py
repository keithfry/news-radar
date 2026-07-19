#!/usr/bin/env python3
"""Compare summarization quality of several Ollama models across recent RSS articles.

Memory-efficient approach: all articles are processed by one model at a time,
then all judging is done by the judge model at the end — so only one large
model is loaded into VRAM at any point.

Usage:
    uv run -m newsradar.ad_detector.compare_models --config path/to/config.toml --topic ai
    uv run -m newsradar.ad_detector.compare_models --config path/to/config.toml --topic ai --hours 48
    uv run -m newsradar.ad_detector.compare_models --config path/to/config.toml --topic ai --out results.txt
"""

import argparse
import json
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ollama

from ..article_fetcher import enrich_with_full_text
from ..config import Config, load_config
from ..feed_fetcher import fetch_all_feeds

MODELS = ["gemma4:e4b", "gemma4:e2b", "gemma4:12b-mlx", "gemma3:4b", "granite3.2:2b"]
JUDGE_MODEL = "nemotron-3-super:cloud"  # qwen3.5:397b-cloud requires paid Ollama subscription (403)
ARTICLE_COUNT = 30
JUDGE_TIMEOUT_SECONDS = 90
JUDGE_MAX_RETRIES = 2

_judge_client = ollama.Client(timeout=JUDGE_TIMEOUT_SECONDS)


def _unload_model(model: str) -> None:
    """Stop a running Ollama model and wait for VRAM to free before continuing."""
    try:
        ollama.generate(model=model, prompt="", keep_alive=0)
    except Exception as e:
        print(f"  (warn) could not unload {model}: {e}", flush=True)
    time.sleep(5)
    try:
        loaded = [m["model"] for m in ollama.ps()["models"]]
    except Exception:
        loaded = []
    if model in loaded:
        print(f"  (warn) {model} still shows loaded after unload+5s wait", flush=True)
    else:
        print(f"  {model} unloaded, VRAM free", flush=True)


def _get_model_size(model: str) -> str:
    """Return human-readable disk size from ollama."""
    try:
        models = ollama.list()["models"]
        for m in models:
            if m["model"] == model:
                size_bytes = m.get("size", 0)
                size_gb = size_bytes / 1e9
                return f"{size_gb:.1f} GB"
    except Exception:
        pass
    return "?"


def _get_param_count(model: str) -> str:
    """Return human-readable parameter count from ollama."""
    try:
        info = ollama.show(model)
        return info.get("details", {}).get("parameter_size", "?")
    except Exception:
        return "?"


def _summarize(
    article_num: int, title: str, text: str, model: str
) -> tuple[str, float]:
    prompt = (
        "Summarize the following article in 3-4 sentences. "
        "Be specific and factual. Return only the summary, no preamble.\n\n"
        f"Title: {title}\n\nContent:\n{text}"
    )
    t0 = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options={"num_ctx": 16384},
    )
    elapsed = time.perf_counter() - t0
    print(f"  [#{article_num}] {model} {elapsed:.2f}s", flush=True)
    return response["message"]["content"].strip(), elapsed


def _judge(article_num: int, title: str, text: str, summaries: dict[str, str]) -> dict:
    sections = "\n\n".join(f"=== Model: {m} ===\n{s}" for m, s in summaries.items())
    prompt = (
        "You are evaluating AI-generated article summaries against the source article. "
        "Score each summary from 0.0 to 10.0 based on factual accuracy against the source "
        "(penalize hallucinated or unsupported claims heavily), clarity, "
        "completeness, and conciseness. Then give a composite ranking.\n\n"
        f"Article #{article_num}: {title}\n\n"
        f"Source content:\n{text}\n\n"
        f"{sections}\n\n"
        "Return a JSON object with this exact shape:\n"
        '{"scores": {"<model_name>": <score>, ...}, "winner": "<model_name>", '
        '"reasoning": "<one sentence>"}'
    )
    t0 = time.perf_counter()
    for attempt in range(1, JUDGE_MAX_RETRIES + 2):
        try:
            response = _judge_client.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                think=False,
                options={"num_ctx": 16384},
            )
            break
        except Exception as e:
            print(
                f"  [#{article_num}] {JUDGE_MODEL} (judge) attempt {attempt} failed: {e}",
                flush=True,
            )
            if attempt == JUDGE_MAX_RETRIES + 1:
                return {
                    "scores": {m: 0.0 for m in summaries},
                    "winner": "unknown",
                    "reasoning": f"judge call failed after {attempt} attempts: {e}",
                }
    elapsed = time.perf_counter() - t0
    print(f"  [#{article_num}] {JUDGE_MODEL} (judge) {elapsed:.2f}s", flush=True)
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {
            "scores": {m: 0.0 for m in summaries},
            "winner": "unknown",
            "reasoning": "parse error",
        }


def _generate_image(
    models: list[str],
    param_counts: dict[str, str],
    sizes: dict[str, str],
    avg_latency: dict[str, float],
    avg_scores: dict[str, float],
    final_scores: dict[str, float],
    out_path: str,
) -> None:
    col_labels = [
        "Model",
        "# Parameters",
        "Disk Size",
        "Avg Latency",
        "Avg Quality\nScore (0–10)",
        "Final Score\n(0–10)",
    ]
    rows = []
    for m in models:
        rows.append(
            [
                m,
                param_counts.get(m, "?"),
                sizes.get(m, "?"),
                f"{avg_latency[m]:.2f}s",
                f"{avg_scores[m]:.2f}",
                f"{final_scores[m]:.2f}",
            ]
        )

    # Sort by final score descending
    rows.sort(key=lambda r: float(r[5]), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 2.8 + len(rows) * 0.7))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)

    # Style header
    for col in range(len(col_labels)):
        cell = table[0, col]
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold")

    # Style rows — highlight winner (row 1 = top score)
    row_colors = ["#e8f4fd", "#ffffff"]
    for row_idx, _ in enumerate(rows, start=1):
        bg = row_colors[0] if row_idx == 1 else row_colors[1]
        for col in range(len(col_labels)):
            cell = table[row_idx, col]
            cell.set_facecolor(bg)
            if col == 5 and row_idx == 1:
                cell.set_text_props(fontweight="bold", color="#0066cc")

    # Formula note — anchored just below the table
    ax.text(
        0.5,
        0.02,
        "Final Score = 0.7 × Avg Quality Score  +  0.3 × Relative Avg Latency",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        va="bottom",
        color="#444444",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )

    fig.suptitle(
        f"Model Comparison — {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_models(config: Config, topic_name: str, hours: int | None, out: str) -> None:
    topic = config.topic(topic_name)
    hours = hours if hours is not None else config.pipeline.lookback_hours

    # --- Fetch articles ---
    print(f"Fetching RSS feeds (last {hours}h)...", flush=True)
    articles, _ = fetch_all_feeds(config, topic, hours=hours)
    articles = [
        a
        for a in articles
        if a.get("title") and (a.get("body") or a.get("summary") or "").strip()
    ]
    articles = articles[:ARTICLE_COUNT]
    enrich_with_full_text(
        articles, config.pipeline.url_workers, config.pipeline.article_body_char_cap, print
    )
    lengths = [len(a.get("body", "") or a.get("summary", "")) for a in articles]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(
        f"Using {len(articles)} articles "
        f"(avg content length: {avg_len:.0f} chars, "
        f"min: {min(lengths, default=0)}, max: {max(lengths, default=0)})\n",
        flush=True,
    )

    # summaries[i] = {model: summary_text}
    summaries: dict[int, dict[str, str]] = {i: {} for i in range(len(articles))}
    # latencies[model] = [elapsed, ...]
    latencies: dict[str, list[float]] = {m: [] for m in MODELS}

    # --- Phase 1: summarize all articles with each model, one model at a time ---
    for model in MODELS:
        print(f"── {model}: summarizing {len(articles)} articles ──", flush=True)
        for i, article in enumerate(articles):
            title = article["title"]
            text = article.get("body", "") or article.get("summary", "")
            summary, elapsed = _summarize(i + 1, title, text, model)
            summaries[i][model] = summary
            latencies[model].append(elapsed)
        _unload_model(model)
        print(flush=True)

    # --- Phase 2: judge all articles sequentially with the judge model ---
    print(f"── {JUDGE_MODEL}: judging {len(articles)} articles ──", flush=True)
    judgments: list[dict] = []
    for i, article in enumerate(articles):
        article_text = article.get("body", "") or article.get("summary", "")
        judgment = _judge(i + 1, article["title"], article_text, summaries[i])
        judgments.append(judgment)
        print(
            f"  [#{i+1}] winner={judgment.get('winner')}  scores={judgment.get('scores')}",
            flush=True,
        )
    print(flush=True)
    _unload_model(JUDGE_MODEL)

    # --- Write output file ---
    lines = [
        f"Model Comparison — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Models:   {', '.join(MODELS)}",
        f"Judge:    {JUDGE_MODEL}",
        f"Articles: {len(articles)}",
        "=" * 80,
        "",
    ]

    for i, article in enumerate(articles):
        j = judgments[i]
        lines += [
            f"Article #{i+1}: {article['title']}",
            f"Source: {article.get('source', '')}",
            "",
        ]
        for model in MODELS:
            score = j.get("scores", {}).get(model, "?")
            lines += [
                f"  [{model}]  score={score}",
                f"  {summaries[i][model]}",
                "",
            ]
        lines += [
            f"  Winner:    {j.get('winner')}",
            f"  Reasoning: {j.get('reasoning')}",
            "-" * 80,
            "",
        ]

    # Aggregate win counts and average scores
    win_counts: dict[str, int] = {m: 0 for m in MODELS}
    score_totals: dict[str, float] = {m: 0.0 for m in MODELS}
    score_counts: dict[str, int] = {m: 0 for m in MODELS}
    for j in judgments:
        winner = j.get("winner")
        if winner in win_counts:
            win_counts[winner] += 1
        for m, s in j.get("scores", {}).items():
            if m in score_totals:
                try:
                    score_totals[m] += float(s)
                    score_counts[m] += 1
                except (TypeError, ValueError):
                    pass

    avg_scores = {
        m: score_totals[m] / score_counts[m] if score_counts[m] else 0.0 for m in MODELS
    }
    avg_latency = {
        m: sum(latencies[m]) / len(latencies[m]) if latencies[m] else 0.0
        for m in MODELS
    }

    # Final score: 70% quality + 30% speed, both normalized 0–10
    # quality component: avg_score is already 0–10
    # speed component: best (lowest) latency = 10, others scaled proportionally
    min_latency = min(avg_latency.values()) or 1.0
    speed_scores = {
        m: (min_latency / avg_latency[m]) * 10 if avg_latency[m] else 0.0
        for m in MODELS
    }
    final_scores = {m: 0.7 * avg_scores[m] + 0.3 * speed_scores[m] for m in MODELS}

    lines += ["SUMMARY", "=" * 80]
    for m in MODELS:
        lines.append(
            f"  {m:30s}  wins={win_counts[m]:3d}  avg_score={avg_scores[m]:.2f}"
            f"  avg_latency={avg_latency[m]:.2f}s  final_score={final_scores[m]:.2f}"
        )
    lines.append("")

    with open(out, "w") as f:
        f.write("\n".join(lines))

    print(f"Results written to {out}\n")
    print("SUMMARY")
    print("=" * 40)
    for m in MODELS:
        print(
            f"  {m:30s}  wins={win_counts[m]}  avg={avg_scores[m]:.2f}"
            f"  latency={avg_latency[m]:.2f}s  final={final_scores[m]:.2f}"
        )

    # --- Generate comparison image ---
    model_sizes = {m: _get_model_size(m) for m in MODELS}
    model_params = {m: _get_param_count(m) for m in MODELS}
    image_path = out.rsplit(".", 1)[0] + ".png" if "." in out else out + ".png"
    _generate_image(
        MODELS, model_params, model_sizes, avg_latency, avg_scores, final_scores, image_path
    )
    print(f"Image written to {image_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to news-radar TOML config file")
    parser.add_argument("--topic", required=True, help="Configured topic name")
    parser.add_argument("--hours", type=int, default=None)
    parser.add_argument("--out", default="compare_models_output.txt")
    args = parser.parse_args()

    config = load_config(args.config)
    compare_models(config, args.topic, args.hours, args.out)


if __name__ == "__main__":
    main()
