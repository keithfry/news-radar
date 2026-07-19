"""Thin Ollama wrapper for all LLM operations in the pipeline."""

import json
import os
import re
import sys
import threading
import time

import httpx
import json_repair
import ollama

from .topics import Topic

_llm_call_lock = threading.Lock()
_llm_call_count = 0
_llm_total_duration = 0.0


def llm_stats() -> tuple[int, float]:
    """Return (call_count, total_duration_seconds) accumulated so far."""
    with _llm_call_lock:
        return _llm_call_count, _llm_total_duration


def unload_all_models() -> list[str]:
    """Evict all loaded Ollama models from VRAM/RAM. Returns names unloaded."""
    loaded = [m.model for m in ollama.ps().models]
    for name in loaded:
        ollama.generate(model=name, prompt="", keep_alive=0)
    return loaded


def _parse_json(text: str) -> dict:
    return json_repair.loads(text)


def _chat(prompt: str, model: str, json_mode: bool = False, think: bool = False, num_ctx: int | None = None, _retries: int = 3) -> str:
    global _llm_call_count, _llm_total_duration

    kwargs: dict = {"think": think}
    if json_mode:
        kwargs["format"] = "json"
    if num_ctx is not None:
        kwargs["options"] = {"num_ctx": num_ctx}

    t0 = time.perf_counter()
    for attempt in range(_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            break
        except (httpx.RemoteProtocolError, httpx.ConnectError) as exc:
            if attempt == _retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"  [llm] Ollama disconnected ({exc.__class__.__name__}), retry {attempt+1}/{_retries-1} in {wait}s...", flush=True)
            time.sleep(wait)
    elapsed = time.perf_counter() - t0

    with _llm_call_lock:
        _llm_call_count += 1
        _llm_total_duration += elapsed
        cnt = _llm_call_count

    print(f"  [llm:{cnt}] {model} {elapsed:.3f}s", flush=True)

    return response["message"]["content"].strip()


_ARTICLE_CHAR_LIMIT = 24000  # ~6k tokens, leaves headroom for prompt+output in a 32k ctx window
_SUMMARIZE_NUM_CTX = 32768


def summarize_title(text: str, model: str) -> str:
    """Derive a concise article title from body text."""
    prompt = (
        "Write a concise, specific headline (under 12 words) for the following article. "
        "Return only the headline, no punctuation at the end, no quotes.\n\n"
        f"Article:\n{text[:_ARTICLE_CHAR_LIMIT]}\n\n"
        "Headline:"
    )
    return _chat(prompt, model, num_ctx=_SUMMARIZE_NUM_CTX).strip('"').strip()


def summarize(title: str, text: str, model: str) -> str:
    """Summarize an article in under 5 sentences."""
    prompt = (
        f"Summarize the following article in under 5 sentences. "
        f"Be specific and factual. Do not start with 'The article' or 'This article'. "
        f"Return only the summary text, no preamble.\n\n"
        f"Title: {title}\n\n"
        f"Content:\n{text[:_ARTICLE_CHAR_LIMIT]}"
    )
    return _chat(prompt, model, num_ctx=_SUMMARIZE_NUM_CTX)


def tag(title: str, summary: str, model: str) -> list[str]:
    """Assign 1-3 tags from the predefined set.

    Valid tag keys: policy, model, agents, safety, robotics, voice, health, research, ethics
    """
    prompt = (
        "Assign 1 to 3 tags to this AI/tech article from the following list ONLY. "
        "Return a JSON object with a single key 'tags' containing a list of tag strings.\n\n"
        "Tag definitions — only use a tag if the article is clearly about that topic:\n"
        "  policy   — government regulation, legislation, corporate AI policy, legal cases\n"
        "  model    — LLMs, model releases, benchmarks, training, quantization, inference, local models\n"
        "  agents   — AI agents, autonomous systems, multi-agent frameworks, agentic workflows\n"
        "  safety   — AI safety, alignment, security vulnerabilities, privacy, guardrails\n"
        "  robotics — physical robots, embodied AI, robotic hardware, autonomous vehicles\n"
        "  voice    — speech recognition, voice assistants, text-to-speech, audio AI\n"
        "  health   — healthcare, medical AI, biotech, drug discovery\n"
        "  research — academic papers, datasets, novel ML techniques, experiments\n"
        "  ethics   — bias, fairness, misinformation, AI misuse, societal impact\n\n"
        f"Title: {title}\n"
        f"Summary: {summary}\n\n"
        'Example response: {"tags": ["model", "research"]}'
    )
    raw = _chat(prompt, model, json_mode=True)
    try:
        data = _parse_json(raw)
        tags = data.get("tags", [])
        valid = {
            "policy",
            "model",
            "agents",
            "safety",
            "robotics",
            "voice",
            "health",
            "research",
            "ethics",
        }
        return [t for t in tags if t in valid][:3]
    except (json.JSONDecodeError, AttributeError):
        print(f"[warn] tag() failed to parse JSON: {raw!r}", file=sys.stderr)
        return ["research"]


def classify_topic(title: str, summary: str, model: str, topic: Topic) -> bool:
    """Return True if the content belongs in this topic's digest.

    Prompt is built entirely from topic.classifier_prompt. On JSON parse failure,
    falls back to topic.fail_open (preserves per-topic fail-open/fail-closed behavior).
    """
    prompt = topic.classifier_prompt + f"\n\nTitle: {title}\nContent: {summary[:3000]}"
    raw = _chat(prompt, model, json_mode=True)
    try:
        return bool(_parse_json(raw).get("relevant", False))
    except (json.JSONDecodeError, AttributeError):
        return topic.fail_open


def classify_ad(title: str, summary: str, model: str) -> tuple[bool, str]:
    """Return (is_ad, reason). Fails open — returns (False, 'parse error') on JSON failure."""
    prompt = (
        "You are a content classifier. Identify promotional, advertising, or non-editorial "
        "content that should be excluded from a curated AI/tech news feed.\n\n"
        "Mark is_ad: true if the item primarily does ONE of these:\n\n"
        "1. REFERRAL PROGRAM — asks reader to share a link for rewards/prizes.\n"
        "   Signals: \"share your link\", \"refer a friend\", \"earn rewards\", \"prizes available\"\n\n"
        "2. LEAD-GEN OFFER — free guide/ebook/template/checklist as lead capture.\n"
        "   Signals: \"free guide\", \"download now\", \"150+ prompts\", \"get access\", \"free stack\"\n\n"
        "3. NEWSLETTER CTA — asks reader to confirm subscription or re-engage to avoid removal.\n"
        "   Signals: \"still interested?\", \"click to confirm\", \"monitoring subscriber activity\", "
        "\"high email costs\", \"vote to stay\"\n\n"
        "4. ADVERTISING PITCH — promotes buying ad slots in a newsletter or marketing platform.\n"
        "   Signals: \"advertise in\", \"reach X million\", \"ROI\", \"pipeline value\", \"ad slot\"\n\n"
        "5. SUBSCRIPTION PRODUCT — paid membership, exclusive access, or recurring service pitch.\n"
        "   Signals: \"exclusive zip code\", \"one agent per city\", \"$X/year\", \"weekly leads\", "
        "\"apply now\", \"acceptance rate\"\n\n"
        "6. PRODUCT FEATURE DISGUISED AS NEWS — product email written to drive feature adoption.\n"
        "   Signals: imperative language (\"start shopping\", \"try it\"), reader is the target "
        "user, written from product POV\n\n"
        "Do NOT mark as ad: news about product launches (neutrally reported by third parties), "
        "research papers, analysis/opinion, technical tutorials, press releases from official "
        "company blogs with no purchase CTA.\n\n"
        "Examples:\n"
        "Title: \"Refer Friends and Get Rewards with The Hustle Program\"\n"
        "Summary: \"Share a unique link. Prizes are available for purchase.\"\n"
        '→ {"is_ad": true, "reason": "referral program with prize incentive"}\n\n'
        "Title: \"Free Guide to 150+ AI Prompts for Solopreneurs\"\n"
        "Summary: \"A free guide provides 150+ plug-and-play AI prompts. Download your free stack.\"\n"
        '→ {"is_ad": true, "reason": "lead-gen gated content offer"}\n\n'
        "Title: \"Top-Tier Tech Marketers Advertise in TLDR Newsletter\"\n"
        "Summary: \"Reach over 7 million tech professionals. Results: $382k in pipeline, 20.1x ROI.\"\n"
        '→ {"is_ad": true, "reason": "newsletter advertising network pitch"}\n\n'
        "Title: \"A faster way to shop\"\n"
        "Summary: \"ChatGPT helps you browse and compare products side-by-side. Start shopping.\"\n"
        '→ {"is_ad": true, "reason": "product feature pitch disguised as editorial"}\n\n'
        "Title: \"Still interested in Tech news?\"\n"
        "Summary: \"Due to high email costs, we monitor subscriber activity. Click to confirm.\"\n"
        '→ {"is_ad": true, "reason": "newsletter re-engagement CTA"}\n\n'
        "Title: \"Exclusive Real Estate Agent Program for Proven Performers Only\"\n"
        "Summary: \"One agent per city. Apply now. 15+ closings required. $2000/year.\"\n"
        '→ {"is_ad": true, "reason": "subscription product pitch with exclusivity framing"}\n\n'
        "Title: \"Gemini 3.5 Flash, Karpathy joins Anthropic, OpenAI Guaranteed Capacity\"\n"
        "Summary: \"Google introduced Gemini 3.5 Flash. Karpathy joins Anthropic.\"\n"
        '→ {"is_ad": false, "reason": "neutral news digest about AI developments"}\n\n'
        "Title: \"GPT-4o Gets New Voice Mode Capabilities\"\n"
        "Summary: \"OpenAI released an update adding real-time voice conversation.\"\n"
        '→ {"is_ad": false, "reason": "factual product capability news"}\n\n'
        "Title: \"Andrej Karpathy Joins Anthropic: What Happens Next\"\n"
        "Summary: \"Karpathy joins Anthropic as a researcher focused on AI capabilities.\"\n"
        '→ {"is_ad": false, "reason": "industry hiring news, third-party analysis"}\n\n'
        f"Now classify:\n"
        f"Title: {title}\n"
        f"Summary: {summary}\n\n"
        'Respond ONLY with JSON: {"is_ad": true/false, "reason": "brief explanation"}'
    )
    raw = _chat(prompt, model, json_mode=True)
    try:
        data = _parse_json(raw)
        return bool(data.get("is_ad", False)), str(data.get("reason", ""))
    except (json.JSONDecodeError, AttributeError):
        print(f"[warn] classify_ad() failed to parse JSON: {raw!r}", file=sys.stderr)
        return False, "parse error"


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "is", "are", "for",
    "on", "at", "by", "with", "that", "this", "from", "its", "it", "be",
    "as", "was", "has", "have", "how", "why", "what", "can", "new", "about",
    "up", "out", "into", "will", "more", "than", "but", "not", "all",
}


def _title_tokens(title: str) -> set[str]:
    """Lowercase words stripped of punctuation, minus stopwords."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _overlap_coefficient(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _keyword_duplicate_groups(items: list[dict], threshold: float = 0.40, min_shared: int = 4) -> list[list[int]]:
    """Fast pre-pass: group items whose title token overlap exceeds threshold.

    Requires at least `min_shared` tokens in common to avoid short-title false positives.
    """
    token_sets = [_title_tokens(item.get("title", "")) for item in items]
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            shared = token_sets[i] & token_sets[j]
            if len(shared) >= min_shared and _overlap_coefficient(token_sets[i], token_sets[j]) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


_DEDUP_BATCH_SIZE = 40
_DEDUP_CONFIDENCE_THRESHOLD = 0.5


def _dedup_batch(batch: list[tuple[int, dict]], model: str) -> set[int]:
    """Send one batch to LLM. Returns original indices to DROP."""
    if len(batch) <= 1:
        return set()

    payload = [{"id": str(orig_i), "title": item["title"]} for orig_i, item in batch]
    prompt = (
        "You are a news deduplication filter. Given a list of article titles, "
        "identify and remove duplicates — articles covering the same story, product release, "
        "or announcement, even if phrased differently or from different sources. "
        "A shared product name + version (e.g. 'Gemma 4 12B') means duplicate.\n\n"
        "Return a JSON object with key 'keep': a list of objects for each UNIQUE article to keep. "
        "Each object: {\"id\": \"<original id>\", \"confidence\": <0.0-1.0>} "
        "where confidence is how certain you are this item is unique. "
        "When duplicates exist, keep the one with the most informative title.\n\n"
        f"Articles:\n{json.dumps(payload, indent=2)}\n\n"
        "Response (JSON only):"
    )

    if model.startswith("claude"):
        import shutil
        import subprocess
        claude_bin = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
        result = subprocess.run(
            [claude_bin, "-p", prompt, "--model", model, "--output-format", "text"],
            capture_output=True, text=True, check=True,
        )
        raw = result.stdout
    else:
        raw = _chat(prompt, model, json_mode=True)

    try:
        keep_list = _parse_json(raw).get("keep", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"[warn] _dedup_batch() failed to parse JSON: {raw!r}", file=sys.stderr)
        return set()

    valid_ids = {str(orig_i) for orig_i, _ in batch}
    kept_ids: set[str] = set()
    for entry in keep_list:
        if not isinstance(entry, dict):
            continue
        item_id = str(entry.get("id", ""))
        confidence = float(entry.get("confidence", 1.0))
        if item_id in valid_ids and confidence >= _DEDUP_CONFIDENCE_THRESHOLD:
            kept_ids.add(item_id)

    # If LLM returned nothing valid, keep everything (safe fallback)
    if not kept_ids:
        print(f"[warn] _dedup_batch() returned no valid ids — keeping all", file=sys.stderr)
        return set()

    return {orig_i for orig_i, _ in batch if str(orig_i) not in kept_ids}


def deduplicate(items: list[dict], model: str) -> list[dict]:
    """Remove near-duplicate items using keyword pre-pass + batched LLM filter."""
    if len(items) <= 1:
        return items

    # Keyword pre-pass disabled — LLM handles all dedup
    # pre_groups = _keyword_duplicate_groups(items)
    # pre_drop: set[int] = set()
    # for group in pre_groups:
    #     best = max(group, key=lambda i: len(items[i].get("summary", "")))
    #     pre_drop.update(i for i in group if i != best)
    # if pre_drop:
    #     print(f"[dedup] keyword pre-pass removed {len(pre_drop)} items", file=sys.stderr)
    pre_drop: set[int] = set()

    surviving = [(orig_i, item) for orig_i, item in enumerate(items) if orig_i not in pre_drop]
    if len(surviving) <= 1:
        return [item for _, item in surviving]

    # Batched LLM pass over surviving items
    llm_drop: set[int] = set()
    for start in range(0, len(surviving), _DEDUP_BATCH_SIZE):
        batch = surviving[start:start + _DEDUP_BATCH_SIZE]
        dropped = _dedup_batch(batch, model)
        llm_drop |= dropped

    if llm_drop:
        print(f"[dedup] LLM pass removed {len(llm_drop)} items", file=sys.stderr)

    all_drop = pre_drop | llm_drop
    return [item for i, item in enumerate(items) if i not in all_drop]


def rank_items(items: list[dict], model: str) -> list[dict]:
    """Rank items by AI/robotics relevance and newsworthiness, 1 = most important.

    Returns a new list sorted by rank with a 'rank' field added to each item.
    Falls back to original order if LLM response cannot be parsed.
    """
    if not items:
        return []

    index_lines = "\n".join(
        f"{i}: {item['title']}" for i, item in enumerate(items)
    )
    prompt = (
        "You are an editor ranking AI and robotics news items by importance and interest "
        "for a daily listener. Rank the following items from most to least important. "
        "Consider: breadth of impact, novelty, practical significance for AI practitioners.\n\n"
        "Return a JSON object with key 'ranked' — a list of original indices in order "
        "from most to least important. Include every index exactly once.\n\n"
        f"Items:\n{index_lines}\n\n"
        'Example response: {"ranked": [3, 0, 2, 1]}'
    )
    raw = _chat(prompt, model, json_mode=True)
    try:
        order: list[int] = _parse_json(raw).get("ranked", [])
        order = [int(x) for x in order]
        # Deduplicate (keep first occurrence) and clamp to valid range
        seen: set[int] = set()
        clean: list[int] = []
        for idx in order:
            if 0 <= idx < len(items) and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        # Append any indices the model dropped, preserving original relative order
        missing = [i for i in range(len(items)) if i not in seen]
        if missing:
            print(f"[warn] rank_items() model omitted {len(missing)} indices, appending at end", file=sys.stderr)
        clean.extend(missing)
        ranked = []
        for rank, orig_idx in enumerate(clean, 1):
            item = dict(items[orig_idx])
            item["rank"] = rank
            ranked.append(item)
        return ranked
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
        print(f"[warn] rank_items() failed ({e}), using original order", file=sys.stderr)
        return [dict(item, rank=i + 1) for i, item in enumerate(items)]


def generate_audio_script(item: dict, model: str) -> str:
    """Generate a ~140-word news-report style spoken script for a single podcast segment.

    Structure: source/title attribution → overview (5-10s) → body (45s) → conclusion (10-15s).
    """
    source = item.get("source", "")
    title = item.get("title", "")
    summary = item.get("summary", "")

    prompt = (
        "Write a spoken podcast news segment in three parts. "
        "Use a factual, third-person news-report tone — no personal opinions, no 'you', no 'we'. "
        "No bullet points. No markdown. No URLs. No 'click here' or 'read more'. "
        "Write as one continuous spoken passage.\n\n"
        f"Source: {source}\n"
        f"Title: {title}\n"
        f"Summary: {summary}\n\n"
        "Structure — write all three parts as a single flowing paragraph:\n"
        f"1. ATTRIBUTION (1 sentence): Begin with 'From {source}, \"{title}\".' "
        "then immediately state in one sentence what this story is about and its key result or finding.\n"
        "2. BODY (about 100 words): Report the details — what was done, how it was accomplished, "
        "the process, the methodology, the specific findings or announcements. Be precise and factual.\n"
        "3. CONCLUSION (about 30 words): State what this means for the AI/tech industry "
        "or what to watch for next. No rhetorical questions.\n\n"
        "Target total: approximately 140 words.\n\n"
        "Spoken segment:"
    )
    return _chat(prompt, model).strip()


def generate_episode_tagline(items: list[dict], model: str) -> str:
    """Generate a 2-4 word episode tagline from top stories, e.g. 'New Gemini, Faster Learning'."""
    top3 = [item["title"] for item in items[:3]]
    top3_str = "\n".join(f"- {t}" for t in top3)
    prompt = (
        "Write a 2 to 4 word episode tagline that captures the key themes from these top stories. "
        "Use title case. No punctuation at start or end. No quotes. Examples: "
        "'New Gemini, Faster Learning' or 'Robots Learn to Walk' or 'Claude 4, AGI Debate'.\n\n"
        f"Top stories:\n{top3_str}\n\n"
        "Tagline:"
    )
    raw = _chat(prompt, model).strip()
    # Take only the first line in case the model leaks explanation text
    first_line = raw.splitlines()[0].strip().strip('"').strip("'")
    return first_line


def generate_outro_script(items: list[dict], date: "datetime", model: str, topic: Topic) -> str:
    """Generate a short, creative, varied sign-off outro for the podcast episode."""
    date_str = f"{date.strftime('%B')} {date.day}, {date.year}"
    topic_desc = topic.display_name
    top_title = items[0]["title"] if items else ""
    prompt = (
        f"Write a short podcast outro (20 to 40 words) for a daily {topic_desc} news digest. "
        f"Date: {date_str}. "
        "Sign off warmly and thank the listener for tuning in. "
        "Be creative and vary the style — it could be playful, punny, enthusiastic, philosophical, "
        "cheeky, warm, or cleverly reference the day's biggest story. "
        "Do not be generic. No markdown. No 'stay tuned'. Speak naturally as if signing off on air. "
        f"Today's top story was about: {top_title}\n\n"
        "Outro:"
    )
    return _chat(prompt, model).strip()


def generate_intro_script(items: list[dict], date: "datetime", model: str, topic: Topic) -> str:
    """Generate a podcast intro mentioning date, item count, and top-3 topics."""
    date_str = f"{date.strftime('%B')} {date.day}, {date.year}"
    top3 = [item["title"] for item in items[:3]]
    top3_str = "\n".join(f"- {t}" for t in top3)
    topic_desc = topic.display_name
    prompt = (
        f"Write a short podcast intro (under 45 words) for a daily {topic_desc} news digest. "
        f"Date: {date_str}. Total items: {len(items)}. "
        f"Mention up to 3 top stories by topic (not exact title). "
        f"Sound natural and welcoming. No markdown. End naturally, don't say 'let's get started'.\n\n"
        f"Top stories:\n{top3_str}\n\n"
        "Intro:"
    )
    return _chat(prompt, model).strip()
