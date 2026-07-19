"""Build the radar HTML digest page.

The LLM produces summaries and tags; this module handles all HTML rendering.
"""

from datetime import datetime
from html import escape

# Tag key → (CSS class, display label)
TAG_META: dict[str, tuple[str, str]] = {
    "policy":   ("t-policy",   "Policy"),
    "model":    ("t-model",    "Models"),
    "agents":   ("t-agents",   "Agents"),
    "safety":   ("t-safety",   "Safety"),
    "robotics": ("t-robotics", "Robotics"),
    "voice":    ("t-voice",    "Voice AI"),
    "health":   ("t-health",   "Healthcare"),
    "research": ("t-research", "Research"),
    "ethics":   ("t-ethics",   "Ethics"),
}

# Subsection assignment: tag → subsection label (first match wins)
_SUBSECTION_ORDER = [
    ("policy",   "Policy &amp; Industry"),
    ("robotics", "Robotics"),
    ("safety",   "Safety, Ethics &amp; Society"),
    ("ethics",   "Safety, Ethics &amp; Society"),
    ("health",   "Safety, Ethics &amp; Society"),
    ("voice",    "Models &amp; Developer Tools"),
    ("model",    "Models &amp; Developer Tools"),
    ("agents",   "Models &amp; Developer Tools"),
    ("research", "Models &amp; Developer Tools"),
]

_SUBSECTION_DISPLAY_ORDER = [
    "Policy &amp; Industry",
    "Models &amp; Developer Tools",
    "Safety, Ethics &amp; Society",
    "Robotics",
]

_CSS = """\
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: #f4f6f9;
    color: #1a1d23;
    line-height: 1.6;
  }
  .page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f3460 100%);
    color: white;
    padding: 52px 24px 44px;
    text-align: center;
  }
  .page-header .eyebrow { font-size: 11px; letter-spacing: 3.5px; text-transform: uppercase; color: #60a5fa; margin-bottom: 14px; font-weight: 600; }
  .page-header h1 { font-size: 34px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 10px; }
  .page-header .dateline { font-size: 15px; color: #94a3b8; }
  .header-stats { display: flex; justify-content: center; gap: 20px; margin-top: 22px; flex-wrap: wrap; }
  .stat-pill { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); border-radius: 20px; padding: 5px 16px; font-size: 12px; color: #cbd5e1; font-weight: 500; }
  .container { max-width: 880px; margin: 0 auto; padding: 44px 20px 64px; }
  .section-header { display: flex; align-items: center; gap: 10px; margin: 52px 0 18px; padding-bottom: 12px; border-bottom: 2px solid #e2e8f0; }
  .section-header .icon { font-size: 20px; }
  .section-header h2 { font-size: 18px; font-weight: 700; color: #0f172a; letter-spacing: -0.2px; }
  .section-header .badge { margin-left: auto; font-size: 11px; font-weight: 700; background: #f1f5f9; color: #64748b; border-radius: 12px; padding: 3px 11px; }
  .subsection { font-size: 10px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; color: #94a3b8; margin: 28px 0 10px; }
  .card { background: white; border-radius: 10px; padding: 20px 22px; margin-bottom: 12px; border: 1px solid #e8edf3; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: box-shadow 0.15s ease; }
  .card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.09); }
  .card .via { font-size: 10.5px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #3b82f6; margin-bottom: 6px; }
  .card h3 { font-size: 15.5px; font-weight: 700; line-height: 1.4; margin-bottom: 9px; color: #0f172a; }
  .card h3 a { color: inherit; text-decoration: none; }
  .card h3 a:hover { color: #2563eb; text-decoration: underline; }
  .card p { font-size: 14px; color: #475569; line-height: 1.65; }
  .card.arxiv { border-left: 3px solid #8b5cf6; }
  .card.arxiv .via { color: #8b5cf6; }
  .card.robotics { border-left: 3px solid #10b981; }
  .card.robotics .via { color: #10b981; }
  .tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
  .tag { font-size: 10.5px; font-weight: 600; padding: 2px 9px; border-radius: 10px; }
  .t-policy   { background:#fee2e2; color:#b91c1c; }
  .t-model    { background:#dbeafe; color:#1d4ed8; }
  .t-agents   { background:#dcfce7; color:#15803d; }
  .t-safety   { background:#fef9c3; color:#a16207; }
  .t-robotics { background:#ede9fe; color:#6d28d9; }
  .t-voice    { background:#ffedd5; color:#c2410c; }
  .t-health   { background:#cffafe; color:#0e7490; }
  .t-research { background:#f1f5f9; color:#475569; }
  .t-ethics   { background:#fce7f3; color:#be185d; }
  .problem-feeds { border: 1px solid #fca5a5; background: #fff7f7; border-radius: 10px; padding: 20px 22px; margin-top: 40px; }
  .problem-feeds h2 { font-size: 15px; font-weight: 700; color: #b91c1c; margin-bottom: 12px; }
  .problem-feeds ul { list-style: none; }
  .problem-feeds li { font-size: 13px; color: #64748b; margin-bottom: 8px; }
  .problem-feeds li a { color: #2563eb; }
  .problem-feeds .error-reason { color: #dc2626; font-style: italic; }
  footer { text-align: center; font-size: 12px; color: #94a3b8; padding: 28px 20px 40px; border-top: 1px solid #e2e8f0; margin-top: 16px; }
  .filter-bar { background: white; border-bottom: 1px solid #e2e8f0; padding: 12px 20px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .filter-chip { font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 12px; cursor: pointer; border: 2px solid transparent; transition: border-color 0.15s, opacity 0.15s; }
  .filter-chip.active { border-color: currentColor; }
  .filter-chip.dimmed { opacity: 0.4; }
  .card.hidden { display: none; }
  .subsection.hidden { display: none; }
  .section-block.hidden { display: none; }
  .podcast-player { margin-top: 22px; display: flex; align-items: center; justify-content: center; gap: 12px; flex-wrap: wrap; }
  .podcast-player audio { max-width: 360px; width: 100%; height: 36px; border-radius: 18px; }
  .podcast-subscribe { font-size: 11px; font-weight: 700; color: #60a5fa; text-decoration: none; letter-spacing: 0.5px; padding: 6px 16px; border: 1px solid rgba(96,165,250,0.4); border-radius: 14px; white-space: nowrap; transition: background 0.15s; }
  .podcast-subscribe:hover { background: rgba(96,165,250,0.1); }"""


def _card(item: dict, extra_class: str = "") -> str:
    cls = f"card {extra_class}".strip()
    tag_keys = " ".join(t for t in item.get("tags", []) if t in TAG_META)
    title_html = escape(item.get("title", ""))
    link = item.get("link", "")
    via = escape(item.get("source", ""))
    summary = escape(item.get("summary", ""))

    title_block = (
        f'<h3><a href="{escape(link)}" target="_blank" rel="noopener">{title_html}</a></h3>'
        if link else
        f"<h3>{title_html}</h3>"
    )

    tags_html = ""
    tags = item.get("tags", [])
    if tags:
        tag_spans = "".join(
            f'<span class="tag {TAG_META[t][0]}">{TAG_META[t][1]}</span>'
            for t in tags if t in TAG_META
        )
        if tag_spans:
            tags_html = f'\n    <div class="tags">{tag_spans}</div>'

    data_tags = f' data-tags="{tag_keys}"' if tag_keys else ""
    chapter_offset = item.get("chapter_start_seconds")
    data_chapter = f' data-chapter-offset="{chapter_offset}"' if chapter_offset is not None else ""
    return (
        f'  <div class="{cls}"{data_tags}{data_chapter}>\n'
        f'    <div class="via">{via}</div>\n'
        f"    {title_block}\n"
        f"    <p>{summary}</p>{tags_html}\n"
        f"  </div>\n"
    )


def _subsection_for(item: dict) -> str:
    tags = item.get("tags", [])
    for tag, label in _SUBSECTION_ORDER:
        if tag in tags:
            return label
    return "Models &amp; Developer Tools"


def _podcast_player_html(mp3_url: str | None, rss_url: str | None) -> str:
    if not mp3_url:
        return ""
    subscribe = (
        f'\n  <a class="podcast-subscribe" href="{rss_url}">🎙 Subscribe</a>'
        if rss_url else ""
    )
    return (
        f'\n  <div class="podcast-player">\n'
        f'    <audio controls preload="none" src="{mp3_url}">'
        f'Your browser does not support audio.</audio>{subscribe}\n'
        f'  </div>'
    )


def generate_html(
    newsletters: list[dict],
    articles: list[dict],
    papers: list[dict],
    errors: list[dict],
    date: datetime,
    topic_label: str = "AI",
    source_label: str = "techradar-agent",
    mp3_url: str | None = None,
    podcast_rss_url: str | None = None,
    og_image_url: str | None = None,
) -> str:
    digest_title = f"{topic_label} Daily Digest"

    day_str = date.strftime("%A, %B %-d, %Y")
    time_str = date.strftime("%-I:%M %p ET")
    dateline = f"{day_str} &mdash; {time_str}"
    date_title = date.strftime("%B %-d, %Y")

    error_count = len(errors)
    stats = (
        f'    <span class="stat-pill">📬 {len(newsletters)} Newsletter{"s" if len(newsletters) != 1 else ""}</span>\n'
        f'    <span class="stat-pill">📰 {len(articles)} Article{"s" if len(articles) != 1 else ""}</span>\n'
        f'    <span class="stat-pill">🔬 {len(papers)} Paper{"s" if len(papers) != 1 else ""}</span>\n'
        f'    <span class="stat-pill">{"✅" if error_count == 0 else "⚠️"} {error_count} Feed Error{"s" if error_count != 1 else ""}</span>\n'
    )

    # Collect tags that actually appear in this digest, sorted by display label
    all_items = newsletters + articles + papers
    used_tags = sorted(
        {t for item in all_items for t in item.get("tags", []) if t in TAG_META},
        key=lambda t: TAG_META[t][1],
    )

    parts: list[str] = []
    og_meta = ""
    if og_image_url:
        og_desc = f"{topic_label} news digest — {date_title}"
        og_meta = (
            f'<meta property="og:title" content="{digest_title} — {date_title}"/>\n'
            f'<meta property="og:description" content="{og_desc}"/>\n'
            f'<meta property="og:image" content="{og_image_url}"/>\n'
            f'<meta property="og:image:width" content="1200"/>\n'
            f'<meta property="og:image:height" content="630"/>\n'
            f'<meta property="og:type" content="website"/>\n'
            f'<meta name="twitter:card" content="summary_large_image"/>\n'
            f'<meta name="twitter:image" content="{og_image_url}"/>\n'
        )

    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{digest_title} &mdash; {date_title}</title>
{og_meta}<style>
{_CSS}
</style>
</head>
<body>

<div class="page-header">
  <div class="eyebrow">⚡ Daily Briefing</div>
  <h1>{digest_title}</h1>
  <div class="dateline">{dateline}</div>
  <div class="header-stats">
{stats}  </div>{_podcast_player_html(mp3_url, podcast_rss_url)}
</div>

""")

    # --- Filter bar ---
    if used_tags:
        chip_html = "\n".join(
            f'  <button class="filter-chip {TAG_META[t][0]}" data-tag="{t}">{TAG_META[t][1]}</button>'
            for t in used_tags
        )
        parts.append(f'<div class="filter-bar">\n{chip_html}\n</div>\n\n')

    parts.append('<div class="container">\n')

    # --- Email Newsletters ---
    if newsletters:
        parts.append('  <div class="section-block">\n')
        parts.append(
            f'  <div class="section-header"><span class="icon">📬</span>'
            f'<h2>Email Newsletters</h2>'
            f'<span class="badge">{len(newsletters)} item{"s" if len(newsletters) != 1 else ""}</span></div>\n\n'
        )
        for item in newsletters:
            parts.append(_card(item))
        parts.append('  </div>\n')

    # --- Blog Posts & Articles ---
    if articles:
        parts.append('\n  <div class="section-block">\n')
        parts.append(
            f'  <div class="section-header"><span class="icon">📰</span>'
            f'<h2>Blog Posts &amp; Articles</h2>'
            f'<span class="badge">{len(articles)} item{"s" if len(articles) != 1 else ""}</span></div>\n'
        )
        subsections: dict[str, list[dict]] = {s: [] for s in _SUBSECTION_DISPLAY_ORDER}
        for item in articles:
            sub = _subsection_for(item)
            if sub not in subsections:
                subsections[sub] = []
            subsections[sub].append(item)

        for sub_label in _SUBSECTION_DISPLAY_ORDER:
            items_in_sub = subsections.get(sub_label, [])
            if not items_in_sub:
                continue
            parts.append(f'\n  <div class="subsection">{sub_label}</div>\n\n')
            for item in items_in_sub:
                extra = "robotics" if "robotics" in item.get("tags", []) else ""
                parts.append(_card(item, extra))
        parts.append('  </div>\n')

    # --- Research Papers ---
    if papers:
        badge = f"{len(papers)} paper{'s' if len(papers) != 1 else ''} · arXiv"
        parts.append('\n  <div class="section-block">\n')
        parts.append(
            f'  <div class="section-header"><span class="icon">🔬</span>'
            f'<h2>Latest Research Papers</h2>'
            f'<span class="badge">{badge}</span></div>\n\n'
        )
        for item in papers:
            parts.append(_card(item, "arxiv"))
        parts.append('  </div>\n')

    # --- Problem Feeds ---
    if errors:
        parts.append('\n  <div class="problem-feeds">\n')
        parts.append('    <h2>⚠️ Problem Feeds</h2>\n    <ul>\n')
        for err in errors:
            src = escape(err.get("source", "Unknown"))
            url = escape(err.get("feed_url", ""))
            reason = escape(err.get("error", "Unknown error"))
            parts.append(
                f'      <li><strong>{src}</strong> — '
                f'<a href="{url}" target="_blank" rel="noopener">{url}</a> — '
                f'<span class="error-reason">{reason}</span></li>\n'
            )
        parts.append('    </ul>\n  </div>\n')

    parts.append("\n</div>\n\n")

    # --- Footer ---
    feed_count = len(articles) + len(papers)
    footer_date = date.strftime("%A, %B %-d, %Y — %-I:%M %p ET")
    parts.append(
        f"<footer>\n"
        f"  {digest_title} &bull; {footer_date} &bull; "
        f"Sources: Gmail + {feed_count} RSS items via {source_label}\n"
        f"</footer>\n\n"
        f"<script>\n(function(){{\n"
        f"  const active = new Set();\n"
        f"  document.querySelectorAll('.filter-chip').forEach(chip => {{\n"
        f"    chip.addEventListener('click', () => {{\n"
        f"      active.has(chip.dataset.tag) ? active.delete(chip.dataset.tag) : active.add(chip.dataset.tag);\n"
        f"      applyFilter();\n"
        f"    }});\n"
        f"  }});\n"
        f"  function applyFilter() {{\n"
        f"    document.querySelectorAll('.filter-chip').forEach(c => {{\n"
        f"      c.classList.toggle('active', active.has(c.dataset.tag));\n"
        f"      c.classList.toggle('dimmed', active.size > 0 && !active.has(c.dataset.tag));\n"
        f"    }});\n"
        f"    document.querySelectorAll('.card').forEach(card => {{\n"
        f"      if (active.size === 0) {{ card.classList.remove('hidden'); return; }}\n"
        f"      const tags = (card.dataset.tags || '').split(' ');\n"
        f"      card.classList.toggle('hidden', !tags.some(t => active.has(t)));\n"
        f"    }});\n"
        f"    document.querySelectorAll('.subsection').forEach(sub => {{\n"
        f"      let el = sub.nextElementSibling, any = false;\n"
        f"      while (el && !el.classList.contains('subsection') && !el.classList.contains('section-header')) {{\n"
        f"        if (el.classList.contains('card') && !el.classList.contains('hidden')) any = true;\n"
        f"        el = el.nextElementSibling;\n"
        f"      }}\n"
        f"      sub.classList.toggle('hidden', !any);\n"
        f"    }});\n"
        f"    document.querySelectorAll('.section-block').forEach(block => {{\n"
        f"      const any = [...block.querySelectorAll('.card')].some(c => !c.classList.contains('hidden'));\n"
        f"      block.classList.toggle('hidden', !any);\n"
        f"    }});\n"
        f"  }}\n"
        f"}})();\n</script>\n"
        f"</body>\n</html>\n"
    )

    return "".join(parts)
