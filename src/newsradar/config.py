"""Config loader — reads a TOML config file + .env for secrets.

No hardcoded identity, output paths, or topics. Secrets (Gmail client id/secret,
ANTHROPIC_API_KEY, etc.) are read only from the environment / .env — never from
the TOML file — so config files are safe to commit to a public repo.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .topics import Topic

DEFAULT_TOKEN_DIR = Path.home() / ".config" / "newsradar"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


@dataclass
class GmailConfig:
    client_id: str = ""
    client_secret: str = ""
    credentials_path: Path | None = None
    token_path: Path = field(default_factory=lambda: DEFAULT_TOKEN_DIR / "token.json")
    scopes: list[str] = field(default_factory=lambda: list(GMAIL_SCOPES))


@dataclass
class SiteConfig:
    title: str = "News Radar"
    base_url: str = ""
    author_name: str = ""
    author_email: str = ""
    # URL path segment between base_url and each topic's output_dir, for when
    # output_root is NOT itself the published docroot (e.g. output_root is a
    # subdirectory of a larger site). Empty (default) means output_root IS the
    # docroot, so URLs are just f"{base_url}/{topic.output_dir}/...".
    public_path_prefix: str = ""


@dataclass
class ModelsConfig:
    summarize_model: str = "llama3.2"
    rank_model: str = "qwen3.5:9b"
    dedup_model: str = "llama3.2"
    ad_detector_model: str = "ad-detector"
    ad_gate_enabled: bool = True


@dataclass
class PipelineConfig:
    lookback_hours: int = 24
    llm_workers: int = 2
    url_workers: int = 10
    article_body_char_cap: int = 20000
    tts_workers: int = 2
    arxiv_max_papers: int = 10


@dataclass
class Config:
    config_path: Path
    repo_root: Path  # directory containing the config file; base for relative paths
    site: SiteConfig
    models: ModelsConfig
    pipeline: PipelineConfig
    gmail: GmailConfig
    output_root: Path
    feeds_csv: Path
    topics: dict[str, Topic]
    publish_hook: str | None = None  # dotted path "module:function", e.g. "hooks.publish:publish"

    def topic(self, name: str) -> Topic:
        try:
            return self.topics[name]
        except KeyError:
            raise ValueError(
                f"Unknown topic {name!r}; configured topics: {sorted(self.topics)}"
            ) from None


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p)


def _bool_env(env_value: str | None, default: bool) -> bool:
    if env_value is None:
        return default
    return env_value not in ("0", "false", "False", "no")


def load_config(config_path: str | Path, env_file: str | Path | None = None) -> Config:
    """Load a news-radar TOML config file plus environment/.env secrets."""
    config_path = Path(config_path).resolve()
    base = config_path.parent

    load_dotenv(env_file if env_file is not None else base / ".env")

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    site_raw = raw.get("site", {})
    site = SiteConfig(
        title=site_raw.get("title", "News Radar"),
        base_url=site_raw.get("base_url", ""),
        author_name=site_raw.get("author_name", ""),
        author_email=site_raw.get("author_email", ""),
        public_path_prefix=site_raw.get("public_path_prefix", "").strip("/"),
    )

    models_raw = raw.get("models", {})
    models = ModelsConfig(
        summarize_model=os.environ.get("SUMMARIZE_MODEL", models_raw.get("summarize_model", "llama3.2")),
        rank_model=os.environ.get("RANK_MODEL", models_raw.get("rank_model", "qwen3.5:9b")),
        dedup_model=os.environ.get("DEDUP_MODEL", models_raw.get("dedup_model", "llama3.2")),
        ad_detector_model=os.environ.get(
            "AD_DETECTOR_MODEL", models_raw.get("ad_detector_model", "ad-detector")
        ),
        ad_gate_enabled=_bool_env(
            os.environ.get("AD_GATE_ENABLED"), bool(models_raw.get("ad_gate_enabled", True))
        ),
    )

    pipeline_raw = raw.get("pipeline", {})
    pipeline = PipelineConfig(
        lookback_hours=int(os.environ.get("LOOKBACK_HOURS", pipeline_raw.get("lookback_hours", 24))),
        llm_workers=int(os.environ.get("LLM_WORKERS", pipeline_raw.get("llm_workers", 2))),
        url_workers=int(os.environ.get("URL_WORKERS", pipeline_raw.get("url_workers", 10))),
        article_body_char_cap=int(
            os.environ.get("ARTICLE_BODY_CHAR_CAP", pipeline_raw.get("article_body_char_cap", 20000))
        ),
        tts_workers=int(os.environ.get("TTS_WORKERS", pipeline_raw.get("tts_workers", 2))),
        arxiv_max_papers=int(pipeline_raw.get("arxiv_max_papers", 10)),
    )

    gmail_raw = raw.get("gmail", {})
    token_path = _resolve_path(
        base, gmail_raw.get("token_path", str(DEFAULT_TOKEN_DIR / "token.json"))
    )
    creds_value = gmail_raw.get("credentials_path")
    gmail = GmailConfig(
        client_id=os.environ.get("GMAIL_CLIENT_ID", ""),
        client_secret=os.environ.get("GMAIL_CLIENT_SECRET", ""),
        credentials_path=_resolve_path(base, creds_value) if creds_value else None,
        token_path=token_path,
    )

    paths_raw = raw.get("paths", {})
    output_root = _resolve_path(base, paths_raw.get("output_root", "output"))
    feeds_csv = _resolve_path(base, paths_raw.get("feeds_csv", "feeds.csv"))

    topics_raw = raw.get("topics", [])
    if not topics_raw:
        raise ValueError(f"Config {config_path} defines no [[topics]] entries")

    topics: dict[str, Topic] = {}
    for t in topics_raw:
        if "name" not in t:
            raise ValueError(f"Topic entry missing required 'name' field: {t}")
        if "classifier_prompt" not in t:
            raise ValueError(f"Topic {t['name']!r} missing required 'classifier_prompt' field")
        kwargs = dict(
            name=t["name"],
            display_name=t.get("display_name", t["name"]),
            feed_category=t.get("feed_category", t["name"]),
            output_dir=t.get("output_dir", t["name"]),
            file_prefix=t.get("file_prefix", f"{t['name']}-radar"),
            classifier_prompt=t["classifier_prompt"],
            keywords=t.get("keywords", []),
            fail_open=t.get("fail_open", True),
        )
        for field_name in ("accent_color", "accent_color_light", "bg_top_color", "bg_mid_color"):
            if field_name in t:
                kwargs[field_name] = tuple(t[field_name])
        topic = Topic(**kwargs)
        topics[topic.name] = topic

    publish_hook = raw.get("publish", {}).get("hook")

    return Config(
        config_path=config_path,
        repo_root=base,
        site=site,
        models=models,
        pipeline=pipeline,
        gmail=gmail,
        output_root=output_root,
        feeds_csv=feeds_csv,
        topics=topics,
        publish_hook=publish_hook,
    )
