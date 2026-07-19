# Setup

This guide covers credentials, local model installation, and what's kept out
of git. It assumes you've already copied `examples/config.example.toml` to
`config.toml` (see the README quickstart) and edited the `[site]`, `[paths]`,
and `[[topics]]` sections for your own deployment.

## Gmail (optional — only needed if you want newsletters mixed into the digest)

news-radar can pull recent emails from a Gmail inbox via the Gmail API (OAuth2,
read-only scope: `https://www.googleapis.com/auth/gmail.readonly`). If you
only want RSS feeds, skip this section and run with `--no-email`.

There are two ways to provide OAuth client credentials — `newsradar.email_fetcher`
tries them in this order:

**Option A — client ID / secret via environment variables.** Create an OAuth
"Desktop app" client in the [Google Cloud Console](https://console.cloud.google.com/apis/credentials),
then set:

```bash
export GMAIL_CLIENT_ID="your-client-id.apps.googleusercontent.com"
export GMAIL_CLIENT_SECRET="your-client-secret"
```

(or put these in a `.env` file next to your `config.toml` — `load_config()`
reads `.env` automatically via `python-dotenv`.)

**Option B — a `credentials.json` file.** Download the OAuth client JSON from
the Cloud Console and point `credentials_path` at it in `config.toml`:

```toml
[gmail]
credentials_path = "secrets/credentials.json"
```

Either way, on first run a browser window opens for the Google consent flow.
The resulting token is cached at the path configured by `[gmail] token_path`
(default: `~/.config/newsradar/token.json`, resolved relative to your config
file if not absolute). Subsequent runs load and silently auto-refresh that
token — no browser interaction needed until the refresh token itself expires
or is revoked.

To force re-authentication (e.g. after changing scopes or revoking access):

```bash
newsradar --config config.toml --refresh-token
```

This deletes the cached token and re-runs the OAuth flow, then exits.

## Ollama (required)

news-radar's classification, summarization, tagging, deduplication, and
ranking steps run against a local [Ollama](https://ollama.com) daemon by
default — no API keys, no per-request cost, everything stays on your machine.

1. [Install Ollama](https://ollama.com/download) for your platform.
2. Pull whatever models you reference in `config.toml`'s `[models]` section:

   ```bash
   ollama pull llama3.2       # summarize_model / dedup_model example
   ollama pull qwen3.5:9b     # rank_model example
   ```

3. Install the packaged ad-detector model (a fine-tuned Modelfile shipped
   with the package — see `src/newsradar/ad_detector/AD_DETECTION.md` for how
   it works and how to retrain it):

   ```bash
   newsradar ad-detector install
   ```

4. Make sure the daemon is running (`ollama serve`, or just launch the Ollama
   app) before running the pipeline.

Tune `[pipeline] llm_workers` in `config.toml` to match Ollama's
`OLLAMA_NUM_PARALLEL` setting — more workers than Ollama can run concurrently
just queues requests without speeding anything up.

### Model names: `provider/model`

Every `[models]` value (`summarize_model`, `rank_model`, `dedup_model`,
`ad_detector_model`) accepts an optional `provider/` prefix — e.g.
`"ollama/llama3.2:3b"` or `"anthropic/claude-haiku-4-5"`. A bare name with no
recognized provider prefix (`"llama3.2:3b"`, or an Ollama registry path like
`"hf.co/user/model"`) is treated as Ollama — Ollama is the default provider,
so existing bare-name configs keep working unchanged.

#### `anthropic/*` models

If you set any model to `"anthropic/<model>"` (e.g.
`dedup_model = "anthropic/claude-haiku-4-5"`), `newsradar.llm._chat_anthropic`
does **not** call the Anthropic API directly and does **not** read
`ANTHROPIC_API_KEY` itself. Instead it shells out to a local `claude` CLI
binary:

```python
claude_bin = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
subprocess.run([claude_bin, "-p", prompt, "--model", model, "--output-format", "text"], ...)
```

So to use this path you need the [Claude Code CLI](https://claude.com/claude-code)
installed and authenticated on the machine running news-radar (`claude` on
your `PATH`, or present at `~/.local/bin/claude`) — however *that* CLI is
configured to authenticate (subscription login or `ANTHROPIC_API_KEY` in its
own environment) is between you and it; news-radar itself never touches an
Anthropic API key or SDK. This is available to *any* pipeline stage, not just
dedup — `summarize_model = "anthropic/claude-haiku-4-5"` works the same way.
If you'd rather call the Anthropic API directly instead of shelling out to the
CLI, the `dedup-anthropic` extra (`pip install news-radar[dedup-anthropic]`)
pulls in the `anthropic` SDK as a dependency, but wiring it in (a new
`_chat_anthropic` implementation) is left to you.

## Kokoro TTS (required for podcast audio; skip with `--no-podcast`)

Podcast generation uses [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M),
an open-weight local TTS model, via the `kokoro` PyPI package (already a
project dependency). The first call to `KPipeline(...)` in
`newsradar.podcast_generator` downloads the model weights from Hugging Face
Hub automatically and caches them locally — no separate install step, just
make sure the machine has network access the first time you generate a
podcast. `ffmpeg` must also be on `PATH` (used to concatenate synthesized WAV
segments into the final MP3).

If you don't want audio at all, pass `--no-podcast` and skip this section
entirely.

## What's gitignored, and why

`.gitignore` excludes:

| Path | Why |
|---|---|
| `token.json` | Cached Gmail OAuth token — grants read access to the configured inbox. |
| `credentials.json` | Gmail OAuth client secret file, if you use Option B above. |
| `.env` | Where `GMAIL_CLIENT_ID` / `GMAIL_CLIENT_SECRET` / any other secrets live for Option A and other env-var overrides. |

`config.toml` itself is **not** gitignored — by design, nothing secret is
ever read from it (see the header comment in `config.py`), so it's meant to
be committed alongside your feeds CSV and published output. Keep it that way:
don't paste real client secrets or tokens into `config.toml`.
