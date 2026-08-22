"""LLM model comparison tooling: summarization quality/latency benchmarking across Ollama models.

Note: `compare_models` is intentionally not imported here — it depends on
`newsradar.article_fetcher`, a heavier optional dependency chain (RSS +
full-text fetching), and running it as `python -m newsradar.model_compare.compare_models`
would otherwise re-import it mid-execution (RuntimeWarning). Import it directly
from `newsradar.model_compare.compare_models` when needed.
"""
