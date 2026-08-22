#!/usr/bin/env bash
# Run the summarization model comparison against the "ai" topic using the
# techradar project's config. See src/newsradar/model_compare/compare_models.py.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run -m newsradar.model_compare.compare_models \
    --config ../techradar/config/config.toml \
    --topic ai \
    --out src/newsradar/model_compare/compare_models_output.txt \
    "$@"
