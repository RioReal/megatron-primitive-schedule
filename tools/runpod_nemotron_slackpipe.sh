#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m tools.slackpipe_hybrid slackpipe "$@"
