#!/usr/bin/env bash
# Run inside nvcr.io/nvidia/pytorch:26.01-py3 with the project's dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m tools.run_slackpipe_nemotron_h8b_pp4 env "$@"
