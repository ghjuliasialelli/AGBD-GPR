#!/bin/bash
# Pack this project's uv environment (.venv + the uv-managed Python it points to) into ONE tar, so cluster
# jobs unpack it to node-local disk ($TMPDIR) instead of importing thousands of small files from the network
# filesystem. Re-run after every `uv sync`; jobs refuse a tar older than uv.lock.
#
#   uv sync --extra notebook && bash scripts/env/pack_env.sh [out.tar]      # from the repo root
#
# Jobs use it through scripts/env/activate_env.sh (sourced by src/kriging/kriging.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."                                             # repo root
OUT=${1:-${ENV_TAR:-/cluster/scratch/$USER/envs/agbd_gpr_env.tar}}
[[ -x .venv/bin/python ]] || { echo "ERROR: no .venv: run 'uv sync' first"; exit 1; }
PYREAL=$(readlink -f .venv/bin/python)                                   # .../cpython-3.12.x-linux-x86_64-gnu/bin/python3.12
BASE=$(dirname "$(dirname "$PYREAL")")
[[ -x "$BASE/bin/$(basename "$PYREAL")" ]] || { echo "ERROR: base python not found under $BASE"; exit 1; }
.venv/bin/python -W ignore -m compileall -q -j 8 .venv/lib > /dev/null 2>&1 || true   # no .pyc compilation in jobs
echo "$(basename "$BASE")" > .venv/.base_python_dir
mkdir -p "$(dirname "$OUT")"
tar -cf "$OUT.part" -C "$(dirname "$BASE")" "$(basename "$BASE")" -C "$PWD" .venv
mv "$OUT.part" "$OUT"
echo "packed $(du -h "$OUT" | cut -f1) -> $OUT (base python: $BASE)"
