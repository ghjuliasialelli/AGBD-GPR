#!/bin/bash
# SOURCE this in a job: unpacks the packed uv env (scripts/env/pack_env.sh) to node-local disk and activates it.
#   ENV_TAR=/path/to/agbd_gpr_env.tar source scripts/env/activate_env.sh <repo_root>
# Falls back to <repo_root>/.venv if ENV_TAR is set to the empty string.
_repo=${1:?usage: source activate_env.sh <repo_root>}
ENV_TAR=${ENV_TAR-/cluster/scratch/$USER/envs/agbd_gpr_env.tar}
if [[ -n "$ENV_TAR" ]]; then
    [[ -f "$ENV_TAR" ]] || { echo "ERROR: $ENV_TAR missing: run 'bash scripts/env/pack_env.sh'"; return 1; }
    [[ "$ENV_TAR" -nt "$_repo/uv.lock" ]] || { echo "ERROR: $ENV_TAR is older than uv.lock: re-run pack_env.sh"; return 1; }
    _t=$(date +%s); _E=${TMPDIR:-/tmp}/agbd_gpr_env_$$; mkdir -p "$_E"; tar -xf "$ENV_TAR" -C "$_E"
    _B=$_E/$(cat "$_E/.venv/.base_python_dir")
    _PY=$(basename "$(readlink "$_E/.venv/bin/python")")
    ln -sfn "$_B/bin/$_PY" "$_E/.venv/bin/python"                         # re-point the venv at the unpacked Python
    sed -i "s|^home = .*|home = $_B/bin|" "$_E/.venv/pyvenv.cfg"
    export VIRTUAL_ENV=$_E/.venv PATH=$_E/.venv/bin:$PATH; unset PYTHONHOME
    python -c "import sys; assert sys.prefix == '$_E/.venv' and sys.base_prefix == '$_B', (sys.prefix, sys.base_prefix)" || return 1
    echo "ENV unpacked $(du -sh "$_E" | cut -f1) to $_E in $(( $(date +%s) - _t )) s"
else
    source "$_repo/.venv/bin/activate"
fi
python -c "import torch, gpytorch, numpy; print('ENV python', __import__('sys').version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpytorch', gpytorch.__version__, 'numpy', numpy.__version__)"
