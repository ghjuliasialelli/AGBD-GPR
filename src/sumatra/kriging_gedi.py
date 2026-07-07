"""
Kriging (GP residual correction) of a SINGLE-raster AGB prediction (e.g. the ESA CCI map),
evaluated against GEDI. Launched by kriging.sh when pred != "ours".

Thin entry point: it parses the CLI arguments and runs the shared pipeline in sumatra/pipeline.py.
See that module (and the README in this folder) for the full step-by-step description.

Run: configure and launch via kriging.sh (do not call the args by hand).
"""

from sumatra.common import parser
from sumatra.pipeline import run_kriging_for_map, set_seeds


if __name__ == '__main__':
    args = parser()
    set_seeds(args.seed)
    run_kriging_for_map(args, args.path_predictions, args.model_name, s2_tile=args.s2_tile)
