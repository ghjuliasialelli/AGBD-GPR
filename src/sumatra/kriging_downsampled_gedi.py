"""
Kriging (GP residual correction) of OUR downsampled merged prediction -- a single 2-band GeoTIFF
(band 1 = AGB, band 2 = STD), e.g. merged_downsampled-100m_composite.tif. Evaluated against GEDI.
Launched by kriging.sh when pred="ours" and downsampled="true".

Functionally identical to kriging_gedi.py (both correct a single raster); only the input raster
differs, and kriging.sh passes it via --path_predictions / --model_name. Both call the shared
pipeline in sumatra/pipeline.py.

Run: configure and launch via kriging.sh (do not call the args by hand).
"""

from sumatra.common import parser
from sumatra.pipeline import run_kriging_for_map, set_seeds


if __name__ == '__main__':
    args = parser()
    set_seeds(args.seed)
    run_kriging_for_map(args, args.path_predictions, args.model_name, s2_tile=args.s2_tile)
