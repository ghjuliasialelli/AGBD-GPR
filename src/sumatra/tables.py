"""
Reproduce the Sumatra comparison tables of the paper (Table 4 at 100 m; Table G.5 at 500 m / 1 km):
pre- and post-kriging RMSE / MAE / ME of each map against the independent Sumatra reference AGB.

Conventions (Revision 2):
  * every map is evaluated as a 2-D array on the EXACT reference grid of the given resolution;
    rasters that are not on it are warped (bilinear) with `warp_to_reference`;
  * coarse resolutions use a dedicated downsampled raster when one exists (our maps: averaged directly from
    the 10 m mosaic by merge_tiles.py), otherwise the 100 m array is block-averaged (`block_mean`);
  * metrics use the pixels valid in the reference, pre and post map ('own' mask, as in the paper), and
    optionally the pixels valid in ALL maps ('common' mask).
"""
import warnings

import numpy as np
import rasterio as rs
from rasterio.enums import Resampling
from rasterio.warp import reproject

from sumatra import paths


def reference(resolution=100):
    with rs.open(paths.reference_agb(resolution)) as r:
        return r.read(1).astype(np.float64), r.transform, r.crs


def band_index(src, name_or_idx):
    """1-based band index from a band description (e.g. 'STD') or an int."""
    if isinstance(name_or_idx, int): return name_or_idx
    assert name_or_idx in src.descriptions, f'{src.name}: no band {name_or_idx!r} in {src.descriptions}'
    return src.descriptions.index(name_or_idx) + 1


def warp_to_reference(path, band=1, resolution=100):
    """Read one band of `path` on the exact reference grid (identity if already on it)."""
    ref, T, crs = reference(resolution)
    with rs.open(path) as s:
        b = band_index(s, band)
        if s.crs == crs and s.shape == ref.shape and s.transform.almost_equals(T):
            a = s.read(b).astype(np.float64)
            if s.nodata is not None and np.isfinite(s.nodata): a[a == s.nodata] = np.nan
            return a
        # Near-identical grids (|pixel-size ratio - 1| < 1 %, e.g. our 100.006 m mosaics vs the 100 m reference):
        # nearest neighbour, i.e. the paper's index alignment without the up-to-16 m drift and WITHOUT the
        # smoothing a bilinear warp would add (which flatters RMSE). Genuinely different grids (CCI, 98.46 m):
        # bilinear, as in the kriging step (common.load_dense_preds).
        ratio = abs(s.transform.a) / abs(T.a)
        resampling = Resampling.nearest if abs(ratio - 1) < 0.01 else Resampling.bilinear
        out = np.full(ref.shape, np.nan, np.float32)
        reproject(rs.band(s, b), out, src_transform=s.transform, src_crs=s.crs,
                  src_nodata=s.nodata if s.nodata is not None else np.nan,
                  dst_transform=T, dst_crs=crs, dst_nodata=np.nan, resampling=resampling)
    return out.astype(np.float64)


def block_mean(x, k, min_valid=0.5):
    H, W = x.shape
    assert H % k == 0 and W % k == 0, (x.shape, k)
    b = x.reshape(H // k, k, W // k, k)
    valid = np.isfinite(b).sum(axis=(1, 3)) / (k * k)
    with warnings.catch_warnings():  # all-NaN blocks are expected at the study-area edge
        warnings.simplefilter('ignore', RuntimeWarning)
        m = np.nanmean(b, axis=(1, 3))
    m[valid < min_valid] = np.nan
    return m


def metrics(pred, ref, mask):
    d = pred[mask] - ref[mask]
    assert d.size > 0, 'empty evaluation mask'
    return dict(n=int(d.size), RMSE=float(np.sqrt(np.mean(d ** 2))), MAE=float(np.mean(np.abs(d))),
                ME=float(np.mean(d)), r=float(np.corrcoef(pred[mask], ref[mask])[0, 1]))


def load_map(spec, resolution):
    """`spec` maps a resolution to a (path, band) tuple; missing coarse resolutions are block-averaged from 100 m."""
    if resolution in spec:
        path, band = spec[resolution]
        return warp_to_reference(path, band, resolution)
    path, band = spec[100]
    return block_mean(warp_to_reference(path, band, 100), resolution // 100)


def sumatra_tables(maps, resolutions=(100, 500, 1000), masks=('own',)):
    """
    maps: {name: {'pre': {res: (path, band)}, 'post': {res: (path, band)}}}
    Returns {(res, name, mask): {'pre': metrics, 'post': metrics}}.
    """
    out = {}
    for res in resolutions:
        ref = reference(res)[0]
        arrs = {n: (load_map(m['pre'], res), load_map(m['post'], res)) for n, m in maps.items()}
        common = np.isfinite(ref)
        for pre, post in arrs.values(): common &= np.isfinite(pre) & np.isfinite(post)
        for n, (pre, post) in arrs.items():
            own = np.isfinite(ref) & np.isfinite(pre) & np.isfinite(post)
            for mk in masks:
                mask = own if mk == 'own' else common
                out[(res, n, mk)] = {'pre': metrics(pre, ref, mask), 'post': metrics(post, ref, mask)}
    return out


def print_tables(tables):
    print(f"{'res':>5} {'map':10} {'mask':6} {'n':>8} | {'RMSE pre':>8} {'post':>7} | {'MAE pre':>7} {'post':>7} | {'ME pre':>7} {'post':>7}")
    for (res, n, mk), v in tables.items():
        print(f"{res:>4}m {n:10} {mk:6} {v['pre']['n']:8d} | {v['pre']['RMSE']:8.2f} {v['post']['RMSE']:7.2f} | "
              f"{v['pre']['MAE']:7.2f} {v['post']['MAE']:7.2f} | {v['pre']['ME']:7.2f} {v['post']['ME']:7.2f}")
