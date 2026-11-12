"""
Merge per-tile Sumatra AGB rasters (10 m, two UTM zones) into one mosaic over the study area, then
average-downsample it onto the reference grids (100 m / 500 m / 1 km).

Ported from the legacy notebook `EcosystemAnalysis/Models/Biomes/Sumatra/get_dense_basemaps.ipynb`
(cells 1-16), which produced the inputs of Table 4 / Table G.5:
  - pre-kriging : <pred_dir>/<tile>_2021_composite.tif      (bands: AGB, STD)        -> merged*_composite.tif
  - post-kriging: <krig_dir>/<tile>_50_2500_dynamic.tif     (bands: AGB, Residuals, STD)
Steps (identical to the legacy code):
  1. crop a 10 px border, reproject to the reference CRS (EPSG:32748, nearest), mask/crop to the study-area
     polygon (the valid-data footprint of the reference raster);
  2. per band, weight every tile with a 2-D Gaussian centred on the (cropped) tile (sigma = extent / 5,
     10 px zero border), merge with method="sum" over the polygon bounds (target-aligned pixels), and divide
     the summed weighted values by the summed weights (feathered blending of the tile overlaps);
  3. read the 10 m mosaic with Resampling.average onto each reference raster's shape.
Revision-2 fix: the legacy merge took band index 1 as "STD"; for kriged tiles that is the RESIDUALS band,
so the legacy `merged*_50_2500_dynamic.tif` "STD" band holds residuals. Bands are now selected by NAME
(band descriptions) and written with descriptions.
Known legacy quirk kept for reproducibility: in step 2 the weight of a tile is summed even where that tile's
value is nodata (rasterio's per-band merge), which slightly down-weights overlap pixels where one tile is
nodata. Reported, not changed.

Usage (from src/):
  PYTHONPATH=. python sumatra/merge_tiles.py --input_dir <dir> --suffix _50_2500_dynamic --bands AGB STD Residuals \
      --out_dir <dir> --name merged_50_2500_dynamic --tmp_dir <dir>
"""
import argparse, gc, os
from os.path import join, isfile

import numpy as np
import rasterio as rs
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from shapely.geometry import shape

from sumatra import paths

TILES = ['47MRV', '48MTE', '48MUE', '47MRU', '48MTD', '48MUD', '47MRT', '48MTC', '48MUC']


def study_area():
    """Study-area polygon = the reference raster's valid-data footprint (legacy: `geoms[4]`)."""
    with rs.open(paths.reference_agb(100)) as src:
        crs, m = src.crs, src.dataset_mask()
        geoms = [shape(g) for g, v in shapes(m, transform=src.transform) if v > 0]
    geom = max(geoms, key=lambda g: g.area)
    return geom, crs


def crop_border(src, border=10):
    w = Window(border, border, src.width - 2 * border, src.height - 2 * border)
    meta = src.meta.copy(); meta.update(height=int(w.height), width=int(w.width), transform=src.window_transform(w))
    mf = MemoryFile()
    with mf.open(**meta) as dst: dst.write(src.read(window=w))
    return mf


def reproject_to(mf, target_crs):
    with mf.open() as src:
        T, W, H = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
        prof = src.profile.copy(); prof.update(crs=target_crs, transform=T, width=W, height=H)
        out = MemoryFile()
        with out.open(**prof) as dst:
            for i in range(1, src.count + 1):
                reproject(rs.band(src, i), rs.band(dst, i), src_transform=src.transform, src_crs=src.crs,
                          dst_transform=T, dst_crs=target_crs, resampling=Resampling.nearest)
    return out


def prepare_tile(path, geom, target_crs, out_path):
    """Steps 1: crop border, reproject, mask to the study area; written to disk (bands keep descriptions)."""
    with rs.open(path) as src:
        descriptions = src.descriptions
        mf = crop_border(src)
        reproj = src.crs != target_crs
    if reproj: mf = reproject_to(mf, target_crs)
    with mf.open() as src:
        data = src.read().astype(np.float32); data[data == -9999.0] = np.nan
        meta = dict(driver='GTiff', count=src.count, dtype='float32', width=src.width, height=src.height,
                    crs=target_crs, transform=src.transform, nodata=np.nan)
    tmp = MemoryFile()
    with tmp.open(**meta) as dst: dst.write(data)
    del data
    with tmp.open() as src:
        img, T = mask(src, shapes=[geom], crop=True)
        meta.update(height=img.shape[1], width=img.shape[2], transform=T)
    with rs.open(out_path, 'w', **meta) as dst:
        dst.write(img)
        for i, d in enumerate(descriptions): dst.set_band_description(i + 1, d or f'Band_{i + 1}')


def weighted(src, band_idx, border=10, div_factor=5):
    """Legacy `calculate_weights` (gaussian): returns a 2-band memfile (value*w, w)."""
    data = src.read(band_idx)
    h, w = data.shape
    x = np.linspace(-h // 2, h // 2, w); y = np.linspace(-w // 2, w // 2, h)
    x, y = np.meshgrid(x, y)
    sx, sy = (x.max() - x.min()) / div_factor, (y.max() - y.min()) / div_factor
    wts = 1. / (2. * np.pi * sx * sy) * np.exp(-(x ** 2 / (2. * sx ** 2) + y ** 2 / (2. * sy ** 2)))
    wts[:border, :] = 0; wts[-border:, :] = 0; wts[:, :border] = 0; wts[:, -border:] = 0
    meta = src.meta.copy(); meta.update(count=2)
    mf = MemoryFile()
    with mf.open(**meta) as dst:
        dst.write((data * wts).astype(meta['dtype']), 1); dst.write(wts.astype(meta['dtype']), 2)
    return mf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--suffix', required=True, help="e.g. '_50_2500_dynamic' or '_2021_composite'")
    ap.add_argument('--bands', nargs='+', required=True, help='band descriptions to merge, in output order')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--name', required=True, help='output stem, e.g. merged_50_2500_dynamic')
    ap.add_argument('--tmp_dir', required=True)
    ap.add_argument('--resolutions', nargs='+', type=int, default=[100, 500, 1000])
    ap.add_argument('--keep_10m', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.tmp_dir, exist_ok=True)

    geom, target_crs = study_area()
    prepared = []
    for t in TILES:
        src_path = join(a.input_dir, f'{t}{a.suffix}.tif')
        assert isfile(src_path), f'missing input {src_path}'
        out = join(a.tmp_dir, f'{t}{a.suffix}.tif')
        print('prepare', t, flush=True)
        prepare_tile(src_path, geom, target_crs, out)
        prepared.append(out)

    merged, T_out = {}, None
    for b in a.bands:
        print('merge band', b, flush=True)
        mfs, dss = [], []
        for p in prepared:
            with rs.open(p) as src:
                assert b in src.descriptions, f'{p} has no band {b!r}: {src.descriptions}'
                mf = weighted(src, src.descriptions.index(b) + 1)
            mfs.append(mf); dss.append(mf.open())
        arr, T = merge(dss, bounds=geom.bounds, nodata=np.nan, method='sum', target_aligned_pixels=True)
        for d in dss: d.close()
        for m in mfs: m.close()
        s, wsum = arr[0], arr[1]; del arr
        bad = np.isnan(s) | np.isnan(wsum) | (wsum == 0)
        out = np.full(s.shape, np.nan, np.float32)
        np.divide(s, wsum, out=out, where=~bad)
        merged[b] = out; T_out = T
        del s, wsum, bad; gc.collect()

    H, W = next(iter(merged.values())).shape
    meta10 = dict(driver='GTiff', height=H, width=W, count=len(a.bands), dtype='float32', crs=target_crs,
                  transform=T_out, nodata=np.nan, compress='deflate', tiled=True, BIGTIFF='YES')
    path10 = join(a.out_dir if a.keep_10m else a.tmp_dir, f'{a.name}.tif')
    with rs.open(path10, 'w', **meta10) as dst:
        for i, b in enumerate(a.bands):
            dst.write(merged[b], i + 1); dst.set_band_description(i + 1, b)
    del merged; gc.collect()
    print('10 m mosaic', path10, (H, W), flush=True)

    for res in a.resolutions:
        with rs.open(paths.reference_agb(res)) as r: ref_shape = r.shape
        with rs.open(path10) as src:
            data = src.read(out_shape=(src.count,) + ref_shape, resampling=Resampling.average)
            T = src.transform * src.transform.scale(src.width / data.shape[-1], src.height / data.shape[-2])
        meta = dict(driver='GTiff', height=ref_shape[0], width=ref_shape[1], count=len(a.bands), dtype='float32',
                    crs=target_crs, transform=T, nodata=np.nan)
        out = join(a.out_dir, f'{a.name.replace("merged", f"merged_downsampled-{res}m")}.tif')
        with rs.open(out, 'w', **meta) as dst:
            for i, b in enumerate(a.bands):
                dst.write(data[i].astype('float32'), i + 1); dst.set_band_description(i + 1, b)
        print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
