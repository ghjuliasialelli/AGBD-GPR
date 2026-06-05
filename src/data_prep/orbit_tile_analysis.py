"""
Analyse Sentinel-2 orbit coverage of MGRS tiles.

Outputs:
  1. For each tile, the number of orbits that cover it.
  2. A barplot of the distribution (how many tiles are covered by 1, 2, … orbits).
  3. For each tile, the minimum set of orbits needed for full coverage.

Usage:
    conda run -n agbd python orbit_tile_analysis.py
"""

from config import DATA_ROOT
import xml.etree.ElementTree as ET
from pathlib import Path
from itertools import combinations

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
DIR = Path(__file__).resolve().parent
KML_PATH = DIR / "S2A_relative_orbit_groundtrack_10Sec.kml"
SHP_PATH = Path(f"{DATA_ROOT}/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp")
OUT_DIR = DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

# Sentinel-2 half-swath width in metres (290 km / 2)
HALF_SWATH_M = 145_000

# Equal-area CRS for buffering
EA_CRS = "EPSG:6933"

# ── 1. Parse orbit ground tracks from KML ────────────────────────────────────
def parse_orbits(kml_path):
    """Return dict {orbit_number: MultiLineString} in EPSG:4326."""
    tree = ET.parse(kml_path)
    ns = "{http://www.opengis.net/kml/2.2}"
    orbits = {}
    for pm in tree.findall(f".//{ns}Placemark"):
        name = pm.find(f"{ns}name").text  # "RELATIVE ORBIT 1"
        orbit_num = int(name.split()[-1])
        lines = []
        for ls in pm.findall(f".//{ns}LineString"):
            coords_text = ls.find(f"{ns}coordinates").text.strip()
            coords = []
            for part in coords_text.split():
                lon, lat, *_ = part.split(",")
                coords.append((float(lon), float(lat)))
            if len(coords) >= 2:
                lines.append(LineString(coords))
        orbits[orbit_num] = unary_union(lines)
    return orbits


# ── 2. Build swath polygons by buffering ground tracks ───────────────────────
def build_swath_polygons(orbits):
    """Buffer each orbit ground track by the half-swath width.
    Returns a GeoDataFrame with columns [orbit, geometry] in EPSG:4326."""
    gdf = gpd.GeoDataFrame(
        {"orbit": list(orbits.keys())},
        geometry=list(orbits.values()),
        crs="EPSG:4326",
    )
    gdf_proj = gdf.to_crs(EA_CRS)
    gdf_proj["geometry"] = gdf_proj.geometry.buffer(HALF_SWATH_M)
    gdf_swath = gdf_proj.to_crs("EPSG:4326")
    # make_valid in case buffering introduced artefacts
    gdf_swath["geometry"] = gdf_swath.geometry.apply(make_valid)
    return gdf_swath


# ── 3. Load MGRS tiles ──────────────────────────────────────────────────────
def load_mgrs_tiles(shp_path):
    """Load MGRS tile geometries, clean Z coords, deduplicate, clip to S2 coverage."""

    def drop_z(geom):
        """Strip Z dimension from geometry."""
        if geom.geom_type == "Polygon":
            return Polygon([(x, y) for x, y, *_ in geom.exterior.coords])
        elif geom.geom_type == "MultiPolygon":
            polys = []
            for p in geom.geoms:
                polys.append(Polygon([(x, y) for x, y, *_ in p.exterior.coords]))
            return MultiPolygon(polys)
        return geom

    gdf = gpd.read_file(shp_path, engine="pyogrio")
    gdf = gdf.rename(columns={"Name": "tile"})
    # Drop Z and dissolve duplicates (some tiles appear twice for antimeridian)
    gdf["geometry"] = gdf.geometry.apply(drop_z)
    gdf = gdf.dissolve(by="tile").reset_index()
    # Clip to S2 latitude range (~-56 to 84)
    bbox = gpd.GeoDataFrame(
        geometry=[Polygon([(-180, -56), (180, -56), (180, 84), (-180, 84)])],
        crs="EPSG:4326",
    )
    gdf = gpd.overlay(gdf, bbox, how="intersection")
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    return gdf


# ── 4. Spatial join: which orbits cover which tiles ──────────────────────────
def find_orbit_tile_intersections(gdf_tiles, gdf_swaths):
    """Return a DataFrame with [tile, orbit] for all intersections."""
    joined = gpd.sjoin(gdf_tiles, gdf_swaths, how="inner", predicate="intersects")
    return joined[["tile", "orbit"]].reset_index(drop=True)


# ── 5. Minimum set of orbits for full coverage (greedy set cover) ────────────
def minimum_orbit_cover(tile_geom, orbit_geoms):
    """Find the smallest set of orbits that fully cover a tile.

    For small orbit counts (≤6), try all subsets by increasing size (exact).
    Otherwise fall back to greedy.
    """
    n = len(orbit_geoms)
    orbit_ids = list(orbit_geoms.keys())
    tile_area = tile_geom.area

    # Pre-compute intersections
    orbit_intersections = {}
    for oid in orbit_ids:
        inter = tile_geom.intersection(orbit_geoms[oid])
        if not inter.is_empty:
            orbit_intersections[oid] = inter

    if not orbit_intersections:
        return []

    # Check single orbits first
    for oid, inter in orbit_intersections.items():
        if inter.area / tile_area > 0.999:
            return [oid]

    # For small n, try exact (all subsets by increasing size)
    if n <= 6:
        for size in range(2, n + 1):
            for subset in combinations(orbit_intersections.keys(), size):
                union = unary_union([orbit_intersections[oid] for oid in subset])
                if union.area / tile_area > 0.999:
                    return list(subset)

    # Greedy fallback
    remaining = tile_geom
    selected = []
    available = dict(orbit_intersections)
    while remaining.area / tile_area > 0.001 and available:
        # Pick the orbit covering the most remaining area
        best_oid = max(available, key=lambda oid: remaining.intersection(available[oid]).area)
        selected.append(best_oid)
        remaining = remaining.difference(available[best_oid])
        remaining = make_valid(remaining)
        del available[best_oid]
    return selected


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Parsing orbit ground tracks from KML ...")
    orbits = parse_orbits(KML_PATH)
    print(f"  {len(orbits)} orbits parsed.")

    print("Building swath polygons (buffering by 145 km) ...")
    gdf_swaths = build_swath_polygons(orbits)
    print("  Done.")

    print("Loading MGRS tile geometries ...")
    gdf_tiles = load_mgrs_tiles(SHP_PATH)
    print(f"  {len(gdf_tiles)} tiles loaded.")

    # ── Q1 & Q2: orbit count per tile ────────────────────────────────────────
    print("Computing orbit-tile intersections (spatial join) ...")
    df_intersections = find_orbit_tile_intersections(gdf_tiles, gdf_swaths)
    print(f"  {len(df_intersections)} (tile, orbit) pairs found.")

    # Group: for each tile, list of orbits
    tile_orbits = df_intersections.groupby("tile")["orbit"].apply(list).to_dict()
    tile_orbit_count = {t: len(o) for t, o in tile_orbits.items()}

    # Save Q1 results
    df_q1 = pd.DataFrame([
        {"tile": t, "num_orbits": len(orbs), "orbits": sorted(orbs)}
        for t, orbs in tile_orbits.items()
    ]).sort_values("tile")
    df_q1.to_csv(OUT_DIR / "tile_orbit_counts.csv", index=False)
    print(f"  Saved per-tile orbit counts to {OUT_DIR / 'tile_orbit_counts.csv'}")

    # Q2: distribution barplot
    counts = pd.Series(tile_orbit_count).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax, edgecolor="black")
    ax.set_xlabel("Number of covering orbits")
    ax.set_ylabel("Number of MGRS tiles")
    ax.set_title("Distribution of Sentinel-2 orbit coverage per MGRS tile")
    for i, (x, y) in enumerate(zip(counts.index, counts.values)):
        ax.text(i, y + max(counts) * 0.01, str(y), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "orbit_coverage_distribution.png", dpi=150)
    print(f"  Saved barplot to {OUT_DIR / 'orbit_coverage_distribution.png'}")

    # ── Q3: minimum orbit set per tile ───────────────────────────────────────
    print("Computing minimum orbit set per tile ...")
    # Build a dict of swath geometries for quick lookup
    swath_geom = dict(zip(gdf_swaths["orbit"], gdf_swaths["geometry"]))
    tile_geom_dict = dict(zip(gdf_tiles["tile"], gdf_tiles["geometry"]))

    results_q3 = []
    for tile, orbs in tqdm(tile_orbits.items(), desc="Min cover"):
        t_geom = tile_geom_dict[tile]
        orbit_geoms = {o: swath_geom[o] for o in orbs}
        min_set = minimum_orbit_cover(t_geom, orbit_geoms)
        results_q3.append({
            "tile": tile,
            "min_orbits_needed": len(min_set),
            "min_orbit_set": sorted(min_set),
        })

    df_q3 = pd.DataFrame(results_q3).sort_values("tile")
    df_q3.to_csv(OUT_DIR / "tile_min_orbit_cover.csv", index=False)
    print(f"  Saved minimum orbit sets to {OUT_DIR / 'tile_min_orbit_cover.csv'}")

    # Summary
    print("\n=== Summary ===")
    print(f"Total tiles: {len(gdf_tiles)}")
    print(f"Tiles with orbit coverage: {len(tile_orbits)}")
    print(f"\nOrbit count distribution:")
    for n_orb, n_tiles in counts.items():
        print(f"  {n_orb} orbit(s): {n_tiles} tiles")
    min_cover_dist = df_q3["min_orbits_needed"].value_counts().sort_index()
    print(f"\nMinimum orbits needed distribution:")
    for n_orb, n_tiles in min_cover_dist.items():
        print(f"  {n_orb} orbit(s): {n_tiles} tiles")


if __name__ == "__main__":
    main()
