# `sumatra/data/` — self-contained data bundle

Everything `get_results.ipynb` needs to run, so the analysis is reproducible from a fresh clone
with no external data and no `AGBD_ENV`. All paths resolve through `sumatra/paths.py`, which
prefers these files and falls back to the full `DATA_ROOT` tree when a file isn't here.

```
inputs/    the maps + reference the notebook opens directly
  merged_downsampled-100m_composite.tif   our downsampled 100 m AGB prediction (pre-kriging)
  CCI_N00E100.tif                         ESA CCI AGB map over Sumatra (pre-kriging)
  GEDI_L4B_AGBD_Sumatra.tif               GEDI L4B 1 km gridded AGB (1 km agreement check)
  L4A_Sumatra.gpkg                        GEDI L4A footprints (kriging reference + metrics)
  agbd_100m.tif / agbd_500m.tif / agbd_1000m.tif   GEDI-derived reference AGB rasters

outputs/   the kriging results the notebook reads back
  kriging-sumatra_gedi_composite-gedi_50.tif             our 10 m map, corrected
  kriging-sumatra_downsampled_gedi_composite-gedi_50.tif our downsampled map, corrected
  kriging-sumatra_cci_gedi-gedi_50.tif                   ESA CCI map, corrected
  splits-sumatra_downsampled_gedi_composite_..._-gedi.pkl  test-footprint split (ours)
  splits-sumatra_cci_gedi_..._-gedi.pkl                    test-footprint split (CCI)
```

Corrected GeoTIFFs have three bands: (1) AGB, (2) residuals, (3) kriging STD.

**Regenerating the outputs.** These are produced by `kriging.sh` on the processing machine (they
need the full-resolution S2 imagery + DEM, which are too large to bundle). To refresh a bundled
output after a new run, copy it from `DATA_ROOT/EcosystemAnalysis/Models/Biomes/Sumatra/` back
into `outputs/` under the same name.

## Original data sources

The files here are redistributed copies provided so the notebook is self-contained. The originals:

- **ESA CCI** biomass map (`CCI_N00E100.tif`) — ESA CCI Biomass product,
  <https://catalogue.ceda.ac.uk/uuid/6429d1aafe1e43b9b414e4a5a7f8b903/>
- **Sumatra reference** AGB rasters (`agbd_{100,500,1000}m.tif`) — from Paul B. May (2024),
  *Remote Sensing of Environment*,
  <https://www.sciencedirect.com/science/article/pii/S0034425724004103>; the reference prediction
  rasters were obtained by contacting the author.
- **GEDI** L4A footprints (`L4A_Sumatra.gpkg`) and L4B 1 km grid (`GEDI_L4B_AGBD_Sumatra.tif`) —
  NASA GEDI mission products.
- `merged_downsampled-100m_composite.tif` and the `kriging-*` / `splits-*` files are produced by
  this repository (BioFiLM inference + the Sumatra kriging pipeline).
