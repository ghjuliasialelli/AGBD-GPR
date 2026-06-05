"""
Compose 6 AGB density maps into a 2x3 panel figure.

Usage:
    Provide each panel as a pre-rendered image (PNG/TIFF).
    Adjust INPUT_PATHS below, then run:
        python compose_figure.py
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from matplotlib.colors import LogNorm


if __name__ == "__main__":

    # Set this to control which set of panels to compose
    TO_PLOT = 'density' # 'maps' or 'zoomed maps' or 'residuals' or 'density'

    # ---------------------------------------------------------------------------
    # Panel composition for map comparison
    # ---------------------------------------------------------------------------
    if 'maps' in TO_PLOT :

        if 'zoom' in TO_PLOT : ZOOM = True
        suffix = '-zoomed' if ZOOM else ''

        INPUT_PATHS = {
            "ours_basemap":   f"figs/ours_basemap{suffix}.png",
            "ours_10m":       f"figs/ours_post10m{suffix}.png",
            "ours_100m":      f"figs/ours_post100m{suffix}.png",
            "esa_basemap":    f"figs/esa_basemap{suffix}.png",
            "esa_postkriging": f"figs/esa_post{suffix}.png",
            "reference":      f"figs/reference{suffix}.png",
        }

        # Colorbar range (match your AGB density scale)
        VMIN, VMAX = 0, 250
        CMAP = "viridis"
        CBAR_LABEL = "AGB Density [t/ha]"

        # Output
        OUTPUT_PATH = f"figs/figure_maps_comparison{suffix}.pdf"  # PDF for vector; also saves a PNG
        DPI = 300

        # Layout
        #   Column 0: Basemap    Column 1: Post-kriging (10 m)   Column 2: Post-kriging (100 m)
        #   Row 0: Ours          Ours basemap   Ours 10 m         Ours 100 m
        #   Row 1: ESA CCI       ESA basemap    ESA post-kriging   Reference

        ROW_LABELS = ["Ours", "ESA CCI"]
        COL_LABELS = ["Basemap", "Post-kriging (10 m)", "Post-kriging (100 m)"]

        PANEL_ORDER = [
            # row 0
            ["ours_basemap", "ours_10m", "ours_100m"],
            # row 1
            ["esa_basemap", "esa_postkriging", "reference"],
        ]

        # Label override for bottom-right panel
        REFERENCE_POS = (1, 2)  # (row, col) of the reference panel


        def load_panel(path: str):
            p = Path(path)
            if not p.exists():
                return None
            return mpimg.imread(str(p))


        def make_figure():
            nrows, ncols = 2, 3

            # Width ratios: 3 equal map columns + thin colorbar column
            fig = plt.figure(figsize=(14, 7.5))
            gs = gridspec.GridSpec(
                nrows, ncols + 1,
                width_ratios=[1, 1, 1, 0.04],
                wspace=0.08,
                hspace=0.15,
                left=0.07, right=0.93, top=0.90, bottom=0.05,
            )

            axes = []
            for r in range(nrows):
                row_axes = []
                for c in range(ncols):
                    ax = fig.add_subplot(gs[r, c])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    key = PANEL_ORDER[r][c]
                    img = load_panel(INPUT_PATHS[key])

                    if img is not None:
                        ax.imshow(img)
                    else:
                        ax.text(
                            0.5, 0.5, f"[{key}]\nnot found",
                            ha="center", va="center", fontsize=9,
                            color="0.5", transform=ax.transAxes,
                        )
                        ax.set_facecolor("0.95")

                    # Column headers (top row only)
                    if r == 0:
                        ax.set_title(COL_LABELS[c], fontsize=11, fontweight="bold", pad=8)

                    # Row labels (left column only)
                    if c == 0:
                        ax.set_ylabel(
                            ROW_LABELS[r], fontsize=11, fontweight="bold",
                            rotation=90, labelpad=12,
                        )

                    # Mark reference panel distinctly
                    if (r, c) == REFERENCE_POS:
                        for spine in ax.spines.values():
                            spine.set_edgecolor("0.3")
                            spine.set_linewidth(1.5)
                            spine.set_linestyle("--")
                        # Override column header for this cell
                        ax.set_title("Reference", fontsize=11, fontstyle="italic", pad=8)

                    row_axes.append(ax)
                axes.append(row_axes)

            # Shared colorbar
            cbar_ax = fig.add_subplot(gs[:, ncols])
            norm = Normalize(vmin=VMIN, vmax=VMAX)
            sm = ScalarMappable(cmap=CMAP, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(CBAR_LABEL, fontsize=10)
            cbar.ax.tick_params(labelsize=9)

            # Save vector (PDF) + raster (PNG)
            fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
            print(f"Saved  {OUTPUT_PATH}")

            png_path = OUTPUT_PATH.replace(".pdf", ".png")
            fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
            print(f"Saved  {png_path}")

            # Also save a TIFF if the journal wants it
            #fig.savefig(tiff_path, dpi=DPI, bbox_inches="tight")

            plt.show()
        
        make_figure()

            
    elif TO_PLOT == 'zoomed maps':
        pass

    elif TO_PLOT == 'residuals':

        INPUT_PATHS = {
            "ours": "figs/ours-both-gedi-field-binned-residuals.png",
            "esa":  "figs/cci-gedi-field-binned-residuals.png",
        }

        DPI = 300
        OUTPUT_PATH = "figs/figure_residuals_comparison"


        def make_figure():
            fig = plt.figure(figsize=(14, 5.5))
            gs = gridspec.GridSpec(
                1, 2,
                width_ratios=[1, 1],
                wspace=0.05,
                left=0.02, right=0.98, top=0.88, bottom=0.12,
            )

            titles = ["Ours", "ESA CCI"]
            keys = ["ours", "esa"]

            for i, (key, title) in enumerate(zip(keys, titles)):
                ax = fig.add_subplot(gs[0, i])
                img = mpimg.imread(INPUT_PATHS[key])
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

            fig.savefig(f"{OUTPUT_PATH}.pdf", dpi=DPI, bbox_inches="tight")
            fig.savefig(f"{OUTPUT_PATH}.png", dpi=DPI, bbox_inches="tight")
            print("Saved")
            plt.show()
        
        make_figure()


    elif TO_PLOT == 'density':

        INPUT_PATHS = {
            "ours_basemap":     "figs/scatter_ours_basemap.png",
            "ours_10m":         "figs/scatter_ours_10m.png",
            "ours_100m":        "figs/scatter_ours_100m.png",
            "esa_basemap":      "figs/scatter_esa_basemap.png",
            "esa_postkriging":  "figs/scatter_esa_post.png",
        }

        PANEL_ORDER = [
            ["ours_basemap", "ours_10m", "ours_100m"],
            ["esa_basemap", "esa_postkriging", None],
        ]

        ROW_LABELS = ["Ours", "ESA CCI"]
        COL_LABELS = ["Basemap", "Post-kriging (10 m)", "Post-kriging (100 m)"]

        CMAP = "Greens"
        VMIN, VMAX = 1, 1e4

        OUTPUT_PATH = "figs/figure_scatter_comparison"
        DPI = 300

        def make_figure():
            nrows, ncols = 2, 3

            fig = plt.figure(figsize=(14, 7.5))
            gs = gridspec.GridSpec(
                nrows, ncols + 1,
                width_ratios=[1, 1, 1, 0.04],
                wspace=0.15,
                hspace=0.20,
                left=0.07, right=0.93, top=0.90, bottom=0.05,
            )

            for r in range(nrows):
                for c in range(ncols):
                    ax = fig.add_subplot(gs[r, c])
                    key = PANEL_ORDER[r][c]

                    if key is not None:
                        p = Path(INPUT_PATHS[key])
                        if p.exists():
                            ax.imshow(mpimg.imread(str(p)))
                        else:
                            ax.text(0.5, 0.5, f"[{key}]\nnot found",
                                    ha="center", va="center", fontsize=9,
                                    color="0.5", transform=ax.transAxes)
                            ax.set_facecolor("0.95")
                    else:
                        ax.set_visible(False)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    if r == 0:
                        ax.set_title(COL_LABELS[c], fontsize=11, fontweight="bold", pad=8)
                    if c == 0:
                        ax.set_ylabel(ROW_LABELS[r], fontsize=11, fontweight="bold",
                                    rotation=90, labelpad=12)

            # Shared colorbar
            cbar_ax = fig.add_subplot(gs[:, ncols])
            norm = LogNorm(vmin=VMIN, vmax=VMAX)
            sm = ScalarMappable(cmap=CMAP, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label("Number of samples", fontsize=10)

            fig.text(0.48, 0.01, "Reference AGBD [Mg/ha]", ha="center", fontsize=12)
            fig.text(0.01, 0.5, "Predicted AGB [Mg/ha]", va="center", rotation=90, fontsize=12)

            fig.savefig(f"{OUTPUT_PATH}.pdf", dpi=DPI, bbox_inches="tight")
            fig.savefig(f"{OUTPUT_PATH}.png", dpi=DPI, bbox_inches="tight")
            print("Saved")
            plt.show()


        make_figure()


    else:
        raise ValueError(f"Invalid TO_PLOT value: {TO_PLOT}")