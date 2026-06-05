"""
One-time migration: embed the fit-time configuration into existing kriging checkpoints.

`predict.py` reads the configuration (matern_nu, aux, norm_*, ...) from the kriging
checkpoint (`{model_name}_{s2_tile}.pt`). Checkpoints produced before this was added do
not contain it; predict.py then falls back to querying Weights & Biases. Run this script
**once** to copy the config from W&B into each checkpoint, after which predict.py needs no
W&B access at all.

Usage:
    export WANDB_ENTITY=<your-entity>
    python tools/backfill_config.py --ckpt_dir /path/to/kriging/checkpoints [--project kriging] [--dry_run]

For every `*.pt` under --ckpt_dir that lacks an embedded 'config', the script derives the
run name from the filename (`{model_name}_{s2_tile}.pt`), fetches that run's config from
W&B, and rewrites the checkpoint with the config embedded.
"""

import argparse
import glob
import os

import torch

# Keys predict.py needs from the configuration.
CONFIG_KEYS = ["arch", "year", "ens_models", "aux", "extra_features", "matern_nu",
               "COMPUTE_VAR", "norm_res", "norm_coords", "coords", "pred_vals",
               "composites", "agb", "norm_aux"]


def build_config(run_config):
    """Pull the keys predict.py needs from a W&B run config (with sensible fallbacks)."""
    cfg = {}
    for k in CONFIG_KEYS:
        cfg[k] = run_config.get(k)
    # 'arch' was not always logged; only one architecture is used in this release.
    if cfg.get("arch") is None:
        cfg["arch"] = "nico_film"
    # ens_models may be logged as a list; predict.py expects the joined string.
    if isinstance(cfg.get("ens_models"), (list, tuple)):
        cfg["ens_models"] = "_".join(cfg["ens_models"])
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True, help="Directory (searched recursively) containing the *.pt kriging checkpoints.")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), help="W&B entity (defaults to $WANDB_ENTITY).")
    p.add_argument("--project", default="kriging", help="W&B project name (default: kriging).")
    p.add_argument("--dry_run", action="store_true", help="Report what would change without writing.")
    args = p.parse_args()

    if not args.entity:
        raise SystemExit("Set --entity or the WANDB_ENTITY environment variable.")

    import wandb
    api = wandb.Api()

    pts = sorted(glob.glob(os.path.join(args.ckpt_dir, "**", "*.pt"), recursive=True))
    print(f"Found {len(pts)} checkpoint(s) under {args.ckpt_dir}")
    done = skipped = failed = 0

    for pt in pts:
        data = torch.load(pt, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and data.get("config") is not None:
            skipped += 1
            continue
        # filename is "{model_name}_{s2_tile}.pt"; s2_tile is the part after the last '_'
        stem = os.path.splitext(os.path.basename(pt))[0]
        model_name = stem.rsplit("_", 1)[0]
        try:
            runs = api.runs(f"{args.entity}/{args.project}", {"display_name": model_name})
            if not runs:
                print(f"  ! no W&B run named '{model_name}' for {pt} -- skipped")
                failed += 1
                continue
            data["config"] = build_config(runs[0].config)
            if args.dry_run:
                print(f"  [dry-run] would embed config into {pt} (run '{model_name}')")
            else:
                torch.save(data, pt)
                print(f"  embedded config into {pt} (run '{model_name}')")
            done += 1
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  ! failed for {pt}: {e}")
            failed += 1

    print(f"\nDone. embedded={done} already-had-config={skipped} failed={failed}"
          + (" (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
