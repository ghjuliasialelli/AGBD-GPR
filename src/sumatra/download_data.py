"""Fetch the Sumatra data (inputs + kriging outputs read by get_results.ipynb) into sumatra/data/.

The files are too large for GitHub. They are in the `Sumatra/` folder of the ETH Research Collection
record, flat (no subfolders). This script puts each file where `paths.py` expects it (data/inputs/ or
data/outputs/) and checks it against `data/MANIFEST.json` (in git: size and md5 of every file).

    python src/sumatra/download_data.py --from ~/Downloads/Sumatra       # the downloaded folder ...
    python src/sumatra/download_data.py --from ~/Downloads/Sumatra.zip   # ... or a zip of it
    python src/sumatra/download_data.py --url URL                        # download a zip of the folder first
    python src/sumatra/download_data.py --check                          # only check the files already here

Maintainers: after changing a data file, rebuild the manifest and a flat folder to upload as `Sumatra/`:
    python src/sumatra/download_data.py --stage /path/to/Sumatra

Standard library only, so it runs before any environment is set up.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from os.path import abspath, basename, dirname, getsize, isdir, isfile, join, relpath

DATA = join(dirname(abspath(__file__)), "data")
MANIFEST = join(DATA, "MANIFEST.json")
SUBDIRS = ("inputs", "outputs")  # where paths.py looks, relative to DATA


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)["files"]


def check():
    """Return the problems with the local files (empty list if all match the manifest)."""
    problems = []
    for rel, meta in load_manifest().items():
        path = join(DATA, rel)
        if not isfile(path):
            problems.append(f"missing: {rel}")
        elif getsize(path) != meta["bytes"]:
            problems.append(f"wrong size: {rel} ({getsize(path)} bytes, expected {meta['bytes']})")
        elif md5(path) != meta["md5"]:
            problems.append(f"wrong md5: {rel} (an older or corrupted copy?)")
    return problems


def install(source):
    """Copy the files of a downloaded Sumatra/ folder (or a zip of it) into data/inputs|outputs, matched by name."""
    by_name = {basename(rel): rel for rel in load_manifest()}
    assert len(by_name) == len(load_manifest()), "file names in the manifest must be unique"
    with tempfile.TemporaryDirectory() as tmp:
        if isfile(source) and zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as z:
                for n in z.namelist():
                    if not n.endswith("/") and basename(n) in by_name:
                        with z.open(n) as src, open(join(tmp, basename(n)), "wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 20)
            source = tmp
        if not isdir(source):
            sys.exit(f"{source} is neither a folder nor a zip file")
        found = {}
        for root, _, names in os.walk(source):
            for n in names:
                if n in by_name:
                    found[n] = join(root, n)
                else:
                    print(f"not used by the notebook, skipped: {n}")
        for n, path in sorted(found.items()):
            dst = join(DATA, by_name[n])
            os.makedirs(dirname(dst), exist_ok=True)
            shutil.copyfile(path, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        print(f"copied {len(found)} of {len(by_name)} files into {DATA}")


def stage(out):
    files = sorted(relpath(join(r, n), DATA) for sub in SUBDIRS for r, _, ns in os.walk(join(DATA, sub)) for n in ns)
    assert files, f"no files under {DATA}/{{{','.join(SUBDIRS)}}}"
    names = [basename(f) for f in files]
    assert len(set(names)) == len(names), "two files share a name; the flat Sumatra/ folder needs unique names"
    manifest = {"note": "Files read by get_results.ipynb, relative to src/sumatra/data/. On the ETH Research "
                        "Collection they sit flat in Sumatra/. Written by download_data.py --stage; do not edit by hand.",
                "files": {rel: {"bytes": getsize(join(DATA, rel)), "md5": md5(join(DATA, rel))} for rel in files}}
    os.makedirs(out, exist_ok=True)
    for rel in files:
        shutil.copyfile(join(DATA, rel), join(out, basename(rel)))
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    total = sum(m["bytes"] for m in manifest["files"].values())
    print(f"staged {len(files)} files ({total / 1e6:.0f} MB) flat in {out}; manifest written to {MANIFEST}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="source", metavar="PATH", help="downloaded Sumatra/ folder, or a zip of it")
    src.add_argument("--url", help="download a zip of the Sumatra/ folder from this link, then install it")
    src.add_argument("--check", action="store_true", help="only check the files already in data/")
    src.add_argument("--stage", metavar="DIR", help="maintainers: refresh the manifest and copy the files flat into DIR")
    args = ap.parse_args()

    if args.stage:
        return stage(abspath(args.stage))
    if args.url:
        with tempfile.TemporaryDirectory() as tmp:
            zpath = join(tmp, "Sumatra.zip")
            print(f"downloading {args.url}")
            with urllib.request.urlopen(args.url) as r, open(zpath, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            install(zpath)
    elif args.source:
        install(abspath(os.path.expanduser(args.source)))
    problems = check()
    for p in problems:
        print(p)
    if problems:
        sys.exit(f"{len(problems)} problem(s) with the files in {DATA}")
    print(f"all {len(load_manifest())} files in {DATA} match the manifest")


if __name__ == "__main__":
    main()
