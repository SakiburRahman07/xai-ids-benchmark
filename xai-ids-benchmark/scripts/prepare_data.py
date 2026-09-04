#!/usr/bin/env python
"""Prepare datasets: download (or instruct manual placement) + cache splits.

Most IDS datasets require a manual download (UNB/UNSW host them behind a form).
This script checks for files under data_raw/<dataset>/ and, if missing, prints
the manual instructions from the config and exits non-zero.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_ids_benchmark.utils import load_configs
from xai_ids_benchmark.data.loaders import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--base", default=".")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    cfgs = load_configs(args.config_dir)
    ds_cfgs = cfgs["datasets"]["datasets"]
    keys = args.datasets or list(ds_cfgs.keys())
    ok = True
    for k in keys:
        try:
            df = load_dataset(k, ds_cfgs[k], base=args.base)
            print(f"[OK] {k}: {df.shape}")
        except FileNotFoundError as e:
            print(f"[MISS] {k}:\n{e}\n", file=sys.stderr)
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
