#!/usr/bin/env python
"""CLI entry for the XAI-IDS benchmark. Run from the repo root.

Examples:
    python scripts/run_benchmark.py --quick              # smoke run
    python scripts/run_benchmark.py                       # full factorial
    python scripts/run_benchmark.py --datasets cicids2017 --models cnn1d
"""
import sys
from pathlib import Path

# Make the src package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_ids_benchmark.runner import main

if __name__ == "__main__":
    main()
