"""Backward-compatible name for the old one-file script.\n\nUse `python -m scripts.run_eda` for the maintained entrypoint.\n"""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_eda import main

if __name__ == '__main__':
    main()
