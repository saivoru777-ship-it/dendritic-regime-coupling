"""Dendritic regime coupling analysis."""

import sys
from pathlib import Path

# Ensure neurostat and input-clustering src are importable
_neurostat_path = str(Path.home() / "research" / "neurostat")
_ic_src_path = str(Path.home() / "research" / "neurostat-input-clustering" / "src")

if _neurostat_path not in sys.path:
    sys.path.insert(0, _neurostat_path)
if _ic_src_path not in sys.path:
    sys.path.insert(0, _ic_src_path)
