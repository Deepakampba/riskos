# src/riskos/defaults.py
from __future__ import annotations

DEFAULT_PD_COL = "PD_T"
DEFAULT_QUARTER_COL = "as_of_quarter"
DEFAULT_GROUP_COL = "Sector"
DEFAULT_EXPOSURE_COL = "Exposure"

DEFAULT_SLOPE_TH = 0.0015
DEFAULT_SEED_PREV_PD_1 = 0.019
DEFAULT_SEED_PREV_PD_2 = 0.018

DEFAULT_TOP_N = 3