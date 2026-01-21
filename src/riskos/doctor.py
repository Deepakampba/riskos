# src/riskos/doctor.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .validation import DataContract, quarter_key


def _try_quarter_parse(series: pd.Series) -> Dict[str, Any]:
    """Non-throwing quarter format check. Returns stats + examples."""
    s = series.astype(str)
    ok_mask = s.str.match(r"^\d{4}-Q[1-4]$")
    bad = s[~ok_mask].unique().tolist()
    return {
        "ok_pct": float(ok_mask.mean() * 100.0),
        "bad_examples": bad[:10],
        "n_bad": int((~ok_mask).sum()),
    }


def doctor_csv(
    path: Path,
    *,
    contract: DataContract = DataContract(),
    allow_pd_gt1: bool = False,
    head_rows: int = 5,
) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    result: Dict[str, Any] = {
        "path": str(path),
        "contract": asdict(contract),
        "rows": int(len(df)),
        "cols": df.columns.tolist(),
    }

    # ---- required columns ----
    req = [contract.quarter_col, contract.group_col, contract.pd_col, contract.exposure_col]
    missing = sorted(list(set(req) - set(df.columns)))
    result["missing_required_cols"] = missing

    if missing:
        # no point going deeper
        result["status"] = "FAIL"
        result["reason"] = f"Missing required columns: {missing}"
        return result

    qcol, gcol, pdcol, expcol = contract.quarter_col, contract.group_col, contract.pd_col, contract.exposure_col

    # ---- basic stats ----
    result["nulls"] = df[req].isna().sum().to_dict()
    result["n_groups"] = int(df[gcol].nunique(dropna=False))
    result["groups"] = df[gcol].dropna().astype(str).unique().tolist()[:20]

    # ---- quarter format ----
    qcheck = _try_quarter_parse(df[qcol])
    result["quarter_format"] = qcheck

    # ---- duplicates on (group, quarter) ----
    keys = [gcol, qcol]
    dup_mask = df.duplicated(keys, keep=False)
    n_dups = int(dup_mask.sum())
    result["dup_rows_on_group_quarter"] = n_dups
    if n_dups:
        result["dup_pct"] = float(n_dups / len(df) * 100.0) if len(df) else 0.0
        result["dup_examples"] = (
            df.loc[dup_mask, keys]
            .head(10)
            .to_dict(orient="records")
        )
    else:
        result["dup_pct"] = 0.0
        result["dup_examples"] = []

    # ---- numeric coercion checks ----
    pd_numeric = pd.to_numeric(df[pdcol], errors="coerce")
    exp_numeric = pd.to_numeric(df[expcol], errors="coerce")

    result["pd_non_numeric_rows"] = int(pd_numeric.isna().sum() - df[pdcol].isna().sum())
    result["exp_non_numeric_rows"] = int(exp_numeric.isna().sum() - df[expcol].isna().sum())

    # ---- ranges + bounds ----
    result["pd_min"] = float(np.nanmin(pd_numeric.to_numpy())) if len(pd_numeric) else float("nan")
    result["pd_max"] = float(np.nanmax(pd_numeric.to_numpy())) if len(pd_numeric) else float("nan")
    result["exp_min"] = float(np.nanmin(exp_numeric.to_numpy())) if len(exp_numeric) else float("nan")
    result["exp_max"] = float(np.nanmax(exp_numeric.to_numpy())) if len(exp_numeric) else float("nan")

    neg_pd = int((pd_numeric < 0).sum(skipna=True))
    neg_exp = int((exp_numeric < 0).sum(skipna=True))
    gt1_pd = int((pd_numeric > 1).sum(skipna=True))

    result["pd_negative_rows"] = neg_pd
    result["exposure_negative_rows"] = neg_exp
    result["pd_gt1_rows"] = gt1_pd

    # ---- preview ----
    result["head_preview"] = df.head(head_rows).to_dict(orient="records")

    # ---- status ----
    problems = []
    if any(v > 0 for v in result["nulls"].values()):
        problems.append("missing_values")
    if qcheck["n_bad"] > 0:
        problems.append("bad_quarter_format")
    if neg_pd > 0:
        problems.append("negative_pd")
    if neg_exp > 0:
        problems.append("negative_exposure")
    if (not allow_pd_gt1) and gt1_pd > 0:
        problems.append("pd_gt1")
    if result["pd_non_numeric_rows"] > 0:
        problems.append("pd_non_numeric")
    if result["exp_non_numeric_rows"] > 0:
        problems.append("exposure_non_numeric")

    result["problems"] = problems
    result["status"] = "OK" if len(problems) == 0 else "WARN"
    return result


def print_doctor_report(report: Dict[str, Any]) -> None:
    print("\nRiskOS Doctor")
    print("=============")
    print(f"File: {report.get('path')}")
    print(f"Rows: {report.get('rows')} | Cols: {len(report.get('cols', []))}")
    print(f"Status: {report.get('status')}")
    if report.get("missing_required_cols"):
        print(f"Missing required columns: {report['missing_required_cols']}")
        return

    c = report["contract"]
    print(f"Contract: quarter={c['quarter_col']} group={c['group_col']} pd={c['pd_col']} exposure={c['exposure_col']}")
    print(f"Groups: {report.get('n_groups')} | Sample: {report.get('groups', [])[:10]}")

    qf = report.get("quarter_format", {})
    print(f"Quarter format OK%: {qf.get('ok_pct', 0):.1f}% | bad={qf.get('n_bad', 0)} | examples={qf.get('bad_examples', [])}")

    print(f"Duplicates on (group, quarter): {report.get('dup_rows_on_group_quarter', 0)} rows ({report.get('dup_pct', 0):.2f}%)")

    print(
        "Ranges: "
        f"PD [{report.get('pd_min'):.6f}, {report.get('pd_max'):.6f}] | "
        f"Exposure [{report.get('exp_min'):.2f}, {report.get('exp_max'):.2f}]"
    )

    print(
        "Issues: "
        f"nulls={report.get('nulls')} | "
        f"pd_non_numeric={report.get('pd_non_numeric_rows')} | "
        f"exp_non_numeric={report.get('exp_non_numeric_rows')} | "
        f"pd<0={report.get('pd_negative_rows')} | "
        f"exp<0={report.get('exposure_negative_rows')} | "
        f"pd>1={report.get('pd_gt1_rows')}"
    )

    probs = report.get("problems", [])
    if probs:
        print(f"Problems detected: {probs}")
    else:
        print("Problems detected: none ✅")