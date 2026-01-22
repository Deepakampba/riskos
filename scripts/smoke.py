#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def die(msg: str, code: int = 1) -> None:
    print(f"❌ {msg}")
    raise SystemExit(code)


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        die(f"Command failed (exit {p.returncode})", p.returncode)


def latest_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        die(f"Base output dir not found: {base_dir}")
    runs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        die(f"No run_* subdir found under: {base_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def read_json(path: Path) -> dict:
    if not path.exists():
        die(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(name: str, got: float, expected: float, tol: float) -> None:
    if abs(got - expected) > tol:
        die(f"{name} out of tolerance. got={got:.6f} expected={expected:.6f} tol={tol:.6f}")
    print(f"✅ {name} within tolerance (got={got:.6f}, expected={expected:.6f}, tol={tol:.6f})")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    golden = repo / "tests" / "golden" / "sample_multi_sector.csv"
    out_base = repo / "outputs" / "smoke" / "portfolio_ok"

    # 1) Run portfolio (happy path)
    run_cmd([
        "riskos", "run",
        "--portfolio",
        "--input", str(golden),
        "--out", str(out_base),
    ])

    run_dir = latest_run_dir(out_base)
    print(f"\nUsing run dir: {run_dir}")

    meta = read_json(run_dir / "meta.json")
    summary = read_json(run_dir / "summary.json")

    # 2) Assertions (stable)
    status = meta.get("status")
    if status != "SUCCESS":
        die(f"meta.status expected SUCCESS, got {status!r}")
    print("✅ meta.status == SUCCESS")

    portfolio = summary.get("portfolio")
    if not isinstance(portfolio, dict):
        die("summary.portfolio missing or not an object")

    n_groups = portfolio.get("n_groups")
    if n_groups != 4:
        die(f"portfolio.n_groups expected 4, got {n_groups!r}")
    print("✅ portfolio.n_groups == 4")

    ear_pct = float(portfolio.get("exposure_at_risk_pct", -999))
    # IMPORTANT: pick a tolerance that won't be flaky.
    # Since this is a fixed golden CSV, we can be tight.
    expected = 37.62459103689112
    tol = 1e-6
    assert_close("portfolio.exposure_at_risk_pct", ear_pct, expected, tol)

    print("\n🎉 JSON assertions passed.")


if __name__ == "__main__":
    main()