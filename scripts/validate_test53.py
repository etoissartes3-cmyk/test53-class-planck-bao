#!/usr/bin/env python3
"""Deterministic smoke checks for the patched Test 53 CLASS build."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def run_class(executable: Path, class_dir: Path, ini_name: str) -> None:
    ini = ROOT / "config" / ini_name
    proc = subprocess.run(
        [str(executable), str(ini)],
        cwd=class_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode:
        print(proc.stdout)
        raise SystemExit(f"CLASS failed for {ini_name}")


def check_planck(planck_dir: Path) -> None:
    sys.path.insert(0, str(planck_dir))
    previous = Path.cwd()
    try:
        os.chdir(planck_dir)
        from planck_lite_py import PlanckLitePy

        likelihood = PlanckLitePy(
            data_directory="data",
            year=2018,
            spectra="TTTEEE",
            use_low_ell_bins=False,
        )
    finally:
        os.chdir(previous)
    if len(likelihood.X_data) != 613:
        raise SystemExit(f"Expected 613 Plik-lite bins, got {len(likelihood.X_data)}")


def check_bao() -> None:
    bao = np.genfromtxt(
        ROOT / "data" / "desi_dr2_bao13.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    if len(bao) != 13:
        raise SystemExit(f"Expected 13 DESI DR2 BAO entries, got {len(bao)}")
    sigma = np.asarray(bao["sigma"], dtype=float)
    covariance = np.diag(sigma**2)
    for i in range(1, 13, 2):
        corr = float(bao["r_MH"][i])
        covariance[i, i + 1] = covariance[i + 1, i] = corr * sigma[i] * sigma[i + 1]
    np.linalg.cholesky(covariance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-dir", type=Path, required=True)
    parser.add_argument("--planck-dir", type=Path, required=True)
    args = parser.parse_args()

    class_dir = args.class_dir.resolve()
    executable = class_dir / "class"
    if not executable.is_file():
        raise SystemExit(f"Missing CLASS executable: {executable}")

    output = class_dir / "output"
    output.mkdir(exist_ok=True)
    for ini in (
        "test53_lcdm_reference.ini",
        "test53_f0_limit.ini",
        "test53_geodesic_bestfit.ini",
    ):
        run_class(executable, class_dir, ini)

    lcdm = np.loadtxt(output / "test53_lcdm_reference_cl_lensed.dat")
    fzero = np.loadtxt(output / "test53_f0_limit_cl_lensed.dat")
    if lcdm.shape != fzero.shape or not np.array_equal(lcdm[:, 0], fzero[:, 0]):
        raise SystemExit("LCDM and f_dyn=0 spectrum grids differ")
    relative = np.abs(
        (fzero[:, 1:] - lcdm[:, 1:]) / np.maximum(np.abs(lcdm[:, 1:]), 1e-30)
    )
    maximum = float(np.nanmax(relative))
    if maximum > 1e-4:
        raise SystemExit(f"f_dyn=0 limit failed: max relative difference={maximum:.6e}")

    check_planck(args.planck_dir.resolve())
    check_bao()
    print(f"PASS: f_dyn=0 vs LCDM max relative difference={maximum:.6e}")
    print("PASS: Planck 2018 Plik-lite TTTEEE high-l has 613 bins")
    print("PASS: DESI DR2 BAO covariance is positive definite for 13 entries")


if __name__ == "__main__":
    main()

