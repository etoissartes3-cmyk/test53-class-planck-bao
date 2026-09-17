#!/usr/bin/env python3
"""MCMC posterior for Test 53 using the joint Planck Plik-lite + DESI BAO likelihood.

The seven cosmological parameters and the Planck calibration nuisance parameter
are sampled together.  The expensive theory prediction is produced by the same
patched CLASS executable and likelihood code used by the deterministic refit.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("CALIBRATION", "profile")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import emcee
import matplotlib

matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt
import numpy as np

import refit_joint_plik_bao as joint


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_mcmc_profile"
OUT.mkdir(exist_ok=True)

NWALKERS = int(os.environ.get("TEST53_MCMC_WALKERS", "32"))
NSTEPS = int(os.environ.get("TEST53_MCMC_STEPS", "400"))
BURNIN = int(os.environ.get("TEST53_MCMC_BURNIN", "100"))
NPROCS = int(os.environ.get("TEST53_MCMC_PROCESSES", "4"))
SEED = int(os.environ.get("TEST53_MCMC_SEED", "530053"))

COSMO_NAMES = joint.MODELS["geo"]["names"]
NAMES = COSMO_NAMES + ["A_planck"]
LABELS = [
    r"$\omega_b$",
    r"$\omega_{idm}$",
    r"$h$",
    r"$\ln(10^{10}A_s)$",
    r"$n_s$",
    r"$\tau$",
    r"$f_{dyn}$",
    r"$A_{Planck}$",
]
LOWER = np.array(joint.MODELS["geo"]["lo"] + [0.98], dtype=float)
UPPER = np.array(joint.MODELS["geo"]["hi"] + [1.02], dtype=float)


def silence_worker_log() -> None:
    """Prevent multiple worker processes from writing the optimizer CSV."""
    try:
        joint.csvf.close()
    except Exception:
        pass
    joint.csvf = open(os.devnull, "w")
    joint.W = csv.writer(joint.csvf)


def log_probability(theta: np.ndarray) -> float:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != LOWER.shape or np.any(theta <= LOWER) or np.any(theta >= UPPER):
        return -np.inf
    result = joint.residual("geo", theta[:-1], start_id=-1, A_override=theta[-1])
    if result is None:
        return -np.inf
    chi2 = float(result[0] @ result[0])
    return -0.5 * chi2 if np.isfinite(chi2) else -np.inf


def split_rhat(chain: np.ndarray) -> np.ndarray:
    """Classical split-Rhat, treating each ensemble walker as a chain."""
    nstep, nwalker, ndim = chain.shape
    half = nstep // 2
    if half < 2:
        return np.full(ndim, np.nan)
    split = np.concatenate([chain[:half], chain[-half:]], axis=1).transpose(1, 0, 2)
    n = split.shape[1]
    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    within = variances.mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    var_hat = ((n - 1) / n) * within + between / n
    return np.sqrt(var_hat / within)


def main() -> None:
    if NWALKERS < 2 * len(NAMES):
        raise SystemExit(f"Need at least {2 * len(NAMES)} walkers for {len(NAMES)} dimensions")
    if NSTEPS <= BURNIN or BURNIN < 0:
        raise SystemExit("MCMC steps must be greater than burn-in")

    fit_path = ROOT / "results_joint_profile" / "fit_summary_joint.json"
    if not fit_path.is_file():
        raise SystemExit("Run the joint-profile optimizer before MCMC")
    fit = json.loads(fit_path.read_text())
    geo = fit["geo"]

    center = np.array([geo["params"][name] for name in COSMO_NAMES] + [geo["A_planck"]])
    fisher = np.array([geo["errors_fisher"][name] for name in COSMO_NAMES] + [joint.ACAL_SIG])
    fallback = np.array(joint.MODELS["geo"]["step"] + [joint.ACAL_SIG])
    scale = np.where(np.isfinite(fisher) & (fisher > 0), 0.25 * fisher, fallback)

    rng = np.random.default_rng(SEED)
    initial = np.empty((NWALKERS, len(NAMES)))
    for i in range(NWALKERS):
        for _ in range(10000):
            candidate = center + rng.normal(size=len(NAMES)) * scale
            if np.all(candidate > LOWER) and np.all(candidate < UPPER):
                initial[i] = candidate
                break
        else:
            raise RuntimeError("Could not initialize walkers inside prior bounds")

    backend_path = OUT / "chain.h5"
    backend = emcee.backends.HDFBackend(backend_path)
    backend.reset(NWALKERS, len(NAMES))

    print(
        f"Starting Test 53 MCMC: walkers={NWALKERS}, steps={NSTEPS}, "
        f"burn-in={BURNIN}, processes={NPROCS}",
        flush=True,
    )
    context = mp.get_context("fork")
    with context.Pool(processes=NPROCS, initializer=silence_worker_log) as pool:
        sampler = emcee.EnsembleSampler(
            NWALKERS,
            len(NAMES),
            log_probability,
            pool=pool,
            backend=backend,
        )
        state = initial
        completed = 0
        while completed < NSTEPS:
            chunk = min(10, NSTEPS - completed)
            state = sampler.run_mcmc(state, chunk, progress=False)
            completed += chunk
            print(f"MCMC progress: {completed}/{NSTEPS} steps", flush=True)

    chain = backend.get_chain()
    logp = backend.get_log_prob()
    post = chain[BURNIN:]
    flat = post.reshape(-1, len(NAMES))
    flat_logp = logp[BURNIN:].reshape(-1)

    try:
        tau = np.asarray(emcee.autocorr.integrated_time(post, tol=0, quiet=True), dtype=float)
    except Exception:
        tau = np.full(len(NAMES), np.nan)
    ess = np.where(np.isfinite(tau) & (tau > 0), flat.shape[0] / tau, np.nan)
    rhat = split_rhat(post)
    acceptance = np.asarray(sampler.acceptance_fraction, dtype=float)
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    best_index = int(np.nanargmax(flat_logp))

    parameters = {}
    for i, name in enumerate(NAMES):
        parameters[name] = {
            "q16": float(q16[i]),
            "median": float(q50[i]),
            "q84": float(q84[i]),
            "minus_1sigma": float(q50[i] - q16[i]),
            "plus_1sigma": float(q84[i] - q50[i]),
            "rhat_split": float(rhat[i]),
            "autocorr_time": float(tau[i]),
            "effective_samples": float(ess[i]),
        }

    finite_rhat = rhat[np.isfinite(rhat)]
    finite_ess = ess[np.isfinite(ess)]
    max_rhat = float(np.max(finite_rhat)) if finite_rhat.size else None
    min_ess = float(np.min(finite_ess)) if finite_ess.size else None
    converged = bool(
        max_rhat is not None
        and min_ess is not None
        and max_rhat < 1.05
        and min_ess >= 400
        and np.all((NSTEPS - BURNIN) >= 50 * tau[np.isfinite(tau)])
    )

    summary = {
        "likelihood": "Planck2018 Plik-lite TTTEEE high-l + DESI DR2 BAO13 + tau prior",
        "sampler": "emcee affine-invariant ensemble",
        "sampled_model": "Test 53 GEO",
        "calibration": "A_planck sampled with Gaussian prior sigma=0.0025",
        "uniform_prior_bounds": dict(zip(NAMES, zip(LOWER.tolist(), UPPER.tolist()))),
        "walkers": NWALKERS,
        "steps": NSTEPS,
        "burnin": BURNIN,
        "posterior_samples": int(flat.shape[0]),
        "mean_acceptance_fraction": float(np.mean(acceptance)),
        "min_acceptance_fraction": float(np.min(acceptance)),
        "max_acceptance_fraction": float(np.max(acceptance)),
        "max_split_rhat": max_rhat,
        "min_effective_samples": min_ess,
        "strict_convergence_passed": converged,
        "convergence_rule": "max split-Rhat < 1.05, min ESS >= 400, chain >= 50 autocorrelation times",
        "parameters": parameters,
        "best_sample": dict(zip(NAMES, map(float, flat[best_index]))),
        "best_sample_chi2": float(-2 * flat_logp[best_index]),
        "derived": {
            "H0_median_km_s_Mpc": float(100 * q50[NAMES.index("h")]),
            "H0_q16_km_s_Mpc": float(100 * q16[NAMES.index("h")]),
            "H0_q84_km_s_Mpc": float(100 * q84[NAMES.index("h")]),
            "probability_f_dyn_gt_0_05": float(np.mean(flat[:, NAMES.index("f_dyn_test53")] > 0.05)),
        },
    }
    (OUT / "posterior_summary.json").write_text(json.dumps(summary, indent=2))

    stride = max(1, flat.shape[0] // 20000)
    with (OUT / "posterior_samples.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(NAMES + ["log_probability"])
        for values, lp in zip(flat[::stride], flat_logp[::stride]):
            writer.writerow([*map(float, values), float(lp)])

    fig, axes = plt.subplots(len(NAMES), 1, figsize=(12, 2.0 * len(NAMES)), sharex=True)
    for i, axis in enumerate(axes):
        axis.plot(chain[:, :, i], color="black", alpha=0.18, linewidth=0.45)
        axis.axvline(BURNIN, color="tab:red", linestyle="--", linewidth=1)
        axis.set_ylabel(LABELS[i])
    axes[-1].set_xlabel("MCMC step")
    fig.tight_layout()
    fig.savefig(OUT / "trace.png", dpi=180)
    plt.close(fig)

    figure = corner.corner(
        flat,
        labels=LABELS,
        truths=center,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4g",
    )
    figure.savefig(OUT / "corner.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2), flush=True)
    if converged:
        print("PASS: strict convergence thresholds satisfied", flush=True)
    else:
        print("WARNING: strict convergence thresholds not yet satisfied; extend the chain", flush=True)


if __name__ == "__main__":
    main()
