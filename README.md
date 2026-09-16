# Test 53: modified CLASS + Planck Plik-lite + DESI DR2 BAO

Reproducible pipeline for the Test 53 geodesic interacting-vacuum model.
The repository pins every external source used by the automated build and
provides a GitHub Actions **Run workflow** entry point.

## What the workflow does

1. Checks out [`kaeonikc/class_iv`](https://github.com/kaeonikc/class_iv) at
   commit `ac627d54e9ce196a08878d1ba33999819925d19c`.
2. Applies the base fix and canonical Test 53 patch.
3. Builds CLASS on a clean Ubuntu runner with GSL.
4. Verifies that `f_dyn_test53 = 0` reproduces the corresponding LambdaCDM
   lensed spectra to a maximum relative difference below `1e-4`.
5. Loads the pinned
   [`planck-lite-py`](https://github.com/heatherprince/planck-lite-py)
   implementation at commit `2c0d0f67e59ce781654cf62dd7fb10757b0e60be`
   and confirms 613 TTTEEE high-ell bins.
6. Validates the 13-entry DESI DR2 BAO vector and its six within-bin
   `DM`-`DH` correlations.
7. Optionally runs the joint numerical fit and uploads the executable,
   provenance, logs, best fits and BAO predictions as a workflow artifact.

## Run on GitHub

Open **Actions → Test 53 reproducibility → Run workflow** and select:

- `smoke`: compile and run all deterministic checks;
- `joint-fixed`: full Plik-lite + BAO fit with `A_planck = 1`;
- `joint-profile`: same fit while profiling the calibration parameter.

For the archived multi-start setup, leave `max_iterations = 30` and
`start_limit = 0`. The joint modes are computationally expensive; `smoke` is
the default so every push remains a fast integrity check.

## Likelihood definition

- Planck 2018 Plik-lite TTTEEE high-ell: 613 bins;
- DESI DR2 BAO: `DV/rd` at `z=0.295`, plus `DM/rd` and `DH/rd` at
  `z=0.510, 0.706, 0.934, 1.321, 1.484, 2.330`;
- Gaussian prior `tau = 0.0544 ± 0.0073`;
- nonlinear corrections disabled in the joint refit.

This is a Plik-lite high-ell analysis with a tau prior. It is **not** the full
Planck likelihood with all low-ell, lensing and nuisance components.

The BAO values and within-bin correlation coefficients are transcribed from
[DESI DR2 Results II, Table 4](https://arxiv.org/abs/2503.14738). The baseline
uses six 2×2 anisotropic covariance blocks and no covariance between different
redshift bins.

## Model implementation

The archived shape is evaluated from `test53_shape.h` and inserted by the
canonical patch. The convention is

```text
rho_c_dot + 3 H rho_c = -Q
V_dot                 =  Q
Q                     =  H dV/dln(a)
```

with

```text
V(z)   = Omega_iv0 H0^2 [1 + f_dyn S(z)]
rho_c  = H0^2 (1+z)^3 [Omega_c0 - Omega_iv0 f_dyn J(z)]
J(z)   = integral_0^z S_z(z')/(1+z')^3 dz'
```

For `z > 2.5`, `S` and `J` are frozen and the derivatives of `S` vanish.
At exactly `f_dyn_test53 = 0`, the parser remaps the interacting matter and
vacuum densities to ordinary CDM and Lambda. This makes the nested limit use
the unmodified CLASS perturbation sector instead of only matching the
background equations.

## Repository layout

```text
.github/workflows/test53.yml       GitHub Actions entry point
patches/                           pinned CLASS modifications
model/test53_shape.h               archived 4097-node shape table
config/                            CLASS inputs and archived Plik-lite starts
data/desi_dr2_bao13.csv            DESI DR2 BAO vector
scripts/build_class.sh             deterministic source build
scripts/validate_test53.py         nested-limit/input smoke checks
scripts/refit_joint_plik_bao.py    joint numerical refit
```

## Local reproduction

On Ubuntu/WSL:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git libgsl-dev python3-pip
python3 -m pip install -r requirements.txt
bash scripts/build_class.sh
git clone https://github.com/heatherprince/planck-lite-py.git .runtime/planck-lite-py
git -C .runtime/planck-lite-py checkout --detach 2c0d0f67e59ce781654cf62dd7fb10757b0e60be
python3 scripts/validate_test53.py \
  --class-dir .runtime/class_iv \
  --planck-dir .runtime/planck-lite-py
```

To run the joint fixed-calibration fit:

```bash
CLASS="$PWD/.runtime/class_iv/class" \
PLANCK_DIR="$PWD/.runtime/planck-lite-py" \
CALIBRATION=fixed \
python3 scripts/refit_joint_plik_bao.py
```

## License and citation

The pipeline code is released under the repository's MIT License. Upstream
CLASS, Planck likelihood material and DESI data retain their respective terms.
Scientific use should cite CLASS, Planck 2018 likelihoods and DESI DR2 Results
II in addition to the Test 53 manuscript.
