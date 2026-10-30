# LinearConvergenceOperatorNormSolver

## Numerical experiments: stochastic Rayleigh–Ritz ascent

The three cleaned notebooks study convergence of the adjoint-free random-plane algorithm for estimating the Euclidean operator norm. They generate the figures and numerical summaries for the local convergence, global convergence, and lower-spectrum experiments.

## Files

| File | Experiment |
| --- | --- |
| `01_local_gap_and_dimension.clean.ipynb` | Local convergence: spectral gap, dimension, and initialization |
| `02_global_entrance_and_contraction.clean.ipynb` | Uniform sphere starts: entrance times and contraction after entrance |
| `03_bulk_spectrum_comparison.clean.ipynb` | Lower-spectrum effects: uniform sphere versus fresh random local starts; fitted-rate ratios and table |
| `operator_norm.py` | Shared simulation and analysis functions; required by all three notebooks |
| `requirements.txt` | Python dependencies |

The original notebooks are retained separately. The cleaned fourth notebook omits the superseded deterministic-start experiments and alternative diagnostic layouts. The exact two-level benchmark is in the separate third notebook and is not part of these three cleaned copies.

## Setup and execution

Use Python 3 with the dependencies listed in `requirements.txt`. From the `spectral_gap_experiments` directory, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open a cleaned notebook and run its cells from top to bottom. Each notebook can be run independently; there is no required order between notebooks. Use **Restart Kernel and Run All Cells** to check reproducibility without relying on variables retained from an earlier session.

The notebooks locate `operator_norm.py` in the current working directory, its ancestors, or a child named `spectral_gap_experiments`. The simplest arrangement is to launch Jupyter from the directory containing the notebooks and `operator_norm.py`. In that arrangement, the path-discovery block can be replaced by `HERE = Path.cwd()`; retain the import of `Path`.

A system LaTeX installation is not required for the plots. The default Matplotlib configuration renders mathematical labels without external LaTeX.

## Default experiment settings

The cleaned notebooks default to the `quick` profile. Change `PROFILE` or `RANDOM_PROFILE` to `full` for the larger configurations specified in each notebook. Larger configurations require more memory and computation.

| Notebook | Configurations | Independent runs per configuration | Iterations | Bootstrap resamples |
| --- | --- | ---: | --- | ---: |
| First | d = 20, 60; gap = 0.5, 0.1, 0.03; eta = 0.01, 0.1, 0.5 | 96 | min(ceil(4 / (-log q_eta)), 2500) | 150 |
| Second | (d, gap) = (20, 0.5), (20, 0.1), (60, 0.1); eta = 0.01, 0.1, 0.5 | 128 | 3500 global; 1200 post-entry | 150 |
| Fourth | d = 100; gap = 0.1; three lower spectra; eta = 0.01, 0.1, 0.5 | 256 | 20000 global; 5000 local | 200 |

The seeds are specified in each notebook. Spectra are fixed within each configuration; the independent runs are algorithm trajectories, not newly sampled matrices. The simulations work in spectral coordinates using diagonal representatives. For this rotationally invariant algorithm, an orthogonal change of basis does not change the distribution of the spectral error when initialization is transformed consistently.

### First notebook: controlled local starts

All trajectories in a configuration begin at the same deterministic vector

    v0 = sqrt(1 - eta/2) u1 + sqrt(eta/2) u2.

The initial spectral error is eta times the spectral gap divided by two. Only the random search directions differ between independent runs. Seeds are reused across eta to couple the search randomness; the initial vector changes with eta.

The figures show normalized arithmetic mean errors, iteration rescaling, and fitted-rate comparisons across dimensions and gaps. In the ordinary-iteration panels, shading below each q0 reference is a reference region. It is not a confidence interval or a guaranteed error envelope. Bootstrap bands are shown in the rescaled panels.

### Second notebook: entrance and post-entry contraction

Initial vectors are uniform on the full unit sphere. Reusing the seed across eta preserves the global trajectories: eta changes the entrance threshold, not the global algorithm. Entrance occurs when the spectral error first falls below eta times the spectral gap.

When every trajectory has entered, the post-entry ensembles restart from all first-entry states using fresh continuation randomness. If some trajectories have not entered by the iteration limit, the notebook omits that post-entry ensemble instead of selecting only successful runs.

The empirical survival probability is the fraction of trajectories that have not entered by iteration k. The red horizontal reference is

    u_N = 1 - 0.05**(1/N).

This is the one-sided 95% binomial upper confidence limit when zero survivors are observed among N independent runs. At nonzero survivor counts, it is used only as a fixed reference level. The red vertical band spans the first crossings of this level across eta; it is not an entrance-time confidence interval or a range at which exactly 95% of runs have entered.

### Third notebook: lower-spectrum comparison

All three spectra share d = 100, largest eigenvalue 1, spectral gap 0.1, and spectral width 1. The remaining eigenvalues are spread, clustered at the second eigenvalue, or clustered at zero.

The main comparison has two panels:

- Uniform initialization on the entire sphere.
- Fresh random initialization in the local error region for eta = 0.1.

The right panel does not reuse first-entry states from the left panel. To generate local starts, the notebook samples a uniform transverse direction, a random leading sign, and a uniform target error between zero and eta times the gap. This is a specified distribution inside the region, not uniform surface measure on that region. Coupled initialization draws give matched initial errors across spectra.

The ratio plot includes all three eta values. The LaTeX table reports fresh random local starts for eta = 0.1, including fitted rates, ratios to the q0 reference exponent, and final relative mean errors. A companion figure with mean plus/minus one sample standard deviation is also exported.

## Errors, reference factors, and fitted rates

The spectral error is

    epsilon_k = lambda_1 - ||A v_k||^2.

Normalized mean-error curves use the arithmetic mean error divided by its initial mean. For deterministic common initial errors, this equals the mean of the individually normalized errors.

The proved local factor on the region epsilon <= eta * delta is

    q_eta = 1 - (1 - eta) * delta / ((d - 1) * Delta).

The comparison reference used in the cleaned figures is its limit as eta tends to zero:

    q0 = 1 - delta / ((d - 1) * Delta),
    a0 = -log(q0).

q0 is a limiting local reference, not a proved error bound for the finite-error initializations, not a global bound, and not a claim of an optimal algorithmic rate. The proved global bounds in the second notebook retain q_eta.

The fitted empirical rate a_hat is the negative least-squares slope of the logarithm of the arithmetic mean error against iteration number. Fits use relative mean errors in [1e-4, 0.5], with at most 400 selected iterations. The reported rate ratio is

    a_hat / (-log(q0)).

It compares logarithmic decay exponents. It is not a ratio of contraction factors or a ratio of errors. Since q0 does not depend on eta, differences between ratios across eta reflect empirical-rate differences.

## Bootstrap intervals

A bootstrap resample selects N complete recorded trajectories, with replacement, from the N original trajectories. It does not simulate additional runs or generate additional matrices. Repeating this procedure B times produces bootstrap mean curves; pointwise 95% percentile intervals use the 2.5th and 97.5th percentiles.

The fourth notebook normalizes each bootstrap mean by its own bootstrap initial mean. The first notebook has the same deterministic initial error for all trajectories. Fitted-rate intervals keep the fitting window selected from the original mean fixed. These intervals are conditional on that window and are not simultaneous bands over all iterations.

Mean plus/minus one sample standard deviation describes variability between individual trajectories; a bootstrap confidence interval describes uncertainty in the estimated mean. They answer different questions.

## Outputs and cached data

Clean exports are written under:

```text
results/clean/01_local/<profile>/etas_.../
results/clean/02_global/<profile>/etas_.../
results/clean/03_random_initialization/<profile>/
```

Exports include PDF/PNG figures and CSV summaries; the fourth notebook also exports the LaTeX table fragment `random_local_table_eta0.1.tex`. Include that fragment in a manuscript using `booktabs`.

The first and fourth notebooks reuse matching simulation caches from the original result directories. The second notebook reruns its ensembles and stores their histories in its clean output directory. The fourth notebook still calculates post-entry diagnostic summaries in CSV form, although the cleaned notebook displays only the current two-panel comparison and categorical ratio plot.

When changing simulation settings, use matching cache files or regenerate the affected caches. Do not reuse histories generated with different parameters. The notebooks contain checks for cache compatibility and numerical consistency.
