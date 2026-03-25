# README.md

# MESS: Multiproposal Elliptical Slice Sampling

This repository implements Elliptical Slice Sampling (ESS) and its multiproposal generalization (MESS) for Bayesian inference in models with Gaussian priors. It accompanies the pre-print [Multiproposal Elliptical Slice Sampling](https://arxiv.org/abs/2602.22358).

---

## Key concepts

### ESS and MESS

- MESS proposes $M$ candidate angles per subiteration and accepts one uniformly or based on a transition matrix.
- ESS is recovered by setting $M=1$.
- All algorithms require: sampling from a Gaussian prior and evaluating a log-likelihood.

---

## Design principles

1. Algorithms are model-agnostic
   - Samplers only need a prior sampler and a log-likelihood.

2. Each problem constructs its own prior
   - Prior mean and covariance are derived in the problem class.
   - Problems considered: examples 1 (GP) and 2 (LR) in Murray et. al (2010), Semi-Blind Deconvolution from Senn et al. (2025, 2026), toy model for solute transport from Glatt-Holtz et al. (2024).

3. Gaussian priors only
   - All problems inherit from GaussianPriorProblem.
   - Gaussian sampling is performed via Cholesky decomposition.

4. Parallelization is user-defined
    - The provided MESS algorithm is better suited to toy problems where likelihood costs are cheap and parallel overhead dominates.
    - For larger problems, users can parallelize the per-proposal likelihood evaluation and distance measurement.
    - Parallelization is not implemented to allow users choose the best strategy for their infrastructure. It should however be a straightforward programming task. 

---

## Repository structure

### src/mess/algorithms

- ess.py: ESS (equivalent to MESS with $M=1$)
- mess.py: MESS with optional LP-based transition matrices
- utils.py: angle sampling, shrinking logic, and helper routines

### src/mess/problems

All problems implement log_likelihood(x) and provide prior construction.

- gp_regression.py: Gaussian process regression
- logistic_regression.py: Bayesian logistic regression
- sbd.py: semi-blind deconvolution inverse problem
- advection_diffusion.py: toy solute transport inverse problem

### src/mess/kernels

- stationary.py: RBF and exponential kernels for GP priors

### src/mess/data

- Synthetic data generators used by the notebooks and tests

---

## Notebooks

The notebooks/ directory reproduces the experiments and figures used in the paper.

- gp_regression.ipynb: baseline ESS/MESS comparison on GP regression.
- logistic_regression_ess_mess.ipynb: Bayesian logistic regression with multiple distance metrics.
- sbd_ess_mess.ipynb: semi-blind deconvolution.
- solute_transport_d10_mess_ellipse_iter10000.ipynb: solute transport toy problem at fixed dimension.
- solute_transport_dim_sweep_shared_draws.ipynb: solute transport dimension sweep with shared draws.
- solute_transport_dim_sweep_shared_draws_lp_compare.ipynb: LP-based transition comparison for the dimension sweep.

---

## Typical usage (for GP regression example)

```python
import numpy as np
from mess.data.gp_regression import generate_gp_regression_data
from mess.problems.gp_regression import GaussianProcessRegression
from mess.algorithms.ess import mess_step

data = generate_gp_regression_data(seed=0)
problem = GaussianProcessRegression(
    X=data["X"],
    y=data["y"],
    length_scale=1.0,
    noise_variance=0.09,
)

x = data["f_init"]
rng = np.random.default_rng(0)
M = 5 # Choose nr of proposals

# To run ESS (MESS with M=1)
for _ in range(1000):
    x, _, _ = ess_step(x, problem, rng)

# To run MESS (Uniform probabilities in acceptance step)
for _ in range(1000):
    x, _, _ = mess_step(x, problem, rng, M=M, use_lp=False)

# To run MESS (Distance-informed transition matrix)
for _ in range(1000):
    x, _, _ = mess_step(x, problem, rng, M=M, use_lp=True, distance_metric='angular')
    #  x, _, _ = mess_step(x, problem, rng, M=M, use_lp=True, distance_metric='euclidean')

```

---

## References

- Glatt-Holtz, Nathan E., Andrew J. Holbrook, Justin A. Krometis, and Cecilia F. Mondaini. 2024. "Parallel MCMC Algorithms: Theoretical Foundations, Algorithm Design, Case Studies." Transactions of Mathematics and Its Applications 8 (2): tnae004. https://academic.oup.com/imatrm/article/8/2/tnae004/7738435.
- Murray, Iain, Ryan Prescott Adams, and David J. C. MacKay. 2010. "Elliptical slice sampling." arXiv preprint. https://arxiv.org/abs/1001.0175.
- Senn, Guillermina, Matt Walker, and Haakon Tjelmeland. 2025. "Scalable Bayesian seismic wavelet estimation." Geophysical Prospecting. https://onlinelibrary.wiley.com/doi/full/10.1111/1365-2478.70026.
- Senn, Guillermina, Hakon Tjelmeland, Nathan E. Glatt-Holtz, Matt Walker, and Andrew J. Holbrook. 2026. "Bayesian Semi-Blind Deconvolution at Scale." arXiv preprint. https://arxiv.org/pdf/2601.09677.