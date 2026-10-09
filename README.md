# LogitTools

[![Build Status](https://github.com/Gkreindler/LogitTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Gkreindler/LogitTools.jl/actions/workflows/CI.yml?query=branch%3Amain)


Package to estimate binary logit models, Bayesian bootstrap (with option to cluster), and write to tables using RegressionTables.jl. Analytic likelihood and gradient + Optim.jl.

Modelled after [GMMTools.jl](https://github.com/Gkreindler/GMMTools.jl).

To install:
```
] add https://github.com/Gkreindler/LogitTools.jl
```

See examples/example.jl for basic usage of `logit2` and `mlogit`.

## Mixed logit: `mlogit_rfx`

`mlogit_rfx` estimates a conditional logit whose coefficients vary across people
(and, optionally, across a person's alternatives), by maximum simulated
likelihood with an analytic gradient.

### Model

Data are in long format: one row per (choice set, option). Three identifiers:

| | meaning |
|---|---|
| `col_group` | the person. The likelihood is a product over people, each integrated over that person's random coefficients. |
| `col_id` | the choice set: the rows one softmax runs over. Unique across people. |
| a term's `level` | the column indexing that term's draws within a person. The default is the person (one draw per person). A column such as a neighborhood code gives one draw per (person, neighborhood), shared by every row of the person in that neighborhood. |

The utility of option `t` is `x_t' beta + sum_m z_tm * a_m`, where `a_m` is the
random part of term `m` drawn for the row's (person, level) cell. For person `i`

    L_i = E[ prod over i's choice sets of P(chosen option) ],

the expectation over all of `i`'s random coefficients, approximated by an
average over `R` draws. The estimator maximises `sum_i w_i log L_i`.

### Random terms

```julia
rfx = [:a,                                    # normal person-level coefficient on :a
       rfx_term(:x; dist = :uniform),         # also :triangular, :lognormal, :neg_lognormal
       rfx_term(:x; level = :nbh_code),       # person-by-neighborhood coefficient on :x
       rfx_term(level = :nbh_code),           # person-by-neighborhood intercept
       rfx_term(:g; mean = false)]            # mean-zero loading on :g (no formula mean)
```

- A term's variable must be in `formula`, which carries the coefficient's mean,
  unless `mean = false`.
- `:normal`, `:uniform` and `:triangular` draws are standardised, so `sd_x` is
  the coefficient's standard deviation under each. A lognormal coefficient is
  `exp(mu + sigma * eta)`, with `mu` the formula coefficient on the log scale.
- Terms are independent of each other unless they are in a correlation block.

### Correlation blocks

`rfx_corr` takes blocks of person-level terms (linear families) whose
coefficients are correlated. A tuple of terms gives an unrestricted correlation
matrix:

```julia
rfx_corr = [(:a, :b)]                         # one correlation parameter
rfx_corr = [(:a, :b, :c)]                     # three
```

The block's correlation matrix is `L * L'`, with `L` lower triangular and rows
of unit length. `rfx_corr_block` can fix entries of `L` at zero, which encodes
conditional independence. In the example below the correlation matrices of
`(F, V, AV)` and of `(F, Y, W)` are each unrestricted, while `(V, AV)` and
`(Y, W)` are independent given `F` (5 standard deviations and 6 coordinates):

```julia
rfx_corr = [rfx_corr_block(:F, :V, :AV, :Y, :W;
                           zero = [(:Y, :V), (:Y, :AV), (:W, :V), (:W, :AV)])]
```

Each free entry of `L` is one parameter in `(-1, 1)`, named `pcor_row__col`.
In an unrestricted block these are partial correlations. Report correlations
and covariances with `mlogit_rfx_correlation(theta, P)` and
`mlogit_rfx_covariance(theta, P)`. Person-by-level terms cannot be in a block;
they are independent of everything else.

### Workflow

```julia
using LogitTools

xs  = [:dur, :pay, :any_fam, :salient]
rfx = [rfx_term(:any_fam), rfx_term(:salient), rfx_term(:any_fam; level = :nbh_code)]

# 1. prepare once: data in numeric form, cells and draws
P  = mlogit_rfx_prep(df, xs, :choice_id, :selected; col_group = :person,
                     rfx, rfx_corr = [(:any_fam, :salient)], ndraws = 2000)

# 2. plain logit for starting values (no random terms)
P0 = mlogit_rfx_prep(df, xs, :choice_id, :selected; col_group = :person)
f0 = mlogit_rfx_fit(P0, zeros(length(xs)))

# 3. mixed logit from several starts
starts = theta0_mlogit_rfx_multistart(P; b0 = f0.theta_hat, nstarts = 5)
fit = mlogit_rfx_fit(P, starts)          # fit.theta_hat, fit.fits_df, fit.extra

# 4. person-clustered sandwich, holding boundary parameters fixed
V = mlogit_rfx_vcov(fit.theta_hat, P; fixed = mlogit_rfx_boundary(fit.theta_hat, P))

# 5. Bayesian bootstrap over people, warm-started at the estimate
W  = mlogit_rfx_boot_weights(P.N, 500; seed = 1)
bt = boot_mlogit_rfx(P, fit.theta_hat; W)   # bt.theta_boot_table, bt.V, bt.boot_fits
```

`mlogit_rfx(df, formula, col_id, col_selected, theta0; kwargs...)` does steps 1
and 3 in one call.

`theta = [means (K, formula order); standard deviations (M, rfx order); block
coordinates (B)]`, named in `P.theta_names`.

### Functions

| function | purpose |
|---|---|
| `mlogit_rfx_prep` | build the prepared model `P` (draws are generated here) |
| `mlogit_rfx_fit` | estimate from one start (vector) or several (one per row) |
| `theta0_mlogit_rfx`, `theta0_mlogit_rfx_multistart` | starting values from coefficient levels `b0` |
| `mlogit_rfx_objective`, `mlogit_rfx_gradient` | simulated negative log-likelihood and its gradient |
| `mlogit_rfx_scores` | per-person scores (`N x npar`) |
| `mlogit_rfx_vcov` | person-clustered sandwich; `fixed` holds parameters at their values |
| `mlogit_rfx_boundary` | flags standard deviations near zero and coordinates near +-1 |
| `mlogit_rfx_boot_weights`, `boot_mlogit_rfx` | Bayesian bootstrap over people |
| `mlogit_rfx_ess` | effective number of draws per person |
| `mlogit_rfx_correlation`, `mlogit_rfx_covariance` | implied correlation / covariance of the random terms |
| `rfx_cell_report` | cell counts behind each random term |

### Practical notes

**Bootstrap.** Each replicate re-estimates with person weights drawn from a flat
Dirichlet, with the simulation draws held fixed. Pass a vector of starting
values for a single (warm) start per replicate, or a matrix for a multi-start
fit of every replicate. Generate `W` once and pass column subsets to run the
bootstrap in chunks; results do not depend on chunking or on `parallel`. With
`parallel = true` the replicates run on `workers()` (`addprocs(n)` and
`@everywhere using LogitTools` first); each worker receives `P` once.

**Boundaries.** Standard deviations are kept positive (`sigma = eps + softplus`)
and block coordinates inside `(-1, 1)` (scaled `tanh`). An estimate can sit at
the boundary, e.g. a standard deviation near zero. At that point the
transformation is flat and an optimiser started there cannot move, so starting
standard deviations below `sd_start_min = 0.01` are raised to it and
coordinates are clipped to `+-corr_start_max = 0.99`. This affects starting
values only. A boundary estimate has no regular curvature; `mlogit_rfx_vcov`
with `fixed = mlogit_rfx_boundary(...)` gives the sandwich with those
parameters held at their values. Neither the sandwich nor the bootstrap is
valid for a parameter at the boundary itself.

**Simulation.** Draws are pseudo-random and antithetic (draw `r + R/2` is minus
draw `r`), generated once in `mlogit_rfx_prep` from `seed` and held fixed, so
the objective is a smooth function of `theta`. Simulation error does not enter
the sandwich or the bootstrap. Check it directly: re-estimate with twice the
draws, and with another seed, and compare. `fit.extra.ess_p10` (10th
percentile of effective draws per person) falls when the integral is hard. Its
dimension is the number of informative cells per person, which
`rfx_cell_report(P)` lists by term.

**Kernels.** `kernel = :auto` uses a fast kernel when every choice set has two
options and no term is lognormal. It computes each antithetic pair of draws at
once. Otherwise the general kernel handles any number of options and all
distributions. Both compute the same likelihood (tested to agree to
roundoff).

**Identification guards.** `mlogit_rfx_prep` refuses terms that cannot be
identified: a loading that is zero everywhere, a term that cancels within every
choice set (e.g. a person-level random intercept), or a term with one row per
cell. It warns when most choice sets carry no information about a term.
