```@meta
CurrentModule = LogitTools
```

# Random coefficients

[`logit2_rfx`](@ref) fits a binary logit in which some coefficients vary across
individuals, by maximum simulated likelihood with an analytic gradient.

## The model

Individuals ``i`` make repeated choices ``t``. ``Z`` denotes the columns of ``X``
that carry random coefficients:

```math
v_{it} = X_{it}'\beta + \sum_m \sigma_m Z_{itm} \eta_{im},
\qquad \eta_i \sim N(0, I_M)
```

Each individual draws one coefficient vector and keeps it for all of their
choices, so the panel dimension is what identifies ``\sigma``. The individual
log likelihood integrates ``\eta_i`` out by simulation:

```math
\log \hat L_i = \log \frac{1}{R} \sum_r \exp\Big( \sum_t \log \Lambda(q_{it} v_{itr}) \Big),
\qquad q_{it} = 2y_{it} - 1
```

and the objective minimised is ``Q(\theta) = -\sum_i \omega_i \log \hat L_i``.

## Estimating

`col_id` is **positional** — the model is undefined without a group identifier.
`rfx` is a **keyword**, so it cannot be silently transposed with the formula.

```julia
using LogitTools

myxs  = [:dur, :total_pay, :dist]
myrfx = [:dur, :dist]

# start beta at the plain logit, sigma at 0.5
logit_fit = logit2(copy(df), myxs, :pick1, zeros(3))
theta0    = theta0_rfx(myxs, myrfx; b0 = logit_fit.theta_hat)

fit = logit2_rfx(df, myxs, :pick1, :personid, theta0;
                 rfx     = myrfx,
                 ndraws  = 1000,
                 seed    = 20260808)
```

`logit2_rfx` does **not** mutate `df`.

### Parameter ordering

``\theta = [\beta;\ \sigma]`` — ``K`` slopes in formula order, then ``M``
standard deviations in the order listed in `rfx`:

```julia
fit.theta_names   # ["dur", "total_pay", "dist", "sd_dur", "sd_dist"]
```

### Why `s0` is never 0.0

``\sigma = 0`` is a *stationary point* of the simulated likelihood — a saddle,
not a minimum. The ``\sigma``-block of the gradient there is exactly zero, so an
optimiser started at ``\sigma_0 = 0`` cannot move. [`theta0_rfx`](@ref) defaults
to `0.5` and [`logit2_rfx`](@ref) rejects a zero start with an explicit error.

### Why every returned `σ` is non-negative

The likelihood satisfies ``Q(\beta, \sigma) = Q(\beta, -\sigma)`` exactly, so
there are ``2^M`` mirror optima and the sign of ``\sigma`` is not identified.
Estimates are canonicalised with `abs()` at the source, in the main fit and in
every bootstrap replicate alike. Without this, `cov(theta_boot_table)` would mix
mirror modes and be meaningless.

## Simulation draws

Draws are antithetic pseudo-random normals, generated **once** in prep and reused
for every function evaluation and every bootstrap replicate — otherwise the
objective would not be a deterministic function of ``\theta`` and LBFGS would
fail.

- `ndraws` must be **even**: draws are used in ``\pm`` pairs. Reflection halves
  simulation variance and is what makes the ``\sigma = 0`` saddle and the
  ``\sigma \to -\sigma`` symmetry exact.
- Each individual gets their **own** ``M \times R`` block. A common draw set
  leaves simulation error perfectly correlated across individuals and an
  ``O(1/\sqrt R)`` component that does not vanish as ``N \to \infty``.
- Same `seed` twice gives bit-identical estimates.

### The ESS diagnostic

A long panel concentrates the posterior over ``\eta_i``, so many draws end up
contributing nothing. The effective number of draws per individual is

```math
\mathrm{ESS}_i = 1 \big/ \sum_r \tau_{ir}^2
```

where ``\tau_{ir}`` are the posterior draw weights. It is reported in `extra`:

```julia
fit.extra.ess_min, fit.extra.ess_p10, fit.extra.ess_median, fit.extra.ess_mean
```

`logit2_rfx` warns when `ess_p10 < 30`. That is a signal to raise `ndraws`, not
a sign the fit is wrong. At ``N = 400``, ``T = 21``, ``R = 1000`` a healthy run
looks like min 52 / p10 182 / median 448.

!!! tip "Check this on real data before reporting"
    The estimates depend on the draw set. Refit at two or three other `seed`
    values and at `2 * ndraws`, and compare the spread in ``\hat\sigma`` against
    `boot_se`. If the seed-to-seed spread is an appreciable fraction of the
    bootstrap standard error, `ndraws` is too small. There is a ready-made block
    at the bottom of `examples/example_logit2_rfx.jl`.

## Bootstrap

[`boot_logit2_rfx`](@ref) resamples at the `col_id` level with group-level
Dirichlet weights. Prep runs once on the master; workers only ever receive
numeric arrays, never the DataFrame.

```julia
using Distributed
addprocs(4)
@everywhere using LogitTools     # the one setup step you must do

fit.vcov = boot_logit2_rfx(df, myxs, :pick1, :personid, theta0;
                           rfx         = myrfx,
                           ndraws      = 1000,
                           seed        = 20260808,
                           nboot       = 500,
                           boot_seed   = 12345,
                           cluster_var = :personid,   # must equal col_id
                           parallel    = true,
                           theta_start = fit.theta_hat)   # warm start
```

Notes:

- `cluster_var`, if given, **must** equal `col_id`. Clustering at any other level
  leaves the group weighting undefined.
- `theta_start = fit.theta_hat` is much faster than starting from `theta0`.
- With the same `boot_seed`, `parallel = true` and `parallel = false` produce
  identical replicates.
- Replicates are dropped from `V` **only** on numerical failure. A replicate
  where ``\hat\sigma_m`` lands near zero is a legitimate draw from the sampling
  distribution; dropping it would bias the standard error downward.
  `theta_boot_table` keeps every row (`NaN` for errored ones); `V` uses the
  survivors.

## Reading the results

```julia
boot_report(fit)
```

```
Bayesian bootstrap for logit2_rfx
  replicates : 500 attempted / 500 used / 0 dropped
  n_obs      : 8400   n_groups: 400   (col_id = :personid)
  T_i        : min 21 / median 21.0 / max 21
  draws      : R = 1000  (seed 20260808)
  ESS        : min 52 / p10 182 / median 448

5×7 DataFrame
 Row │ param      is_sd  estimate   boot_se    ci_lo      ci_hi       share_near_zero
─────┼────────────────────────────────────────────────────────────────────────────────
   1 │ dur        false  -0.15058   0.040756   -0.228955  -0.0666318            NaN
   2 │ total_pay  false   0.10613   0.0251078   0.056659   0.154777             NaN
   3 │ dist       false  -0.261511  0.0322079  -0.32382   -0.198853             NaN
   4 │ sd_dur      true   0.645628  0.0425514   0.55676    0.718435               0.0
   5 │ sd_dist     true   0.370021  0.0427818   0.288202   0.455247               0.0
```

**For the `σ` rows, read the percentile CI, not the standard error.** The `abs()`
canonicalisation folds the sampling distribution, so it is skewed — increasingly
so the closer ``\sigma`` sits to zero — and a symmetric ``\pm 1.96 \cdot se``
interval is the wrong summary. A fit where `sd_x` came out at `0.015` with
`boot_se` `0.200` would put most of that Wald interval below zero, while the
percentile CI stays in the right place.

`share_near_zero` — the share of converged replicates with
``\hat\sigma_m <`` `sd_tol` — is the more informative statistic for whether
homogeneity is rejected.

!!! warning "Boundary problem"
    A confidence interval for a variance parameter that touches zero is a
    boundary problem. Neither the percentile CI nor a Wald test is calibrated
    exactly at ``\sigma_m = 0`` (for a single ``\sigma`` the LR statistic is a
    ``\tfrac12\chi^2_0 + \tfrac12\chi^2_1`` mixture). Read such an interval as
    "cannot reject homogeneity", not as a calibrated interval.

### Tables

`regtable(fit)` still works, since `logit2_rfx` returns an ordinary `MLEFit`.

[`regtable_rfx`](@ref) additionally prints a **percentile confidence interval
under the `sd_` rows** while keeping standard errors under the `β` rows, which is
the presentation the previous paragraph argues for:

```julia
regtable_rfx(fit)
```

```
----------------------
x1            0.601***
               (0.152)
x2           -0.895***
               (0.133)
x3            0.512***
               (0.096)
sd_x1         0.852***
        (0.619, 1.084)
sd_x3            0.077
        (0.003, 0.560)
----------------------
N                  640
----------------------
```

`sd_x3` shows why it matters: an estimate of `0.077` with a bootstrap standard
error of `0.170` gives a Wald interval of `(-0.256, 0.410)`, mostly outside the
parameter space, while the percentile interval `(0.003, 0.560)` does not.

```julia
regtable_rfx(fit; ci_levels = [5, 95])   # 90% interval
regtable_rfx(fit; ci_for_sd = false)     # conventional: standard errors throughout
regtable_rfx(fit1, fit2)                 # several models, each with its own CIs
```

Other keywords (`render = LatexTable()`, `labels`, …) are forwarded to `regtable`.

!!! note
    `ci_for_sd = true` needs RegressionTables 0.7 or newer — the first version
    whose `below_statistic` receives the coefficient index, and so can vary by
    row. On 0.6.x it throws an explanatory error; use `ci_for_sd = false`.

## Guards

These are hard errors, each naming the offending values:

| Condition | Why |
|---|---|
| `rfx` name not in `formula`, or duplicated | typo |
| distribution other than `:normal` | not implemented yet |
| `length(theta0) != K + M` | use [`theta0_rfx`](@ref) |
| any ``\sigma_0 = 0`` | saddle point |
| `ndraws` odd | antithetic pairing |
| every group a singleton | ``\sigma`` not identified without repeated choices |
| `weights` not constant within `col_id` | the weight multiplies the group log likelihood |
| choice column not 0/1 | as `logit2` |
| `cluster_var != col_id` | group weighting undefined |
| `parallel = true` with no workers, or `LogitTools` missing on one | names the worker ids |

And two warnings: `median(T_i) < 3` (weak identification on very short panels),
and an `rfx` variable with less than 1% of its variance within group (a random
coefficient on a near group-invariant regressor is near-collinear with a random
intercept).

## Performance

The draw dimension is collapsed *before* the design matrix is touched:

```math
\bar e_{it} = \sum_r \tau_{ir} e_{itr} \quad O(T_i R), \qquad
\partial/\partial\beta = \sum_t \bar e_{it} X_{it} \quad O(T_i K)
```

Building an augmented per-draw design matrix instead would cost
``O(N_{obs} R K)`` — about a 90× penalty at ``N_{obs} = 8400``, ``K = 100``,
``R = 1000``, which makes the bootstrap infeasible. Only the ``\sigma``-gradient
needs the draw dimension, and ``M`` is small.

At ``N = 400``, ``T = 21``, ``K = 3``, ``R = 1000`` a single fit takes a few
seconds; 500 bootstrap replicates take roughly 10 minutes on 4 workers.

Memory is ``M \cdot R \cdot N`` floats for the draws — 6.4 MB at ``M = 2``,
``R = 1000``, ``N = 400``.

## Extension points

Hooks are in place; the implementations are not.

- `scheme = :halton` / `:gauss_hermite` in [`LogitTools._make_draws`](@ref).
  Everything flows through a `(nodes, log_weights)` pair, so only the node
  generator changes.
- Lognormal random coefficients via `rfx = [:dur => :lognormal]`. `rfx` is
  normalised to `Vector{Pair{Symbol,Symbol}}` internally already. Needed if a
  ratio of two random coefficients is ever reported, since a normal/normal ratio
  has no moments.
- Threading over groups within a fit (needs per-thread buffer sets; the
  bootstrap currently takes the parallelism).
- `logitn_rfx` for the multinomial case — the draw machinery and posterior weight
  structure carry over unchanged.
