# LogitTools

[![Build Status](https://github.com/Gkreindler/LogitTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Gkreindler/LogitTools.jl/actions/workflows/CI.yml?query=branch%3Amain)

Package to estimate discrete choice models with an analytic likelihood and gradient
(via Optim.jl), a Bayesian bootstrap (with option to cluster), and tables through
RegressionTables.jl.

| Model | Estimation | Inference |
|---|---|---|
| Binary logit | `logit2` | `boot_logit2` |
| Multinomial logit | `mlogit` | `boot_mlogit` |
| Binary logit with **random coefficients** | `logit2_rfx` | `boot_logit2_rfx` |
| Multinomial logit with **random coefficients**, including **option-level** random effects | `mlogit_rfx` | `boot_mlogit_rfx` |

All return an `MLEFit`, so `vcov`, `cis` and `regtable` work the same way across them.

Modelled after [GMMTools.jl](https://github.com/Gkreindler/GMMTools.jl).

## Install

```julia
] add https://github.com/Gkreindler/LogitTools.jl
```

## Binary logit

```julia
using LogitTools

myxs = [:dur, :total_pay, :dist]

fit = logit2(choices_df, myxs, :pick1, zeros(length(myxs)))

fit.vcov = boot_logit2(choices_df, myxs, :pick1, zeros(length(myxs));
                       cluster_var = :clusterid, nboot = 500)

regtable(fit)
```

The bootstrap runs in parallel with `parallel = true`, which gives bit-for-bit
identical results to the serial path:

```julia
using Distributed
addprocs(4)
@everywhere using LogitTools

fit.vcov = boot_logit2(choices_df, myxs, :pick1, zeros(length(myxs));
                       cluster_var = :clusterid, nboot = 500, parallel = true)
```

## Worked example: random coefficients

`logit2_rfx` lets coefficients vary across individuals:

$$v_{it} = X_{it}'\beta + \sum_m \sigma_m Z_{itm}\eta_{im}, \qquad \eta_i \sim N(0, I_M)$$

Each individual draws one coefficient vector and keeps it for all of their choices,
so a panel is what identifies the standard deviations `σ`. Estimation is by maximum
simulated likelihood with an analytic gradient.

### Simulate a panel

400 people, 21 choices each, with random coefficients on `dur` and `dist`:

```julia
using LogitTools, DataFrames, Random, LogExpFunctions

Random.seed!(20260808)
N, T = 400, 21
df = DataFrame(personid  = repeat(1:N, inner = T),
               dur       = randn(N*T),
               total_pay = randn(N*T),
               dist      = randn(N*T))

beta  = [-0.10, 0.10, -0.30]      # dur, total_pay, dist
sigma = [ 0.70, 0.40]             # sd of the random coefficients on dur, dist

b_dur  = repeat(beta[1] .+ sigma[1] .* randn(N), inner = T)
b_dist = repeat(beta[3] .+ sigma[2] .* randn(N), inner = T)

v = @. b_dur * df.dur + beta[2] * df.total_pay + b_dist * df.dist
df.pick1 = Float64.(rand(N*T) .< logistic.(v))

myxs  = [:dur, :total_pay, :dist]
myrfx = [:dur, :dist]
```

### 1. Starting values

Start `β` at the plain logit and `σ` at 0.5. Never start `σ` at 0.0 — it is a
saddle point of the simulated likelihood, and `logit2_rfx` rejects it.

```julia
logit_fit = logit2(copy(df), myxs, :pick1, zeros(3))
theta0    = theta0_rfx(myxs, myrfx; b0 = logit_fit.theta_hat)
# [-0.1236, 0.0967, -0.232, 0.5, 0.5]
```

### 2. Fit

`personid` is **positional** (the model is undefined without a group id).
`rfx` is a **keyword**, so it cannot be silently transposed with the formula.

```julia
fit = logit2_rfx(df, myxs, :pick1, :personid, theta0;
                 rfx    = myrfx,
                 ndraws = 1000,
                 seed   = 20260808)
```

```
  dur         -0.1506        (true -0.10)
  total_pay    0.1061        (true  0.10)
  dist        -0.2615        (true -0.30)
  sd_dur       0.6456        (true  0.70)
  sd_dist      0.3700        (true  0.40)
converged = true, iterations = 10
ESS: min 52.3  p10 182.1  median 448.4  (R = 1000)
```

`logit2_rfx` does **not** mutate `df`. Parameters are ordered `[β; σ]`, named
`["dur", "total_pay", "dist", "sd_dur", "sd_dist"]`.

The **ESS** line is the effective number of draws per person. A long panel
concentrates the posterior over `ηᵢ`, so many draws contribute nothing; the fit
warns if the 10th percentile drops below 30, which means raise `ndraws`.

### 3. Bootstrap

Resamples at the `personid` level. Workers never receive the DataFrame — only
numeric arrays — so the one setup step is loading the package everywhere.

```julia
using Distributed
addprocs(4)
@everywhere using LogitTools

fit.vcov = boot_logit2_rfx(df, myxs, :pick1, :personid, theta0;
                           rfx         = myrfx,
                           ndraws      = 1000,
                           seed        = 20260808,
                           nboot       = 200,
                           boot_seed   = 12345,
                           cluster_var = :personid,     # must equal the group id
                           parallel    = true,
                           theta_start = fit.theta_hat) # warm start; much faster
```

### 4. Report

```julia
boot_report(fit)
```

```
Bayesian bootstrap for logit2_rfx
  replicates : 200 attempted / 200 used / 0 dropped
  n_obs      : 8400   n_groups: 400   (col_id = :personid)
  T_i        : min 21 / median 21.0 / max 21
  draws      : R = 1000  (seed 20260808)
  ESS        : min 52 / p10 182 / median 448

5×7 DataFrame
 Row │ param      is_sd  estimate   boot_se    ci_lo       ci_hi       share_near_zero
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ dur        false  -0.15058   0.0398415  -0.221442   -0.0612216            NaN
   2 │ total_pay  false   0.10613   0.0251664   0.0540375   0.15097              NaN
   3 │ dist       false  -0.261511  0.0316297  -0.322111   -0.198472             NaN
   4 │ sd_dur      true   0.645628  0.0396107   0.570609    0.709419               0.0
   5 │ sd_dist     true   0.370021  0.0428762   0.294295    0.463987               0.0
```

All five true values fall inside their 95% percentile intervals.

`regtable_rfx(fit)` prints a **95% percentile CI under the `sd_` rows** and keeps
standard errors under the `β` rows:

```
----------------------
dur           -0.151***
                (0.040)
total_pay      0.106***
                (0.025)
dist          -0.262***
                (0.032)
sd_dur         0.646***
         (0.571, 0.709)
sd_dist        0.370***
         (0.294, 0.464)
----------------------
N                  8,400
----------------------
```

```julia
regtable_rfx(fit; ci_levels = [5, 95])   # 90% interval
regtable_rfx(fit; ci_for_sd = false)     # standard errors throughout
```

(Needs RegressionTables 0.7+; `ci_for_sd = false` works on any version.)

Plain `regtable(fit)` also still works, since `logit2_rfx` returns an ordinary
`MLEFit`:

```
---------------------
dur         -0.151***
              (0.040)
total_pay    0.106***
              (0.025)
dist        -0.262***
              (0.032)
sd_dur       0.646***
              (0.040)
sd_dist      0.370***
              (0.043)
---------------------
N               8,400
---------------------
```

### Reading the `σ` rows

**Lead with the percentile CI, not the standard error.** The sign of `σ` is not
identified — the likelihood satisfies `Q(β, σ) = Q(β, -σ)` — so estimates are
canonicalised to `σ ≥ 0`. That folds the sampling distribution and makes it
skewed, increasingly so the closer `σ` sits to zero, and a symmetric `±1.96·se`
interval becomes the wrong summary. A `σ̂` of `0.015` with `boot_se` `0.200`
would put most of its Wald interval below zero.

`share_near_zero` — the share of converged replicates with `σ̂ₘ` below `sd_tol` —
is the more informative statistic for whether homogeneity is rejected.

A confidence interval for a variance parameter that touches zero is a boundary
problem, so read such an interval as "cannot reject homogeneity" rather than as
a calibrated interval.

### Before reporting results on real data

Estimates depend on the simulation draws. Refit at two or three other `seed`
values and at `2 * ndraws`, and compare the spread in `σ̂` against `boot_se`. If
the seed-to-seed spread is an appreciable fraction of the bootstrap standard
error, `ndraws` is too small. There is a ready-made block at the bottom of
`examples/example_logit2_rfx.jl`.

## Examples

- `examples/example.jl` — binary logit and its bootstrap
- `examples/example_mlogit.jl` — multinomial logit
- `examples/example_logit2_rfx.jl` — random coefficients end to end

## Documentation

Docstrings are available at the REPL (`?logit2_rfx`). To build the HTML docs
locally:

```julia
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

then open `docs/build/index.html`.

## Todo

- add other discrete choice models
- use actual formula from StatsAPI
- fixed effects
- code asymptotic variance covariance (?)
- random coefficients: Halton / Gauss-Hermite draws, lognormal coefficients,
  and `logitn_rfx` for the multinomial case (hooks are in place)
