```@meta
CurrentModule = LogitTools
```

# LogitTools.jl

Discrete choice estimation in Julia: binary logit, multinomial logit, and binary
logit with **independent normal random coefficients**, each with an analytic
gradient and a Bayesian bootstrap. Results print through
[RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl).

Modelled after [GMMTools.jl](https://github.com/Gkreindler/GMMTools.jl).

## Installation

```julia
] add https://github.com/Gkreindler/LogitTools.jl
```

## What is here

| Model | Estimation | Inference |
|---|---|---|
| Binary logit | [`logit2`](@ref) | [`boot_logit2`](@ref) |
| Multinomial logit | [`mlogit`](@ref) | [`boot_mlogit`](@ref) |
| Binary logit, random coefficients | [`logit2_rfx`](@ref) | [`boot_logit2_rfx`](@ref) |

All three return an `MLEFit`, so `vcov`, `cis` and `regtable` work the same way
across them.

## Quick start

```julia
using LogitTools, DataFrames

fit = logit2(df, [:dur, :total_pay, :dist], :pick1, zeros(3))

fit.vcov = boot_logit2(df, [:dur, :total_pay, :dist], :pick1, zeros(3);
                       cluster_var = :clusterid, nboot = 500)

regtable(fit)
```

To let two of those coefficients vary across individuals:

```julia
theta0 = theta0_rfx([:dur, :total_pay, :dist], [:dur, :dist]; b0 = fit.theta_hat)

rfx_fit = logit2_rfx(df, [:dur, :total_pay, :dist], :pick1, :personid, theta0;
                     rfx = [:dur, :dist], ndraws = 1000)
```

See [Random coefficients](logit2_rfx.md) for the full walkthrough, including the
bootstrap and how to read the standard deviation parameters.

## Runnable examples

- `examples/example.jl` — binary logit and its bootstrap
- `examples/example_mlogit.jl` — multinomial logit
- `examples/example_logit2_rfx.jl` — random coefficients end to end

## Roadmap

- other discrete choice models
- use actual formula from StatsAPI
- fixed effects
- asymptotic variance-covariance (?)
- Halton / Gauss-Hermite draws, lognormal random coefficients, `logitn_rfx`
  (hooks are in place; see [Extension points](logit2_rfx.md#Extension-points))
