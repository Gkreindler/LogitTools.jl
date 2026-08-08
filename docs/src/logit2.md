```@meta
CurrentModule = LogitTools
```

# Binary logit

[`logit2`](@ref) estimates

```math
\Pr(y_{i} = 1) = \Lambda(X_i'\beta), \qquad \Lambda(u) = \frac{1}{1 + e^{-u}}
```

by maximum likelihood, with an analytic gradient passed to `Optim.LBFGS`.

## Estimation

```julia
using LogitTools, DataFrames

myxs = [:dur, :total_pay, :dist]

fit = logit2(choices_df, myxs, :pick1, zeros(length(myxs)))

fit.theta_hat     # coefficients, in formula order
fit.converged
fit.obj_value     # minus log likelihood at the optimum
```

The choice column must contain only `0` and `1`. An optional `weights` keyword
takes a column name.

!!! note
    `logit2` converts the regressor columns to `Float64` **in place** in the
    DataFrame you pass. Pass `copy(df)` if that matters to you.
    [`logit2_rfx`](@ref) does not do this.

## Bayesian bootstrap

[`boot_logit2`](@ref) draws Dirichlet weights and re-estimates. `cluster_var`
resamples at the cluster level rather than the row level.

```julia
fit.vcov = boot_logit2(choices_df, myxs, :pick1, zeros(length(myxs));
                       cluster_var = :clusterid,
                       nboot = 500)

fit.vcov.V                   # covariance matrix
fit.vcov.theta_boot_table    # nboot x nparam replicate estimates
cis(fit)                     # percentile confidence intervals
```

`boot_logit2` appends `bw1 … bwN` weight columns to the DataFrame you pass.

## Multinomial logit

[`mlogit`](@ref) takes a group identifier and a 0/1 selection column, one row
per alternative:

```julia
fit = mlogit(df, [:x1, :x2], :uniqueid, :selected, zeros(2))
fit.vcov = boot_mlogit(df, [:x1, :x2], :uniqueid, :selected, zeros(2); nboot = 500)
```

## Tables

Any `MLEFit` renders through RegressionTables:

```julia
regtable(fit)
regtable(fit1, fit2, fit3)      # several models side by side
```
