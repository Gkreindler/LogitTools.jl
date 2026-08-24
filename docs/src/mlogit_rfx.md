```@meta
CurrentModule = LogitTools
```

# Option-level random effects

[`mlogit_rfx`](@ref) is the random-coefficient version of [`mlogit`](@ref). It
does everything [`logit2_rfx`](@ref) does, and adds the one thing the binary,
wide layout cannot express: a random effect that lives on the **option**.

## Why a third level

`logit2_rfx` knows a single identifier. `col_id` is simultaneously the panel unit
and the level at which every draw lives, so every random coefficient is a taste
that a person carries into all of their choices.

An option-level effect is different. It is person `i`'s idiosyncratic taste for a
particular *alternative* — a neighborhood, a product, a brand — shared across
every choice in which `i` meets that alternative, and different from the taste
they have for the alternative sitting next to it. `mlogit_rfx` therefore separates
three things:

| | meaning |
|---|---|
| `col_group` | integration unit (the individual). One simulated integral per group. |
| `col_id` | choice set: the rows the softmax runs over. Nested in `col_group`. |
| a term's `level` | the identifier at which *that* term's draw varies. Defaults to `col_group`. |

A term whose `level` varies *within* a choice set is an option-level random
effect. A term at `col_group` is exactly a `logit2_rfx` random coefficient.

## The model

For group ``i``, choice set ``c``, option ``j``, draw ``r``:

```math
v_{cjr} = \sum_k \tilde{x}_{cjk}\,\beta_k
        + \sum_m Z_{cjm}\, A_m\!\left(\mathrm{cell}_m(cj),\, r\right)
```

```math
\ell_{cr} = v_{c,\mathrm{sel}(c),r} - \log\!\!\sum_{j \in c} e^{v_{cjr}},
\qquad
\log \hat{L}_i = \log\!\!\sum_r \exp\!\Big( \sum_{c \in i} \ell_{cr} + \log w_r \Big)
```

and the objective is ``Q(\theta) = -\sum_i \omega_i \log \hat{L}_i`` with
``\theta = [\mu\ (K);\ \sigma\ (M)]``, minimised by LBFGS with an analytic
gradient. `A_m` is the per-cell coefficient contribution:

| `dist` | ``A_m(g,r)`` | support |
|---|---|---|
| `:normal` | ``\sigma_m \eta_m[g,r]`` | ``\mathbb{R}`` |
| `:lognormal` | ``\exp(\mu_m + \sigma_m \eta_m[g,r])`` | ``(0,\infty)`` |
| `:neg_lognormal` | ``-\exp(\mu_m + \sigma_m \eta_m[g,r])`` | ``(-\infty,0)`` |

A **cell** is a `(group, level value)` pair. That indexing is the whole trick: the
draw is `ν_{i,ℓ}`, so the level column is interacted with `col_group` and the
likelihood stays a sum of independent group terms. A random effect genuinely
common to all individuals — a `ξ_ℓ` with no `i` — cannot be fitted this way,
because nothing would factorise.

!!! warning "The level is read *inside* an individual"
    `nbh_code` takes the same values for different people, and `mlogit_rfx` does
    **not** give them a shared draw. It estimates the spread of person-specific
    tastes for each alternative, not an alternative fixed effect with a
    distribution over the population. Put the mean effect of an alternative in
    `formula` as a dummy if you want it.

## Specifying terms

```julia
rfx = [:any_fam,                              # agent-level normal coefficient
       :dur => :neg_lognormal,                # agent-level lognormal coefficient
       rfx_term(level = :nbh_code),           # option-level random INTERCEPT
       rfx_term(:salient; level = :nbh_code)] # option-level random slope
```

Plain `Symbol` and `Pair` entries mean exactly what they mean in `logit2_rfx`, so
existing `rfx` vectors carry over unchanged. [`rfx_term`](@ref) adds the level and
the intercept form.

An intercept term (`var = nothing`) has **no** matching formula coefficient: its
mean is fixed at zero, which is the right normalisation because the mean effect of
a level value is either a fixed effect in `formula` or not identified at all. Its
parameter is named `sd_1|level`, following the `lme4` convention; a slope term is
`sd_var|level`.

The same variable may carry two terms at two levels — that is a variance
decomposition, `β_{i,ℓ} = μ + σ_A ν_i + σ_B ν_{i,ℓ}` — and both `σ`'s are
identified as long as the loading varies within cells.

## Binary choice

A choice set of two rows *is* a binary logit, and that is how you get an
option-level effect into a binary model: in the wide, differenced `logit2` layout
there is no single column holding "this row's alternative", so a level has nothing
to point at. Reshape to long — two rows per choice, the alternatives' attributes
undifferenced, one column naming each row's alternative — and it works.

The two models agree to floating-point noise on that case, with the **same draws**:
the cell layout is built so that a model whose every term sits at the group level
consumes exactly `logit2_rfx`'s RNG stream. The test suite checks the objective and
the full gradient agree to `1e-14`, which is a real cross-check of the kernel
rather than a simulation-noise comparison.

## What is not identified

Three failures are caught in prep rather than left to produce a plausible number:

1. **The term cancels in the softmax.** Only relative utilities matter inside a
   choice set, so a term contributes nothing whenever `Z_cj · A(cell(cj))` is
   constant across the set's options. A **group-level random intercept** is the
   leading example — it shifts every option equally — and so is a random
   coefficient on a regressor that is constant within the choice set. Error when it
   holds for every set, warning when it holds for most.
2. **One row per cell.** The draw then perturbs a single utility and is not
   separable from the extreme-value noise already there. Error; a merely low median
   gets a warning.
3. **A slope collinear with an intercept at the same level.** A loading that
   barely varies inside its own cells is a rescaled intercept, so the two
   variances trade off. Warning, raised only when an intercept term at that level
   is actually in the model.

[`rfx_cell_report`](@ref) is the table behind all three.

## Simulation error

This is the part that differs most from `logit2_rfx` in practice.

**The integral's dimension is the number of cells in a group, not the number of
terms.** An option-level effect over 20 alternatives is a 20-dimensional integral
being done with the same `R` draws, so `ess_p10` falls much faster in `R` than the
binary case would lead you to expect. Read `cells_per_group_max` from
`rfx_cell_report`, **summed over the terms**, alongside `extra.ess_p10`: a large
dimension with a small ESS says the draw set is too thin for the model, not that
the model is wrong. The `ess_p10` warning already reports that summed figure.

Everything the `logit2_rfx` page says about choosing `ndraws` still applies, and
applies harder:

- **ESS alone is not enough.** Draws are held fixed across bootstrap replicates —
  they must be, or the objective is not a deterministic function of `θ` and LBFGS
  fails — so simulation error never enters `boot_se`.
- Because `σ` is canonicalised with `abs()`, noise around a true zero always reads
  as a *positive* number: too few draws bias a small `σ` **upward**.
- **The real test is refitting at other seeds** and at `2 × ndraws`, and comparing
  the spread in `σ̂` against `boot_se`. Monte Carlo error falls like `1/√R`; if the
  seed-to-seed spread does not fall, the parameter is weakly identified and more
  draws will not help.

## Estimating

```julia
using LogitTools, Optim

myxs  = [:fam, :salient, :tfx, :dist]
myrfx = [:salient, rfx_term(level = :nbh_code)]

# a plain fit first: it is the starting point, and the comparison
plain = mlogit(df, myxs, :uniqueid_choice, :selected, zeros(length(myxs));
               optim_options = Optim.Options(iterations = 100_000, g_tol = 1e-6))

th0 = theta0_mlogit_rfx(myxs, myrfx; b0 = plain.theta_hat, col_group = :uniqueid)

fit = mlogit_rfx(df, myxs, :uniqueid_choice, :selected, th0;
                 col_group = :uniqueid, rfx = myrfx, ndraws = 400,
                 optim_options = Optim.Options(iterations = 100_000, g_tol = 1e-6))

fit.vcov = boot_mlogit_rfx(df, myxs, :uniqueid_choice, :selected, th0;
                           col_group = :uniqueid, rfx = myrfx, ndraws = 400,
                           nboot = 500, cluster_var = :uniqueid,
                           theta_start = fit.theta_hat, parallel = true)

boot_report(fit)
rfx_cell_report(fit)
regtable_rfx(fit)
```

`boot_mlogit_rfx` resamples at `col_group` and nowhere else: a replicate reweights
whole groups, because the group log-likelihood is the independent unit. Resampling
choice sets would break the factorisation the random effects rely on, and
`cluster_var` is checked against `col_group` for exactly that reason.

## Reporting

An `mlogit_rfx` fit is an ordinary `MLEFit` whose `extra` carries the same `K` /
`rfx` / `rfx_cols` fields a `logit2_rfx` fit does, so the whole reporting layer
works on it unchanged: [`boot_report`](@ref), [`regtable_rfx`](@ref),
[`rfx_level_moments`](@ref), [`boot_vcov!`](@ref). Standard errors go under the
`β` rows and bootstrap **percentile** intervals under the `sd_` rows, with no
significance stars on the latter — see the [Random coefficients](@ref) page for
why a Wald test is the wrong instrument at a variance parameter's boundary.

[`rfx_cell_report`](@ref) is the one report with no `logit2_rfx` counterpart,
because there every draw sits on the panel unit by construction.

## Multi-start

As with `logit2_rfx`, pass a matrix of starts — one per **row** — and the best
usable optimum is returned, with every attempt in `fit.fits_df`:

```julia
th0m = theta0_mlogit_rfx_multistart(myxs, myrfx; b0 = plain.theta_hat,
                                    nstarts = 50, col_group = :uniqueid)
fit = mlogit_rfx(df, myxs, :uniqueid_choice, :selected, th0m;
                 col_group = :uniqueid, rfx = myrfx, ndraws = 400, parallel = true)
fit.extra.n_distinct_optima
```

This matters more here than in the binary case whenever two `σ`'s can trade off —
a group-level slope and an option-level intercept on correlated regressors is
exactly that situation. A single fit's `converged = true` says an optimum was
reached, not that it was the maximum.

## Changes to `mlogit`

Bringing `mlogit` up to the same standard, all of it verified **bitwise** against
the previous implementation (`test/test_mlogit.jl` keeps that implementation
verbatim as a permanent oracle):

- The choice-set grouping is computed **once**, in prep. It used to call
  `transform!(groupby(df, col_id), …)` on the caller's DataFrame on every function
  evaluation, which both mutated the caller's data and cost a fresh grouping a few
  thousand times per fit. Measured ~4.6× faster on the objective at 12,000 rows.
- `mlogit` and `boot_mlogit` no longer mutate `data_df` at all — no `u_comp`, `pi`,
  `log_sum_exp` or `__group_count` columns, and no in-place `Float64` conversion
  of the formula columns. (`bbw!` still appends its `bw*` columns, unchanged.)
- `optim_options` on `mlogit` and `boot_mlogit`. **Check `fit.converged`**: the
  default LBFGS cap is 1000 iterations and a full set of alternative dummies can
  run past it, returning a fit that still tabulates.
- `parallel = false` on `boot_mlogit`, mirroring `boot_logit2`. Serial and
  parallel are bitwise identical, and workers never receive the DataFrame.
