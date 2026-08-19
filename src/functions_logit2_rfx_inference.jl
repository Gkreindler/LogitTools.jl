
###############################################################################
# Bayesian bootstrap for logit2_rfx.
#
# Group-level Dirichlet weights, parallelised over replicates with Distributed.
# Prep runs once on the master; workers never see the DataFrame.
###############################################################################

"""
    _rfx_boot_weights(N, nboot, boot_seed) -> Matrix{Float64}

Group-level Bayesian bootstrap (Dirichlet) weights, `N × nboot`, each column
normalised to mean 1.

Deliberately does not use `bbw!`: that mutates the user's DataFrame with `nboot`
extra columns, and master-side DataFrame mutations would never propagate to
workers.
"""
function _rfx_boot_weights(N::Int, nboot::Int, boot_seed::Int)
    rng = MersenneTwister(boot_seed)
    W = Matrix{Float64}(undef, N, nboot)
    for b in 1:nboot
        w = rand(rng, Dirichlet(N, 1.0))
        W[:, b] .= w .* (N / sum(w))     # mean 1; global scale is irrelevant to the argmax
    end
    return W
end

"""
    boot_logit2_rfx(data_df, formula, choice, col_id, theta0; kwargs...) -> MLEvcov

Bayesian bootstrap for [`logit2_rfx`](@ref), resampling at the `col_id` (random
coefficient group) level.

# Keywords
- `rfx`, `ndraws`, `seed`, `weights`, `optim_options`: as in `logit2_rfx`.
  Simulation draws are held fixed across replicates.
- `nboot = 500`: number of bootstrap replicates.
- `boot_seed = 12345`: seed for the Dirichlet weights. With the same `boot_seed`,
  `parallel = true` and `parallel = false` give identical results.
- `cluster_var = nothing`: if given, must equal `col_id`.
- `parallel = true`: distribute replicates over `workers()`. Requires
  `addprocs(n)` and `@everywhere using LogitTools`.
- `theta_start = nothing`: warm start; defaults to `theta0`. Passing the main fit's
  `theta_hat` is usually much faster. May also be an `nstarts × npar` **matrix**, in
  which case every replicate is itself multi-started and its best optimum is kept.
  That costs `nstarts` times the runtime and is the right call when the specification
  has local optima: a single warm start hands every replicate the same basin, so the
  replicates inherit the point estimate's basin rather than exploring the one their own
  resample prefers, and the resulting spread understates or distorts the sampling
  variation. Keep this modest (4-8 starts) — it multiplies a run that is already
  `nboot` fits.
- `rethrow_errors = false`: by default a replicate that throws is captured as
  `errored` and the run continues. Set `true` to let the first exception
  propagate with its stacktrace. When replicates are failing, re-run with
  `parallel = false, nboot = 1, rethrow_errors = true` for a readable error.
- `mydebug = false`: print per-replicate progress (serial path only).

# Notes
Prep runs once on the master and rides along in the `pmap` closure, so workers
never receive the DataFrame — only the numeric arrays in `RfxPrep`. A
`CachingPool` serialises that closure once per worker rather than once per task.

Replicates are dropped from `V` only on numerical failure (errored or
non-converged). Replicates where `σ̂ₘ` lands near zero are legitimate draws from
the sampling distribution; dropping them would bias the standard error downward.
`boot_report` counts them instead.

`theta_boot_table` retains **all** rows (`NaN` for errored replicates); `V` is
computed on survivors only.
"""
function boot_logit2_rfx(
        data_df,
        formula,
        choice,
        col_id::Symbol,
        theta0;
        rfx = Symbol[],
        ndraws::Int = 1000,
        seed::Int = 20260808,
        weights::Union{Nothing, Symbol, String} = nothing,
        nboot::Int = 500,
        boot_seed::Int = 12345,
        cluster_var = nothing,
        parallel::Bool = true,
        theta_start = nothing,
        optim_options::Optim.Options = Optim.Options(),
        rethrow_errors::Bool = false,
        mydebug::Bool = false)

    if !isnothing(cluster_var) && Symbol(cluster_var) != col_id
        error("cluster_var = :$(Symbol(cluster_var)) must equal col_id = :$col_id. " *
              "Clustering has to be at the random-coefficient group level, or the " *
              "group weighting is undefined.")
    end

    nboot >= 2 || error("nboot must be at least 2; got $nboot")

    # check the cluster before prep, so a misconfigured parallel run fails
    # immediately rather than after prep has already been paid for
    parallel && _check_boot_workers()

    # prep once, on the master
    P, gw_user = _prep_logit2_rfx(data_df, formula, choice, col_id, rfx,
                                  ndraws, seed, weights)

    theta0s  = _rfx_theta0_matrix(theta0, P)
    th_start = isnothing(theta_start) ? theta0s : _rfx_theta0_matrix(theta_start, P)
    nstart   = size(th_start, 1)

    # group-level Dirichlet weights
    W = _rfx_boot_weights(P.N, nboot, boot_seed)

    # user weights (if any) multiply the bootstrap weights
    Wfull = isnothing(gw_user) ? W : (W .* gw_user)

    # The closure captures P (design matrices + draws) and Wfull. CachingPool
    # serialises it once per worker; each task then transmits only an Int.
    # With several starts per replicate the inner fit must run SERIALLY: the outer pmap
    # already owns every worker, and nesting would deadlock on the same pool. Each
    # replicate then pays nstart fits, so the run is nstart times longer -- which is the
    # price of not letting every replicate inherit one basin from a single warm start.
    task = if nstart == 1
        ths = vec(th_start)
        b -> _logit2_rfx(P, ths, Vector{Float64}(view(Wfull, :, b)), optim_options;
                         rethrow_errors = rethrow_errors)
    else
        b -> _logit2_rfx_multi(P, th_start, Vector{Float64}(view(Wfull, :, b)),
                               optim_options; parallel = false,
                               rethrow_errors = rethrow_errors,
                               warn_multi = false)
    end

    fits = if parallel
        pmap(task, CachingPool(workers()), 1:nboot)
    else
        map(1:nboot) do b
            mydebug && println("bootstrapping rfx, replicate=", b)
            task(b)
        end
    end

    return _assemble_rfx_boot(fits, P, nboot)
end

"""
    _assemble_rfx_boot(fits, P, nboot) -> MLEvcov

Assemble the bootstrap replicates. Keeps every row of `theta_boot_table`; computes
`V` on converged, non-errored rows only.
"""
function _assemble_rfx_boot(fits, P::RfxPrep, nboot::Int)

    npar = P.K + P.M
    theta_boot_table = Matrix{Float64}(undef, nboot, npar)
    for (b, f) in enumerate(fits)
        theta_boot_table[b, :] .= f.theta_hat
    end

    keep = [(!f.errored) && f.converged for f in fits]
    nkeep = sum(keep)

    # Separate the two failure modes: they have completely different fixes.
    nerr    = count(f -> f.errored, fits)
    noconv  = count(f -> (!f.errored) && (!f.converged), fits)

    if nkeep < 2
        msg = "only $nkeep of $nboot bootstrap replicates were usable " *
              "($nerr errored, $noconv ran but did not converge); " *
              "cannot compute a covariance matrix.\n"
        if nerr > 0
            firsterr = fits[findfirst(f -> f.errored, fits)].error_message
            msg *= "First error was:\n    " * replace(firsterr, "\n" => "\n    ") * "\n" *
                   "Re-run with rethrow_errors = true to get the full stacktrace."
        else
            msg *= "Nothing threw, so this is an optimiser problem: try a better " *
                   "theta_start (e.g. the main fit's theta_hat), or raise the " *
                   "iteration limit via optim_options = Optim.Options(iterations = 5_000)."
        end
        error(msg)
    end

    if nkeep < nboot
        @warn "$(nboot - nkeep) of $nboot bootstrap replicates were dropped " *
              "($nerr errored, $noconv did not converge); " *
              "V is computed from the remaining $nkeep." *
              (nerr > 0 ? "\nFirst error: " *
                          first(split(fits[findfirst(f -> f.errored, fits)].error_message, "\n")) : "")
    end

    return MLEvcov(
        method = :bayesian_bootstrap,
        theta_boot_table = theta_boot_table,   # ALL rows, NaN for errored
        V = cov(theta_boot_table[keep, :]),    # converged rows only
        boot_fits = fits
    )
end


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

# ---- lognormal: from the estimated log scale to reported level moments ------

"""
    _rfx_to_level(θ, K, rfx_pairs, rfx_cols) -> Vector{Float64}

Map `θ = [μ; σ]` onto the vector that gets *reported*. Entries belonging to a
`:normal` coefficient are copied through unchanged; for a lognormal rfx variable
`m` sitting at formula position `k = rfx_cols[m]`,

    out[k]     = ±exp(μ_m + σ_m²/2)                      = E[β_m]
    out[K + m] =  exp(μ_m + σ_m²/2)·sqrt(exp(σ_m²) - 1)  = SD[β_m]

so that a table row holds a level coefficient and a level standard deviation
whatever the distribution, and a lognormal column is comparable with a normal one
row by row.

`expm1(σ²)` rather than `exp(σ²) - 1`: the latter loses most of its significant
digits at the small σ a near-homogeneous coefficient produces, which is exactly
where the number matters.
"""
function _rfx_to_level(θ, K::Int, rfx_pairs, rfx_cols)
    out = collect(Float64, θ)
    for (m, (_, d)) in enumerate(rfx_pairs)
        _rfx_is_log(d) || continue
        k  = rfx_cols[m]
        μ  = θ[k]
        σ  = θ[K + m]
        m1 = exp(μ + σ^2 / 2)                    # E|β|
        out[k]     = _rfx_sign(d) * m1
        out[K + m] = m1 * sqrt(max(expm1(σ^2), 0.0))
    end
    return out
end

"""
    _rfx_log_meta(fit) -> (K, rfx_pairs, rfx_cols) | nothing

`nothing` unless `fit` actually has a lognormal random coefficient, in which case
this is everything the level transform needs. Fits serialised before the
lognormal families existed carry only `:normal` entries and so return `nothing`
here — that is what keeps the reporting path for existing results untouched.
"""
function _rfx_log_meta(fit::MLEFit)
    e = fit.extra
    isnothing(e) && return nothing
    (hasproperty(e, :rfx) && hasproperty(e, :K)) || return nothing
    any(_rfx_is_log(last(p)) for p in e.rfx) || return nothing

    cols = _rfx_cols_of(fit)
    isnothing(cols) && error(
        "this fit has lognormal random coefficients but the formula position of each " *
        "rfx variable could not be recovered from extra.rfx_cols or theta_names, so " *
        "the level moments cannot be located. Refit with the current LogitTools.")
    return (e.K, e.rfx, cols)
end

"""
    _rfx_cols_of(fit) -> Vector{Int} | nothing

Formula index of each rfx variable. Prefers `extra.rfx_cols`; for fits serialised
before that field existed, recovers it by matching the rfx variable names against
the first `K` entries of `theta_names`. `nothing` when neither works — callers
that only need this for labelling should degrade rather than fail.
"""
function _rfx_cols_of(fit::MLEFit)
    e = fit.extra
    isnothing(e) && return nothing
    hasproperty(e, :rfx_cols) && return collect(Int, e.rfx_cols)
    (hasproperty(e, :rfx) && hasproperty(e, :K)) || return nothing
    isnothing(fit.theta_names) && return nothing

    heads = fit.theta_names[1:e.K]
    cols  = Int[]
    for (v, _) in e.rfx
        k = findfirst(==(string(v)), heads)
        isnothing(k) && return nothing
        push!(cols, k)
    end
    return cols
end

"""
    rfx_level_moments(fit; ci_levels = [2.5, 97.5]) -> DataFrame

Implied moments of the coefficient itself, for each **lognormal** random
coefficient in `fit`: `mean = ±exp(μ + σ²/2)`, `median = ±exp(μ)` and
`SD = exp(μ + σ²/2)·sqrt(exp(σ²) - 1)`, each with a bootstrap standard error and
percentile interval computed by transforming the replicates one by one (not by a
delta method — these transforms are strongly nonlinear in σ).

!!! warning "Read the interval, not `boot_se`"
    `boot_se` on a `mean` or `SD` row is reported for completeness and should
    generally **not** be quoted. `exp(μ + σ²/2)` has a heavy right tail whenever `σ`
    is not sharply identified, because the simulated likelihood has a flat ridge
    along which a large `σ` is offset by a very negative `μ`; a few replicates from
    that ridge dominate the second moment while leaving the percentiles alone. In
    practice `boot_se` can be four orders of magnitude larger than the whole
    percentile interval *without anything overflowing at all*. `ci_lo`/`ci_hi` are
    the summary to use.

`n_nonfinite` counts the replicates whose transform was not even representable
(`exp` overflowed). It is the extreme end of the same phenomenon, not the only sign
of it — a zero count does **not** certify `boot_se`. When the count is nonzero,
`boot_se` is computed on the finite replicates so it is never `NaN`, but it may
still be `Inf`, and the affected percentile bound is `Inf` too if the count exceeds
the tail probability (1 replicate in 20 is 5%, above a 2.5% tail; 1 in 500 is not).

Zero rows when `fit` has no lognormal random coefficient, so it is safe to call
unconditionally. `boot_report` appends these rows automatically.
"""
function rfx_level_moments(fit::MLEFit; ci_levels = [2.5, 97.5])

    out = DataFrame(variable = Symbol[], dist = Symbol[], quantity = String[],
                    estimate = Float64[], boot_se = Float64[],
                    ci_lo = Float64[], ci_hi = Float64[], n_nonfinite = Int[])

    meta = _rfx_log_meta(fit)
    isnothing(meta) && return out
    K, rfx_pairs, rcols = meta

    isnothing(fit.vcov) && error("fit has no vcov; run boot_logit2_rfx first")
    keep = _boot_keep_rows(fit)
    sum(keep) >= 2 || error("fewer than 2 usable bootstrap replicates")
    kept = fit.vcov.theta_boot_table[keep, :]
    B    = size(kept, 1)

    keptL = _rfx_boot_to_level(kept, K, rfx_pairs, rcols)
    lvl   = _rfx_to_level(fit.theta_hat, K, rfx_pairs, rcols)
    lo, hi = ci_levels[1], ci_levels[2]

    for (m, (v, d)) in enumerate(rfx_pairs)
        _rfx_is_log(d) || continue
        k = rcols[m]
        s = _rfx_sign(d)

        # median = ±exp(μ): not one of the two rows the table shows, so it is
        # transformed here rather than in _rfx_to_level.
        med      = s * exp(fit.theta_hat[k])
        med_boot = [s * exp(kept[b, k]) for b in 1:B]

        for (q, est, col) in (("mean",   lvl[k],     collect(view(keptL, :, k))),
                              ("median", med,        med_boot),
                              ("SD",     lvl[K + m], collect(view(keptL, :, K + m))))
            # Percentiles need the full column; the standard error is computed on the
            # finite part, because a replicate in the flat mu/sigma ridge maps to an
            # unrepresentable level moment. n_nonfinite is how many, and a nonzero
            # count means boot_se should not be quoted -- read ci_lo/ci_hi instead.
            fin  = filter(isfinite, col)
            nbad = length(col) - length(fin)
            push!(out, (variable = v, dist = d, quantity = q, estimate = est,
                        boot_se = length(fin) >= 2 ? std(fin) : NaN,
                        ci_lo = percentile(col, lo), ci_hi = percentile(col, hi),
                        n_nonfinite = nbad))
        end
    end

    return out
end

"""
    _rfx_log_param_stats(fit, ci_levels) -> Vector{NamedTuple}

The *estimated* log-scale parameters of each lognormal random coefficient in
`fit`: `mu` and `sd` (of `log|β|`) with their bootstrap standard errors, plus a
percentile interval for `sd`. One entry per lognormal variable, empty otherwise.

These come off the **untransformed** replicates, unlike `_rfx_table_stats`, which
reports level moments.
"""
function _rfx_log_param_stats(fit::MLEFit, ci_levels)
    out = NamedTuple[]
    meta = _rfx_log_meta(fit)
    isnothing(meta) && return out
    K, rfx_pairs, rcols = meta

    kept = nothing
    if !isnothing(fit.vcov) && !isnothing(fit.vcov.theta_boot_table)
        keep = _boot_keep_rows(fit)
        sum(keep) >= 2 && (kept = fit.vcov.theta_boot_table[keep, :])
    end
    lo, hi = ci_levels[1], ci_levels[2]

    for (m, (v, d)) in enumerate(rfx_pairs)
        _rfx_is_log(d) || continue
        k, j = rcols[m], K + m
        push!(out, (
            var   = v,
            dist  = d,
            mu    = Float64(fit.theta_hat[k]),
            mu_se = isnothing(kept) ? NaN : std(view(kept, :, k)),
            mu_lo = isnothing(kept) ? NaN : percentile(view(kept, :, k), lo),
            mu_hi = isnothing(kept) ? NaN : percentile(view(kept, :, k), hi),
            sd    = Float64(fit.theta_hat[j]),
            sd_se = isnothing(kept) ? NaN : std(view(kept, :, j)),
            sd_lo = isnothing(kept) ? NaN : percentile(view(kept, :, j), lo),
            sd_hi = isnothing(kept) ? NaN : percentile(view(kept, :, j), hi)))
    end
    return out
end

"""Format one number at a fixed number of decimals, so 0.30 does not print as 0.3."""
_rfx_num(x, d::Int) = Printf.format(Printf.Format("%.$(d)f"), x)

"""
    _rfx_log_param_rows(fits, labels, ci_levels, digits, digits_stats, ci_for_sd)

`extralines` rows carrying the estimated log-scale parameters: one row for `mu`
and one for `sigma` of `log|β|`, per lognormal variable, per column.

The main table prints level moments, which is what makes a lognormal column
comparable with a normal one — but those are not the parameters that were
estimated, and a reader cannot invert `E[β]` and `SD[β]` back to `mu` and `sigma`
without being told the transform. These rows put the estimates themselves in the
table, which is the other half of the usual convention (Revelt & Train 1998;
Train's textbook tables report `mu` and `sigma` and the implied moments together).

With `ci_for_sd = true` (the default) both rows carry a **percentile interval** in
square brackets; with `ci_for_sd = false` both carry a bootstrap **standard error**
in parentheses.

For `sigma` the interval is right for the reason the level SD rows give: its sign is
unidentified, the estimate is canonicalised with `abs()`, and `sigma = 0` sits on the
boundary of the parameter space, so a symmetric interval is not calibrated.

For `mu` the case is less automatic — its sign *is* identified and it is not near a
boundary, so a standard error would be the textbook choice. The interval is the
default anyway because `mu` and `sigma` trade off along a flat ridge of the simulated
likelihood: a replicate that lands there has a large `sigma` offset by a very negative
`mu`, and a handful of such draws dominates `mu`'s bootstrap standard error while
leaving its percentiles alone. When `sigma` is well identified the two agree, and
`ci_for_sd = false` gives the conventional standard errors throughout.

Estimate and below-statistic share one line here rather than taking two, because
these are auxiliary parameters in a panel below the table and a two-line block per
parameter would double an already long footer.
"""
function _rfx_log_param_rows(fits, labels, ci_levels, digits::Int, digits_stats::Int,
                             ci_for_sd::Bool)

    per = [_rfx_log_param_stats(f, ci_levels) for f in fits]
    all(isempty, per) && return Vector{Vector{String}}()

    # Variable order: first appearance across columns, so a table whose lognormal
    # columns carry different rfx sets still reads top to bottom.
    vars = Symbol[]
    for p in per, r in p
        r.var in vars || push!(vars, r.var)
    end

    lab(v) = (s = string(v); isnothing(labels) ? s : string(get(labels, s, s)))

    n    = length(fits)
    rows = Vector{Vector{String}}()
    push!(rows, vcat(["Log-scale parameters of \$\\log\\beta\$"], fill("", n)))

    for v in vars
        murow = vcat(["\\quad \$\\mu\$ " * lab(v)], fill("", n))
        sdrow = vcat(["\\quad \$\\sigma_{\\log}\$ " * lab(v)], fill("", n))

        for (c, p) in enumerate(per)
            i = findfirst(r -> r.var === v, p)
            isnothing(i) && continue          # this column is normal, or lacks v
            r = p[i]

            below(est, se, lo, hi) =
                if ci_for_sd && isfinite(lo) && isfinite(hi)
                    " [" * _rfx_num(lo, digits_stats) * ", " *
                           _rfx_num(hi, digits_stats) * "]"
                elseif isfinite(se)
                    " (" * _rfx_num(se, digits_stats) * ")"
                else
                    ""
                end

            murow[c + 1] = _rfx_num(r.mu, digits) *
                           below(r.mu, r.mu_se, r.mu_lo, r.mu_hi)
            sdrow[c + 1] = _rfx_num(r.sd, digits) *
                           below(r.sd, r.sd_se, r.sd_lo, r.sd_hi)
        end

        push!(rows, murow)
        push!(rows, sdrow)
    end

    return rows
end

"""Apply `_rfx_to_level` row by row to a `nboot × npar` replicate matrix."""
function _rfx_boot_to_level(tbl::AbstractMatrix, K::Int, rfx_pairs, rfx_cols)
    out = similar(tbl, Float64)
    for b in axes(tbl, 1)
        out[b, :] .= _rfx_to_level(view(tbl, b, :), K, rfx_pairs, rfx_cols)
    end
    return out
end

"""
    _rfx_boot_se(kept) -> (se, n_nonfinite)

Per-column bootstrap standard error, computed over the **finite** replicates only,
with the count of what had to be excluded.

Why this is needed rather than a plain `std`. A lognormal level moment is
`exp(mu + sigma^2/2)`. The simulated likelihood has a flat ridge along which a large
`sigma` is offset by a very negative `mu` — a legitimately *converged* replicate can
sit at `sigma = 79`, `mu = -149` — and the level moment there is astronomically
large or `Inf`. `std` over a column containing one `Inf` is `NaN`, which would reach
a table as a printed `NaN`, and even without an `Inf` a handful of `1e14` draws puts
the standard error at `1e12`.

Filtering to finite values keeps the number computable; the count is what makes the
problem visible instead of silent. Note that this does **not** rescue the standard
error as a *summary*: when the count is nonzero, the honest reading is that the
level moment's bootstrap variance is not usefully finite, and the percentile
interval — which a few tail draws cannot move — is the statistic to report. That is
why `ci_for_sd = true` routes those rows to intervals.
"""
function _rfx_boot_se(kept::AbstractMatrix)
    npar = size(kept, 2)
    se   = Vector{Float64}(undef, npar)
    nbad = zeros(Int, npar)
    for j in 1:npar
        c   = view(kept, :, j)
        fin = [x for x in c if isfinite(x)]
        nbad[j] = length(c) - length(fin)
        se[j]   = length(fin) >= 2 ? std(fin) : NaN
    end
    return se, nbad
end


# ---- mixed below-statistic: SE for β rows, percentile CI for σ rows --------

"""
    RfxUnderStat <: RegressionTables.AbstractUnderStatistic

The value printed under a coefficient in [`regtable_rfx`](@ref): a standard error
for the `β` rows (a scalar) and a percentile confidence interval for the `sd_`
rows (a pair). RegressionTables' `below_statistic` is otherwise uniform across
the whole table.
"""
struct RfxUnderStat <: RegressionTables.AbstractUnderStatistic
    val::Union{Float64, Tuple{Float64,Float64}}
    small_as_lt::Bool
end
RfxUnderStat(val) = RfxUnderStat(val, true)

"""
    _rfx_fmt(render, u, digits, small_as_lt) -> String

Render one number. With `small_as_lt`, a nonzero value too small to show at this
many digits prints as `<0.01` (at `digits = 2`) rather than as `0.00`, so a
confidence bound near the boundary is not mistaken for an exact zero. Negative
values of the same size print as `>-0.01`.

LaTeX needs `\$<\$`: a bare `<` in text mode renders as the wrong glyph.
"""
# Above this magnitude a below-statistic is printed in scientific notation instead
# of in full. A bootstrap standard error of 1.8e74 -- which is what the level mean of
# a weakly identified lognormal coefficient produces -- is 75 characters at two
# decimals and would wreck the column; "1.8e74" is four, and says the same thing.
const _RFX_BIG = 1e5

"""One significant digit in compact scientific notation: `1.8e74`, not `1.8e+74`."""
function _rfx_sci(u)
    s = Printf.format(Printf.Format("%.1e"), u)
    s = replace(s, "e+0" => "e", "e-0" => "e-")
    return replace(s, "e+" => "e")
end

function _rfx_fmt(render, u, digits, small_as_lt::Bool)
    # Guard the two ways a below-statistic can be unprintable before anything else.
    # These arise for real: a lognormal level moment is exp(mu + sigma^2/2), and
    # replicates on the flat mu/sigma ridge send its bootstrap variance to something
    # not representable. Printing "n.a." or "1.8e74" says "not estimable" without
    # silently degrading the number or breaking the table.
    isfinite(u) || return "n.a."
    abs(u) >= _RFX_BIG && return _rfx_sci(u)

    s = Base.repr(render, u; digits, commas = false)

    # Decide from the rendered string, not from a reimplemented rounding rule:
    # that way the cutoff is exactly where this renderer prints all zeros, for
    # any `digits`. (Comparing against 10.0^-digits / 2 is off by an ulp, so
    # 0.005 -- which renders as 0.01 -- would be misflagged.)
    if small_as_lt && u != 0 && isfinite(u) && all(c -> !isdigit(c) || c == '0', s)
        islatex = render isa RegressionTables.AbstractLatex
        sym = u > 0 ? (islatex ? "\$<\$" : "<") : (islatex ? "\$>\$-" : ">-")
        # The tight bound. A value prints as 0.00 exactly when |u| < 0.005, so
        # "<0.005" is what is actually known; "<0.01" would be true but loose.
        # It needs one more decimal than the rest of the column.
        edge = Base.repr(render, 0.5 * 10.0^(-digits); digits = digits + 1,
                         commas = false)
        return sym * edge
    end
    return s
end

# A scalar (a standard error) renders in parentheses like StdError; a pair (a
# confidence interval) renders in SQUARE BRACKETS, so that a table mixing the two
# — β rows with standard errors, σ rows with intervals — is readable without
# having to consult the notes.
function Base.repr(render::RegressionTables.AbstractRenderType, x::RfxUnderStat;
                   digits = RegressionTables.default_digits(render, 0.0), args...)
    v = x.val
    if v isa Tuple
        return "[" * _rfx_fmt(render, v[1], digits, x.small_as_lt) * ", " *
                     _rfx_fmt(render, v[2], digits, x.small_as_lt) * "]"
    end
    return RegressionTables.below_decoration(render,
                                             _rfx_fmt(render, v, digits, x.small_as_lt))
end

"""
    RfxBelowStatistic

Callable passed as `below_statistic`. RegressionTables invokes it as
`(rr, k)` for coefficient `k` of model `rr`, which is what makes per-row
dispatch possible. Keyed by model identity so a multi-model table stays correct.

With `ci_for_sd = true` the `σ` rows get a percentile interval; otherwise every
row gets its standard error.
"""
struct RfxBelowStatistic
    tbl::IdDict{Any, NamedTuple}
    ci_for_sd::Bool
    small_as_lt::Bool
    mean_stat::Symbol      # :se or :ci, for a lognormal E[β] row
end
RfxBelowStatistic(tbl, ci_for_sd) = RfxBelowStatistic(tbl, ci_for_sd, true, :se)
RfxBelowStatistic(tbl, ci_for_sd, small_as_lt) =
    RfxBelowStatistic(tbl, ci_for_sd, small_as_lt, :se)

function (f::RfxBelowStatistic)(rr, k::Int; vargs...)
    d = f.tbl[rr]

    # Three cases, and only the middle one is a judgement call.
    #
    #   sigma / SD rows      -> interval when ci_for_sd: sign unidentified, abs()
    #                           canonicalised, sigma = 0 on the boundary.
    #   lognormal E[b] rows  -> standard error by default (mean_stat = :se), so the
    #                           row matches the beta rows of the normal columns it
    #                           sits beside. E[b] is a mean with an identified sign,
    #                           which is the textbook case for a standard error; the
    #                           cost is that the bootstrap variance of exp(mu +
    #                           sigma^2/2) is fragile when sigma is loosely pinned,
    #                           and such an SE prints in scientific notation rather
    #                           than being quietly replaced. mean_stat = :ci opts out.
    #   everything else      -> standard error, exactly as before.
    use_ci = if d.is_sd[k]
        f.ci_for_sd
    elseif d.is_log_row[k]
        f.mean_stat === :ci
    else
        false
    end

    return use_ci ?
        RfxUnderStat((d.ci_lo[k], d.ci_hi[k]), f.small_as_lt) :
        RfxUnderStat(d.se[k], f.small_as_lt)
end

# A variance large enough that coef/se rounds to a t-statistic of zero, so
# RegressionTables' p-value comes out at ~1 and no significance stars are drawn.
const _RFX_NO_STARS_VAR = 1e24

"""
    _rfx_table_stats(fit, ci_levels) -> (; is_sd, is_log_row, coef, se, ci_lo, ci_hi)

Everything one fit contributes to a table row: the displayed point estimate, its
standard error, and its percentile interval.

`coef` is `theta_hat` unless the fit has a lognormal random coefficient, in which
case the two rows belonging to it hold `E[β]` and `SD[β]` instead of `μ` and
`σ_log` — see [`_rfx_to_level`](@ref). The bootstrap replicates are then put
through the same transform *before* the standard error and the interval are taken,
so all three numbers in a row describe the same quantity. `is_log_row` marks the
transformed rows, because a Wald test against zero is vacuous there: `E[β]` and
`SD[β]` are positive by construction under a lognormal.
"""
function _rfx_table_stats(fit::MLEFit, ci_levels)
    npar = length(fit.theta_hat)
    e = fit.extra
    K = isnothing(e) ? npar : e.K
    is_sd = [j > K for j in 1:npar]

    meta = _rfx_log_meta(fit)          # nothing unless a lognormal rfx is present

    is_log_row = falses(npar)
    if !isnothing(meta)
        Kl, rfx_pairs, rcols = meta
        for (m, (_, d)) in enumerate(rfx_pairs)
            _rfx_is_log(d) || continue
            is_log_row[rcols[m]] = true
            is_log_row[Kl + m]   = true
        end
    end

    coef = isnothing(meta) ? collect(Float64, fit.theta_hat) :
                             _rfx_to_level(fit.theta_hat, meta...)

    # Kept replicates, on the reported scale.
    kept = nothing
    if !isnothing(fit.vcov) && !isnothing(fit.vcov.theta_boot_table)
        nbootpar = size(fit.vcov.theta_boot_table, 2)
        nbootpar == npar || error(
            "theta_boot_table has $nbootpar columns but the fit has $npar parameters. " *
            "This vcov belongs to a different model.")
        keep = _boot_keep_rows(fit)
        if sum(keep) >= 2
            kept = fit.vcov.theta_boot_table[keep, :]
            isnothing(meta) || (kept = _rfx_boot_to_level(kept, meta...))
        end
    end

    # With no lognormal row this is diag(vcov(fit)) exactly as before. With one,
    # vcov(fit) describes μ and σ_log and so says nothing about the SE of the
    # transformed quantity; the replicates do. Same convention either way, since
    # V is itself cov(theta_boot_table[keep, :]).
    se = if isnothing(meta)
        V = vcov(fit)
        [sqrt(max(V[j, j], 0.0)) for j in 1:npar]
    else
        isnothing(kept) && error(
            "a fit with lognormal random coefficients needs bootstrap replicates to be " *
            "tabulated: the standard error of the implied level moment cannot be read " *
            "off vcov(fit), which describes μ and σ_log. Run boot_logit2_rfx first.")
        s, nbad = _rfx_boot_se(kept)
        if any(nbad[is_log_row] .> 0)
            j = findfirst(j -> is_log_row[j] && nbad[j] > 0, 1:npar)
            @warn "the implied level moment of a lognormal coefficient overflowed in " *
                  "$(nbad[j]) of $(size(kept, 1)) bootstrap replicates (first affected " *
                  "row: $(isnothing(fit.theta_names) ? j : fit.theta_names[j])). Those " *
                  "replicates sit in the flat μ/σ ridge, where exp(μ + σ²/2) is not " *
                  "representable, so that row's bootstrap standard error is not " *
                  "usefully finite -- it will print in scientific notation, which is " *
                  "the table saying the second moment does not exist rather than " *
                  "reporting a precision. The percentile interval is unaffected: see " *
                  "rfx_level_moments, or pass lognormal_mean_stat = :ci to put the " *
                  "interval in the table instead."
        end
        s
    end

    ci_lo = fill(NaN, npar)
    ci_hi = fill(NaN, npar)
    if !isnothing(kept)
        for j in 1:npar
            ci_lo[j] = percentile(view(kept, :, j), ci_levels[1])
            ci_hi[j] = percentile(view(kept, :, j), ci_levels[2])
        end
    end

    return (; is_sd, is_log_row, coef, se, ci_lo, ci_hi)
end

"""
    regtable_rfx(fits::MLEFit...; ci_for_sd = true, ci_levels = [2.5, 97.5],
                 stars_for_sd = false, stars_for_lognormal = false, kwargs...)

`regtable` for random-coefficient fits, with four differences from the generic
`MLEFit` path:

0. A **lognormal** random coefficient is reported on the level scale: its two
   rows hold `E[β] = ±exp(μ + σ²/2)` and `SD[β]`, not the `μ` and `σ_log` that
   were estimated, with the bootstrap replicates transformed before the standard
   error and the interval are taken. This is what makes a lognormal column
   comparable with a normal one row by row, since the log-scale parameters are on
   a different scale from every other coefficient in the table. The estimated
   `μ` and `σ_log` are **not** dropped: with `log_params = true` (the default) they
   are appended as `extralines`, one line each per lognormal variable — see
   [`_rfx_log_param_rows`](@ref) for the below-statistic they carry. The full set of
   level moments, including the median, is in [`rfx_level_moments`](@ref), and
   `boot_report` carries both scales.

   The `E[β]` row carries a **standard error** in parentheses, like the `β` rows of
   the normal columns it sits beside: `E[β]` is a mean with an identified sign, which
   is the textbook case for a standard error. The `SD[β]` row keeps the **percentile
   interval** in square brackets, as the `σ` rows do.

   Be aware of what that costs. `E[β] = exp(μ + σ²/2)` and the simulated likelihood
   has a flat ridge along which a large `σ` is offset by a very negative `μ`; a
   replicate converging there maps to a level mean of `1e14` or to `Inf`, and the
   bootstrap variance stops being usefully finite while the percentiles stay put. Such
   a standard error prints in compact scientific notation (`1.8e74`) or as `n.a.`,
   never as a plausible-looking number, and a warning names the row. That is the table
   reporting that the second moment does not exist, not a precision. Pass
   `lognormal_mean_stat = :ci` to put the percentile interval on that row instead, or
   read `rfx_level_moments`, which always carries both.

   Such a fit needs bootstrap replicates and RegressionTables 0.7+, and by default
   carries no significance stars on either of its two main rows — `E[β]` and `SD[β]`
   are positive by construction, so a Wald test against zero there is vacuous. Pass
   `stars_for_lognormal = true` to restore them.
1. The covariance type is reported through the public `vcov_method` helper
   rather than the hardcoded `Vcov.simple()`.
2. With `ci_for_sd = true` (the default) the `sd_` rows print a **percentile
   confidence interval** from the bootstrap replicates, in square brackets, while
   the `β` rows keep their standard error in parentheses.
3. With `stars_for_sd = false` (the default) the `sd_` rows carry **no
   significance stars**.

Points 2 and 3 are the statistically meaningful ones, and they have the same
cause. Because `σ`'s sign is not identified, estimates are canonicalised with
`abs()`, which folds the sampling distribution and makes it skewed — so a
symmetric `±1.96·se` interval is the wrong summary for those rows, increasingly
so the closer `σ` sits to zero. For the same reason a Wald test against zero is
not calibrated there: `σ = 0` is on the boundary of the parameter space, where
the LR statistic is a `½χ²₀ + ½χ²₁` mixture rather than `χ²₁`. Stars on those
rows would invite exactly the reading they cannot support, so they are off by
default; read the interval instead.

Pass `ci_for_sd = false` for standard errors throughout,
`stars_for_sd = true` to restore the conventional stars on the `sd_` rows, and
`stars_for_lognormal = true` for those on a lognormal coefficient's two rows. Any
other keyword is forwarded to `regtable` — including `digits` and `digits_stats`, which set the
number of digits for the estimates and for the below-statistics respectively.
Note that RegressionTables only applies `digits_stats` when `digits` is also
given, so pass both.

`small_as_lt = true` (the default) prints a below-statistic that is nonzero but
too small to show at this many digits as `<0.005` (at `digits_stats = 2`) rather
than as `0.00`, so a confidence bound sitting near the boundary is not mistaken
for an exact zero. `0.00` is printed exactly when the value is below half of the
last digit, so `<0.005` is the tight bound; it carries one more decimal than the
rest of the column. It applies to every below-statistic — the `β` standard errors
as well as the `σ` intervals. Pass `small_as_lt = false` for plain rounding.
It has no effect on the `ci_for_sd = false, stars_for_sd = true` path, which
delegates entirely to `regtable`.

!!! note
    Per-row behaviour (`ci_for_sd = true` or `stars_for_sd = false`) needs
    RegressionTables 0.7 or newer, the first version whose `below_statistic`
    receives the coefficient index. On 0.6.x this throws with an explanatory
    message; use `ci_for_sd = false, stars_for_sd = true` there.

# Example
```julia
fit.vcov = boot_logit2_rfx(df, myxs, :pick1, :personid, theta0; rfx = myrfx)

regtable_rfx(fit)                                  # SE under β, 95% CI under σ, no σ stars
regtable_rfx(fit; digits = 2, digits_stats = 2)     # two digits everywhere
regtable_rfx(fit; ci_levels = [5, 95])              # 90% interval
regtable_rfx(fit; ci_for_sd = false, stars_for_sd = true)   # conventional table
```
"""
function regtable_rfx(fits::MLEFit...; ci_for_sd::Bool = true,
                      ci_levels = [2.5, 97.5], stars_for_sd::Bool = false,
                      stars_for_lognormal::Bool = false,
                      log_params::Bool = true,
                      lognormal_mean_stat::Symbol = :se,
                      small_as_lt::Bool = true, kwargs...)

    isempty(fits) && error("regtable_rfx needs at least one MLEFit")
    lognormal_mean_stat in (:se, :ci) || error(
        "lognormal_mean_stat must be :se or :ci, got :$lognormal_mean_stat")

    # Stats first, then the rendering path, then the models. The order matters:
    # star suppression writes a huge variance onto the vcov diagonal, and that is
    # invisible only on the path where every below-statistic comes from `stats`
    # instead of from that matrix. So the decision has to be made before the
    # models are built.
    ds = [_rfx_table_stats(fit, ci_levels) for fit in fits]

    any_sd  = any(any(d.is_sd)      for d in ds)
    any_log = any(any(d.is_log_row) for d in ds)

    # A lognormal fit always takes the per-row path, even with
    # stars_for_lognormal = true: its displayed coefficients are level moments,
    # whose standard errors live in `stats` and not in vcov(fit).
    plain = (!any_sd || (!ci_for_sd && stars_for_sd)) && !any_log

    models = LogitRegModel[]
    stats  = IdDict{Any, NamedTuple}()

    for (fit, d) in zip(fits, ds)
        m = LogitRegModel(fit)                   # existing constructor, unmodified

        # Suppressing stars on the σ rows: RegressionTables derives its p-values
        # from coef / sqrt(diag(vcov)), so reporting a huge variance for those
        # rows drives the t-statistic to ~0 and the p-value to ~1. The number
        # itself is never displayed -- those rows' below-statistic comes from
        # `stats`, not from this matrix -- so nothing else changes. Lognormal rows
        # get the same treatment by default: E[β] and SD[β] are positive by
        # construction, so a Wald test against zero there means nothing.
        vc = m.vcov
        if !plain
            nostar = falses(length(d.is_sd))
            stars_for_sd        || (nostar .|= d.is_sd)
            stars_for_lognormal || (nostar .|= d.is_log_row)
            if any(nostar)
                vc = copy(vc)
                for j in findall(nostar)
                    vc[j, j] = _RFX_NO_STARS_VAR
                end
            end
        end

        m_rfx = LogitRegModel(
            coef        = d.coef,
            vcov        = vc,
            vcov_type   = vcov_method(fit),      # <- the public helper
            esample     = m.esample,
            fe          = m.fe,
            fekeys      = m.fekeys,
            coefnames   = m.coefnames,
            responsename = m.responsename,
            contrasts   = m.contrasts,
            nobs        = m.nobs,
            dof         = m.dof,
            dof_fes     = m.dof_fes,
            dof_residual = m.dof_residual,
            rss         = m.rss,
            tss         = m.tss,
            F           = m.F,
            p           = m.p,
            iterations  = m.iterations,
            converged   = m.converged,
        )
        push!(models, m_rfx)
        stats[m_rfx] = d
    end

    # The plain regtable path is enough only when there is nothing to vary by
    # row: no σ rows at all, or σ rows that want the ordinary standard error and
    # the ordinary stars, and no lognormal coefficient. It is also the only path
    # that works on RegressionTables 0.6.
    if plain
        return RegressionTables.regtable(models...; render = AsciiTable(), kwargs...)
    end

    _rfx_check_below_statistic_api()

    if ci_for_sd
        for m in models
            d = stats[m]
            if any(d.is_sd) && !all(isfinite, d.ci_lo[d.is_sd])
                error("regtable_rfx(ci_for_sd = true) needs bootstrap replicates for the " *
                      "sd_ rows, but this fit has no usable theta_boot_table. Run " *
                      "boot_logit2_rfx first, or pass ci_for_sd = false.")
            end
        end
    end

    kw = Dict{Symbol,Any}(kwargs)

    # The estimated log-scale parameters go into `extralines`, ahead of whatever the
    # caller passed: they are estimates, and the caller's lines are usually
    # specification descriptors ("Random coefficients", "Simulation draws"), which
    # read better underneath. `digits`/`digits_stats` follow the table.
    if log_params && any_log
        dg  = get(kw, :digits, 3)
        dgs = get(kw, :digits_stats, dg)
        rows = _rfx_log_param_rows(fits, get(kw, :labels, nothing), ci_levels,
                                   dg, dgs, ci_for_sd)
        if !isempty(rows)
            user = get(kw, :extralines, nothing)
            kw[:extralines] = isnothing(user) ? rows : vcat(rows, collect(user))
        end
    end

    return RegressionTables.regtable(models...;
                                     render = AsciiTable(),
                                     below_statistic = RfxBelowStatistic(stats, ci_for_sd,
                                                                         small_as_lt,
                                                                         lognormal_mean_stat),
                                     kw...)
end

"""Feature-detect the RegressionTables API that per-row below statistics need."""
function _rfx_check_below_statistic_api()
    ok = hasmethod(RegressionTables.StdError,
                   Tuple{RegressionTables.RegressionModel, Int})
    ok || error(
        "regtable_rfx with ci_for_sd = true or stars_for_sd = false requires " *
        "RegressionTables 0.7 or newer: older versions call below_statistic with " *
        "(se, coef, dof) and no coefficient index, so the statistic cannot vary by " *
        "row. Installed version is $(pkgversion(RegressionTables)). Upgrade " *
        "RegressionTables, or call " *
        "regtable_rfx(fit; ci_for_sd = false, stars_for_sd = true).")
    return nothing
end

"""
    _boot_keep_rows(fit) -> BitVector

Which rows of `theta_boot_table` to use. Prefers the convergence flags in
`boot_fits`; falls back to non-`NaN` rows when only the table was saved.
"""
function _boot_keep_rows(fit::MLEFit)
    tbl = fit.vcov.theta_boot_table
    if isnothing(fit.vcov.boot_fits)
        @warn "fit.vcov.boot_fits is nothing (results were probably reloaded from disk " *
              "with only the table saved). Falling back to all rows without NaN."
        return [all(isfinite, view(tbl, b, :)) for b in 1:size(tbl, 1)]
    end
    return [(!f.errored) && f.converged for f in fit.vcov.boot_fits]
end

"""
    boot_vcov!(fit; ) -> MLEFit

Refilter the bootstrap replicates and refresh `fit.vcov.V` in place.
"""
function boot_vcov!(fit::MLEFit)
    isnothing(fit.vcov) && error("fit has no vcov; run boot_logit2_rfx first")

    tbl  = fit.vcov.theta_boot_table
    keep = _boot_keep_rows(fit)
    sum(keep) >= 2 || error("fewer than 2 usable bootstrap replicates; cannot compute V")

    fit.vcov.V = cov(tbl[keep, :])
    return fit
end

"""
    boot_report(fit; ci_levels = [2.5, 97.5], sd_tol = 1e-3) -> DataFrame

One row per parameter, with a header block summarising the run.

Two columns describe *what* each row is: `dist` (`:none` for a plain coefficient,
otherwise the random coefficient's distribution) and `scale`. For a lognormal
random coefficient `scale == "log"`, meaning `estimate` is `μ` or `σ` of `log|β|`
rather than a level — the estimated parameters, which is what
[`rfx_level_moments`](@ref) does not give you. The implied level moments
(`E[.]`, `median[.]`, `SD[.]`) are appended as extra rows with `scale == "level"`,
so the CSV carries both scales and neither has to be reconstructed by hand.

For the `σ` rows, **lead with the percentile CI rather than the standard error**.
The `abs()` canonicalisation folds the sampling distribution, so it is skewed —
increasingly so the closer `σ` sits to zero — and a symmetric `±1.96·se` interval
is the wrong summary. `share_near_zero` (the share of converged replicates with
`σ̂ₘ < sd_tol`) is the more informative statistic for whether homogeneity is
rejected.

!!! note "Boundary problem"
    A confidence interval for a variance parameter that touches zero is a boundary
    problem, so neither the percentile CI nor a Wald test is calibrated exactly at
    `σₘ = 0` (for a single `σ` the LR statistic is a `½χ²₀ + ½χ²₁` mixture). Read
    such an interval as "cannot reject homogeneity" rather than as a calibrated
    interval.
"""
function boot_report(fit::MLEFit; ci_levels = [2.5, 97.5], sd_tol = 1e-3)

    isnothing(fit.vcov) && error("fit has no vcov; run boot_logit2_rfx first")

    tbl   = fit.vcov.theta_boot_table
    nboot = size(tbl, 1)
    keep  = _boot_keep_rows(fit)
    used  = sum(keep)
    used >= 2 || error("fewer than 2 usable bootstrap replicates")

    kept = tbl[keep, :]

    npar  = length(fit.theta_hat)
    names_ = isnothing(fit.theta_names) ? ["theta_$i" for i in 1:npar] : fit.theta_names

    # which rows are sd_ parameters
    e = fit.extra
    K = isnothing(e) ? npar : e.K
    is_sd = [j > K for j in 1:npar]

    # ---- header block ------------------------------------------------------
    println("Bayesian bootstrap for logit2_rfx")
    println("  replicates : $nboot attempted / $used used / $(nboot - used) dropped")
    print("  n_obs      : $(fit.n_obs)")
    if !isnothing(e)
        println("   n_groups: $(e.n_groups)   (col_id = :$(e.col_id))")
        println("  T_i        : min $(e.Ti_min) / median $(e.Ti_median) / max $(e.Ti_max)")
        println("  draws      : R = $(e.R)  (seed $(e.seed))")
        @printf("  ESS        : min %.0f / p10 %.0f / median %.0f\n",
                e.ess_min, e.ess_p10, e.ess_median)
    else
        println()
    end
    println()

    # ---- table -------------------------------------------------------------
    lo, hi = ci_levels[1], ci_levels[2]

    boot_se = [std(view(kept, :, j)) for j in 1:npar]
    ci_lo   = [percentile(view(kept, :, j), lo) for j in 1:npar]
    ci_hi   = [percentile(view(kept, :, j), hi) for j in 1:npar]
    share_nz = [is_sd[j] ? mean(view(kept, :, j) .< sd_tol) : NaN for j in 1:npar]

    # ---- which distribution each row belongs to, and on which scale ---------
    # For a lognormal coefficient, `estimate` is μ or σ of log|β| sitting under a
    # row name ("any_fam", "sd_any_fam") that reads like a level. `scale` is what
    # stops that being misread; the level moments are appended below.
    rfx_pairs = (!isnothing(e) && hasproperty(e, :rfx)) ? collect(e.rfx) :
                                                          Pair{Symbol,Symbol}[]
    rcols = _rfx_cols_of(fit)
    dist  = fill(:none, npar)
    scale = fill("level", npar)
    for (m, (v, d)) in enumerate(rfx_pairs)
        dist[K + m] = d
        _rfx_is_log(d) && (scale[K + m] = "log")
        if !isnothing(rcols)
            dist[rcols[m]] = d
            _rfx_is_log(d) && (scale[rcols[m]] = "log")
        end
    end

    out = DataFrame(
        param           = names_,
        dist            = dist,
        scale           = scale,
        is_sd           = is_sd,
        estimate        = collect(Float64, fit.theta_hat),
        boot_se         = boot_se,
        ci_lo           = ci_lo,
        ci_hi           = ci_hi,
        share_near_zero = share_nz,
        n_nonfinite     = zeros(Int, npar),   # only the level-moment rows can overflow
    )

    # ---- implied level moments for the lognormal coefficients ---------------
    lm = rfx_level_moments(fit; ci_levels = ci_levels)
    if nrow(lm) > 0
        println("  scales     : rows with scale = \"log\" hold μ and σ of log|β|;")
        println("               the E[.] / median[.] / SD[.] rows are level moments.")
        println()
        for r in eachrow(lm)
            push!(out, (param = "$(r.quantity)[$(r.variable)]",
                        dist  = r.dist,
                        scale = "level",
                        is_sd = r.quantity == "SD",
                        estimate = r.estimate,
                        boot_se  = r.boot_se,
                        ci_lo    = r.ci_lo,
                        ci_hi    = r.ci_hi,
                        share_near_zero = NaN,
                        n_nonfinite = r.n_nonfinite))
        end
        if any(lm.n_nonfinite .> 0)
            println("  WARNING    : a level moment overflowed in up to " *
                    "$(maximum(lm.n_nonfinite)) replicate(s); boot_se is not usable")
            println("               for those rows -- read ci_lo/ci_hi instead.")
            println()
        end
    end

    return out
end
