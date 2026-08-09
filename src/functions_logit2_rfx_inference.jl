
###############################################################################
# Bayesian bootstrap for logit2_rfx.
#
# Group-level Dirichlet weights, parallelised over replicates with Distributed.
# Prep runs once on the master; workers never see the DataFrame.
###############################################################################

"""
    _check_rfx_workers()

Verify that every worker can see `LogitTools`. Names the offending workers.
"""
function _check_rfx_workers()
    nprocs() > 1 || error(
        "parallel = true but nprocs() == 1: there are no worker processes. " *
        "Run addprocs(n) followed by @everywhere using LogitTools, or pass parallel = false.")

    bad = Int[]
    for w in workers()
        ok = try
            remotecall_fetch(() -> isdefined(Main, :LogitTools), w)
        catch
            false
        end
        ok || push!(bad, w)
    end

    isempty(bad) || error(
        "LogitTools is not loaded on worker(s) $bad. Run: @everywhere using LogitTools")

    return nothing
end

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
- `theta_start = nothing`: warm start; defaults to `theta0`. Passing the main
  fit's `theta_hat` is usually much faster.
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
    parallel && _check_rfx_workers()

    # prep once, on the master
    P, gw_user = _prep_logit2_rfx(data_df, formula, choice, col_id, rfx,
                                  ndraws, seed, weights)

    theta0 = _check_theta0_rfx(theta0, P)
    th_start = isnothing(theta_start) ? theta0 : _check_theta0_rfx(theta_start, P)

    # group-level Dirichlet weights
    W = _rfx_boot_weights(P.N, nboot, boot_seed)

    # user weights (if any) multiply the bootstrap weights
    Wfull = isnothing(gw_user) ? W : (W .* gw_user)

    # The closure captures P (design matrices + draws) and Wfull. CachingPool
    # serialises it once per worker; each task then transmits only an Int.
    task = b -> _logit2_rfx(P, th_start, Vector{Float64}(view(Wfull, :, b)), optim_options;
                            rethrow_errors = rethrow_errors)

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

"""
    regtable_rfx(fit; kwargs...)

`regtable` for an rfx fit, reporting the covariance type through the existing
public helper `vcov_method` instead of the hardcoded `Vcov.simple()` that the
generic `MLEFit` path uses.

This is a separate entry point rather than a change to `functions_regtable.jl`:
`regtable(fit)` keeps its current behaviour for every existing model.
"""
function regtable_rfx(fit::MLEFit; kwargs...)
    m = LogitRegModel(fit)                       # existing constructor, unmodified
    m_rfx = LogitRegModel(
        coef        = m.coef,
        vcov        = m.vcov,
        vcov_type   = vcov_method(fit),          # <- the public helper
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
    return RegressionTables.regtable(m_rfx; render = AsciiTable(), kwargs...)
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

    return DataFrame(
        param           = names_,
        is_sd           = is_sd,
        estimate        = fit.theta_hat,
        boot_se         = boot_se,
        ci_lo           = ci_lo,
        ci_hi           = ci_hi,
        share_near_zero = share_nz,
    )
end
