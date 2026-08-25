using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using Revise
using LinearAlgebra
using Random
using DataFrames
using Distributions
using LogExpFunctions
using StatsBase
using Distributed

using LogitTools

# ============================================================================
# Simulate a panel: N individuals, T choices each, three regressors, with
# random coefficients on two of them.
# ============================================================================

    Random.seed!(20260808)

    N = 400          # individuals
    T = 21           # choice situations per individual
    nobs = N * T

    beta_true  = [-0.10, 0.10, -0.30]     # dur, total_pay, dist
    sigma_true = [ 0.70, 0.40]            # sd of the random coefficients on dur, dist

    choices_df = DataFrame(
        personid = repeat(1:N, inner = T),
        dur      = randn(nobs),
        total_pay = randn(nobs),
        dist     = randn(nobs),
    )

    # individual-specific coefficient deviations (constant within personid)
    dev_dur  = repeat(sigma_true[1] .* randn(N), inner = T)
    dev_dist = repeat(sigma_true[2] .* randn(N), inner = T)

    choices_df.v = @. (beta_true[1] + dev_dur)  * choices_df.dur +
                       beta_true[2]             * choices_df.total_pay +
                      (beta_true[3] + dev_dist) * choices_df.dist

    choices_df.pick1 = Float64.(rand(nobs) .< logistic.(choices_df.v))

    myxs   = [:dur, :total_pay, :dist]
    myrfx  = [:dur, :dist]

# ============================================================================
# 1. Plain logit, for starting values
# ============================================================================

    logit_fit = logit2(copy(choices_df), myxs, :pick1, zeros(length(myxs)))
    println("plain logit beta = ", round.(logit_fit.theta_hat, digits = 4))

# ============================================================================
# 2. Assemble theta0: beta from the plain logit, sigma at 0.5
#    (never 0.0 -- sigma = 0 is a saddle point, and logit2_rfx rejects it)
# ============================================================================

    theta0 = theta0_rfx(myxs, myrfx; b0 = logit_fit.theta_hat)
    println("theta0 = ", round.(theta0, digits = 4))

# ============================================================================
# 3. Main random-coefficients fit
# ============================================================================

    @time fit = logit2_rfx(
        choices_df,
        myxs,
        :pick1,
        :personid,          # col_id is POSITIONAL: the model is undefined without it
        theta0;
        rfx    = myrfx,     # rfx is a KEYWORD, so it cannot be transposed with myxs
        ndraws = 1000,
        seed   = 20260808)

    println()
    println("converged = ", fit.converged, "  iterations = ", fit.iterations)
    for (nm, th) in zip(fit.theta_names, fit.theta_hat)
        println("  ", rpad(nm, 12), round(th, digits = 4))
    end
    println("  true:      beta = ", beta_true, "   sigma = ", sigma_true)

    # ESS diagnostic: the effective number of draws per individual. A long panel
    # concentrates the posterior over eta_i, so many draws contribute nothing.
    # Warn territory is ess_p10 < 30.
    println()
    println("effective draws per individual (R = ", fit.extra.R, "):")
    println("  min    = ", round(fit.extra.ess_min,    digits = 1))
    println("  p10    = ", round(fit.extra.ess_p10,    digits = 1))
    println("  median = ", round(fit.extra.ess_median, digits = 1))
    println("  mean   = ", round(fit.extra.ess_mean,   digits = 1))

# ============================================================================
# 4. Bayesian bootstrap, parallel over replicates
#
#    The workers never receive the DataFrame: prep runs once on the master and
#    only the numeric arrays ride along in the pmap closure. The one thing you
#    must do is load LogitTools everywhere.
# ============================================================================

    addprocs(4)
    @everywhere using LogitTools

    @time fit.vcov = boot_logit2_rfx(
        choices_df,
        myxs,
        :pick1,
        :personid,
        theta0;
        rfx         = myrfx,
        ndraws      = 1000,
        seed        = 20260808,
        nboot       = 500,
        boot_seed   = 12345,
        cluster_var = :personid,       # must equal col_id
        parallel    = true,
        theta_start = fit.theta_hat)   # warm start: much faster than theta0

# ============================================================================
# 5. Report
#
#    For the sigma rows, read the percentile CI rather than the standard error:
#    sigma is constrained positive and zero is a boundary, so the sampling
#    distribution is skewed near zero.
#    share_near_zero is the better statistic for whether homogeneity is rejected.
# ============================================================================

    rep = boot_report(fit)
    show(stdout, rep; allrows = true, allcols = true)
    println()

    # regtable still works (logit2_rfx returns an ordinary MLEFit)
    regtable(fit) |> display

    # rfx-specific rendering, reporting the covariance type via vcov_method
    regtable_rfx(fit) |> display

    rmprocs(workers())

# ============================================================================
# 6. Seed-stability check
#
#    THIS IS THE CHECK TO RUN ON REAL DATA BEFORE REPORTING RESULTS.
#
#    The estimates are a simulated likelihood, so they depend on the draw set.
#    Refit at other seeds and at twice the draws, and compare the spread in
#    sigma-hat against boot_se. If the seed-to-seed spread is an appreciable
#    fraction of the bootstrap standard error, ndraws is too small.
#
#    Left commented out because it costs several more fits.
# ============================================================================

# let
#     sds = Float64[]
#     for s in [20260808, 11111, 22222]
#         f = logit2_rfx(choices_df, myxs, :pick1, :personid, theta0;
#                        rfx = myrfx, ndraws = 1000, seed = s)
#         push!(sds, f.theta_hat[end-1:end]...)
#         println("seed $s: sigma = ", round.(f.theta_hat[end-1:end], digits = 4))
#     end
#
#     # and at twice the draws
#     f2 = logit2_rfx(choices_df, myxs, :pick1, :personid, theta0;
#                     rfx = myrfx, ndraws = 2000, seed = 20260808)
#     println("ndraws = 2000: sigma = ", round.(f2.theta_hat[end-1:end], digits = 4))
#
#     rep = boot_report(fit)
#     println()
#     println("seed-to-seed spread in sigma vs bootstrap se:")
#     for (j, row) in enumerate(eachrow(rep[rep.is_sd, :]))
#         spread = std(sds[j:2:end])
#         println("  ", rpad(row.param, 12),
#                 " seed sd = ", round(spread, digits = 4),
#                 "   boot_se = ", round(row.boot_se, digits = 4),
#                 "   ratio = ", round(spread / row.boot_se, digits = 3))
#     end
# end
