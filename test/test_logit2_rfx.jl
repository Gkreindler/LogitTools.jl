using LogitTools
using Test
using DataFrames
using Random
using Distributions
using LinearAlgebra
using Statistics
using FiniteDiff
using Optim
using RegressionTables
using Distributed   # top level: @everywhere is expanded at parse time

const LT = LogitTools

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"""Balanced panel with K regressors and optional true random coefficients."""
function _rfx_testdata(; N = 60, T = 5, K = 3, seed = 42,
                         beta = nothing, sigma_true = nothing, rfx_idx = Int[])
    rng  = MersenneTwister(seed)
    nobs = N * T
    df = DataFrame(personid = repeat(1:N, inner = T))
    for k in 1:K
        df[!, Symbol("x", k)] = randn(rng, nobs)
    end

    X = Matrix(df[:, [Symbol("x", k) for k in 1:K]])
    b = isnothing(beta) ? zeros(K) : beta
    v = X * b
    if !isnothing(sigma_true)
        for (j, k) in enumerate(rfx_idx)
            dev = repeat(sigma_true[j] .* randn(rng, N), inner = T)
            v .+= dev .* X[:, k]
        end
    end
    df.pick1 = Float64.(rand(rng, nobs) .< (1 ./ (1 .+ exp.(-v))))
    return df
end

"""Prep + buffers + non-trivial group weights, for kernel-level tests."""
function _rfx_kernel_setup(; K = 3, rfx = [:x1, :x2], R = 64, N = 60, T = 5,
                             seed = 42, weighted = true)
    df = _rfx_testdata(N = N, T = T, K = K, seed = seed)
    formula = [Symbol("x", k) for k in 1:K]
    P, _ = LT._prep_logit2_rfx(df, formula, :pick1, :personid, rfx, R, 20260808, nothing)
    gw  = weighted ? (0.3 .+ 1.5 .* rand(MersenneTwister(7), P.N)) : nothing
    buf = LT.RfxBuffers(P)
    return P, buf, gw
end

# ---------------------------------------------------------------------------

@testset "logit2_rfx" begin

    # -----------------------------------------------------------------------
    @testset "gradient vs finite differences" begin
        # generic parameter points: not at the truth, not at sigma = 0.
        # Several points and several M, because most indexing bugs in the
        # kernel show up on exactly one dimension.
        for M in 1:3
            rfx = [Symbol("x", k) for k in 1:M]
            P, buf, gw = _rfx_kernel_setup(K = 4, rfx = rfx)

            rng = MersenneTwister(1000 + M)
            points = [[0.4, -0.7, 0.2, 0.9], randn(rng, 4), 2 .* randn(rng, 4), fill(0.15, 4)]
            sigmas = [fill(0.6, M), abs.(randn(rng, M)) .+ 0.2,
                      [1.4, 0.3, 0.8][1:M], -[0.5, 1.1, 0.25][1:M]]

            for (b, s) in zip(points, sigmas)
                θ = [b; s]
                G = zeros(length(θ))
                LT._rfx_fg!(true, G, copy(θ), P, buf, gw)
                fd = FiniteDiff.finite_difference_gradient(
                        t -> LT._rfx_fg!(true, nothing, collect(t), P, buf, gw), θ)
                @test maximum(abs.(G .- fd) ./ max.(1.0, abs.(fd))) < 1e-6
            end
        end
    end

    @testset "gradient with ragged panel" begin
        rng = MersenneTwister(99)
        N, K = 50, 4
        Ti = rand(rng, 2:9, N)
        df = DataFrame(personid = vcat([fill(i, Ti[i]) for i in 1:N]...))
        nobs = nrow(df)
        for k in 1:K; df[!, Symbol("x", k)] = randn(rng, nobs); end
        df.pick1 = Float64.(rand(rng, nobs) .< 0.5)
        df = df[shuffle(rng, 1:nobs), :]

        formula = [Symbol("x", k) for k in 1:K]
        P, _ = LT._prep_logit2_rfx(df, formula, :pick1, :personid, [:x1, :x3], 64, 20260808, nothing)
        gw  = 0.3 .+ 1.5 .* rand(MersenneTwister(7), P.N)
        buf = LT.RfxBuffers(P)

        θ = [0.3, -0.6, 0.45, 0.1, 0.8, 0.35]
        G = zeros(6)
        LT._rfx_fg!(true, G, copy(θ), P, buf, gw)
        fd = FiniteDiff.finite_difference_gradient(
                t -> LT._rfx_fg!(true, nothing, collect(t), P, buf, gw), θ)
        @test maximum(abs.(G .- fd) ./ max.(1.0, abs.(fd))) < 1e-6
    end

    # -----------------------------------------------------------------------
    @testset "saddle point at sigma = 0" begin
        # tests the antithetic draw construction, not just the gradient
        P, buf, gw = _rfx_kernel_setup()
        θ = [0.4, -0.7, 0.2, 0.0, 0.0]
        G = zeros(5)
        LT._rfx_fg!(true, G, copy(θ), P, buf, gw)
        @test maximum(abs.(G[4:5])) < 1e-12

        # and the beta-block reduces exactly to the plain logit gradient
        wobs = zeros(size(P.xmatrix, 1))
        for (i, rg) in enumerate(P.ranges); wobs[rg] .= gw[i]; end
        gplain = LT.minus_grad(θ[1:3], P.yvec, P.xmatrix, similar(P.yvec), wobs)
        @test maximum(abs.(G[1:3] .- vec(gplain))) < 1e-10
    end

    @testset "mirror symmetry Q(b,s) == Q(b,-s)" begin
        P, buf, gw = _rfx_kernel_setup()
        Qp = LT._rfx_fg!(true, nothing, [0.4, -0.7, 0.2,  0.75, -0.4], P, buf, gw)
        Qm = LT._rfx_fg!(true, nothing, [0.4, -0.7, 0.2, -0.75,  0.4], P, buf, gw)
        @test abs(Qp - Qm) < 1e-12
    end

    @testset "draws are antithetic and mean zero" begin
        eta, logw = LT._make_draws(3, 64, 10, 20260808)
        @test size(eta) == (3, 64, 10)
        @test maximum(abs.(sum(eta, dims = 2))) < 1e-12
        @test all(logw .≈ -log(64))
        @test eta[:, 1:32, :] ≈ -eta[:, 33:64, :]
        @test_throws ErrorException LT._make_draws(2, 63, 5, 1)          # odd
        @test_throws ErrorException LT._make_draws(2, 64, 5, 1; scheme = :halton)
    end

    # -----------------------------------------------------------------------
    @testset "nesting: rfx = Symbol[] reproduces logit2" begin
        df = _rfx_testdata(N = 80, T = 4, K = 3, seed = 11, beta = [0.5, -0.8, 0.3])
        myxs = [:x1, :x2, :x3]

        f_plain = logit2(copy(df), myxs, :pick1, zeros(3))
        f_rfx   = logit2_rfx(df, myxs, :pick1, :personid, zeros(3); rfx = Symbol[], ndraws = 10)

        @test maximum(abs.(f_plain.theta_hat .- f_rfx.theta_hat)) < 1e-8
        @test abs(f_plain.obj_value - f_rfx.obj_value) < 1e-10
        @test f_rfx.theta_names == ["x1", "x2", "x3"]
    end

    # -----------------------------------------------------------------------
    @testset "does not mutate data_df" begin
        rng = MersenneTwister(3)
        N, T = 40, 4; nobs = N * T
        df = DataFrame(personid = repeat(1:N, inner = T),
                       x1 = randn(rng, nobs), x2 = randn(rng, nobs))
        df.pick1 = Int.(rand(rng, nobs) .< 0.5)            # deliberately Int
        df.wt    = repeat(rand(rng, N), inner = T)
        ref = deepcopy(df)

        logit2_rfx(df, [:x1, :x2], :pick1, :personid, [0.1, 0.1, 0.5];
                   rfx = [:x1], ndraws = 20, weights = :wt)

        @test isequal(df, ref)
        @test eltype(df.pick1) == eltype(ref.pick1)        # no in-place retyping
        @test names(df) == names(ref)                       # no bw* columns
    end

    # -----------------------------------------------------------------------
    @testset "theta0_rfx" begin
        @test theta0_rfx([:a, :b, :c], [:a, :c]) == [0.0, 0.0, 0.0, 0.5, 0.5]
        @test theta0_rfx([:a, :b], [:a]; b0 = [1.0, 2.0], s0 = [0.3]) == [1.0, 2.0, 0.3]
        @test_throws ErrorException theta0_rfx([:a, :b], [:a]; b0 = [1.0])
        @test_throws ErrorException theta0_rfx([:a, :b], [:a]; s0 = [0.3, 0.3])
    end

    # -----------------------------------------------------------------------
    @testset "guards" begin
        df = _rfx_testdata(N = 40, T = 4, K = 3, seed = 5)
        myxs = [:x1, :x2, :x3]
        ok4 = [0.0, 0.0, 0.0, 0.5]

        # rfx name not in formula
        @test_throws ErrorException logit2_rfx(df, myxs, :pick1, :personid, ok4; rfx = [:nope])
        # duplicate rfx
        @test_throws ErrorException logit2_rfx(df, myxs, :pick1, :personid,
                                               [0.0, 0.0, 0.0, 0.5, 0.5]; rfx = [:x1, :x1])
        # unsupported distribution
        @test_throws ErrorException logit2_rfx(df, myxs, :pick1, :personid, ok4;
                                               rfx = [:x1 => :lognormal])
        # wrong theta0 length
        @test_throws ErrorException logit2_rfx(df, myxs, :pick1, :personid, zeros(2); rfx = [:x1])
        # sigma0 == 0 (saddle)
        @test_throws ErrorException logit2_rfx(df, myxs, :pick1, :personid,
                                               [0.0, 0.0, 0.0, 0.0]; rfx = [:x1])
        # odd ndraws
        @test_throws ErrorException logit2_rfx(df, myxs, :pick1, :personid, ok4;
                                               rfx = [:x1], ndraws = 101)
        # every group a singleton
        let dfs = _rfx_testdata(N = 50, T = 1, K = 3, seed = 6)
            @test_throws ErrorException logit2_rfx(dfs, myxs, :pick1, :personid, ok4;
                                                   rfx = [:x1], ndraws = 20)
        end
        # weights not constant within group
        let dfw = copy(df)
            dfw.wt = rand(MersenneTwister(2), nrow(dfw))
            @test_throws ErrorException logit2_rfx(dfw, myxs, :pick1, :personid, ok4;
                                                   rfx = [:x1], ndraws = 20, weights = :wt)
        end
        # choice not 0/1
        let dfb = copy(df)
            dfb.pick1 = dfb.pick1 .+ 0.5
            @test_throws ErrorException logit2_rfx(dfb, myxs, :pick1, :personid, ok4;
                                                   rfx = [:x1], ndraws = 20)
        end
        # cluster_var != col_id
        @test_throws ErrorException boot_logit2_rfx(df, myxs, :pick1, :personid, ok4;
                                                    rfx = [:x1], cluster_var = :x2,
                                                    nboot = 2, parallel = false, ndraws = 20)
        # parallel with no workers
        if nprocs() == 1
            @test_throws ErrorException boot_logit2_rfx(df, myxs, :pick1, :personid, ok4;
                                                        rfx = [:x1], nboot = 2,
                                                        parallel = true, ndraws = 20)
        end
    end

    # -----------------------------------------------------------------------
    @testset "determinism and canonical sigma" begin
        df = _rfx_testdata(N = 60, T = 6, K = 3, seed = 21, beta = [0.6, -0.9, 0.4],
                           sigma_true = [0.7, 0.4], rfx_idx = [1, 3])
        myxs = [:x1, :x2, :x3]
        th0 = theta0_rfx(myxs, [:x1, :x3])

        f1 = logit2_rfx(df, myxs, :pick1, :personid, th0; rfx = [:x1, :x3], ndraws = 100, seed = 7)
        f2 = logit2_rfx(df, myxs, :pick1, :personid, th0; rfx = [:x1, :x3], ndraws = 100, seed = 7)
        @test f1.theta_hat == f2.theta_hat                       # same seed -> identical

        # sigma is canonicalised regardless of the sign of the start
        f3 = logit2_rfx(df, myxs, :pick1, :personid, [th0[1:3]; -0.5; -0.5];
                        rfx = [:x1, :x3], ndraws = 100, seed = 7)
        @test all(f3.theta_hat[4:5] .>= 0)
        @test all(f1.theta_hat[4:5] .>= 0)
    end

    # -----------------------------------------------------------------------
    @testset "ESS diagnostic" begin
        df = _rfx_testdata(N = 60, T = 6, K = 3, seed = 21, beta = [0.6, -0.9, 0.4],
                           sigma_true = [0.7, 0.4], rfx_idx = [1, 3])
        myxs = [:x1, :x2, :x3]
        R = 100
        fit = logit2_rfx(df, myxs, :pick1, :personid, theta0_rfx(myxs, [:x1, :x3]);
                         rfx = [:x1, :x3], ndraws = R, seed = 7)

        P, _ = LT._prep_logit2_rfx(df, myxs, :pick1, :personid, [:x1, :x3], R, 7, nothing)
        ess = LT._rfx_ess(fit.theta_hat, P, LT.RfxBuffers(P))

        @test length(ess) == P.N
        @test all(isfinite, ess)
        @test all(1 .< ess .<= R)                                # in (1, R]
        @test fit.extra.ess_min ≈ minimum(ess)
        @test fit.extra.ess_median ≈ median(ess)
        @test fit.extra.M == 2 && fit.extra.K == 3 && fit.extra.R == R
        @test fit.extra.n_groups == 60
        @test fit.n_obs == nrow(df)
    end

    # -----------------------------------------------------------------------
    @testset "bootstrap (serial)" begin
        df = _rfx_testdata(N = 60, T = 6, K = 3, seed = 21, beta = [0.6, -0.9, 0.4],
                           sigma_true = [0.7, 0.4], rfx_idx = [1, 3])
        myxs = [:x1, :x2, :x3]
        th0  = theta0_rfx(myxs, [:x1, :x3])

        fit = logit2_rfx(df, myxs, :pick1, :personid, th0;
                         rfx = [:x1, :x3], ndraws = 100, seed = 7)
        fit.vcov = boot_logit2_rfx(df, myxs, :pick1, :personid, th0;
                                   rfx = [:x1, :x3], ndraws = 100, seed = 7,
                                   nboot = 20, boot_seed = 777, parallel = false,
                                   theta_start = fit.theta_hat, cluster_var = :personid)

        @test fit.vcov.method == :bayesian_bootstrap
        @test size(fit.vcov.theta_boot_table) == (20, 5)
        @test size(fit.vcov.V) == (5, 5)
        @test issymmetric(fit.vcov.V)
        # every returned sigma is canonical, in every replicate
        @test all(fit.vcov.theta_boot_table[:, 4:5] .>= 0)

        rep = boot_report(fit)
        @test nrow(rep) == 5
        @test rep.param == ["x1", "x2", "x3", "sd_x1", "sd_x3"]
        @test rep.is_sd == [false, false, false, true, true]
        @test all(rep.ci_lo .<= rep.ci_hi)
        @test all(isnan, rep.share_near_zero[1:3])               # beta rows
        @test all(0 .<= rep.share_near_zero[4:5] .<= 1)          # sigma rows
        @test rep.estimate == fit.theta_hat

        V0 = copy(fit.vcov.V)
        boot_vcov!(fit)
        @test fit.vcov.V ≈ V0                                    # idempotent

        @test regtable(fit) !== nothing                           # still renders
        @test regtable_rfx(fit) !== nothing
        @test LT.vcov_method(fit) isa LT.bayesian_bootstrap
    end

    # -----------------------------------------------------------------------
    @testset "regtable_rfx mixed SE / CI" begin
        df = _rfx_testdata(N = 60, T = 6, K = 3, seed = 21, beta = [0.6, -0.9, 0.4],
                           sigma_true = [0.7, 0.4], rfx_idx = [1, 3])
        myxs = [:x1, :x2, :x3]
        th0  = theta0_rfx(myxs, [:x1, :x3])

        fit = logit2_rfx(df, myxs, :pick1, :personid, th0;
                         rfx = [:x1, :x3], ndraws = 100, seed = 7)
        fit.vcov = boot_logit2_rfx(df, myxs, :pick1, :personid, th0;
                                   rfx = [:x1, :x3], ndraws = 100, seed = 7,
                                   nboot = 20, boot_seed = 777, parallel = false,
                                   theta_start = fit.theta_hat)

        # ci_for_sd = false works on every RegressionTables version
        @test regtable_rfx(fit; ci_for_sd = false) !== nothing

        has_api = hasmethod(RegressionTables.StdError,
                            Tuple{RegressionTables.RegressionModel, Int})
        if has_api
            @test regtable_rfx(fit) !== nothing
            @test regtable_rfx(fit; ci_levels = [5, 95]) !== nothing

            # the printed CI must equal boot_report's percentile CI
            rep = boot_report(fit)
            st  = LT._rfx_table_stats(fit, [2.5, 97.5])
            @test st.is_sd == [false, false, false, true, true]
            @test st.ci_lo ≈ rep.ci_lo
            @test st.ci_hi ≈ rep.ci_hi
            @test st.se   ≈ rep.boot_se          # sqrt(diag(V)) == std of replicates

            # the under-statistic dispatches per row
            bs = LT.RfxBelowStatistic(IdDict{Any, NamedTuple{(:is_sd, :se, :ci_lo, :ci_hi),
                    Tuple{Vector{Bool}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}())
            m = LogitTools.LogitRegModel(fit)
            bs.tbl[m] = st
            @test bs(m, 1).val isa Float64            # beta row -> scalar SE
            @test bs(m, 4).val isa Tuple              # sd_ row  -> pair
            @test bs(m, 4).val == (st.ci_lo[4], st.ci_hi[4])
            @test bs(m, 1).val ≈ st.se[1]

            # several fits in one table
            @test regtable_rfx(fit, fit) !== nothing

            # a fit with no bootstrap cannot produce percentile CIs
            nb = logit2_rfx(df, myxs, :pick1, :personid, th0;
                            rfx = [:x1, :x3], ndraws = 100, seed = 7)
            @test_throws ErrorException regtable_rfx(nb)
            @test regtable_rfx(nb; ci_for_sd = false) !== nothing
        end

        # an M = 0 fit has no sd_ rows: falls back to the plain table
        f0 = logit2_rfx(df, myxs, :pick1, :personid, zeros(3); rfx = Symbol[], ndraws = 10)
        f0.vcov = boot_logit2_rfx(df, myxs, :pick1, :personid, zeros(3);
                                  rfx = Symbol[], ndraws = 10, nboot = 20,
                                  boot_seed = 777, parallel = false)
        @test regtable_rfx(f0) !== nothing

        # a vcov from a different model is caught with a clear message
        f0bad = logit2_rfx(df, myxs, :pick1, :personid, zeros(3); rfx = Symbol[], ndraws = 10)
        f0bad.vcov = fit.vcov                       # 5 params vs 3
        @test_throws ErrorException LT._rfx_table_stats(f0bad, [2.5, 97.5])
    end

    @testset "rethrow_errors" begin
        df = _rfx_testdata(N = 40, T = 5, K = 3, seed = 21, beta = [0.6, -0.9, 0.4],
                           sigma_true = [0.7, 0.4], rfx_idx = [1, 3])
        myxs = [:x1, :x2, :x3]
        th0  = theta0_rfx(myxs, [:x1, :x3])

        P, _ = LT._prep_logit2_rfx(df, myxs, :pick1, :personid, [:x1, :x3], 50, 7, nothing)
        # poison the draws so the kernel throws
        Pbad = LT.RfxPrep(P.xmatrix, P.zmatrix, P.yvec, P.q, P.ranges,
                          fill(NaN, size(P.eta)), P.logw, P.group_ids,
                          P.K, P.M, P.R, P.N, P.Tmax, P.rfx_pairs,
                          P.theta_names, P.col_id, P.seed)

        # default: captured into errored / error_message, never throws
        f = LT._logit2_rfx(Pbad, th0, nothing, Optim.Options())
        @test f.errored
        @test !f.converged
        @test all(isnan, f.theta_hat)
        @test !isempty(f.error_message)

        # rethrow_errors = true: the original exception propagates
        @test_throws Exception LT._logit2_rfx(Pbad, th0, nothing, Optim.Options();
                                              rethrow_errors = true)

        # the public entry points accept and forward the keyword
        @test :rethrow_errors in Base.kwarg_decl(first(methods(logit2_rfx)))
        @test :rethrow_errors in Base.kwarg_decl(first(methods(boot_logit2_rfx)))

        # a normal fit is unaffected by the default
        good = logit2_rfx(df, myxs, :pick1, :personid, th0;
                          rfx = [:x1, :x3], ndraws = 50, seed = 7)
        @test !good.errored
    end

    @testset "assembly error distinguishes failure modes" begin
        df = _rfx_testdata(N = 40, T = 5, K = 3, seed = 21, beta = [0.6, -0.9, 0.4],
                           sigma_true = [0.7, 0.4], rfx_idx = [1, 3])
        myxs = [:x1, :x2, :x3]
        th0  = theta0_rfx(myxs, [:x1, :x3])

        # iterations = 0 -> every replicate runs but none converges
        err = try
            boot_logit2_rfx(df, myxs, :pick1, :personid, th0;
                            rfx = [:x1, :x3], ndraws = 50, seed = 7,
                            nboot = 4, parallel = false,
                            optim_options = Optim.Options(iterations = 0))
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("0 errored", err)
        @test occursin("did not converge", err)
        @test occursin("optimiser problem", err)   # points at the right fix
    end

    # -----------------------------------------------------------------------
    @testset "boot_report fallback and failure handling" begin
        df = _rfx_testdata(N = 60, T = 6, K = 3, seed = 21, beta = [0.6, -0.9, 0.4],
                           sigma_true = [0.7, 0.4], rfx_idx = [1, 3])
        myxs = [:x1, :x2, :x3]
        th0  = theta0_rfx(myxs, [:x1, :x3])

        fit = logit2_rfx(df, myxs, :pick1, :personid, th0;
                         rfx = [:x1, :x3], ndraws = 100, seed = 7)
        fit.vcov = boot_logit2_rfx(df, myxs, :pick1, :personid, th0;
                                   rfx = [:x1, :x3], ndraws = 100, seed = 7,
                                   nboot = 20, boot_seed = 777, parallel = false,
                                   theta_start = fit.theta_hat)

        # a run reloaded from disk with only the table: warn once, fall back to
        # all non-NaN rows
        fit.vcov.boot_fits = nothing
        rep = (@test_logs (:warn,) match_mode = :any boot_report(fit))
        @test nrow(rep) == 5
        @test all(isfinite, rep.boot_se)

        # errored replicates keep their row (as NaN) but are excluded from V
        tbl = fit.vcov.theta_boot_table
        tbl[3, :] .= NaN
        keep = [all(isfinite, view(tbl, b, :)) for b in 1:size(tbl, 1)]
        @test sum(keep) == 19
        boot_vcov!(fit)
        @test all(isfinite, fit.vcov.V)          # NaN row did not poison V
        @test size(fit.vcov.V) == (5, 5)
    end

    # -----------------------------------------------------------------------
    # Guarded: addprocs inside Pkg.test()'s sandbox has to inherit a temporary
    # environment and is a known source of flakiness. This is the single most
    # valuable test for the Distributed plumbing, so it must exist and must be
    # run deliberately -- just not on every CI invocation, where a worker
    # timeout would produce a red badge for no reason.
    #
    #     LOGITTOOLS_TEST_PARALLEL=1 julia --project=. -e 'using Pkg; Pkg.test()'
    if get(ENV, "LOGITTOOLS_TEST_PARALLEL", "0") == "1"
        @testset "serial == parallel" begin
            addprocs(2)
            try
                @everywhere using LogitTools

                df = _rfx_testdata(N = 60, T = 6, K = 3, seed = 21, beta = [0.6, -0.9, 0.4],
                                   sigma_true = [0.7, 0.4], rfx_idx = [1, 3])
                myxs = [:x1, :x2, :x3]
                th0  = theta0_rfx(myxs, [:x1, :x3])
                kw = (; rfx = [:x1, :x3], ndraws = 100, seed = 7,
                        nboot = 20, boot_seed = 777, cluster_var = :personid)

                vser = boot_logit2_rfx(df, myxs, :pick1, :personid, th0; parallel = false, kw...)
                vpar = boot_logit2_rfx(df, myxs, :pick1, :personid, th0; parallel = true,  kw...)

                @test vser.theta_boot_table == vpar.theta_boot_table   # bitwise
                @test vser.V == vpar.V
            finally
                rmprocs(workers())
            end
        end
    end

    # -----------------------------------------------------------------------
    # Slow: the recovery test at realistic dimensions (~3 s plus compilation).
    @testset "recovery (slow)" begin
        N, T, K = 400, 21, 3
        beta  = [0.6, -0.9, 0.4]
        sigma = [0.7, 0.4]
        df = _rfx_testdata(N = N, T = T, K = K, seed = 20260808,
                           beta = beta, sigma_true = sigma, rfx_idx = [1, 3])
        myxs = [:x1, :x2, :x3]

        plain = logit2(copy(df), myxs, :pick1, zeros(K))
        th0   = theta0_rfx(myxs, [:x1, :x3]; b0 = plain.theta_hat)
        fit   = logit2_rfx(df, myxs, :pick1, :personid, th0;
                           rfx = [:x1, :x3], ndraws = 500, seed = 20260808)

        @test fit.converged
        @test !fit.errored
        @test maximum(abs.(fit.theta_hat[4:5] .- sigma)) < 0.1
        @test maximum(abs.(fit.theta_hat[1:3] .- beta)) < 0.15
        @test fit.extra.ess_p10 > 30
    end
end
