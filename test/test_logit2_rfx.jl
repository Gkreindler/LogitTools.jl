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
        # unsupported distribution (:normal, :lognormal, :neg_lognormal are the set)
        @test_throws ErrorException logit2_rfx(df, myxs, :pick1, :personid, ok4;
                                               rfx = [:x1 => :uniform])
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
                    Tuple{Vector{Bool}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}(),
                    true)   # ci_for_sd
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

        # small values render as <0.005 (the tight bound) rather than 0.00
        @testset "small_as_lt" begin
            asc, lat = AsciiTable(), LatexTable()

            # exactly at the rounding boundary: 0.005 prints as 0.01, so it must
            # NOT be flagged; anything below it prints as 0.00, so it must be
            @test LT._rfx_fmt(asc, 0.005,  2, true) == "0.01"
            @test LT._rfx_fmt(asc, 0.0049, 2, true) == "<0.005"
            @test LT._rfx_fmt(asc, 0.003,  2, true) == "<0.005"
            @test LT._rfx_fmt(asc, 0.12,   2, true) == "0.12"

            # an exact zero stays an exact zero, and NaN is untouched
            @test LT._rfx_fmt(asc, 0.0, 2, true) == "0.00"
            @test LT._rfx_fmt(asc, NaN, 2, true) == "NaN"

            # negatives of the same size
            @test LT._rfx_fmt(asc, -0.003, 2, true) == ">-0.005"

            # LaTeX needs math mode around the inequality
            @test LT._rfx_fmt(lat, 0.003,  2, true) == "\$<\$0.005"
            @test LT._rfx_fmt(lat, -0.003, 2, true) == "\$>\$-0.005"

            # the threshold follows `digits`
            @test LT._rfx_fmt(asc, 0.0004, 3, true) == "<0.0005"
            @test LT._rfx_fmt(asc, 0.0006, 3, true) == "0.001"

            # switch it off
            @test LT._rfx_fmt(asc, 0.003, 2, false) == "0.00"

            # end to end: on by default, suppressible
            if has_api
                s_on  = sprint(show, regtable_rfx(fit; digits = 2, digits_stats = 2))
                s_off = sprint(show, regtable_rfx(fit; digits = 2, digits_stats = 2,
                                                  small_as_lt = false))
                st = LT._rfx_table_stats(fit, [2.5, 97.5])
                if any(0 .< st.ci_lo[st.is_sd] .< 0.005)   # only if a bound is that small
                    @test occursin("<0.005", s_on)
                    @test !occursin("<0.005", s_off)
                end
                @test !occursin("<0.005", s_off)
            end
        end

        # stars on sd_ rows, and digits for the below-statistics
        if has_api
            st = LT._rfx_table_stats(fit, [2.5, 97.5])

            # default: no stars on the sd_ rows; stars_for_sd = true restores them
            s_nostar = sprint(show, regtable_rfx(fit; digits = 2, digits_stats = 2))
            s_star   = sprint(show, regtable_rfx(fit; digits = 2, digits_stats = 2,
                                                 stars_for_sd = true))
            sd_lines_nostar = filter(l -> occursin("sd_", l), split(s_nostar, "\n"))
            sd_lines_star   = filter(l -> occursin("sd_", l), split(s_star,   "\n"))
            @test !any(l -> occursin("*", l), sd_lines_nostar)
            @test any(l -> occursin("*", l), sd_lines_star)
            # beta rows keep their stars either way
            @test any(l -> occursin("*", l), filter(l -> startswith(strip(l), "x1"),
                                                    split(s_nostar, "\n")))

            # digits_stats = 2 rounds both the SEs and the CI bounds; SEs come in
            # parentheses and intervals in square brackets
            @test occursin("(" * string(round(st.se[1], digits = 2)) * ")", s_nostar)
            @test occursin("[" * string(round(st.ci_lo[4], digits = 2)) * ", " *
                           string(round(st.ci_hi[4], digits = 2)) * "]", s_nostar)
            # an interval is never wrapped in parentheses
            @test !occursin("(" * string(round(st.ci_lo[4], digits = 2)) * ", ", s_nostar)
            # ... and 3 digits (the default) gives a different rendering
            s3 = sprint(show, regtable_rfx(fit; digits = 3, digits_stats = 3))
            @test s3 != s_nostar

            # suppressing stars must not disturb the displayed standard errors:
            # with ci_for_sd = false the sd_ rows show their real SE, not the
            # inflated variance used to kill the stars
            s_se = sprint(show, regtable_rfx(fit; digits = 2, digits_stats = 2,
                                             ci_for_sd = false))
            @test occursin(string(round(st.se[4], digits = 2)), s_se)
            @test !occursin("1.0e12", s_se)
            @test !occursin("Inf", s_se)

            # the caller's fit is never mutated
            V44 = fit.vcov.V[4, 4]
            regtable_rfx(fit)
            @test fit.vcov.V[4, 4] == V44
            @test V44 < 1.0

            # fully conventional path still works (and is the only 0.6-safe one)
            @test regtable_rfx(fit; ci_for_sd = false, stars_for_sd = true) !== nothing
        end

        # sd_ detection is positional (extra.K), not name-based, so renaming
        # coefficients cannot move the CI onto the wrong row
        if has_api
            st = LT._rfx_table_stats(fit, [2.5, 97.5])
            @test st.is_sd == [false, false, false, true, true]

            pretty = Dict("x1" => "Duration", "x2" => "Payment", "x3" => "Distance",
                          "sd_x1" => "σ Duration", "sd_x3" => "σ Distance")
            for kw in ((; labels = pretty),
                       (; labels = Dict("x1" => "Duration")),        # partial
                       (; transform_labels = Dict("x" => "var")),
                       (; order = ["sd_x1", "sd_x3"]))               # sd rows first
                s = sprint(show, regtable_rfx(fit; kw...))
                # the sd_ rows still carry a CI pair, the beta rows a lone SE
                @test occursin(string(round(st.ci_lo[4], digits = 3)), s)
                @test occursin(string(round(st.ci_hi[4], digits = 3)), s)
                @test count(==(','), s) >= 2      # one comma per CI, two sd rows
            end
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
        # poison the draws so the kernel throws. Copy field by field rather than
        # listing the constructor positionally, so adding a field to RfxPrep does
        # not break this test.
        Pbad = LT.RfxPrep((f === :eta ? fill(NaN, size(P.eta)) : getfield(P, f)
                           for f in fieldnames(LT.RfxPrep))...)

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

    # -----------------------------------------------------------------------
    # Lognormal / neg-lognormal random coefficients
    # -----------------------------------------------------------------------
    @testset "lognormal" begin

        """Prep + buffers + group weights for an explicit rfx spec."""
        function _ln_setup(rfx; K = 4, R = 64, N = 60, T = 5, seed = 42)
            df = _rfx_testdata(N = N, T = T, K = K, seed = seed)
            formula = [Symbol("x", k) for k in 1:K]
            P, _ = LT._prep_logit2_rfx(df, formula, :pick1, :personid, rfx, R,
                                       20260808, nothing)
            gw = 0.3 .+ 1.5 .* rand(MersenneTwister(7), P.N)
            return P, LT.RfxBuffers(P), gw
        end

        # -------------------------------------------------------------------
        @testset "the all-normal path is structurally untouched" begin
            P, _, _ = _ln_setup([:x1, :x2])
            @test !P.any_log
            @test P.xlin === P.xmatrix           # same memory, not a zeroed copy
            @test P.rfx_islog == [false, false]
            @test P.rfx_cols  == [1, 2]
            @test P.rfx_sgn   == [1.0, 1.0]
        end

        @testset "xlin zeroes exactly the lognormal columns" begin
            P, _, _ = _ln_setup([:x1 => :lognormal, :x3 => :normal])
            @test P.any_log
            @test P.rfx_islog == [true, false]
            @test P.rfx_cols  == [1, 3]
            @test all(P.xlin[:, 1] .== 0)                # lognormal: out of the linear part
            @test P.xlin[:, 2] == P.xmatrix[:, 2]
            @test P.xlin[:, 3] == P.xmatrix[:, 3]        # normal: still in the linear part
            @test P.xlin[:, 4] == P.xmatrix[:, 4]
            @test P.zmatrix[:, 1] == P.xmatrix[:, 1]     # Z keeps the real column
            @test P.rfx_sgn == [1.0, 1.0]

            Pn, _, _ = _ln_setup([:x2 => :neg_lognormal])
            @test Pn.rfx_sgn == [-1.0]
            @test all(Pn.xlin[:, 2] .== 0)
        end

        # -------------------------------------------------------------------
        @testset "gradient vs finite differences" begin
            specs = [[:x1 => :lognormal],
                     [:x1 => :neg_lognormal],
                     [:x1 => :lognormal, :x2 => :lognormal],
                     [:x1 => :lognormal, :x3 => :normal],
                     [:x2 => :normal, :x1 => :neg_lognormal, :x3 => :lognormal]]

            for (si, rfx) in enumerate(specs)
                P, buf, gw = _ln_setup(rfx)
                M = length(rfx)
                rng = MersenneTwister(3000 + si)
                for _ in 1:4
                    # mu is on the LOG scale, so keep it modest: exp() of a wild
                    # draw would swamp the finite-difference comparison.
                    θ = [0.5 .* randn(rng, 4); 0.2 .+ 0.6 .* abs.(randn(rng, M))]
                    G = zeros(length(θ))
                    LT._rfx_fg!(true, G, copy(θ), P, buf, gw)
                    fd = FiniteDiff.finite_difference_gradient(
                            t -> LT._rfx_fg!(true, nothing, collect(t), P, buf, gw), θ)
                    @test maximum(abs.(G .- fd) ./ max.(1.0, abs.(fd))) < 1e-6
                end
            end
        end

        @testset "gradient with a ragged panel" begin
            rng = MersenneTwister(451)
            N, K = 50, 4
            Ti = rand(rng, 2:9, N)
            df = DataFrame(personid = vcat([fill(i, Ti[i]) for i in 1:N]...))
            nobs = nrow(df)
            for k in 1:K; df[!, Symbol("x", k)] = randn(rng, nobs); end
            df.pick1 = Float64.(rand(rng, nobs) .< 0.5)
            df = df[shuffle(rng, 1:nobs), :]

            formula = [Symbol("x", k) for k in 1:K]
            rfx = [:x1 => :lognormal, :x3 => :normal]
            P, _ = LT._prep_logit2_rfx(df, formula, :pick1, :personid, rfx, 64,
                                       20260808, nothing)
            gw  = 0.3 .+ 1.5 .* rand(MersenneTwister(7), P.N)
            buf = LT.RfxBuffers(P)

            θ = [-0.4, -0.6, 0.45, 0.1, 0.7, 0.35]
            G = zeros(6)
            LT._rfx_fg!(true, G, copy(θ), P, buf, gw)
            fd = FiniteDiff.finite_difference_gradient(
                    t -> LT._rfx_fg!(true, nothing, collect(t), P, buf, gw), θ)
            @test maximum(abs.(G .- fd) ./ max.(1.0, abs.(fd))) < 1e-6
        end

        # -------------------------------------------------------------------
        @testset "mirror symmetry Q(mu, s) == Q(mu, -s)" begin
            # Holds for the lognormal families too, because antithetic draws make
            # the draw set symmetric. This is what justifies keeping the abs()
            # canonicalisation and the percentile CIs on the sigma rows.
            P, buf, gw = _ln_setup([:x1 => :lognormal, :x2 => :neg_lognormal])
            Qp = LT._rfx_fg!(true, nothing, [0.3, -0.2, 0.4, 0.1,  0.7, -0.45], P, buf, gw)
            Qm = LT._rfx_fg!(true, nothing, [0.3, -0.2, 0.4, 0.1, -0.7,  0.45], P, buf, gw)
            @test abs(Qp - Qm) < 1e-12
        end

        @testset "sigma = 0 is still a stationary point" begin
            P, buf, gw = _ln_setup([:x1 => :lognormal, :x3 => :neg_lognormal])
            G = zeros(6)
            LT._rfx_fg!(true, G, [0.3, -0.2, 0.4, 0.1, 0.0, 0.0], P, buf, gw)
            @test maximum(abs.(G[5:6])) < 1e-10
        end

        @testset "sigma = 0 reduces to plain logit at beta = +/-exp(mu)" begin
            # The strongest check on the linear index: at sigma = 0 a lognormal
            # coefficient is the constant +/-exp(mu), so the objective must equal
            # the plain logit objective at that beta. Catches a double count (xlin
            # not zeroed) or a dropped level, in either direction.
            P, buf, gw = _ln_setup([:x1 => :lognormal, :x3 => :neg_lognormal])
            μ = [0.35, -0.7, 0.2, 0.9]
            Q = LT._rfx_fg!(true, nothing, [μ; 0.0; 0.0], P, buf, gw)

            β = copy(μ)
            β[1] =  exp(μ[1])
            β[3] = -exp(μ[3])
            wobs = zeros(size(P.xmatrix, 1))
            for (i, rg) in enumerate(P.ranges); wobs[rg] .= gw[i]; end
            Qplain = LT.minus_ll(β, P.yvec, P.xmatrix, similar(P.yvec), wobs)
            @test abs(Q - Qplain) < 1e-9
        end

        # -------------------------------------------------------------------
        @testset "theta0_rfx converts a level b0 to the log scale" begin
            myxs = [:x1, :x2, :x3]
            b0   = [0.8, -0.5, 1.6]
            s0   = [0.6, 0.4]

            th = theta0_rfx(myxs, [:x1 => :lognormal, :x2 => :neg_lognormal];
                            b0 = b0, s0 = s0)
            # the IMPLIED MEAN coefficient equals the level b0 that went in
            @test  exp(th[1] + s0[1]^2 / 2) ≈ b0[1]
            @test -exp(th[2] + s0[2]^2 / 2) ≈ b0[2]
            @test th[3] == b0[3]                       # not a random coefficient
            @test th[4:5] == s0

            # normal coefficients are passed through exactly as before
            @test theta0_rfx(myxs, [:x1, :x2]; b0 = b0, s0 = s0) == [b0; s0]
            @test theta0_rfx(myxs, [:x1 => :normal, :x2]; b0 = b0, s0 = s0) == [b0; s0]

            # b0 must carry the sign the support allows, and cannot be zero
            @test_throws ErrorException theta0_rfx(myxs, [:x2 => :lognormal];     b0 = b0)
            @test_throws ErrorException theta0_rfx(myxs, [:x1 => :neg_lognormal]; b0 = b0)
            @test_throws ErrorException theta0_rfx(myxs, [:x1 => :lognormal])   # b0 = zeros
        end

        # -------------------------------------------------------------------
        @testset "_rfx_to_level matches simulation" begin
            K     = 3
            pairs = [:x1 => :lognormal, :x3 => :neg_lognormal]
            cols  = [1, 3]
            θ     = [-0.3, 0.55, 0.4, 0.65, 0.25]
            lvl   = LT._rfx_to_level(θ, K, pairs, cols)

            @test lvl[2] == θ[2]                          # untouched
            rng = MersenneTwister(4242)
            for (m, (_, d)) in enumerate(pairs)
                k = cols[m]
                s = d === :neg_lognormal ? -1.0 : 1.0
                draws = s .* exp.(θ[k] .+ θ[K + m] .* randn(rng, 1_000_000))
                @test isapprox(lvl[k],     mean(draws); rtol = 1e-2)
                @test isapprox(lvl[K + m], std(draws);  rtol = 2e-2)
            end
            # SD is positive on both supports; the mean carries the sign
            @test lvl[1] > 0 && lvl[3] < 0
            @test lvl[K + 1] > 0 && lvl[K + 2] > 0

            # tiny sigma: SD -> sigma * |E|, which is where exp(s^2) - 1 would have
            # lost every significant digit and expm1 does not
            small = LT._rfx_to_level([0.0, 0.0, 0.0, 1e-7, 1e-7], K, pairs, cols)
            @test isapprox(small[K + 1], 1e-7; rtol = 1e-6)
        end

        # -------------------------------------------------------------------
        @testset "overflow is reported, not silently NaN" begin
            P, buf, gw = _ln_setup([:x1 => :lognormal])
            θ = [800.0, 0.0, 0.0, 0.0, 0.5]        # exp(800) = Inf
            @test_throws ErrorException LT._rfx_fg!(true, zeros(5), copy(θ), P, buf, gw)

            f = LT._logit2_rfx(P, θ, gw, Optim.Options())
            @test f.errored
            @test occursin("lognormal", f.error_message)
            @test all(isnan, f.theta_hat)
        end

        # -------------------------------------------------------------------
        @testset "reporting: level moments in regtable_rfx and boot_report" begin
            N, T, K = 120, 8, 3
            rng = MersenneTwister(99)
            df = DataFrame(personid = repeat(1:N, inner = T))
            for k in 1:K; df[!, Symbol("x", k)] = randn(rng, N * T); end
            X  = Matrix(df[:, [:x1, :x2, :x3]])
            b1 = repeat(exp.(log(0.8) .+ 0.5 .* randn(rng, N)), inner = T)
            v  = b1 .* X[:, 1] .- 0.9 .* X[:, 2] .+ 0.4 .* X[:, 3]
            df.pick1 = Float64.(rand(rng, N * T) .< (1 ./ (1 .+ exp.(-v))))

            myxs  = [:x1, :x2, :x3]
            plain = logit2(copy(df), myxs, :pick1, zeros(K))

            myrfx = [:x1 => :lognormal]
            th0   = theta0_rfx(myxs, myrfx; b0 = plain.theta_hat)
            fit   = logit2_rfx(df, myxs, :pick1, :personid, th0;
                               rfx = myrfx, ndraws = 100, seed = 7)
            @test fit.converged && !fit.errored
            fit.vcov = boot_logit2_rfx(df, myxs, :pick1, :personid, th0;
                                       rfx = myrfx, ndraws = 100, seed = 7,
                                       nboot = 20, parallel = false,
                                       theta_start = fit.theta_hat)
            @test fit.extra.rfx      == [:x1 => :lognormal]
            @test fit.extra.rfx_cols == [1]
            @test fit.theta_names    == ["x1", "x2", "x3", "sd_x1"]   # names unchanged

            # ---- rfx_level_moments -----------------------------------------
            lm = rfx_level_moments(fit)
            @test nrow(lm) == 3
            @test lm.quantity == ["mean", "median", "SD"]
            @test all(lm.variable .== :x1)
            @test lm.estimate[1] ≈ exp(fit.theta_hat[1] + fit.theta_hat[4]^2 / 2)
            @test lm.estimate[2] ≈ exp(fit.theta_hat[1])
            @test all(lm.estimate .> 0)
            @test all(lm.ci_lo .<= lm.estimate .<= lm.ci_hi)
            @test all(lm.boot_se .> 0)
            @test lm.n_nonfinite == [0, 0, 0]     # this fit is well behaved

            # a normal fit gets zero rows, so the call is safe unconditionally
            th0n = theta0_rfx(myxs, [:x1]; b0 = plain.theta_hat)
            fitn = logit2_rfx(df, myxs, :pick1, :personid, th0n;
                              rfx = [:x1], ndraws = 100, seed = 7)
            fitn.vcov = boot_logit2_rfx(df, myxs, :pick1, :personid, th0n;
                                        rfx = [:x1], ndraws = 100, seed = 7,
                                        nboot = 20, parallel = false,
                                        theta_start = fitn.theta_hat)
            @test nrow(rfx_level_moments(fitn)) == 0

            # ---- the table shows LEVEL moments, not mu / sigma_log ----------
            d = LT._rfx_table_stats(fit, [2.5, 97.5])
            @test d.is_log_row == [true, false, false, true]
            @test d.is_sd      == [false, false, false, true]
            @test d.coef[1] ≈ exp(fit.theta_hat[1] + fit.theta_hat[4]^2 / 2)
            @test d.coef[2:3] == fit.theta_hat[2:3]
            @test d.coef[4] ≈ d.coef[1] * sqrt(expm1(fit.theta_hat[4]^2))
            @test all(isfinite, d.se) && all(d.se .> 0)
            @test all(d.ci_lo .<= d.coef .<= d.ci_hi)

            # a normal fit's stats are unchanged: coef is theta_hat, se is diag(V)
            dn = LT._rfx_table_stats(fitn, [2.5, 97.5])
            @test dn.coef == collect(Float64, fitn.theta_hat)
            @test !any(dn.is_log_row)
            @test dn.se ≈ [sqrt(LT.vcov(fitn)[j, j]) for j in 1:4]

            # renders, and carries a bracketed interval on the SD row
            tab = sprint(show, regtable_rfx(fit))
            @test occursin("x1", tab)
            @test occursin("[", tab)
            # stars are suppressed on the two lognormal rows by default
            @test sprint(show, regtable_rfx(fit; stars_for_lognormal = true)) != tab
            @test !isnothing(regtable_rfx(fit; render = LatexTable()))

            # ---- estimated log-scale parameters in extralines ---------------
            ps = LT._rfx_log_param_stats(fit, [2.5, 97.5])
            @test length(ps) == 1
            @test ps[1].var == :x1
            @test ps[1].mu == fit.theta_hat[1]          # the ESTIMATED mu, untransformed
            @test ps[1].sd == fit.theta_hat[4]
            @test ps[1].mu_se > 0 && ps[1].sd_se > 0
            @test ps[1].sd_lo <= ps[1].sd <= ps[1].sd_hi
            @test isempty(LT._rfx_log_param_stats(fitn, [2.5, 97.5]))   # normal fit

            @test ps[1].mu_lo <= ps[1].mu <= ps[1].mu_hi

            rows = LT._rfx_log_param_rows([fit], nothing, [2.5, 97.5], 2, 2, true)
            @test length(rows) == 3                     # header + mu + sigma
            @test all(r -> length(r) == 2, rows)        # label + one column
            @test occursin("mu", rows[2][1])
            @test occursin("[", rows[2][2])             # ci_for_sd: mu gets an interval
            @test occursin("[", rows[3][2])             # sigma likewise
            @test startswith(rows[2][2], LT._rfx_num(fit.theta_hat[1], 2))
            # ci_for_sd = false switches both to standard errors
            rows_se = LT._rfx_log_param_rows([fit], nothing, [2.5, 97.5], 2, 2, false)
            @test occursin("(", rows_se[2][2]) && !occursin("[", rows_se[2][2])
            @test occursin("(", rows_se[3][2]) && !occursin("[", rows_se[3][2])
            # labels are honoured
            rows_lab = LT._rfx_log_param_rows([fit], Dict("x1" => "First"),
                                              [2.5, 97.5], 2, 2, true)
            @test occursin("First", rows_lab[2][1])
            # a column with no lognormal coefficient gets a blank cell
            rows2 = LT._rfx_log_param_rows([fitn, fit], nothing, [2.5, 97.5], 2, 2, true)
            @test all(r -> length(r) == 3, rows2)
            @test rows2[2][2] == "" && rows2[2][3] != ""

            # the rows reach the rendered table, ahead of the caller's extralines
            tab_lp = sprint(show, regtable_rfx(fit; digits = 2, digits_stats = 2,
                                               extralines = [["Draws", "100"]]))
            @test occursin("Log-scale parameters", tab_lp)
            @test findfirst("Log-scale parameters", tab_lp)[1] <
                  findfirst("Draws", tab_lp)[1]
            # opt out
            @test !occursin("Log-scale parameters",
                            sprint(show, regtable_rfx(fit; log_params = false)))
            # a purely normal table is untouched by the feature
            @test !occursin("Log-scale parameters", sprint(show, regtable_rfx(fitn)))

            # ---- boot_report carries BOTH scales ---------------------------
            rep = boot_report(fit)
            @test "dist"  in names(rep)
            @test "scale" in names(rep)
            @test rep.scale == ["log", "level", "level", "log", "level", "level", "level"]
            @test rep.dist[1] == :lognormal && rep.dist[4] == :lognormal
            @test rep.dist[2] == :none
            @test rep.estimate[1] == fit.theta_hat[1]     # the ESTIMATED mu, not E[b]
            @test rep.estimate[4] == fit.theta_hat[4]     # the ESTIMATED sigma_log
            @test nrow(rep) == 4 + 3
            @test rep.param[5:7] == ["mean[x1]", "median[x1]", "SD[x1]"]
            @test rep.is_sd[5:7] == [false, false, true]

            @test "n_nonfinite" in names(rep)
            @test all(rep.n_nonfinite .== 0)

            # a normal fit keeps one row per parameter, every scale "level"
            repn = boot_report(fitn)
            @test nrow(repn) == 4
            @test all(repn.scale .== "level")
            @test repn.dist == [:normal, :none, :none, :normal]

            # ---- overflow in the level transform ----------------------------
            # A converged replicate can sit in the flat mu/sigma ridge, where a large
            # sigma is offset by a very negative mu; exp(mu + sigma^2/2) is then not
            # representable. Reported statistics must degrade visibly, never silently.
            @testset "level transform overflow" begin
                se, nbad = LT._rfx_boot_se([1.0 2.0; 3.0 Inf; 5.0 4.0])
                @test nbad == [0, 1]
                @test se[1] ≈ std([1.0, 3.0, 5.0])
                @test se[2] ≈ std([2.0, 4.0])          # finite part only, not NaN
                @test isfinite(se[2])

                # fewer than two finite values is NaN, not an error
                se2, nbad2 = LT._rfx_boot_se(reshape([Inf, Inf, 1.0], 3, 1))
                @test nbad2 == [2] && isnan(se2[1])

                # end to end: poison one replicate so its level moment overflows
                fbad = deepcopy(fit)
                fbad.vcov.theta_boot_table[1, 4] = 40.0   # sigma_log = 40 -> exp(800)
                lmb = rfx_level_moments(fbad)
                @test lmb.n_nonfinite[1] == 1            # mean[x1] overflowed
                @test lmb.n_nonfinite[3] == 1            # SD[x1] too
                @test lmb.n_nonfinite[2] == 0            # median = exp(mu), unaffected
                @test !any(isnan, lmb.boot_se)            # finite-filtered rules out NaN (Inf still possible)
                @test all(isfinite, lmb.ci_lo)           # the lower bound is untouched

                # A percentile is robust to the overflowing draws only while their
                # SHARE stays below the tail probability. Here 1 of 20 replicates is
                # 5% > 2.5%, so the 97.5th percentile is genuinely Inf -- an honest
                # "not bounded above by the bootstrap", not a bug. n_nonfinite is how
                # the caller knows to expect it.
                @test !isfinite(lmb.ci_hi[1])
                @test isfinite(lmb.ci_hi[2])             # median row unaffected
                # widen the tail past the overflow share and the bound returns
                lmb90 = rfx_level_moments(fbad; ci_levels = [5.0, 90.0])
                @test all(isfinite, lmb90.ci_hi)

                # the table still renders and warns rather than printing a bare NaN
                dbad = (@test_logs (:warn, r"overflow") LT._rfx_table_stats(fbad, [5.0, 90.0]))
                @test all(isfinite, dbad.se)
                @test all(isfinite, dbad.ci_lo) && all(isfinite, dbad.ci_hi)
            end
        end

        # -------------------------------------------------------------------
        @testset "recovery with a lognormal coefficient (slow)" begin
            N, T, K = 400, 21, 3
            μ1, σ1  = log(0.7), 0.5
            βrest   = [-0.9, 0.4]

            rng = MersenneTwister(77)
            df = DataFrame(personid = repeat(1:N, inner = T))
            for k in 1:K; df[!, Symbol("x", k)] = randn(rng, N * T); end
            X  = Matrix(df[:, [:x1, :x2, :x3]])
            b1 = repeat(exp.(μ1 .+ σ1 .* randn(rng, N)), inner = T)
            v  = b1 .* X[:, 1] .+ βrest[1] .* X[:, 2] .+ βrest[2] .* X[:, 3]
            df.pick1 = Float64.(rand(rng, N * T) .< (1 ./ (1 .+ exp.(-v))))

            myxs  = [:x1, :x2, :x3]
            plain = logit2(copy(df), myxs, :pick1, zeros(K))
            th0   = theta0_rfx(myxs, [:x1 => :lognormal]; b0 = plain.theta_hat)
            fit   = logit2_rfx(df, myxs, :pick1, :personid, th0;
                               rfx = [:x1 => :lognormal], ndraws = 500, seed = 20260808)

            @test fit.converged && !fit.errored
            @test abs(fit.theta_hat[1] - μ1) < 0.12       # mu of log(beta)
            @test abs(fit.theta_hat[4] - σ1) < 0.12       # sigma of log(beta)
            @test maximum(abs.(fit.theta_hat[2:3] .- βrest)) < 0.15
        end
    end
end
