using LogitTools
using Test
using DataFrames
using Random
using LinearAlgebra
using Statistics
using Distributions
using FiniteDiff
using LogExpFunctions
using Distributed

const LT = LogitTools

"""
Long-format panel: `N` people x `S` choice sets, `J` options per set drawn from
`A` neighborhoods (so person-by-neighborhood cells recur across sets). Options
carry continuous `x1`, `x2` and binary `a`, `b`, `c`. If `beta` is given,
choices follow a logit with a correlated person effect on `(a, b)` and a
person-by-neighborhood effect on `x1`; otherwise they are random.
"""
function mrfx_data(; N = 40, S = 6, J = 2, A = 5, seed = 11, beta = nothing,
                   sa = 0.8, sb = 0.6, rho = 0.5, so = 0.5)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    cid = 0
    for i in 1:N
        ea, eb = randn(rng), randn(rng)
        ua, ub = sa * ea, sb * (rho * ea + sqrt(1 - rho^2) * eb)
        nu = so .* randn(rng, A)
        for s in 1:S
            cid += 1
            opts = randperm(rng, A)[1:J]
            v = zeros(J)
            these = NamedTuple[]
            for (k, n) in enumerate(opts)
                o = (pid = i, cid = cid, nbh = n, x1 = randn(rng), x2 = randn(rng),
                     a = Float64(rand(rng) < 0.5), b = Float64(rand(rng) < 0.5),
                     c = Float64(rand(rng) < 0.4))
                if !isnothing(beta)
                    v[k] = beta[1] * o.x1 + beta[2] * o.x2 + (beta[3] + ua) * o.a +
                           (beta[4] + ub) * o.b + nu[n] * o.x1
                end
                push!(these, o)
            end
            sel = isnothing(beta) ? rand(rng, 1:J) : argmax(v .+ rand(rng, Gumbel(), J))
            for (k, o) in enumerate(these)
                push!(rows, merge(o, (sel = Float64(k == sel),)))
            end
        end
    end
    df = DataFrame(rows)
    df.xpos = 0.5 .+ rand(MersenneTwister(seed + 1), nrow(df))   # positive regressor
    return df
end

const XS = [:x1, :x2, :a, :b, :c]

prep(df; kw...) = mlogit_rfx_prep(df, XS, :cid, :sel; col_group = :pid, ndraws = 32, kw...)

fd_grad(P, th; w = nothing) =
    FiniteDiff.finite_difference_gradient(x -> mlogit_rfx_objective(x, P; weights = w), th)

"""Interior parameter vector: means, then SDs, then block coordinates."""
theta_for(P; seed = 5) = (rng = MersenneTwister(seed);
    vcat(0.3 .* randn(rng, P.K), 0.3 .+ 0.6 .* rand(rng, P.M), 0.6 .* rand(rng, P.B) .- 0.3))

@testset "mlogit_rfx" begin

    df  = mrfx_data()
    df3 = mrfx_data(J = 3, seed = 12)

    @testset "terms, blocks and names" begin
        P = prep(df; rfx = [:a, rfx_term(:b), rfx_term(:x1; level = :nbh),
                            rfx_term(level = :nbh), rfx_term(:xpos; mean = false)])
        @test P.theta_names == ["x1", "x2", "a", "b", "c", "sd_a", "sd_b", "sd_x1|nbh",
                                "sd_1|nbh", "sd_xpos"]
        @test (P.K, P.M, P.B) == (5, 5, 0)

        P = prep(df; rfx = [:a, :b, :c],
                 rfx_corr = [rfx_corr_block(:a, :b, :c; zero = [(:a, :c)])])
        @test P.B == 2
        @test P.theta_names[end-1:end] == ["pcor_b__a", "pcor_c__b"]
        @test prep(df; rfx = [:a, :b], rfx_corr = [(:a, :b)]).B == 1
        @test prep(df; rfx = [:a, :b, :c], rfx_corr = [(:a, :b, :c)]).B == 3

        @test_throws ErrorException prep(df; rfx = [:a, :a])
        @test_throws ErrorException prep(df; rfx = [:xpos])                  # not in formula
        @test_throws ErrorException prep(df; rfx = [:a, rfx_term(:b; level = :nbh)],
                                         rfx_corr = [(:a, "b|nbh")])         # block needs person level
        @test_throws ErrorException prep(df; rfx = [:a, :b, :c],
                                         rfx_corr = [(:a, :b), (:b, :c)])    # overlapping blocks
        @test_throws ErrorException prep(df; rfx = [:a, :b => :lognormal],
                                         rfx_corr = [(:a, :b)])
        @test_throws ErrorException prep(df; rfx = [rfx_term()])             # person intercept cancels
        @test_throws ErrorException prep(df; rfx = [:a], ndraws = 31)
    end

    @testset "data guards" begin
        bad = copy(df); bad.sel[1:2] .= 1.0
        @test_throws ErrorException prep(bad; rfx = [:a])
        bad = copy(df); bad.cid[bad.pid .== 2] .= 1                         # set id shared by two people
        @test_throws ErrorException prep(bad; rfx = [:a])
        bad = copy(df); bad.w = Float64.(1:nrow(bad))
        @test_throws ErrorException prep(bad; rfx = [:a], weights = :w)       # not constant in group
    end

    @testset "gradient = finite differences" begin
        specs = [
            (rfx = [:a, :b, rfx_term(:x1; level = :nbh)], rfx_corr = []),
            (rfx = [rfx_term(:a; dist = :uniform), rfx_term(:x2; dist = :triangular),
                    rfx_term(level = :nbh)], rfx_corr = []),
            (rfx = [:a, :b, :c, rfx_term(:x2; level = :nbh)],
             rfx_corr = [rfx_corr_block(:a, :b, :c; zero = [(:c, :b)])]),
            (rfx = [:a, :b, :c], rfx_corr = [(:a, :b, :c)]),
            (rfx = [rfx_term(:a; dist = :uniform), rfx_term(:b; dist = :uniform)],
             rfx_corr = [(:a, :b)]),
            (rfx = [rfx_term(:xpos; mean = false), :a], rfx_corr = [(:xpos, :a)]),
        ]
        w = 0.2 .+ rand(MersenneTwister(3), 40); w[4] = 0.0
        for d in (df, df3), sp in specs, kernel in (:general, :binary)
            d === df3 && kernel === :binary && continue
            P = prep(d; sp..., kernel)
            th = theta_for(P)
            @test mlogit_rfx_gradient(th, P) ≈ fd_grad(P, th) rtol = 1e-5 atol = 1e-6
            @test mlogit_rfx_gradient(th, P; weights = w) ≈ fd_grad(P, th; w) rtol = 1e-5 atol = 1e-6
        end

        # lognormal families (general kernel only)
        for d in (df, df3)
            P = mlogit_rfx_prep(d, [:x1, :x2, :xpos, :a], :cid, :sel; col_group = :pid,
                                rfx = [:xpos => :lognormal, :a, rfx_term(:x1; level = :nbh)],
                                ndraws = 32)
            @test P.kernel === :general
            th = vcat(0.2, -0.1, -0.5, 0.3, 0.4, 0.6, 0.3)
            @test mlogit_rfx_gradient(th, P) ≈ fd_grad(P, th) rtol = 1e-5 atol = 1e-6
            P = mlogit_rfx_prep(d, [:x1, :xpos], :cid, :sel; col_group = :pid,
                                rfx = [:xpos => :neg_lognormal], ndraws = 32)
            th = [0.2, -0.5, 0.4]
            @test mlogit_rfx_gradient(th, P) ≈ fd_grad(P, th) rtol = 1e-5 atol = 1e-6
        end

        # near the boundary of the coordinates
        P = prep(df; rfx = [:a, :b, :c], rfx_corr = [rfx_corr_block(:a, :b, :c; zero = [(:c, :b)])])
        th = theta_for(P); th[end] = 0.995; th[P.K + 2] = 1e-4
        @test mlogit_rfx_gradient(th, P) ≈ fd_grad(P, th) rtol = 1e-4 atol = 1e-5
    end

    @testset "kernels agree" begin
        for sp in [(rfx = [:a, :b, rfx_term(:x1; level = :nbh)], rfx_corr = [(:a, :b)]),
                   (rfx = [rfx_term(:a; dist = :triangular)], rfx_corr = []),
                   (rfx = [], rfx_corr = [])]
            Pg = prep(df; sp..., kernel = :general)
            Pb = prep(df; sp..., kernel = :binary)
            th = theta_for(Pg)
            @test mlogit_rfx_objective(th, Pg) ≈ mlogit_rfx_objective(th, Pb) rtol = 1e-12
            @test mlogit_rfx_gradient(th, Pg) ≈ mlogit_rfx_gradient(th, Pb) rtol = 1e-10 atol = 1e-12
            Pg.M > 0 && @test mlogit_rfx_ess(th, Pg) ≈ mlogit_rfx_ess(th, Pb) rtol = 1e-10
        end
        @test prep(df; rfx = [:a]).kernel === :binary
        @test prep(df3; rfx = [:a]).kernel === :general
        @test_throws ErrorException prep(df3; rfx = [:a], kernel = :binary)
    end

    @testset "nesting and invariances" begin
        # no random terms = conditional logit
        P0 = prep(df3)
        b = [0.3, -0.2, 0.5, 0.1, -0.4]
        ll = 0.0
        for g in groupby(df3, :cid)
            v = Matrix(g[:, XS]) * b
            ll += v[findfirst(==(1.0), g.sel)] - logsumexp(v)
        end
        @test mlogit_rfx_objective(b, P0) ≈ -ll rtol = 1e-12

        # zero standard deviations (at the floor) = conditional logit
        P = prep(df3; rfx = [:a, rfx_term(:x1; level = :nbh)])
        @test mlogit_rfx_objective(vcat(b, [eps(), eps()]), P) ≈ -ll rtol = 1e-10

        # a block with all coordinates zero = independent terms with the same draws
        Pi = prep(df; rfx = [:a, :b, :c])
        Pc = prep(df; rfx = [:a, :b, :c], rfx_corr = [(:a, :b, :c)])
        th = theta_for(Pi)
        @test mlogit_rfx_objective(th, Pi) ≈ mlogit_rfx_objective(vcat(th, zeros(3)), Pc) rtol = 1e-12

        # input row order does not matter
        Pp = prep(df[randperm(MersenneTwister(1), nrow(df)), :];
                  rfx = [:a, rfx_term(:x1; level = :nbh)])
        Pq = prep(df; rfx = [:a, rfx_term(:x1; level = :nbh)])
        th = theta_for(Pq)
        @test mlogit_rfx_objective(th, Pp) ≈ mlogit_rfx_objective(th, Pq) rtol = 1e-12
    end

    @testset "correlation blocks" begin
        # F, V, AV, Y, W with (V, AV) and (Y, W) independent given F
        df5 = copy(df); df5.d = Float64.(rand(MersenneTwister(2), nrow(df5)) .< 0.5)
        P = mlogit_rfx_prep(df5, [:x1, :x2, :a, :b, :c, :d, :xpos], :cid, :sel; col_group = :pid,
                            rfx = [:x2, :a, :b, :c, :d],
                            rfx_corr = [rfx_corr_block(:x2, :a, :b, :c, :d;
                                        zero = [(:c, :a), (:c, :b), (:d, :a), (:d, :b)])],
                            ndraws = 32)
        @test P.B == 6
        th = theta_for(P; seed = 9)
        C = mlogit_rfx_correlation(th, P)
        @test diag(C) ≈ ones(5)
        @test isposdef(Symmetric(C))
        for x in (2, 3), y in (4, 5)              # Corr(x, y) = Corr(x, F) Corr(F, y)
            @test C[x, y] ≈ C[x, 1] * C[1, y] atol = 1e-12
        end
        V = mlogit_rfx_covariance(th, P)
        s = th[P.K+1:P.K+P.M]
        @test V ≈ C .* (s * s')
        @test mlogit_rfx_gradient(th, P) ≈ fd_grad(P, th) rtol = 1e-5 atol = 1e-6

        # the parametrisation reaches any correlation of a full block
        P2 = prep(df; rfx = [:a, :b, :c], rfx_corr = [(:a, :b, :c)])
        target = [1.0 0.5 -0.3; 0.5 1.0 0.2; -0.3 0.2 1.0]
        Lt = cholesky(target).L
        z1, z2 = Lt[2, 1], Lt[3, 1]
        z3 = Lt[3, 2] / sqrt(1 - z2^2)
        th = vcat(zeros(5), ones(3), [z1, z2, z3])
        @test mlogit_rfx_correlation(th, P2) ≈ target
    end

    @testset "scores and sandwich" begin
        P = prep(df; rfx = [:a, :b], rfx_corr = [(:a, :b)])
        th = theta_for(P)
        S = mlogit_rfx_scores(th, P)
        @test vec(sum(S, dims = 1)) ≈ -mlogit_rfx_gradient(th, P) rtol = 1e-10

        # with no random terms the sandwich is the clustered logit sandwich
        P0 = prep(df3)
        f0 = mlogit_rfx_fit(P0, zeros(5); optim_options = LT.Optim.Options(g_tol = 1e-10))
        b = f0.theta_hat
        H = zeros(5, 5); sc = zeros(40, 5)
        for g in groupby(df3, :cid)
            X = Matrix(g[:, XS]); p = softmax(X * b); y = g.sel
            H .+= X' * (Diagonal(p) - p * p') * X
            sc[g.pid[1], :] .+= X' * (y .- p)
        end
        Vc = inv(H) * (sc' * sc) * inv(H) * (40 / 39)
        @test mlogit_rfx_vcov(b, P0) ≈ Vc rtol = 1e-5

        # fixed parameters drop out
        fixed = falses(length(th)); fixed[end] = true
        V = mlogit_rfx_vcov(th, P; fixed)
        @test all(iszero, V[end, :]) && all(iszero, V[:, end])
        th2 = copy(th); th2[P.K + 1] = 1e-6; th2[end] = 1 - 1e-6
        @test mlogit_rfx_boundary(th2, P) == BitVector([falses(P.K); true; false; true])
    end

    @testset "estimation" begin
        d = mrfx_data(N = 300, S = 10, beta = [1.0, -0.5, 0.8, 0.4], seed = 21)
        P = mlogit_rfx_prep(d, [:x1, :x2, :a, :b], :cid, :sel; col_group = :pid,
                            rfx = [:a, :b, rfx_term(:x1; level = :nbh)],
                            rfx_corr = [(:a, :b)], ndraws = 200)
        P0 = mlogit_rfx_prep(d, [:x1, :x2, :a, :b], :cid, :sel; col_group = :pid)
        f0 = mlogit_rfx_fit(P0, zeros(4))
        @test f0.converged

        starts = theta0_mlogit_rfx_multistart(P; b0 = f0.theta_hat, nstarts = 3)
        @test starts[1, :] == theta0_mlogit_rfx(P; b0 = f0.theta_hat)
        fit = mlogit_rfx_fit(P, starts)
        @test fit.converged
        @test nrow(fit.fits_df) == 3
        @test fit.obj_value == minimum(fit.fits_df.obj_value[fit.fits_df.converged])
        @test fit.extra.n_at_best >= 2
        @test maximum(abs.(fit.theta_hat[1:4] .- [1.0, -0.5, 0.8, 0.4])) < 0.25
        @test abs(fit.theta_hat[5] - 0.8) < 0.35 && abs(fit.theta_hat[6] - 0.6) < 0.35
        @test maximum(abs.(mlogit_rfx_gradient(fit.theta_hat, P)[1:4])) < 1e-3

        # wrapper = prep + fit
        f2 = mlogit_rfx(d, [:x1, :x2, :a, :b], :cid, :sel, fit.theta_hat; col_group = :pid,
                        rfx = [:a, :b, rfx_term(:x1; level = :nbh)], rfx_corr = [(:a, :b)],
                        ndraws = 200)
        @test f2.obj_value ≈ fit.obj_value rtol = 1e-8

        # a start at the boundary is moved inside, and still converges
        th = copy(fit.theta_hat); th[5] = 1e-12; th[end] = 1 - 1e-9
        @test mlogit_rfx_fit(P, th).obj_value ≈ fit.obj_value rtol = 1e-6

        # lognormal starting values are levels
        th0 = theta0_mlogit_rfx([:x1, :xpos], [:xpos => :neg_lognormal]; b0 = [0.1, -2.0], s0 = 0.5)
        @test th0[2] ≈ log(2.0) - 0.125
        @test_throws ErrorException theta0_mlogit_rfx([:x1, :xpos], [:xpos => :lognormal];
                                                      b0 = [0.1, -2.0])

        # a failing start is recorded, not thrown
        Pl = mlogit_rfx_prep(df, [:x1, :xpos], :cid, :sel; col_group = :pid,
                             rfx = [:xpos => :lognormal], ndraws = 8)
        bad = mlogit_rfx_fit(Pl, [0.0, 800.0, 1.0])
        @test bad.errored && occursin("overflowed", bad.error_message)
        @test_throws ErrorException mlogit_rfx_fit(Pl, [0.0, 800.0, 1.0]; rethrow_errors = true)
    end

    @testset "bootstrap" begin
        P = prep(df; rfx = [:a, :b], rfx_corr = [(:a, :b)])
        th = mlogit_rfx_fit(P, theta0_mlogit_rfx(P)).theta_hat
        W = mlogit_rfx_boot_weights(P.N, 4; seed = 1)
        @test size(W) == (40, 4) && all(sum(W, dims = 1) .≈ 40)
        @test W == mlogit_rfx_boot_weights(P.N, 4; seed = 1)

        bt = boot_mlogit_rfx(P, th; W)
        @test size(bt.theta_boot_table) == (4, length(th))
        @test all(f.converged for f in bt.boot_fits)
        @test bt.V ≈ cov(bt.theta_boot_table)
        # replicate b is the weighted fit with W[:, b]; subsets reproduce it
        @test bt.theta_boot_table[2, :] ≈
              mlogit_rfx_fit(P, th; weights = W[:, 2]).theta_hat
        sub = boot_mlogit_rfx(P, th; W = W[:, 3:4])
        @test sub.theta_boot_table ≈ bt.theta_boot_table[3:4, :]

        # multi-start within each replicate
        ms = boot_mlogit_rfx(P, vcat(th', theta0_mlogit_rfx(P)'); W = W[:, 1:2])
        @test all(nrow(f.fits_df) == 2 for f in ms.boot_fits)
        @test all(ms.theta_boot_table[b, :] ≈ bt.theta_boot_table[b, :] for b in 1:2)
    end

    # Starts worker processes, so it runs only on request:
    # LOGITTOOLS_TEST_PARALLEL=1 julia --project -e 'using Pkg; Pkg.test()'
    if get(ENV, "LOGITTOOLS_TEST_PARALLEL", "0") == "1"
        @testset "parallel = serial" begin
            procs_ = addprocs(2; exeflags = "--project=$(Base.active_project())")
            try
                @everywhere procs_ using LogitTools
                P = prep(df; rfx = [:a, :b], rfx_corr = [(:a, :b)])
                starts = theta0_mlogit_rfx_multistart(P; nstarts = 3)
                @test mlogit_rfx_fit(P, starts; parallel = true).theta_hat ==
                      mlogit_rfx_fit(P, starts).theta_hat
                th = mlogit_rfx_fit(P, starts).theta_hat
                W = mlogit_rfx_boot_weights(P.N, 4)
                @test boot_mlogit_rfx(P, th; W, parallel = true).theta_boot_table ==
                      boot_mlogit_rfx(P, th; W).theta_boot_table
                # repeated parallel calls with different prepared models
                for rfx in ([:a], [:a, :b, rfx_term(:x1; level = :nbh)], [:b])
                    Q = prep(df; rfx)
                    s = theta0_mlogit_rfx_multistart(Q; nstarts = 2)
                    @test mlogit_rfx_fit(Q, s; parallel = true).theta_hat ==
                          mlogit_rfx_fit(Q, s).theta_hat
                end
                @test all(isempty(remotecall_fetch(() -> LogitTools._MRFX_SHARED, wk))
                          for wk in workers())     # nothing left on the workers
            finally
                rmprocs(procs_)
            end
        end
    end
end
