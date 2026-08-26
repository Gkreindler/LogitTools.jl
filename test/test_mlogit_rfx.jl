using LogitTools
using Test
using DataFrames
using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsBase
using FiniteDiff
using Optim
using LogExpFunctions
using RegressionTables
using Distributed

const LTR = LogitTools

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"""
Long-format multinomial panel: `N` individuals x `S` choice sets each, `J` of `A`
alternatives per set. An alternative recurs across an individual's choice sets, so
a `(uniqueid, alt)` cell holds several rows -- which is what an option-level random
effect needs in order to be identified.

`sigma_g` puts a true group-level random coefficient on `x1`; `sigma_o` a true
option-level random intercept on `:alt`.
"""
function _mrfx_testdata(; N = 50, S = 4, J = 3, A = 4, seed = 13, ragged = false,
                          beta = nothing, sigma_g = nothing, sigma_o = nothing,
                          sigma_net = nothing, sigma_v = nothing, rho = 0.0)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    cid = 0
    nu = randn(rng, N)
    nv = randn(rng, N)
    xi = randn(rng, N, A)
    for i in 1:N
        Si = ragged ? max(2, S - (i % 3)) : S
        for s in 1:Si
            cid += 1
            Ji = ragged ? max(2, J - (s % 2)) : J
            for a in randperm(rng, A)[1:Ji]
                push!(rows, (uniqueid = i, setid = cid, alt = a,
                             x1 = randn(rng), x2 = randn(rng), x3 = randn(rng),
                             nu = nu[i], xi = xi[i, a]))
            end
        end
    end
    df = DataFrame(rows)
    df.xnet = df.x1 .+ df.x3

    v = zeros(nrow(df))
    if !isnothing(beta)
        v .+= beta[1] .* df.x1 .+ beta[2] .* df.x2 .+ beta[3] .* df.x3
    end
    isnothing(sigma_g) || (v .+= sigma_g .* df.nu .* df.x1)
    isnothing(sigma_o) || (v .+= sigma_o .* df.xi)
    if !isnothing(sigma_net) || !isnothing(sigma_v)
        (!isnothing(sigma_net) && !isnothing(sigma_v)) || error(
            "sigma_net and sigma_v must be supplied together")
        abs(rho) < 1 || error("rho must be strictly inside (-1,1)")
        for rg in groupby(df, :uniqueid)
            i = first(rg.uniqueid)
            a = sigma_net * nu[i]
            vv = sigma_v * (rho * nu[i] + sqrt(1 - rho^2) * nv[i])
            rows_i = parentindices(rg)[1]
            v[rows_i] .+= a .* df.xnet[rows_i] .+ vv .* df.x2[rows_i]
        end
    end

    df.selected = zeros(Float64, nrow(df))
    for g in groupby(df, :setid)
        if isnothing(beta) && isnothing(sigma_g) && isnothing(sigma_o) &&
           isnothing(sigma_net) && isnothing(sigma_v)
            g.selected[rand(rng, 1:nrow(g))] = 1.0     # a valid 0/1 pattern is enough
        else
            g.selected[argmax(v[parentindices(g)[1]] .+ rand(rng, Gumbel(), nrow(g)))] = 1.0
        end
    end

    # a regressor that is constant within a choice set, and a row-unique id:
    # both are used to exercise the identification guards
    transform!(groupby(df, :setid), :x1 => (u -> fill(first(u), length(u))) => :xset)
    df.rowid = 1:nrow(df)
    return select(df, Not([:nu, :xi]))
end

const _RXS = [:x1, :x2, :x3]

"""Prep + buffers + non-trivial group weights, for kernel-level tests."""
function _mrfx_kernel_setup(; rfx = [:x1], R = 64, weighted = true, kwargs...)
    df = _mrfx_testdata(; kwargs...)
    P, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid, rfx,
                                R, 20260808, nothing)
    gw  = weighted ? (0.3 .+ 1.5 .* rand(MersenneTwister(7), P.N)) : nothing
    return P, LTR.MlogitRfxBuffers(P), gw
end

_mrfx_obj(P, buf, gw) = th -> LTR._mlogit_rfx_fg!(true, nothing,
                                                  collect(Float64, th), P, buf, gw)
function _mrfx_grad(P, buf, gw, th)
    G = zeros(length(th))
    LTR._mlogit_rfx_fg!(true, G, collect(Float64, th), P, buf, gw)
    return G
end

"""A fit that is guaranteed to fail, for the error-capture tests."""
function _mlogit_rfx_failing_fit(df; rethrow_errors = false)
    P, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid,
                                [:x1 => :lognormal], 8, 1, nothing)
    # a mu this large overflows exp() on the first evaluation
    return LTR._mlogit_rfx(P, [800.0, 0.0, 0.0, 1.0], nothing, Optim.Options();
                           rethrow_errors = rethrow_errors)
end

# ---------------------------------------------------------------------------

@testset "mlogit_rfx" begin

    # -----------------------------------------------------------------------
    @testset "gradient vs finite differences" begin
        # Generic parameter points: not at the truth, not at sigma = 0. Several
        # specifications, because most indexing bugs in the cell machinery only
        # show up with more than one term or more than one level.
        specs = [
            ("M=1 group-level slope",        [:x1],                                    false),
            ("M=1 option-level intercept",   [rfx_term(level = :alt)],                 false),
            ("M=2 group + option intercept", [:x1, rfx_term(level = :alt)],            false),
            ("M=3 group + int + slope",      [:x1, rfx_term(level = :alt),
                                              rfx_term(:x2; level = :alt)],            false),
            ("M=3 ragged panel",             [:x2, rfx_term(level = :alt),
                                              rfx_term(:x3; level = :alt)],            true),
        ]
        for (label, rfx, ragged) in specs, weighted in (false, true)
            P, buf, gw = _mrfx_kernel_setup(rfx = rfx, weighted = weighted,
                                            ragged = ragged, seed = ragged ? 77 : 13)
            M = P.M
            for sgn in (1.0, -1.0), sc in (0.35, 0.9)
                th = vcat([0.4, -0.3, 0.2], (sgn * sc) .* (0.6 .+ 0.2 .* collect(1:M)))
                ga = _mrfx_grad(P, buf, gw, th)
                gn = FiniteDiff.finite_difference_gradient(_mrfx_obj(P, buf, gw), th,
                                                           Val{:central})
                @test maximum(abs.(ga .- gn) ./ max.(1.0, abs.(gn))) < 1e-6
            end
        end
    end

    # -----------------------------------------------------------------------
    @testset "correlated person block: gradient and zero-correlation nesting" begin
        df = _mrfx_testdata(N = 24, ragged = true)
        rfx = [rfx_term(:xnet; mean = false), :x2,
               rfx_term(level = :alt)]
        rc = [(:xnet, :x2)]

        Pc, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid,
                                     rfx, 96, 20260808, nothing, rc)
        gw = 0.3 .+ 1.5 .* rand(MersenneTwister(7), Pc.N)
        @test Pc.B == 1
        @test Pc.corr_pairs == [(1, 2)]
        @test Pc.theta_names[end] == "cor_xnet__x2"
        @test Pc.rfx_cols[1] == 0
        @test Pc.zmatrix[:, 1] == df.xnet

        for rho in (-0.55, 0.0, 0.45)
            th = [0.4, -0.3, 0.2, 0.65, 0.8, 0.5, rho]
            buf = LTR.MlogitRfxBuffers(Pc)
            ga = _mrfx_grad(Pc, buf, gw, th)
            gn = FiniteDiff.finite_difference_gradient(_mrfx_obj(Pc, buf, gw), th,
                                                       Val{:central})
            @test maximum(abs.(ga .- gn) ./ max.(1.0, abs.(gn))) < 2e-6

            # Check the constrained optimiser-coordinate chain rule separately;
            # a correct public gradient can still be wired incorrectly here.
            phi = LTR._mlogit_rfx_unconstrained_start(th, Pc.K, Pc.M, Pc.B)
            tw = similar(th); gt = similar(th)
            kernel = (F, G, theta) -> LTR._mlogit_rfx_fg!(
                F, G, theta, Pc, LTR.MlogitRfxBuffers(Pc), gw)
            gphi = zeros(length(phi))
            LTR._mlogit_rfx_unconstrained_fg!(true, gphi, phi, Pc.K, Pc.M, Pc.B,
                                              tw, gt, kernel)
            objphi = p -> LTR._mlogit_rfx_unconstrained_fg!(
                true, nothing, collect(p), Pc.K, Pc.M, Pc.B, tw, gt, kernel)
            gnphi = FiniteDiff.finite_difference_gradient(objphi, phi, Val{:central})
            @test maximum(abs.(gphi .- gnphi) ./ max.(1.0, abs.(gnphi))) < 2e-6
        end

        # At rho = 0 the correlated construction must be the old independent
        # model exactly, using the same cells and the same draws.
        Pi, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid,
                                     rfx, 96, 20260808, nothing)
        @test Pi.eta == Pc.eta
        thi = [0.4, -0.3, 0.2, 0.65, 0.8, 0.5]
        thc = [thi; 0.0]
        Gi = _mrfx_grad(Pi, LTR.MlogitRfxBuffers(Pi), gw, thi)
        Gc = _mrfx_grad(Pc, LTR.MlogitRfxBuffers(Pc), gw, thc)
        Qi = _mrfx_obj(Pi, LTR.MlogitRfxBuffers(Pi), gw)(thi)
        Qc = _mrfx_obj(Pc, LTR.MlogitRfxBuffers(Pc), gw)(thc)
        @test Qi == Qc
        @test Gi == Gc[1:end-1]
    end

    # -----------------------------------------------------------------------
    @testset "correlated person block: draw covariance and persistence" begin
        df = _mrfx_testdata(N = 3)
        rfx = [rfx_term(:xnet; mean = false), :x2,
               rfx_term(level = :alt)]
        Pc, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid,
                                     rfx, 20_000, 19, nothing, [(:x2, :xnet)])

        # Reversing the names is canonicalised, so finite draws do not change.
        @test Pc.corr_pairs == [(1, 2)]
        i = 1
        crng = Pc.cell_ranges[i]
        etai = view(Pc.eta, crng, :)
        ct = view(Pc.cell_term, crng)
        cc = view(Pc.corr_cells, i, :)
        Ai = zeros(length(crng), Pc.R)
        sigma = [0.7, 1.1, 0.4]
        rho = -0.6
        LTR._mlogit_rfx_fill_A!(Ai, zeros(Pc.K), sigma, [rho], etai, ct, cc, Pc)
        cp, cq = cc[1], cc[2]
        @test abs(std(view(Ai, cp, :)) - sigma[1]) < 0.02
        @test abs(std(view(Ai, cq, :)) - sigma[2]) < 0.02
        @test abs(cor(view(Ai, cp, :), view(Ai, cq, :)) - rho) < 0.02

        # One group-level cell per person means the same two coefficient draws
        # are reused across every one of that person's choice sets.
        for rg in Pc.ranges, m in 1:2
            @test length(unique(view(Pc.cellloc, rg, m))) == 1
        end
    end

    # -----------------------------------------------------------------------
    @testset "gradient vs finite differences, lognormal" begin
        for rfx in ([:x1 => :lognormal, rfx_term(level = :alt)],
                    [:x1 => :neg_lognormal, rfx_term(:x2; level = :alt)],
                    [:x2 => :lognormal])
            P, buf, gw = _mrfx_kernel_setup(rfx = rfx, weighted = true)
            M = P.M
            # a lognormal term's mu is on the LOG scale; keep exp() sane
            b0 = [any(m -> P.rfx_cols[m] == k && P.rfx_islog[m], 1:M) ? -0.7 :
                  (0.3 * k - 0.2) for k in 1:P.K]
            for sgn in (1.0, -1.0), sc in (0.35, 0.9)
                th = vcat(b0, (sgn * sc) .* (0.6 .+ 0.2 .* collect(1:M)))
                ga = _mrfx_grad(P, buf, gw, th)
                gn = FiniteDiff.finite_difference_gradient(_mrfx_obj(P, buf, gw), th,
                                                           Val{:central})
                @test maximum(abs.(ga .- gn) ./ max.(1.0, abs.(gn))) < 1e-6
            end
        end
    end

    # -----------------------------------------------------------------------
    @testset "saddle point at sigma = 0" begin
        rfx = [:x1, rfx_term(level = :alt), rfx_term(:x2; level = :alt)]
        P, buf, gw = _mrfx_kernel_setup(rfx = rfx, weighted = true)
        th = vcat([0.4, -0.3, 0.2], zeros(P.M))
        g = _mrfx_grad(P, buf, gw, th)

        # sigma = 0 is a stationary point: exactly, thanks to antithetic draws
        @test maximum(abs.(g[P.K+1:end])) < 1e-12

        # and the beta-block there is plain mlogit's gradient
        df = _mrfx_testdata()
        _, xm, yv, G = LTR._prep_mlogit(df, _RXS, :setid, :selected, nothing)
        sc = LTR.MlogitScratch(length(yv), G.Tmax)
        # unweighted, to compare against the unweighted mlogit gradient
        gu = _mrfx_grad(P, LTR.MlogitRfxBuffers(P), nothing, th)
        gm = vec(LTR.mlogit_minus_grad([0.4, -0.3, 0.2], yv, xm, G, sc, nothing))
        @test maximum(abs.(gu[1:P.K] .- gm)) < 1e-10
    end

    # -----------------------------------------------------------------------
    @testset "mirror symmetry Q(mu, s) == Q(mu, -s)" begin
        # Flipping sigma swaps the two antithetic halves of the draw set, so the
        # per-draw log-likelihood vector is an EXACT permutation. Only the final
        # logsumexp over that permuted vector re-orders a sum, so the objective
        # agrees to a relative 1e-16 rather than bit for bit.
        for rfx in ([:x1, rfx_term(level = :alt)],
                    [:x1 => :lognormal, rfx_term(:x2; level = :alt)])
            P, buf, gw = _mrfx_kernel_setup(rfx = rfx, weighted = true)
            b = P.any_log ? [-0.6, -0.3, 0.2] : [0.4, -0.3, 0.2]
            Qp = _mrfx_obj(P, buf, gw)(vcat(b, [ 0.7,  0.45]))
            Qm = _mrfx_obj(P, buf, gw)(vcat(b, [-0.7, -0.45]))
            Qone = _mrfx_obj(P, buf, gw)(vcat(b, [-0.7, 0.45]))
            @test abs(Qp - Qm) < 1e-12 * max(1.0, abs(Qp))
            @test abs(Qp - Qone) > 1e-8
        end
    end

    # -----------------------------------------------------------------------
    @testset "draws are antithetic and mean zero" begin
        P, _, _ = _mrfx_kernel_setup(rfx = [:x1, rfx_term(level = :alt)], R = 64)
        half = P.R ÷ 2
        @test P.eta[:, 1:half] == .-P.eta[:, half+1:end]
        @test maximum(abs.(sum(P.eta, dims = 2))) < 1e-12
        @test all(P.logw .== -log(P.R))
        @test_throws ErrorException LTR._make_cell_draws([2, 2], 7, 1)
    end

    # -----------------------------------------------------------------------
    @testset "nesting: rfx = [] reproduces mlogit" begin
        df = _mrfx_testdata()
        a = mlogit(df, _RXS, :setid, :selected, zeros(3))
        b = mlogit_rfx(df, _RXS, :setid, :selected, zeros(3);
                       col_group = :uniqueid, rfx = [], ndraws = 4)
        @test maximum(abs.(a.theta_hat .- b.theta_hat)) < 1e-8
        @test abs(a.obj_value - b.obj_value) < 1e-10
        @test b.extra.M == 0
    end

    # -----------------------------------------------------------------------
    @testset "binary case reproduces logit2_rfx exactly" begin
        # A choice set of two options, with option 0 carrying all-zero regressors
        # and option 1 the differenced ones, IS the binary logit that logit2_rfx
        # fits on the wide layout. The cell draws are laid out so that the two
        # models consume the identical RNG stream, so this compares numbers, not
        # just distributions -- which is what makes it a real cross-check of the
        # whole kernel rather than a simulation-noise comparison.
        rng = MersenneTwister(5)
        N, T, K = 30, 6, 3
        wide = DataFrame(personid = repeat(1:N, inner = T), choiceid = 1:(N*T))
        for k in 1:K
            wide[!, Symbol("d", k)] = randn(rng, N*T)
        end
        v = 0.5 .* wide.d1 .- 0.3 .* wide.d2
        wide.pick1 = Float64.(rand(rng, N*T) .< (1 ./ (1 .+ exp.(-v))))

        long = DataFrame()
        for r in eachrow(wide)
            push!(long, (personid = r.personid, choiceid = r.choiceid, opt = 0,
                         d1 = 0.0, d2 = 0.0, d3 = 0.0,
                         selected = r.pick1 == 0 ? 1.0 : 0.0))
            push!(long, (personid = r.personid, choiceid = r.choiceid, opt = 1,
                         d1 = r.d1, d2 = r.d2, d3 = r.d3,
                         selected = r.pick1 == 1 ? 1.0 : 0.0))
        end

        xs = [:d1, :d2, :d3]
        for rfx in ([:d1], [:d1, :d2])
            M = length(rfx)
            Pw, _ = LTR._prep_logit2_rfx(wide, xs, :pick1, :personid, rfx,
                                         128, 20260808, nothing)
            Pl, _ = LTR._prep_mlogit_rfx(long, xs, :choiceid, :selected, :personid,
                                         rfx, 128, 20260808, nothing)

            # the same draws, not merely the same distribution
            for i in 1:Pw.N, m in 1:M
                c = first(Pl.cell_ranges[i]) + m - 1
                @test Pw.eta[m, :, i] == Pl.eta[c, :]
            end

            bw = LTR.RfxBuffers(Pw)
            bl = LTR.MlogitRfxBuffers(Pl)
            gw = 0.4 .+ 1.2 .* rand(MersenneTwister(3), Pw.N)
            for th in (vcat([0.5, -0.3, 0.1], fill(0.6, M)),
                       vcat([-0.2, 0.7, -0.4], fill(0.25, M)))
                for w in (nothing, gw)
                    Gw = zeros(K+M); Gl = zeros(K+M)
                    Qw = LTR._rfx_fg!(true, Gw, th, Pw, bw, w)
                    Ql = LTR._mlogit_rfx_fg!(true, Gl, th, Pl, bl, w)
                    @test abs(Qw - Ql) < 1e-10 * max(1.0, abs(Qw))
                    @test maximum(abs.(Gw .- Gl)) < 1e-9
                end
            end
        end
    end

    # -----------------------------------------------------------------------
    @testset "does not mutate data_df" begin
        df = _mrfx_testdata()
        before_names = copy(names(df))
        before_types = Dict(c => eltype(df[!, c]) for c in before_names)
        snapshot = copy(df.x1)

        mlogit_rfx(df, _RXS, :setid, :selected, vcat(zeros(3), [0.5]);
                   col_group = :uniqueid, rfx = [rfx_term(level = :alt)], ndraws = 8)

        @test names(df) == before_names
        @test all(eltype(df[!, c]) == before_types[c] for c in before_names)
        @test df.x1 == snapshot
    end

    # -----------------------------------------------------------------------
    @testset "rfx term normalisation and naming" begin
        syms = [:x1, :x2, :x3]
        t = LTR._normalize_mlogit_rfx([:x1, :x2 => :lognormal,
                                       rfx_term(level = :alt),
                                       rfx_term(:x3; level = :alt)],
                                      syms, :uniqueid)
        @test [u.name for u in t] == ["x1", "x2", "1|alt", "x3|alt"]
        @test [u.col  for u in t] == [1, 2, 0, 3]
        @test [u.at_group for u in t] == [true, true, false, false]
        @test [u.dist for u in t] == [:normal, :lognormal, :normal, :normal]

        # a bare symbol / pair must still mean exactly what it means in logit2_rfx
        @test [u.name for u in LTR._normalize_mlogit_rfx([:x1, :x2], syms, :g)] ==
              ["x1", "x2"]

        @test rfx_term() == (var = nothing, level = nothing, dist = :normal)
        @test rfx_term(:x1; level = :alt, dist = :lognormal) ==
              (var = :x1, level = :alt, dist = :lognormal)
        @test rfx_term("x1"; level = "alt") == (var = :x1, level = :alt, dist = :normal)
    end

    # -----------------------------------------------------------------------
    @testset "theta_names and parameter ordering" begin
        df = _mrfx_testdata()
        rfx = [:x1, rfx_term(level = :alt), rfx_term(:x2; level = :alt)]
        P, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid, rfx,
                                    8, 1, nothing)
        @test P.theta_names == ["x1", "x2", "x3", "sd_x1", "sd_1|alt", "sd_x2|alt"]
        @test P.K == 3 && P.M == 3
        @test P.rfx_pairs == [:x1 => :normal, Symbol("1|alt") => :normal,
                              Symbol("x2|alt") => :normal]
        @test P.rfx_cols == [1, 0, 2]                # 0 == intercept, no formula slot
    end

    # -----------------------------------------------------------------------
    @testset "cell structure" begin
        df = _mrfx_testdata(N = 20, S = 4, J = 3, A = 4)
        rfx = [:x1, rfx_term(level = :alt)]
        P, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid, rfx,
                                    8, 1, nothing)

        # group-level term: exactly one cell per group
        @test all(length(unique(view(P.cellloc, rg, 1))) == 1 for rg in P.ranges)
        # option-level term: as many cells as the group has distinct alternatives
        for (i, rg) in enumerate(P.ranges)
            ids = df.uniqueid .== P.group_ids[i]
            @test length(unique(view(P.cellloc, rg, 2))) ==
                  length(unique(df.alt[ids]))
        end
        # cells are contiguous per group, and term-major inside it
        @test sum(length, P.cell_ranges) == length(P.cell_term)
        for rg in P.cell_ranges
            @test issorted(P.cell_term[rg])
        end
        @test size(P.eta, 1) == length(P.cell_term)

        # the loading of an intercept term is 1, of a slope term the regressor
        @test all(P.zmatrix[:, 2] .== 1.0)
        @test P.zmatrix[:, 1] == P.xmatrix[:, 1]

        s = P.cell_stats
        @test length(s) == 2
        @test s[1].term == "x1" && s[1].cells_per_group_max == 1
        @test s[2].term == "1|alt" && s[2].cells_per_group_max >= 2
        @test all(x.cancel_share == 0.0 for x in s)
    end

    # -----------------------------------------------------------------------
    @testset "results do not depend on the input row order" begin
        # Cells are numbered by SORTED level value, not by order of first
        # appearance, so shuffling the caller's rows cannot re-pair cells with
        # draws. Without that, re-sorting a DataFrame would move every sigma by
        # simulation noise for no visible reason.
        df  = _mrfx_testdata(N = 30)
        dfs = df[randperm(MersenneTwister(808), nrow(df)), :]
        rfx = [:x1, rfx_term(level = :alt), rfx_term(:x2; level = :alt)]

        P,  _ = LTR._prep_mlogit_rfx(df,  _RXS, :setid, :selected, :uniqueid, rfx,
                                     64, 20260808, nothing)
        Ps, _ = LTR._prep_mlogit_rfx(dfs, _RXS, :setid, :selected, :uniqueid, rfx,
                                     64, 20260808, nothing)

        # the same groups, the same cells, the same draws
        @test P.group_ids == Ps.group_ids
        @test P.cell_ranges == Ps.cell_ranges
        @test P.cell_term == Ps.cell_term
        @test P.eta == Ps.eta

        # and the same objective and gradient, up to summation order
        th = vcat([0.4, -0.3, 0.2], [0.6, 0.45, 0.3])
        G1 = zeros(6); G2 = zeros(6)
        Q1 = LTR._mlogit_rfx_fg!(true, G1, th, P,  LTR.MlogitRfxBuffers(P),  nothing)
        Q2 = LTR._mlogit_rfx_fg!(true, G2, th, Ps, LTR.MlogitRfxBuffers(Ps), nothing)
        @test abs(Q1 - Q2) < 1e-10 * max(1.0, abs(Q1))
        @test maximum(abs.(G1 .- G2)) < 1e-9

        th0 = vcat(zeros(3), [0.5, 0.5, 0.5])
        f1 = mlogit_rfx(df,  _RXS, :setid, :selected, th0;
                        col_group = :uniqueid, rfx = rfx, ndraws = 64)
        f2 = mlogit_rfx(dfs, _RXS, :setid, :selected, th0;
                        col_group = :uniqueid, rfx = rfx, ndraws = 64)
        @test maximum(abs.(f1.theta_hat .- f2.theta_hat)) < 1e-6
    end

    # -----------------------------------------------------------------------
    @testset "xlin zeroes exactly the lognormal columns" begin
        df = _mrfx_testdata()
        rfx = [:x1 => :lognormal, rfx_term(level = :alt)]
        P, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid, rfx,
                                    8, 1, nothing)
        @test P.any_log
        @test all(P.xlin[:, 1] .== 0.0)
        @test P.xlin[:, 2:end] == P.xmatrix[:, 2:end]
        @test P.zmatrix[:, 1] == P.xmatrix[:, 1]     # the loading is NOT zeroed

        # with no lognormal term xlin must be the same object, not a copy
        P2, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid,
                                     [:x1], 8, 1, nothing)
        @test !P2.any_log
        @test P2.xlin === P2.xmatrix
    end

    # -----------------------------------------------------------------------
    @testset "guards" begin
        df = _mrfx_testdata()
        th4 = vcat(zeros(3), [0.5])
        th5 = vcat(zeros(3), [0.5, 0.5])
        xs  = [:x1, :x2, :xset]
        call(rfx, th; kwargs...) = mlogit_rfx(df, xs, :setid, :selected, th;
                                              col_group = :uniqueid, rfx = rfx,
                                              ndraws = 8, kwargs...)

        # a group-level random intercept shifts every option equally: it cancels
        @test_throws ErrorException call([rfx_term(level = :uniqueid)], th4)
        # ... and so does a random coefficient on a set-constant regressor
        @test_throws ErrorException call([:xset], th4)
        # one row per cell: nothing to separate the effect from the logit error
        @test_throws ErrorException call([rfx_term(level = :rowid)], th4)
        # a lognormal term has no mu to estimate without a formula variable
        @test_throws ErrorException call([rfx_term(level = :alt, dist = :lognormal)], th4)
        @test_throws ErrorException call([:x1, :x1], th5)
        @test_throws ErrorException call([rfx_term(:x1; level = :alt),
                                          rfx_term(:x1; level = :alt)], th5)
        @test_throws ErrorException call([:x1 => :lognormal,
                                          rfx_term(:x1; level = :alt)], th5)
        @test_throws ErrorException call([:nope], th4)
        @test_throws ErrorException call([:x1 => :cauchy], th4)
        @test_throws ErrorException call([3.7], th4)
        @test_throws ErrorException call([(var = :x1, levl = :alt)], th4)
        @test_throws ErrorException call([rfx_term(:xnet)], th4)
        @test_throws ErrorException call(
            [rfx_term(:xnet; mean = false, dist = :lognormal)], th4)
        @test_throws ErrorException call([rfx_term(level = :alt)], vcat(zeros(3), [0.0]))
        @test_throws ErrorException call([rfx_term(level = :alt)], zeros(3))
        @test_throws ErrorException mlogit_rfx(df, xs, :setid, :selected, th4;
                                               col_group = :uniqueid,
                                               rfx = [rfx_term(level = :alt)],
                                               ndraws = 7)
        # a missing column is named
        @test_throws ErrorException mlogit_rfx(df, xs, :setid, :selected, th4;
                                               col_group = :nope,
                                               rfx = [rfx_term(level = :alt)],
                                               ndraws = 8)

        # a choice-set id that restarts inside each person is not nested
        d2 = _mrfx_testdata()
        transform!(groupby(d2, :uniqueid), :setid => (v -> denserank(v)) => :setnum)
        @test_throws ErrorException mlogit_rfx(d2, xs, :setnum, :selected, th4;
                                               col_group = :uniqueid,
                                               rfx = [rfx_term(level = :alt)],
                                               ndraws = 8)

        # exactly one selected option per choice set
        d3 = _mrfx_testdata()
        r = findfirst(t -> d3.setid[t] == d3.setid[1] && d3.selected[t] == 0.0,
                      1:nrow(d3))
        d3.selected[r] = 1.0
        @test_throws ErrorException mlogit_rfx(d3, xs, :setid, :selected, th4;
                                               col_group = :uniqueid,
                                               rfx = [rfx_term(level = :alt)],
                                               ndraws = 8)

        # weights must be constant within the integration unit
        d4 = _mrfx_testdata()
        d4.w = 0.5 .+ rand(MersenneTwister(1), nrow(d4))
        @test_throws ErrorException mlogit_rfx(d4, xs, :setid, :selected, th4;
                                               col_group = :uniqueid,
                                               rfx = [rfx_term(level = :alt)],
                                               weights = :w, ndraws = 8)
        # a constant-within-group weight column is fine
        transform!(groupby(d4, :uniqueid),
                   :w => (v -> fill(first(v), length(v))) => :wg)
        @test mlogit_rfx(d4, xs, :setid, :selected, th4;
                         col_group = :uniqueid, rfx = [rfx_term(level = :alt)],
                         weights = :wg, ndraws = 8).converged isa Bool

        # Correlation blocks are named, normal, group-level, and disjoint.
        crfx = [rfx_term(:xnet; mean = false), :x2]
        cth  = [zeros(3); 0.5; 0.5; 0.0]
        @test mlogit_rfx(df, _RXS, :setid, :selected, cth;
                         col_group = :uniqueid, rfx = crfx,
                         rfx_corr = [(:xnet, :x2)], ndraws = 8).converged isa Bool
        for badcorr in ([(:nope, :x2)], [(:xnet, :xnet)], [(1, 2, 3)])
            @test_throws ErrorException mlogit_rfx(
                df, _RXS, :setid, :selected, cth;
                col_group = :uniqueid, rfx = crfx, rfx_corr = badcorr, ndraws = 8)
        end
        @test_throws ErrorException mlogit_rfx(
            df, _RXS, :setid, :selected, [zeros(3); fill(0.5, 3); 0.0];
            col_group = :uniqueid,
            rfx = [rfx_term(:xnet; mean = false), :x2,
                   rfx_term(:x1; level = :alt)],
            rfx_corr = [(:xnet, Symbol("x1|alt"))], ndraws = 8)
        @test_throws ErrorException mlogit_rfx(
            df, _RXS, :setid, :selected, [zeros(3); fill(0.5, 3); 0.0; 0.0];
            col_group = :uniqueid,
            rfx = [rfx_term(:xnet; mean = false), :x2, :x3],
            rfx_corr = [(:xnet, :x2), (:x2, :x3)], ndraws = 8)
        @test_throws ErrorException mlogit_rfx(
            df, _RXS, :setid, :selected, cth;
            col_group = :uniqueid, rfx = [:x1 => :lognormal, :x2],
            rfx_corr = [(:x1, :x2)], ndraws = 8)
        @test_throws ErrorException mlogit_rfx(
            df, _RXS, :setid, :selected, [zeros(3); 0.5; 0.5; 1.0];
            col_group = :uniqueid, rfx = crfx,
            rfx_corr = [(:xnet, :x2)], ndraws = 8)
    end

    # -----------------------------------------------------------------------
    @testset "theta0_mlogit_rfx" begin
        rfx = [:x1, rfx_term(level = :alt)]
        th = theta0_mlogit_rfx(_RXS, rfx; b0 = [0.3, -0.2, 0.1], col_group = :uniqueid)
        @test th == [0.3, -0.2, 0.1, 0.5, 0.5]
        @test length(theta0_mlogit_rfx(_RXS, rfx; s0 = [0.7, 0.2],
                                       col_group = :uniqueid)) == 5
        @test theta0_mlogit_rfx(_RXS, rfx; s0 = [0.7, 0.2],
                                col_group = :uniqueid)[4:5] == [0.7, 0.2]
        @test_throws ErrorException theta0_mlogit_rfx(_RXS, rfx; b0 = [0.1, 0.2])
        @test_throws ErrorException theta0_mlogit_rfx(_RXS, rfx; s0 = [0.5])
        @test_throws ErrorException theta0_mlogit_rfx(_RXS, rfx; s0 = [0.0, 0.5])
        @test_throws ErrorException theta0_mlogit_rfx(_RXS, rfx; s0 = [-0.5, 0.5])

        # a lognormal start is converted from the level scale to the log scale
        s0 = 0.4
        thl = theta0_mlogit_rfx(_RXS, [:x1 => :lognormal]; b0 = [2.0, 0.0, 0.0],
                                s0 = [s0], col_group = :uniqueid)
        @test thl[1] ≈ log(2.0) - s0^2 / 2
        @test exp(thl[1] + s0^2 / 2) ≈ 2.0            # implied mean is the level b0
        # ... and the wrong sign is caught before any optimisation
        @test_throws ErrorException theta0_mlogit_rfx(_RXS, [:x1 => :lognormal];
                                                      b0 = [-2.0, 0.0, 0.0])
        @test_throws ErrorException theta0_mlogit_rfx(_RXS, [:x1 => :lognormal])
        thn = theta0_mlogit_rfx(_RXS, [:x1 => :neg_lognormal]; b0 = [-2.0, 0.0, 0.0],
                                s0 = [s0])
        @test thn[1] ≈ log(2.0) - s0^2 / 2

        crfx = [rfx_term(:xnet; mean = false), :x2]
        rc = [(:xnet, :x2)]
        thc = theta0_mlogit_rfx(_RXS, crfx; b0 = [0.3, -0.2, 0.1],
                                 s0 = [0.7, 0.9], rfx_corr = rc, corr0 = [-0.4],
                                 col_group = :uniqueid)
        @test thc == [0.3, -0.2, 0.1, 0.7, 0.9, -0.4]
        @test_throws ErrorException theta0_mlogit_rfx(
            _RXS, crfx; rfx_corr = rc, corr0 = Float64[], col_group = :uniqueid)
        @test_throws ErrorException theta0_mlogit_rfx(
            _RXS, crfx; rfx_corr = rc, corr0 = [1.0], col_group = :uniqueid)
    end

    # -----------------------------------------------------------------------
    @testset "determinism and positive sigma" begin
        df = _mrfx_testdata()
        rfx = [:x1, rfx_term(level = :alt)]
        th0 = vcat(zeros(3), [0.5, 0.5])
        a = mlogit_rfx(df, _RXS, :setid, :selected, th0;
                       col_group = :uniqueid, rfx = rfx, ndraws = 64)
        b = mlogit_rfx(df, _RXS, :setid, :selected, th0;
                       col_group = :uniqueid, rfx = rfx, ndraws = 64)
        @test a.theta_hat == b.theta_hat             # bit-identical refit
        @test a.obj_value == b.obj_value

        # a different seed gives a different (but nearby) answer
        c = mlogit_rfx(df, _RXS, :setid, :selected, th0;
                       col_group = :uniqueid, rfx = rfx, ndraws = 64, seed = 99)
        @test c.theta_hat != a.theta_hat

        @test all(a.theta_hat[4:5] .> 0)
        @test_throws ErrorException mlogit_rfx(
            df, _RXS, :setid, :selected, vcat(zeros(3), [-0.5, 0.5]);
            col_group = :uniqueid, rfx = rfx, ndraws = 64)

        # Regression test for the old post-hoc abs() bug.
        P, gw = LTR._prep_mlogit_rfx(
            df, _RXS, :setid, :selected, :uniqueid, rfx, 64, 20260808, nothing)
        qhat = LTR._mlogit_rfx_fg!(
            true, nothing, a.theta_hat, P, LTR.MlogitRfxBuffers(P), gw)
        @test a.obj_value ≈ qhat atol = 1e-9 rtol = 1e-12
    end

    # -----------------------------------------------------------------------
    @testset "correlated fit determinism and stored objective" begin
        df = _mrfx_testdata(N = 35)
        rfx = [rfx_term(:xnet; mean = false), :x2,
               rfx_term(level = :alt)]
        rc = [(:xnet, :x2)]
        th0 = theta0_mlogit_rfx(_RXS, rfx; rfx_corr = rc,
                                 corr0 = [-0.25], col_group = :uniqueid)
        a = mlogit_rfx(df, _RXS, :setid, :selected, th0;
                       col_group = :uniqueid, rfx = rfx, rfx_corr = rc,
                       ndraws = 64)
        b = mlogit_rfx(df, _RXS, :setid, :selected, th0;
                       col_group = :uniqueid, rfx = rfx,
                       rfx_corr = [(:x2, :xnet)], ndraws = 64)
        @test a.theta_hat == b.theta_hat
        @test a.obj_value == b.obj_value
        @test all(a.theta_hat[4:6] .> 0)
        @test abs(a.theta_hat[7]) < 1

        P, gw = LTR._prep_mlogit_rfx(
            df, _RXS, :setid, :selected, :uniqueid, rfx, 64, 20260808,
            nothing, rc)
        qhat = LTR._mlogit_rfx_fg!(
            true, nothing, a.theta_hat, P, LTR.MlogitRfxBuffers(P), gw)
        @test a.obj_value ≈ qhat atol = 1e-9 rtol = 1e-12
        @test a.extra.B == 1
        @test a.extra.corr_parameterization === :scaled_tanh
    end

    # -----------------------------------------------------------------------
    @testset "extra and ESS diagnostic" begin
        df = _mrfx_testdata()
        rfx = [:x1, rfx_term(level = :alt)]
        fit = mlogit_rfx(df, _RXS, :setid, :selected, vcat(zeros(3), [0.5, 0.5]);
                         col_group = :uniqueid, rfx = rfx, ndraws = 64)
        e = fit.extra
        @test e.model == :mlogit_rfx
        @test e.sigma_parameterization === :softplus
        @test e.K == 3 && e.M == 2 && e.R == 64
        @test e.col_id == :uniqueid            # the integration unit
        @test e.col_set == :setid               # the softmax group
        @test e.n_groups == length(unique(df.uniqueid))
        @test e.n_sets == length(unique(df.setid))
        # ESS lands in (1, R]. The upper end is attained exactly when sigma_hat is
        # 0 (every draw equally weighted), where 1/(R*(1/R)^2) can round to one
        # ulp above R -- hence the tolerance rather than a bare <=.
        @test 1 < e.ess_min <= e.ess_median <= e.R * (1 + 1e-12)
        @test e.ess_p10 <= e.ess_median
        @test e.Ti_max == maximum(combine(groupby(df, :uniqueid), nrow => :n).n)
        @test length(e.cell_stats) == 2

        # the report is a DataFrame with one row per term
        r = rfx_cell_report(fit)
        @test nrow(r) == 2
        @test r.term == ["x1", "1|alt"]
        # a logit2_rfx fit has no cell structure to report
        l2 = logit2_rfx(DataFrame(personid = repeat(1:20, inner = 5),
                                  x1 = randn(MersenneTwister(1), 100),
                                  pick1 = Float64.(rand(MersenneTwister(2), 100) .< 0.5)),
                        [:x1], :pick1, :personid, [0.0, 0.5];
                        rfx = [:x1], ndraws = 8)
        @test_throws ErrorException rfx_cell_report(l2)
    end

    # -----------------------------------------------------------------------
    @testset "bootstrap (serial)" begin
        df = _mrfx_testdata(N = 40)
        rfx = [:x1, rfx_term(level = :alt)]
        plain = mlogit(df, _RXS, :setid, :selected, zeros(3))
        th0 = theta0_mlogit_rfx(_RXS, rfx; b0 = plain.theta_hat, col_group = :uniqueid)
        fit = mlogit_rfx(df, _RXS, :setid, :selected, th0;
                         col_group = :uniqueid, rfx = rfx, ndraws = 64)
        v = boot_mlogit_rfx(df, _RXS, :setid, :selected, th0;
                            col_group = :uniqueid, rfx = rfx, ndraws = 64,
                            nboot = 12, parallel = false, theta_start = fit.theta_hat,
                            cluster_var = :uniqueid)
        fit.vcov = v

        @test size(v.theta_boot_table) == (12, 5)
        @test size(v.V) == (5, 5)
        @test v.method == :bayesian_bootstrap
        @test length(v.boot_fits) == 12
        # every returned sigma is positive
        @test all(all(v.theta_boot_table[b, 4:5] .> 0) for b in 1:12
                  if all(isfinite, view(v.theta_boot_table, b, :)))

        # a fixed boot_seed is reproducible
        v2 = boot_mlogit_rfx(df, _RXS, :setid, :selected, th0;
                             col_group = :uniqueid, rfx = rfx, ndraws = 64,
                             nboot = 12, parallel = false, theta_start = fit.theta_hat)
        @test v.theta_boot_table == v2.theta_boot_table

        @test_throws ErrorException boot_mlogit_rfx(
            df, _RXS, :setid, :selected, th0;
            col_group = :uniqueid, rfx = rfx, ndraws = 64, nboot = 1, parallel = false)

        # the shared reporting layer works on an mlogit_rfx fit unchanged
        rep = boot_report(fit)
        @test nrow(rep) == 5
        @test rep.param == ["x1", "x2", "x3", "sd_x1", "sd_1|alt"]
        @test rep.is_sd == [false, false, false, true, true]
        @test all(isfinite, rep.boot_se)
        @test all(isfinite, rep.ci_lo) && all(isfinite, rep.ci_hi)
        @test all(isnan, rep.share_near_zero[1:3])
        @test all(!isnan, rep.share_near_zero[4:5])

        @test size(boot_vcov!(fit).vcov.V) == (5, 5)
        @test nrow(rfx_level_moments(fit)) == 0      # no lognormal term

        # regtable_rfx renders, with the CI landing on the sd_ rows
        s = sprint(io -> show(io, regtable_rfx(fit; digits = 3, digits_stats = 3)))
        @test occursin("sd_x1", s)
        @test occursin("sd_1|alt", s)
        @test occursin("[", s)                       # an interval was printed

        # Correlation is bootstrapped and reported as a signed parameter, not
        # misclassified as a nonnegative standard deviation.
        crfx = [rfx_term(:xnet; mean = false), :x2,
                rfx_term(level = :alt)]
        rc = [(:xnet, :x2)]
        cth0 = theta0_mlogit_rfx(_RXS, crfx; b0 = plain.theta_hat,
                                  rfx_corr = rc, col_group = :uniqueid)
        cfit = mlogit_rfx(df, _RXS, :setid, :selected, cth0;
                          col_group = :uniqueid, rfx = crfx, rfx_corr = rc,
                          ndraws = 64)
        cfit.vcov = boot_mlogit_rfx(
            df, _RXS, :setid, :selected, cth0;
            col_group = :uniqueid, rfx = crfx, rfx_corr = rc, ndraws = 64,
            nboot = 8, parallel = false, theta_start = cfit.theta_hat)
        @test size(cfit.vcov.theta_boot_table) == (8, 7)
        @test all(abs.(cfit.vcov.theta_boot_table[:, 7]) .< 1)
        crep = boot_report(cfit)
        @test crep.param[end] == "cor_xnet__x2"
        @test crep.is_sd == [false, false, false, true, true, true, false]
        @test isnan(crep.share_near_zero[end])

        # A matrix theta_start multi-starts every bootstrap replicate. This is
        # the production escape hatch when a correlated specification has more
        # than one basin: because row 1 is the single warm start, the selected
        # objective can never be worse replicate by replicate.
        cstarts = theta0_mlogit_rfx_multistart(
            _RXS, crfx; b0 = plain.theta_hat, nstarts = 3,
            rfx_corr = rc, col_group = :uniqueid, seed = 91)
        one = boot_mlogit_rfx(
            df, _RXS, :setid, :selected, cth0;
            col_group = :uniqueid, rfx = crfx, rfx_corr = rc, ndraws = 32,
            nboot = 4, boot_seed = 77, parallel = false,
            theta_start = vec(cstarts[1, :]))
        many = boot_mlogit_rfx(
            df, _RXS, :setid, :selected, cth0;
            col_group = :uniqueid, rfx = crfx, rfx_corr = rc, ndraws = 32,
            nboot = 4, boot_seed = 77, parallel = false, theta_start = cstarts)
        @test all(many.boot_fits[b].obj_value <= one.boot_fits[b].obj_value + 1e-10
                  for b in 1:4)
        @test all(many.boot_fits[b].extra.n_starts == 3 for b in 1:4)
    end

    # -----------------------------------------------------------------------
    @testset "multi-start" begin
        df = _mrfx_testdata(N = 40)
        rfx = [:x1, rfx_term(level = :alt)]
        plain = mlogit(df, _RXS, :setid, :selected, zeros(3))
        th0 = theta0_mlogit_rfx(_RXS, rfx; b0 = plain.theta_hat, col_group = :uniqueid)
        th0m = theta0_mlogit_rfx_multistart(_RXS, rfx; b0 = plain.theta_hat,
                                            nstarts = 5, col_group = :uniqueid, seed = 3)
        @test size(th0m) == (5, 5)
        @test th0m[1, :] == th0                       # row 1 is the single-start default
        @test all(th0m[:, 4:5] .> 0)

        single = mlogit_rfx(df, _RXS, :setid, :selected, th0;
                            col_group = :uniqueid, rfx = rfx, ndraws = 64)
        multi = mlogit_rfx(df, _RXS, :setid, :selected, th0m;
                           col_group = :uniqueid, rfx = rfx, ndraws = 64)
        @test nrow(multi.fits_df) == 5
        @test multi.extra.n_starts == 5
        @test multi.extra.n_usable_starts >= 1
        @test multi.obj_value == multi.extra.obj_best
        # a multi-start fit can never be worse than the single start it contains
        @test multi.obj_value <= single.obj_value + 1e-10
        @test count(multi.fits_df.is_best) == 1

        @test_throws ErrorException theta0_mlogit_rfx_multistart(
            _RXS, rfx; nstarts = 0, col_group = :uniqueid)
        @test_throws ErrorException theta0_mlogit_rfx_multistart(
            _RXS, rfx; s_range = (0.0, 1.0), col_group = :uniqueid)
        # a matrix with the wrong number of columns is caught with a pointed message
        @test_throws ErrorException mlogit_rfx(df, _RXS, :setid, :selected,
                                               zeros(3, 4); col_group = :uniqueid,
                                               rfx = rfx, ndraws = 8)

        crfx = [rfx_term(:xnet; mean = false), :x2,
                rfx_term(level = :alt)]
        rc = [(:xnet, :x2)]
        cth0 = theta0_mlogit_rfx(_RXS, crfx; b0 = plain.theta_hat,
                                  rfx_corr = rc, corr0 = [-0.3],
                                  col_group = :uniqueid)
        cth0m = theta0_mlogit_rfx_multistart(
            _RXS, crfx; b0 = plain.theta_hat, nstarts = 5,
            rfx_corr = rc, corr0 = [-0.3], corr_range = (-0.6, 0.6),
            col_group = :uniqueid, seed = 8)
        @test size(cth0m) == (5, 7)
        @test cth0m[1, :] == cth0
        @test all(cth0m[:, 4:6] .> 0)
        @test all(abs.(cth0m[:, 7]) .< 1)
        @test_throws ErrorException theta0_mlogit_rfx_multistart(
            _RXS, crfx; rfx_corr = rc, corr_range = (-1.0, 0.5),
            col_group = :uniqueid)
    end

    # -----------------------------------------------------------------------
    @testset "rethrow_errors" begin
        df = _mrfx_testdata()
        rfx = [:x1]
        # a lognormal mu large enough to overflow exp() must be reported, not
        # silently turned into NaN and a "converged" fit at garbage
        P, _ = LTR._prep_mlogit_rfx(df, _RXS, :setid, :selected, :uniqueid,
                                    [:x1 => :lognormal], 8, 1, nothing)
        buf = LTR.MlogitRfxBuffers(P)
        @test_throws ErrorException LTR._mlogit_rfx_fg!(
            true, zeros(4), [800.0, 0.0, 0.0, 1.0], P, buf, nothing)

        bad = _mlogit_rfx_failing_fit(df)
        @test bad.errored
        @test !bad.converged
        @test all(isnan, bad.theta_hat)
        @test !isempty(bad.error_message)
        @test_throws Exception _mlogit_rfx_failing_fit(df; rethrow_errors = true)
    end

    # -----------------------------------------------------------------------
    if get(ENV, "LOGITTOOLS_TEST_PARALLEL", "0") == "1"
        @testset "serial == parallel" begin
            nprocs() == 1 && addprocs(2)
            @everywhere using LogitTools
            df = _mrfx_testdata(N = 30)
            rfx = [:x1, rfx_term(level = :alt)]
            th0 = vcat(zeros(3), [0.5, 0.5])
            vs = boot_mlogit_rfx(df, _RXS, :setid, :selected, th0;
                                 col_group = :uniqueid, rfx = rfx, ndraws = 32,
                                 nboot = 8, boot_seed = 4242, parallel = false)
            vp = boot_mlogit_rfx(df, _RXS, :setid, :selected, th0;
                                 col_group = :uniqueid, rfx = rfx, ndraws = 32,
                                 nboot = 8, boot_seed = 4242, parallel = true)
            # A second distributed call in the same process exercises the
            # scoped CachingPool cleanup. Large production preps must not leave
            # stale closures resident on workers between stages.
            vp2 = boot_mlogit_rfx(df, _RXS, :setid, :selected, th0;
                                  col_group = :uniqueid, rfx = rfx, ndraws = 32,
                                  nboot = 8, boot_seed = 4242, parallel = true)
            @test vs.theta_boot_table == vp.theta_boot_table
            @test vs.V == vp.V
            @test vp.theta_boot_table == vp2.theta_boot_table
            @test vp.V == vp2.V
        end
    end

    # -----------------------------------------------------------------------
    @testset "recovery (slow)" begin
        # The point of the whole exercise: a KNOWN option-level random effect has
        # to come back, and plain mlogit -- which cannot see it -- has to be
        # visibly attenuated by ignoring it.
        df = _mrfx_testdata(N = 300, S = 6, J = 4, A = 6, seed = 2024,
                            beta = [0.8, -0.5, 0.3], sigma_g = 0.7, sigma_o = 0.9)
        opts = Optim.Options(iterations = 5_000, g_tol = 1e-6)
        rfx = [:x1, rfx_term(level = :alt)]

        plain = mlogit(df, _RXS, :setid, :selected, zeros(3); optim_options = opts)
        th0 = theta0_mlogit_rfx(_RXS, rfx; b0 = plain.theta_hat, col_group = :uniqueid)
        fit = mlogit_rfx(df, _RXS, :setid, :selected, th0;
                         col_group = :uniqueid, rfx = rfx, ndraws = 400,
                         optim_options = opts)

        @test fit.converged
        @test maximum(abs.(fit.theta_hat[1:3] .- [0.8, -0.5, 0.3])) < 0.15
        @test maximum(abs.(fit.theta_hat[4:5] .- [0.7, 0.9])) < 0.2
        # ignoring the random effect attenuates the slope towards zero
        @test abs(plain.theta_hat[1] - 0.8) > abs(fit.theta_hat[1] - 0.8)
    end

    # -----------------------------------------------------------------------
    @testset "correlated person and option-effect recovery (slow)" begin
        truth_b = [0.7, -0.45, 0.25]
        truth_s = [0.65, 0.85, 0.55]
        truth_rho = -0.45
        df = _mrfx_testdata(N = 500, S = 8, J = 4, A = 6, seed = 90210,
                            beta = truth_b, sigma_net = truth_s[1],
                            sigma_v = truth_s[2], rho = truth_rho,
                            sigma_o = truth_s[3])
        opts = Optim.Options(iterations = 5_000, g_tol = 1e-6)
        rfx = [rfx_term(:xnet; mean = false), :x2,
               rfx_term(level = :alt)]
        rc = [(:xnet, :x2)]

        plain = mlogit(df, _RXS, :setid, :selected, zeros(3);
                       optim_options = opts)
        th0 = theta0_mlogit_rfx(_RXS, rfx; b0 = plain.theta_hat,
                                 rfx_corr = rc, corr0 = [0.0],
                                 col_group = :uniqueid)
        fit = mlogit_rfx(df, _RXS, :setid, :selected, th0;
                         col_group = :uniqueid, rfx = rfx, rfx_corr = rc,
                         ndraws = 600, optim_options = opts)

        @test fit.converged
        @test maximum(abs.(fit.theta_hat[1:3] .- truth_b)) < 0.18
        @test maximum(abs.(fit.theta_hat[4:6] .- truth_s)) < 0.22
        @test abs(fit.theta_hat[7] - truth_rho) < 0.25
        @test sign(fit.theta_hat[7]) == sign(truth_rho)
    end
end
