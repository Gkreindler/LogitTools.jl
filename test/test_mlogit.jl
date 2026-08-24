#=
Regression tests for the modernised `mlogit`.

`_mlogit_reference` below is the implementation from before the refactor, kept
VERBATIM as a permanent oracle -- the same device `test_boot_logit2_parallel.jl`
uses for `boot_logit2`. The refactor moved the choice-set grouping out of the
per-evaluation path (it used to call `transform!(groupby(df, col_id), ...)` on the
caller's DataFrame a few thousand times per fit, mutating it on the way), and the
claim is that this is arithmetically inert: every group's rows are still visited
in their original order and every reduction still runs over the same array in the
same order.

So these tests assert **bitwise** equality, not approximate equality. An
approximate test would pass even if the refactor had quietly changed the
summation order, and the published tables were produced with the old code.
=#

using LogitTools
using Test
using DataFrames
using Random
using Statistics
using StatsBase
using LinearAlgebra
using LogExpFunctions
using Optim

const LTM = LogitTools

# ---------------------------------------------------------------------------
# the pre-refactor implementation, verbatim
# ---------------------------------------------------------------------------

function _ref_mlogit_minus_ll(theta, yvec, xmatrix, df, col_id, u_comp, weights)
    mul!(u_comp, xmatrix, theta)
    df.u_comp .= u_comp
    transform!(groupby(df, col_id), :u_comp => logsumexp => :log_sum_exp)
    @. u_comp = @. yvec .* (u_comp .- df.log_sum_exp)
    if !isnothing(weights)
        @. u_comp .*= weights
    end
    return -sum(u_comp)
end

function _ref_mlogit_minus_grad(theta, yvec, xmatrix, df, col_id, u_comp, weights)
    mul!(u_comp, xmatrix, theta)
    df.pi .= u_comp
    transform!(groupby(df, col_id), :pi => softmax => :pi)
    if isnothing(weights)
        return - sum((yvec .- df.pi) .* xmatrix , dims=1)
    else
        return - sum((yvec .- df.pi) .* weights .* xmatrix , dims=1)
    end
end

function _ref_prep_mlogit(data_df, formula, col_id, col_selected, weights=nothing)
    if !isnothing(weights)
        temp_df = combine(groupby(data_df, col_id), weights => sum => :weights_sum)
        weights_sum = sum(temp_df.weights_sum)
        wvec = data_df[!, weights] ./ weights_sum
    else
        wvec = nothing
    end
    for mycol=formula
        if !(eltype(data_df[!, mycol]) == Float64)
            data_df[!, mycol] = convert.(Float64, data_df[:, mycol])
        end
    end
    xmatrix = Matrix(data_df[:, formula])
    yvec = data_df[:, col_selected]
    eltype(yvec) == Float64 || (yvec = convert.(Float64, yvec))
    all((yvec .== 0.0) .| (yvec .== 1.0)) ||
        error("choice column should have 0's and 1's only")
    u_comp = copy(yvec)
    transform!(groupby(data_df, col_id),
               col_selected => (x -> length(x)) => :__group_count)
    return wvec, xmatrix, yvec, u_comp
end

function _ref_mlogit_inner(; xmatrix, df, col_id, yvec, u_comp, theta0, wvec=nothing)
    f = theta ->   _ref_mlogit_minus_ll(theta, yvec, xmatrix, df, col_id, u_comp, wvec)
    g = theta -> _ref_mlogit_minus_grad(theta, yvec, xmatrix, df, col_id, u_comp, wvec)
    opt = optimize(f, g, theta0, LBFGS(), inplace=false)
    return (theta = Optim.minimizer(opt), obj = Optim.minimum(opt),
            converged = Optim.converged(opt), iterations = Optim.iterations(opt))
end

function _ref_mlogit(data_df, formula, col_id, col_selected, theta0; myweights=nothing)
    wvec, xmatrix, yvec, u_comp =
        _ref_prep_mlogit(data_df, formula, col_id, col_selected, myweights)
    return _ref_mlogit_inner(xmatrix=xmatrix, df=data_df, col_id=col_id, yvec=yvec,
                             u_comp=u_comp, theta0=theta0, wvec=wvec)
end

function _ref_boot_mlogit(data_df, formula, col_id, col_selected, theta0;
                          nboot=20, cluster_var=nothing)
    LTM.bbw!(data_df, nboot; cluster_var=cluster_var, mydebug=false)
    _, xmatrix, yvec, u_comp =
        _ref_prep_mlogit(data_df, formula, col_id, col_selected, nothing)
    tbl = zeros(nboot, length(theta0))
    for i = 1:nboot
        r = _ref_mlogit_inner(xmatrix=xmatrix, df=data_df, col_id=col_id, yvec=yvec,
                              u_comp=u_comp, theta0=theta0,
                              wvec=data_df[:, "bw" * string(i)])
        tbl[i, :] .= r.theta
    end
    return tbl
end

# ---------------------------------------------------------------------------
# data: deliberately NOT sorted by the choice-set id, and with an Int regressor,
# because those are the two things the refactor changed how it handles
# ---------------------------------------------------------------------------
function _mlogit_testdata(; N = 60, S = 3, J = 4, seed = 99, shuffle = true)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    for i in 1:N, s in 1:S, j in 1:J
        push!(rows, (uniqueid = i, setid = i * 100 + s, alt = j))
    end
    df = DataFrame(rows)
    df.x1 = randn(rng, nrow(df))
    df.x2 = rand(rng, -3:3, nrow(df))       # Int column on purpose
    df.x3 = randn(rng, nrow(df))
    df.selected = zeros(Float64, nrow(df))
    for g in groupby(df, :setid)
        g.selected[rand(rng, 1:nrow(g))] = 1.0
    end
    df.w = repeat(0.5 .+ rand(rng, N), inner = S * J)
    return shuffle ? df[randperm(MersenneTwister(4), nrow(df)), :] : df
end

const _MXS  = [:x1, :x2, :x3]
const _MTH0 = zeros(3)

@testset "mlogit" begin

    @testset "bitwise identical to the pre-refactor implementation" begin
        for (label, wcol) in [("unweighted", nothing), ("weighted", :w)]
            dref = _mlogit_testdata()
            dnew = _mlogit_testdata()
            r = _ref_mlogit(dref, _MXS, :setid, :selected, _MTH0; myweights = wcol)
            f = mlogit(dnew, _MXS, :setid, :selected, _MTH0; myweights = wcol)

            @test f.theta_hat  == r.theta        # bitwise, not approx
            @test f.obj_value  == r.obj
            @test f.iterations == r.iterations
            @test f.converged  == r.converged
        end
    end

    @testset "objective and gradient are bitwise identical" begin
        dref = _mlogit_testdata()
        dnew = _mlogit_testdata()
        wref, xref, yref, ucref = _ref_prep_mlogit(dref, _MXS, :setid, :selected, :w)
        _, xnew, ynew, G = LTM._prep_mlogit(dnew, _MXS, :setid, :selected, :w)
        wnew = LTM._prep_mlogit(dnew, _MXS, :setid, :selected, :w)[1]
        sc = LTM.MlogitScratch(length(ynew), G.Tmax)

        @test xnew == xref
        @test ynew == yref
        @test wnew == collect(wref)

        for th in ([0.0, 0.0, 0.0], [0.4, -0.25, 0.7], [-1.3, 0.9, 0.05])
            for (wr, wn) in ((nothing, nothing), (wref, wnew))
                @test LTM.mlogit_minus_ll(th, ynew, xnew, G, sc, wn) ==
                      _ref_mlogit_minus_ll(th, yref, xref, dref, :setid, ucref, wr)
                @test LTM.mlogit_minus_grad(th, ynew, xnew, G, sc, wn) ==
                      _ref_mlogit_minus_grad(th, yref, xref, dref, :setid, ucref, wr)
            end
        end
    end

    @testset "boot_mlogit bitwise identical, clustered and unclustered" begin
        for cv in (:uniqueid, nothing)
            dref = _mlogit_testdata()
            dnew = _mlogit_testdata()
            Random.seed!(20260824)
            ref = _ref_boot_mlogit(dref, _MXS, :setid, :selected, _MTH0;
                                   nboot = 10, cluster_var = cv)
            Random.seed!(20260824)
            v = boot_mlogit(dnew, _MXS, :setid, :selected, _MTH0;
                            nboot = 10, cluster_var = cv)
            @test v.theta_boot_table == ref
            # bbw! still appends its columns, as before
            @test "bw1" in names(dnew)
        end
    end

    @testset "boot_mlogit: serial == parallel" begin
        if nprocs() > 1
            d1 = _mlogit_testdata(); d2 = _mlogit_testdata()
            Random.seed!(31)
            a = boot_mlogit(d1, _MXS, :setid, :selected, _MTH0;
                            nboot = 8, cluster_var = :uniqueid, parallel = false)
            Random.seed!(31)
            b = boot_mlogit(d2, _MXS, :setid, :selected, _MTH0;
                            nboot = 8, cluster_var = :uniqueid, parallel = true)
            @test a.theta_boot_table == b.theta_boot_table
        else
            @test_throws ErrorException boot_mlogit(
                _mlogit_testdata(), _MXS, :setid, :selected, _MTH0;
                nboot = 2, parallel = true)
        end
    end

    @testset "does not mutate data_df" begin
        d = _mlogit_testdata()
        before_names = copy(names(d))
        before_types = Dict(c => eltype(d[!, c]) for c in before_names)
        before_x2 = copy(d.x2)

        mlogit(d, _MXS, :setid, :selected, _MTH0)

        @test names(d) == before_names           # no :u_comp, :pi, :log_sum_exp,
        @test d.x2 == before_x2                  # no :__group_count
        @test all(eltype(d[!, c]) == before_types[c] for c in before_names)
    end

    @testset "optim_options is honoured" begin
        d = _mlogit_testdata()
        capped = mlogit(d, _MXS, :setid, :selected, _MTH0;
                        optim_options = Optim.Options(iterations = 1))
        @test capped.iterations <= 1
        @test capped.iteration_limit_reached
        @test !capped.converged

        free = mlogit(d, _MXS, :setid, :selected, _MTH0)
        @test free.converged
        # the default must be exactly what optimize() used to do implicitly
        @test free.theta_hat ==
              mlogit(d, _MXS, :setid, :selected, _MTH0;
                     optim_options = Optim.Options()).theta_hat
    end

    @testset "MlogitGroups preserves the original within-group order" begin
        d = _mlogit_testdata()
        G = LTM.MlogitGroups(d.setid)
        @test length(G.ranges) == length(unique(d.setid))
        @test sum(length, G.ranges) == nrow(d)
        @test sort(G.perm) == 1:nrow(d)
        for rg in G.ranges
            idx = G.perm[rg]
            @test issorted(idx)                          # original order kept
            @test length(unique(d.setid[idx])) == 1       # one choice set per range
        end
        @test G.Tmax == maximum(length, G.ranges)
    end

    @testset "guards" begin
        d = _mlogit_testdata()
        bad = copy(d)
        bad.selected = fill(0.5, nrow(bad))
        @test_throws ErrorException mlogit(bad, _MXS, :setid, :selected, _MTH0)
        @test_throws ErrorException LTM.MlogitGroups(Int[])
    end
end
