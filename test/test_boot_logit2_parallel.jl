using LogitTools
using Test
using DataFrames
using Random
using Distributions
using Optim
using StatsBase
using Distributed   # top level: @everywhere is expanded at parse time

const LT = LogitTools

"""Clustered binary-choice data."""
function _bl2_testdata(; n = 900, seed = 20260808)
    Random.seed!(seed)
    df = DataFrame(clusterid = Int.(ceil.((1:n) ./ 10)),
                   x1 = randn(n), x2 = randn(n), x3 = randn(n))
    df.pick1 = Float64.(rand(n) .< 1 ./ (1 .+ exp.(-(0.5 .* df.x1 .- 0.8 .* df.x2 .+ 0.3 .* df.x3))))
    return df
end

"""
`_boot_logit2` exactly as it was on `main`: one shared `u_comp` buffer, weights
read out of the DataFrame column by column. The refactored implementation must
reproduce this bit for bit.
"""
function _boot_logit2_reference(data_df, formula, choice, theta0;
                                nboot = 500, cluster_var = nothing)
    LT.bbw!(data_df, nboot; cluster_var = cluster_var, mydebug = false)
    _, xmatrix, yvec, u_comp = LT._prep_logit2(data_df, formula, choice, nothing)

    theta_boot_table = zeros(nboot, length(theta0))
    all_boot_fits = []
    for i in 1:nboot
        boot_fit = LT._logit2(xmatrix = xmatrix, yvec = yvec, u_comp = u_comp,
                              theta0 = theta0, wvec = data_df[:, "bw" * string(i)])
        theta_boot_table[i, :] .= boot_fit.theta_hat
        push!(all_boot_fits, boot_fit)
    end

    return LT.MLEvcov(method = :bayesian_bootstrap,
                      theta_boot_table = theta_boot_table,
                      V = cov(theta_boot_table),
                      boot_fits = all_boot_fits)
end

@testset "boot_logit2 parallel" begin

    myxs = [:x1, :x2, :x3]
    nboot = 20

    # -----------------------------------------------------------------------
    @testset "serial path is unchanged from main" begin
        for cv in (:clusterid, nothing)
            d1 = _bl2_testdata(); Random.seed!(999)
            ref = _boot_logit2_reference(d1, myxs, :pick1, zeros(3);
                                         nboot = nboot, cluster_var = cv)

            d2 = _bl2_testdata(); Random.seed!(999)
            new = boot_logit2(d2, myxs, :pick1, zeros(3);
                              nboot = nboot, cluster_var = cv)

            @test new.theta_boot_table == ref.theta_boot_table     # bitwise
            @test new.V == ref.V
            @test new.method == ref.method
            @test length(new.boot_fits) == nboot
            # Table4_boott.jl reads boot_fits[b].weights -- must be preserved
            @test all(new.boot_fits[b].weights == ref.boot_fits[b].weights for b in 1:nboot)
            @test all(new.boot_fits[b].theta_hat == ref.boot_fits[b].theta_hat for b in 1:nboot)
            @test new.boot_fits[1].weights isa Vector{Float64}
            # bbw! still appends its bw* columns to the caller's DataFrame
            @test names(d1) == names(d2)
            @test count(nm -> startswith(nm, "bw"), names(d2)) == nboot
        end
    end

    # -----------------------------------------------------------------------
    @testset "weights matrix round-trips the bw columns" begin
        df = _bl2_testdata(); Random.seed!(7)
        LT.bbw!(df, nboot; cluster_var = :clusterid)
        W = LT._boot_weights_matrix(df, nboot)
        @test size(W) == (nrow(df), nboot)
        @test all(W[:, b] == df[!, "bw" * string(b)] for b in 1:nboot)
        # each column is normalised to mean 1 by bbw!
        @test all(≈(1.0), [mean(W[:, b]) for b in 1:nboot])
    end

    # -----------------------------------------------------------------------
    @testset "old internal signature still works" begin
        df = _bl2_testdata(); Random.seed!(11)
        LT.bbw!(df, nboot; cluster_var = :clusterid)
        _, xmatrix, yvec, u_comp = LT._prep_logit2(df, myxs, :pick1, nothing)
        v = LT._boot_logit2(nboot, xmatrix, yvec, u_comp, zeros(3), df, false)
        @test size(v.theta_boot_table) == (nboot, 3)
        @test v.method == :bayesian_bootstrap
    end

    # -----------------------------------------------------------------------
    @testset "optim_options" begin
        # the default must reproduce the old no-options call exactly
        d1 = _bl2_testdata(); f1 = logit2(d1, myxs, :pick1, zeros(3))
        d2 = _bl2_testdata(); f2 = logit2(d2, myxs, :pick1, zeros(3);
                                          optim_options = Optim.Options())
        @test f1.theta_hat == f2.theta_hat
        @test f1.obj_value == f2.obj_value
        @test f1.iterations == f2.iterations

        d3 = _bl2_testdata(); Random.seed!(5)
        v1 = boot_logit2(d3, myxs, :pick1, zeros(3); nboot = 10, cluster_var = :clusterid)
        d4 = _bl2_testdata(); Random.seed!(5)
        v2 = boot_logit2(d4, myxs, :pick1, zeros(3); nboot = 10, cluster_var = :clusterid,
                         optim_options = Optim.Options())
        @test v1.theta_boot_table == v2.theta_boot_table

        # and the options actually reach the optimiser
        d5 = _bl2_testdata()
        f3 = logit2(d5, myxs, :pick1, zeros(3);
                    optim_options = Optim.Options(iterations = 1))
        @test f3.iterations == 1
        @test f3.iteration_limit_reached
        @test !f3.converged

        d6 = _bl2_testdata(); Random.seed!(5)
        v3 = boot_logit2(d6, myxs, :pick1, zeros(3); nboot = 5, cluster_var = :clusterid,
                         optim_options = Optim.Options(iterations = 1))
        @test all(f.iterations == 1 for f in v3.boot_fits)
    end

    # -----------------------------------------------------------------------
    @testset "parallel = true with no workers errors" begin
        if nprocs() == 1
            df = _bl2_testdata()
            @test_throws ErrorException boot_logit2(df, myxs, :pick1, zeros(3);
                                                    nboot = 4, parallel = true)
            # and it fails before mutating the DataFrame
            @test !any(nm -> startswith(nm, "bw"), names(df))
        end
    end

    # -----------------------------------------------------------------------
    # Guarded for the same reason as the rfx serial-vs-parallel test: addprocs
    # inside Pkg.test()'s sandbox is a known flakiness source.
    #     LOGITTOOLS_TEST_PARALLEL=1 julia --project=. -e 'using Pkg; Pkg.test()'
    if get(ENV, "LOGITTOOLS_TEST_PARALLEL", "0") == "1"
        @testset "serial == parallel" begin
            addprocs(2)
            try
                @everywhere using LogitTools
                for cv in (:clusterid, nothing)
                    d1 = _bl2_testdata(); Random.seed!(4242)
                    vs = boot_logit2(d1, myxs, :pick1, zeros(3);
                                     nboot = nboot, cluster_var = cv, parallel = false)
                    d2 = _bl2_testdata(); Random.seed!(4242)
                    vp = boot_logit2(d2, myxs, :pick1, zeros(3);
                                     nboot = nboot, cluster_var = cv, parallel = true)

                    @test vs.theta_boot_table == vp.theta_boot_table   # bitwise
                    @test vs.V == vp.V
                    @test all(vs.boot_fits[b].weights == vp.boot_fits[b].weights
                              for b in 1:nboot)
                    @test all(vs.boot_fits[b].theta_hat == vp.boot_fits[b].theta_hat
                              for b in 1:nboot)
                    @test count(nm -> startswith(nm, "bw"), names(d2)) == nboot
                end
            finally
                rmprocs(workers())
            end
        end
    end
end
