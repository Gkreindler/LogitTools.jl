using Pkg
# Pkg.activate("./examples/env/")
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using Revise
using LinearAlgebra # for identity matrix "I"
using CSV
using Random
 
using DataFrames
using FixedEffectModels # for benchmarking
using RegressionTables

using LogitTools
using LogExpFunctions
using Distributions

using Optim

using StatsBase

using Plots

# make up some binary choice data
    Random.seed!(123)

    n = 5000

    choices_df = DataFrame(
        uniqueid = 1:n,
        clusterid = Int64.(ceil.((1:n) ./ 10)),
        dur1 = rand(Exponential(1), n),
        dur2 = rand(Exponential(1), n),
        total_pay1 = rand(Uniform(0, 10), n),
        total_pay2 = rand(Uniform(0, 10), n),
        dist1 = rand(Uniform(0, 8), n),
        dist2 = rand(Uniform(0, 8), n)
    )

    choices_df.dur = choices_df.dur1 .- choices_df.dur2
    choices_df.total_pay = choices_df.total_pay1 .- choices_df.total_pay2
    choices_df.dist = choices_df.dist1 .- choices_df.dist2

    beta_dur = -0.1
    beta_total_pay = 0.1
    beta_dist = -0.3

    choices_df.u1 = @. beta_dur * choices_df.dur1 + 
                       beta_total_pay * choices_df.total_pay1 + 
                       beta_dist * choices_df.dist1

    choices_df.u2 = @. beta_dur * choices_df.dur2 + 
                       beta_total_pay * choices_df.total_pay2 + 
                       beta_dist * choices_df.dist2

    choices_df.probs = softmax(hcat(choices_df.u1, choices_df.u2), dims=2)[:, 1]

    choices_df.pick1 = (rand(n) .< choices_df.probs) .+ 0

### Analysis
    myxs = [:dur, :total_pay, :dist]

    # initial conditions
    theta0 = zeros(length(myxs))

    # estimate
    logit_fit = logit2(choices_df, myxs, :pick1, theta0)

    # without variance-covariance matrix
    # logit_fit.vcov = zeros(3,3)
    # regtable(logit_fit)

    # bootstrap
    @time logit_fit.vcov = boot_logit2(
        choices_df, 
        myxs, 
        :pick1,
        theta0,
        cluster_var=:clusterid,
        nboot=500);

    regtable(logit_fit) |> display

    percentile(logit_fit.vcov.theta_boot_table[:, 1], [2.5, 97.5]) |> display
    histogram(logit_fit.vcov.theta_boot_table[:, 1], bins=50) |> display
    describe(logit_fit.vcov.theta_boot_table[:, 1])
