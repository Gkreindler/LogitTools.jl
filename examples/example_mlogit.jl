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

using FiniteDiff

# make up some MULTINOMIAL choice data
    Random.seed!(123)

    n = 500
    m = 5

    # person
    u_df = DataFrame(
        uniqueid = 1:n,
        clusterid = Int64.(ceil.((1:n) ./ 10)),
    )

    # choice
    c_df = DataFrame(
        idx = 1:m
    )

    # together
    choices_df = crossjoin(u_df, c_df)

    choices_df.dur = rand(Exponential(1), n*m)
    choices_df.total_pay = rand(Uniform(0, 10), n*m)
    choices_df.dist = rand(Uniform(0, 8), n*m)

    beta_dur = -0.1
    beta_total_pay = 0.1
    beta_dist = -0.3

    theta_true = [beta_dur, beta_total_pay, beta_dist]

    choices_df.u  = @. beta_dur * choices_df.dur + 
                       beta_total_pay * choices_df.total_pay + 
                       beta_dist * choices_df.dist

    transform!(groupby(choices_df, :uniqueid), :u => softmax => :probs)

    # draw choice randomly based on probabilities
    transform!(groupby(choices_df, :uniqueid), :probs => (x -> (1:length(x)) .== sample(1:length(x), Weights(x))) => :selected)
    choices_df.selected = choices_df.selected .+ 0.0
    
### Analysis
    myformula = [:dur, :total_pay, :dist]

    # initial conditions
    theta0 = zeros(length(myformula))


### TODO: debug
#     data_df = choices_df 
#     formula = myformula
#     col_id = :uniqueid
#     col_selected = :selected
#     # theta0 = theta0
#     myweights = nothing

#     wvec, xmatrix, yvec, u_comp =
#          LogitTools._prep_mlogit(data_df, formula, col_id, col_selected, myweights)

#     df = data_df
    
#     # define objective function (minus log likelihood)
#     f = theta ->   LogitTools.mlogit_minus_ll(theta, yvec, xmatrix, df, col_id, u_comp, wvec)
#     g = theta -> LogitTools.mlogit_minus_grad(theta, yvec, xmatrix, df, col_id, u_comp, wvec)

#     f(theta0)
# #     f(theta_true)

#     # compute gradient of f with finite difference
#     g_fd = theta -> FiniteDiff.finite_difference_jacobian(f, theta)

#     g(theta0) |> display
#     g_fd(theta0) |> display

#     @assert maximum(g(theta0) .- g_fd(theta0)) < 1e-4

    # estimate
    # @profview 
    choices_df.myweights = ones(Float64, n*m)
    logit_fit = mlogit(choices_df, myformula, :uniqueid, :selected, theta0, myweights=:myweights)

    # print results without variance-covariance matrix
    regtable(logit_fit) |> display

    # bootstrap
    @time logit_fit.vcov = boot_mlogit(
        choices_df, 
        myformula, 
        :uniqueid,
        :selected,
        theta0,
        cluster_var=:clusterid,
        nboot=500,
        mydebug=true);

    regtable(logit_fit) |> display

    percentile(logit_fit.vcov.theta_boot_table[:, 1], [2.5, 97.5]) |> display
    histogram(logit_fit.vcov.theta_boot_table[:, 1], bins=50) |> display
    describe(logit_fit.vcov.theta_boot_table[:, 1])
