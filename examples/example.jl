using Pkg
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

# make up some binary choice data
    Random.seed!(123)

    n = 1000

    choices_df = DataFrame(
        uniqueid = 1:n,
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
    
    theta0 = zeros(length(myxs))

    # @time mym = logit2(data_df, myxs, :y, theta0, weights=:w)
    # Optim.minimizer(mym) |> display

    logit_fit = logit2(choices_df, myxs, :pick1, theta0)
    # theta_hat = Optim.minimizer(mym)

    # logit_fit.vcov = Dict(
    #     :method => :simple,
    #     :V => zeros(3,3)
    # )
    logit_fit.vcov = zeros(3,3)
    

    # LogitTools.LogitRegModel(logit_fit)

    regtable(logit_fit)

    # @time theta_boot = boot_logit2(
    #     choices_df, 
    #     myxs, 
    #     :pick1,
    #     theta0,
    #     nboot=500)

    # percentile(theta_boot[:, 1], [2.5, 97.5])

    # myline = "\n" * lpad("Coefficient", 20, " ") * " | Coef  |  CI 95%"
    # println(myline)
    # println("----------------------------------------------")
    # for i=1:8
    #     myline = ""
    #     myline *= lpad(string(myxs[i]), 20, " ")
    #     myline *= lpad(@sprintf(" %5.2f ", theta_hat[i]), 9, " ")

    #     li, ui = percentile(theta_boot[:, i], [2.5, 97.5])

    #     myline *= lpad(@sprintf("[%5.2f, %5.2f]", li, ui), 14, " ")

    #     println( myline)
    # end

