
Base.@kwdef mutable struct MLEFit
    theta0::Vector         # initial conditions   (vector of size P or K x P matrix for K sets of initial conditions)
    theta_hat::Vector      # estimated parameters (vector of size P)
    theta_names::Union{Vector{String}, Nothing}
    theta_factors::Union{Vector{Float64}, Nothing} = nothing # nothing or a vector of length P with factors for each parameter. Parameter theta[i] was replaced by theta[i] * theta_factors[i] before optimization

    n_obs = nothing # number of observations (N)

    # estimation parameters
    weights=nothing # Vector of size N or nothing
    
    # optimization results
    obj_value::Number
    errored::Bool = false
    error_message::String = ""
    converged::Bool
    iterations::Union{Integer, Missing}
    iteration_limit_reached::Union{Bool, Missing}
    time_it_took::Union{Float64, Missing}

    # results from multiple initial conditions (DataFrame)
    fits_df = nothing
    # idx = nothing # aware of which iteration number this is

    # variance covariance matrix
    vcov = nothing
end


# https://www.statlect.com/fundamentals-of-statistics/logistic-model-maximum-likelihood

"""
utility for choosing the 2nd option minus utility for choosing the 1st option
"""
function compute_utility(theta::Vector{Float64}, xmatrix::Matrix{Float64})
    xmatrix * theta
end


"""
minus log likelhood
"""
function minus_ll(
        theta::Vector{Float64}, 
        yvec::Vector{Float64}, 
        xmatrix::Matrix{Float64}, 
        u_comp::Vector{Float64},
        weights::Union{Nothing, Vector{Float64}})
    
    # inplace multiplication
    # u_comp = compute_utility(theta, xmatrix)
    mul!(u_comp, xmatrix, theta)

    # @. u_comp = @. yvec .* u_comp .- log.(1 .+ exp.(u_comp))
    @. u_comp = @. yvec .* u_comp .- log1pexp.(u_comp)
    

    if !isnothing(weights)
        @. u_comp .*= weights
    end
    return -sum(u_comp)
end

function probs(u::Vector{Float64})
    @. 1.0 / (1.0 + exp(-u))
end

function minus_grad(
        theta::Vector{Float64}, 
        yvec::Vector{Float64}, 
        xmatrix::Matrix{Float64}, 
        u_comp::Vector{Float64},
        weights::Union{Nothing, Vector{Float64}})
    
    # inplace multiplication
    mul!(u_comp, xmatrix, theta)

    @. u_comp = (yvec - logistic.(u_comp))

    if isnothing(weights)
        return - transpose(u_comp) * xmatrix 
    else
        return - transpose(u_comp .* weights) * xmatrix
    end

end

function _prep_logit2(
    data_df, 
    formula, 
    choice,
    weights::Union{Nothing, Symbol, String}=nothing)

    # weights
    if !isnothing(weights)
        wvec = data_df[:, weights]
    else
        wvec = nothing
    end

    # all floats
        for mycol=formula
            if !(eltype(data_df[!, mycol]) == Float64) 
                data_df[!, mycol] = convert.(Float64, data_df[:, mycol])
            end
        end

    # prepare matrix (of regressors)
        xmatrix = Matrix(data_df[:, formula])

    # prepare outcome
        yvec = data_df[:, choice]
        eltype(yvec) == Float64 || (yvec = convert.(Float64, yvec))
        all((yvec .== 0.0) .| (yvec .== 1.0)) || error("choice column should have 0's and 1's only")

        u_comp = copy(yvec)

    return wvec, xmatrix, yvec, u_comp
end

"""
Estimate a binary logit model with MLE
    formula = vector of 
"""
function logit2(
    data_df, 
    formula, 
    choice,
    theta0; 
    weights::Union{Nothing, Symbol, String}=nothing)

    wvec, xmatrix, yvec, u_comp = _prep_logit2(data_df, formula, choice, weights)

    # estimate
    return _logit2(
            xmatrix=xmatrix, 
            yvec=yvec, 
            u_comp=u_comp,
            theta0=theta0,
            wvec=wvec)
end

"""
    the inner function (to not repeat prep when bootstrapping)
"""
function _logit2(;
    xmatrix::Matrix{Float64}, 
    yvec::Vector{Float64}, 
    u_comp::Vector{Float64},
    theta0::Vector{Float64},
    wvec::Union{Nothing, Vector{Float64}}=nothing)

    # define objective function (minus log likelihood)
        f = theta -> minus_ll(theta, yvec, xmatrix, u_comp, wvec)
        g = theta -> minus_grad(theta, yvec, xmatrix, u_comp, wvec)

    # estimate
    time_it_took = @elapsed opt_results = optimize(f, g, theta0, LBFGS(), inplace=false)

    # return an MLEFit object
    return MLEFit(
        theta0=theta0,
        theta_hat=Optim.minimizer(opt_results),
        theta_names=nothing,
        n_obs=length(yvec),
        weights=wvec,
        obj_value=Optim.minimum(opt_results),
        converged=Optim.converged(opt_results),
        iterations=Optim.iterations(opt_results),
        iteration_limit_reached=Optim.iteration_limit_reached(opt_results),
        time_it_took=time_it_took
    )
end


"""
Generate nboot columns with bayesian bootstrap weights, optionally clustering by cluster_var
"""
function bbw!(data_df, nboot; cluster_var=nothing, mydebug=false)

    if !isnothing(cluster_var)
        cluster_values = unique(data_df[:, cluster_var])

        ### ___idx_cluster___ has the index in the cluster_values vector of this row's value        
            # drop column if already in the df
            ("___idx_cluster___" in names(data_df)) && select!(data_df, Not("___idx_cluster___"))

            # join
            temp_df = DataFrame(string(cluster_var) => cluster_values, "___idx_cluster___" => 1:length(cluster_values))
            leftjoin!(data_df, temp_df, on=cluster_var)
    end

    n = nrow(data_df)
    for i=1:nboot
        mydebug && println("constructing bayesian bootstrap weights, column=", i)
        
        if !isnothing(cluster_var)

            cluster_level_weights = rand(Dirichlet(length(cluster_values), 1.0))  

            # one step "join" to get the weight for the appropriate cluster
            data_df[!, "bw" * string(i)] .= cluster_level_weights[data_df.___idx_cluster___]    
        else
            data_df[!, "bw" * string(i)] = rand(Dirichlet(nrow(data_df), 1.0))
        end

        s = sum(data_df[!, "bw" * string(i)])
        data_df[!, "bw" * string(i)] *= n/s 
    end

    return
end

function _boot_logit2(
            nboot::Int64, 
            xmatrix::Matrix{Float64}, 
            yvec::Vector{Float64}, 
            u_comp::Vector{Float64},
            theta0::Vector{Float64},
            data_df::DataFrame,
            mydebug::Bool=false)

    # store results
    theta_boot = zeros(nboot, length(theta0))

    # estimate model for each bootstrap column of weights
    for i=1:nboot
        mydebug && println("bootstrapping, column=", i)

        x = _logit2(
            xmatrix=xmatrix, 
            yvec=yvec, 
            u_comp=u_comp,
            theta0=theta0,
            wvec=data_df[:, "bw" * string(i)])

        theta_boot[i, :] .=  Optim.minimizer(x)
    end

    return theta_boot
end

"""
    bootstrap for logit2
"""
function boot_logit2(
    data_df, 
    formula, 
    choice,
    theta0;
    nboot=500,
    cluster_var=nothing,
    mydebug=false)

    # pre-compute a lot of weights columns
    bbw!(data_df, nboot; cluster_var=cluster_var, mydebug=mydebug)

    # prep only once
    _, xmatrix, yvec, u_comp = _prep_logit2(data_df, formula, choice, nothing)

    return _boot_logit2(nboot, xmatrix, yvec, u_comp, theta0, data_df, mydebug)    

    theta_boot
end


"""
 1. copy data to all cores
 2. run is_significant() with pmap
 3. power = share of results
"""
# function power_logit2(idx, 
#             data_df, 
#             formula, 
#             choice_stub,
#             theta0;
#             nboot=500,
#             cluster_var=nothing,
#             ci_level=[2.5, 97.5],
#             mydebug=false)

#     println("power iteration ", idx)
#     theta_boot = boot_logit2(
#         data_df, 
#         formula, 
#         string(choice_stub) * string(idx),
#         theta0;
#         nboot=nboot,
#         cluster_var=cluster_var,
#         mydebug=mydebug)

#     # significant means 0.0 is not within the 95% CI
#     endpoints = percentile(theta_boot[:, 2] ./ theta_boot[:, 1], ci_level)      
#     return 0.0 ∉ ClosedInterval{Float64}(endpoints...)
# end
function _power_logit2(idx, 
    nboot::Int64, 
    xmatrix::Matrix{Float64}, 
    choice_stub::String, 
    u_comp::Vector{Float64},
    theta0::Vector{Float64},
    data_df::DataFrame,
    mydebug::Bool=false)

    # select outcome number "idx"
    yvec = data_df[:, string(choice_stub) * string(idx)]

    println("power iteration ", idx)
    theta_boot = _boot_logit2(
                    nboot, 
                    xmatrix, 
                    yvec, 
                    u_comp,
                    theta0,
                    data_df,
                    mydebug)

    # # significant means 0.0 is not within the 95% CI
    # endpoints = percentile(theta_boot[:, 2] ./ theta_boot[:, 1], ci_level)      
    # return 0.0 ∉ ClosedInterval{Float64}(endpoints...)
    return theta_boot
end


function power_logit2(    
    data_df, 
    formula, 
    choice_stub::String,
    theta0;
    nruns=100,
    nboot=500,
    cluster_var=nothing,
    mydebug=false)

    # pre-compute a lot of weights columns
    bbw!(data_df, nboot; cluster_var=cluster_var, mydebug=mydebug)

    # prep only once
    _, xmatrix, _, u_comp = _prep_logit2(data_df, formula, choice_stub * "1", nothing)

    return pmap( idx -> _power_logit2(idx, nboot, xmatrix, choice_stub, u_comp, theta0, data_df, mydebug), 1:nruns)
end

function get_power(all_results; ci_level=[2.5, 97.5], idx1, idx2)
    # # significant means 0.0 is not within the 95% CI

    is_significant = zeros(length(all_results))
    for i=1:length(all_results)
        myresult = all_results[i]

        endpoints = percentile(myresult[:, idx1] ./ myresult[:, idx2], ci_level)      
        is_significant[i] = 0.0 ∉ ClosedInterval{Float64}(endpoints...)
    end

    return mean(is_significant)
end

function get_power2(all_results; ci_level=[2.5, 97.5], idx1)
    # # significant means 0.0 is not within the 95% CI

    is_significant = zeros(length(all_results))
    for i=1:length(all_results)
        myresult = all_results[i]

        endpoints = percentile(myresult[:, idx1], ci_level)      
        is_significant[i] = 0.0 ∉ ClosedInterval{Float64}(endpoints...)
    end

    return mean(is_significant)
end

function get_power_extend(all_results; h0 = 0.0, ci_level=[2.5, 97.5], idx1, idx2)
    # # significant means 0.0 is not within the 95% CI

    is_significant = zeros(length(all_results))
    for i=1:length(all_results)
        myresult = all_results[i]

        endpoints = percentile(myresult[:, idx1] ./ myresult[:, idx2], ci_level)      
        is_significant[i] = h0 ∉ ClosedInterval{Float64}(endpoints...)
    end

    return mean(is_significant)
end

function get_power_extend2(all_results; h0 = 0.0, ci_level=[2.5, 97.5], idx1)
    # # significant means 0.0 is not within the 95% CI

    is_significant = zeros(length(all_results))
    for i=1:length(all_results)
        myresult = all_results[i]

        endpoints = percentile(myresult[:, idx1], ci_level)      
        is_significant[i] = h0 ∉ ClosedInterval{Float64}(endpoints...)
    end

    return mean(is_significant)
end

function getcoefficient(all_boot_theta, idx; nruns = 100, nboot = 500)
    coef = []
    for i = 1:nruns
        push!(coef, all_boot_theta[i][:,idx])
    end
    return mean(vcat(coef...))
end