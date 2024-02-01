
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
    formula, # TODO: replace this with actual formula from StatsAPI
    choice,
    theta0; 
    weights::Union{Nothing, Symbol, String}=nothing)

    wvec, xmatrix, yvec, u_comp = _prep_logit2(data_df, formula, choice, weights)

    # estimate
    myfit = _logit2(
            xmatrix=xmatrix, 
            yvec=yvec, 
            u_comp=u_comp,
            theta0=theta0,
            wvec=wvec)
    
    myfit.theta_names = string.(formula)

    return myfit
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


