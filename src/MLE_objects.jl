
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

Base.@kwdef mutable struct MLEvcov
    method::Symbol
    V  = nothing
    theta_boot_table 
    boot_fits = nothing
    # W = nothing
    # J = nothing
    # Σ = nothing
    # boot_fits_dict = nothing
end