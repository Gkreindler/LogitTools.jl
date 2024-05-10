

"""
minus log likelhood
the data is in long format, already sorted by uniqueid and option
"""
function mlogit_minus_ll(
        theta::Vector{Float64}, 
        yvec::Vector{Float64}, 
        xmatrix::Matrix{Float64}, 
        df::DataFrame,
        col_id::Union{Symbol, String},
        u_comp::Vector{Float64},
        weights::Union{Nothing, Vector{Float64}})
    
    # Step 1) compute linear utility with inplace multiplication
    mul!(u_comp, xmatrix, theta)

    # Step 2) compute the log(sum(exp())) term in the log likelhood
    df.u_comp .= u_comp
    transform!(groupby(df, col_id), :u_comp => logsumexp => :log_sum_exp)

    # Step 3) compute log likelihood for each observation (only non-zero for the chosen option)
    @. u_comp = @. yvec .* (u_comp .- df.log_sum_exp)

    # Step 4) apply weights (they are at the id level so ok to do here) # ! check
    if !isnothing(weights)
        @. u_comp .*= weights
    end
    return -sum(u_comp)
end

"""
Analytic expression for the gradient ∇f(θ) of the minus log likelihood function f(θ) for the multinomial logit model
"""
function mlogit_minus_grad(
        theta::Vector{Float64}, 
        yvec::Vector{Float64}, 
        xmatrix::Matrix{Float64}, 
        df::DataFrame,
        col_id::Union{Symbol, String},
        u_comp::Vector{Float64},
        weights::Union{Nothing, Vector{Float64}})
    
    # Step 1) compute linear utility with inplace multiplication
    mul!(u_comp, xmatrix, theta)
    
    # Step 2) compute the probabilities
    df.pi .= u_comp
    transform!(groupby(df, col_id), :pi => softmax => :pi) # = Π

    # Step 3) compute the commmon term in the gradient X_i' * Π
    if isnothing(weights)
        return - sum((yvec .- df.pi) .* xmatrix , dims=1) 
    else
        return - sum((yvec .- df.pi) .* weights .* xmatrix , dims=1)
    end
end

function _prep_mlogit(
    data_df, 
    formula, 
    col_id,
    col_selected,
    weights::Union{Nothing, Symbol, String}=nothing)

    # weights
    if !isnothing(weights)
        temp_df = combine(groupby(data_df, col_id), weights => sum => :weights_sum)
        weights_sum = sum(temp_df.weights_sum)
        
        # weights should sum up to 1 at individual level
        wvec = data_df[!, weights] ./ weights_sum
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
        yvec = data_df[:, col_selected]
        eltype(yvec) == Float64 || (yvec = convert.(Float64, yvec))
        all((yvec .== 0.0) .| (yvec .== 1.0)) || error("choice column should have 0's and 1's only")

        u_comp = copy(yvec)

    # group size (handy later)
        transform!(groupby(data_df, col_id), col_selected => (x -> length(x)) => :__group_count)

    return wvec, xmatrix, yvec, u_comp
end

"""
Estimate a binary logit model with MLE
    formula = vector of 
"""
function mlogit(
    data_df, 
    formula, # TODO: replace this with actual formula from StatsAPI
    col_id,
    col_selected,
    theta0; 
    myweights::Union{Nothing, Symbol, String}=nothing)

    wvec, xmatrix, yvec, u_comp = _prep_mlogit(data_df, formula, col_id, col_selected, myweights)

    # estimate
    myfit = _mlogit(
            xmatrix=xmatrix, 
            df=data_df,
            # formula=formula,
            col_id=col_id,
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
function _mlogit(;
    xmatrix::Matrix{Float64}, 
    df::DataFrame,
    col_id::Union{Symbol, String},
    yvec::Vector{Float64}, 
    u_comp::Vector{Float64},
    theta0::Vector{Float64},
    wvec::Union{Nothing, Vector{Float64}}=nothing)

    # define objective function (minus log likelihood)
        f = theta ->   mlogit_minus_ll(theta, yvec, xmatrix, df, col_id, u_comp, wvec)
        g = theta -> mlogit_minus_grad(theta, yvec, xmatrix, df, col_id, u_comp, wvec)

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


