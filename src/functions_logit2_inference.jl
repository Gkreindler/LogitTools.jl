# Base.@kwdef mutable struct MLEvcov
#     method::Symbol
#     V  = nothing
#     theta_boot_table 
#     boot_fits = nothing
#     # W = nothing
#     # J = nothing
#     # Σ = nothing
#     # boot_fits_dict = nothing
# end

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
    theta_boot_table = zeros(nboot, length(theta0))
    all_boot_fits = []

    # estimate model for each bootstrap column of weights
    for i=1:nboot
        mydebug && println("bootstrapping, column=", i)

        boot_fit = _logit2(
            xmatrix=xmatrix, 
            yvec=yvec, 
            u_comp=u_comp,
            theta0=theta0,
            wvec=data_df[:, "bw" * string(i)])

            theta_boot_table[i, :] .=  boot_fit.theta_hat

        push!(all_boot_fits, boot_fit)
    end

    my_boot_vcov = MLEvcov(
            method = :bayesian_bootstrap,
            theta_boot_table = theta_boot_table,
            V = cov(theta_boot_table),
            boot_fits = all_boot_fits
            )

    return my_boot_vcov
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
end




# function getcoefficient(all_boot_theta, idx; nruns = 100, nboot = 500)
#     coef = []
#     for i = 1:nruns
#         push!(coef, all_boot_theta[i][:,idx])
#     end
#     return mean(vcat(coef...))
# end