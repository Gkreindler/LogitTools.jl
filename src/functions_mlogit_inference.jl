


function _boot_mlogit(
            nboot::Int64, 
            xmatrix::Matrix{Float64}, 
            data_df::DataFrame,
            col_id::Union{Symbol, String},
            yvec::Vector{Float64}, 
            u_comp::Vector{Float64},
            theta0::Vector{Float64},
            mydebug::Bool=false)

    # store results
    theta_boot_table = zeros(nboot, length(theta0))
    all_boot_fits = []

    # estimate model for each bootstrap column of weights
    for i=1:nboot
        mydebug && println("bootstrapping, column=", i)

        boot_fit = _mlogit(
            xmatrix=xmatrix, 
            df=data_df,
            col_id=col_id,
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
function boot_mlogit(
    data_df, 
    formula, 
    col_id,
    col_selected,
    theta0;
    nboot=500,
    cluster_var=nothing,
    mydebug=false)

    # pre-compute a lot of weights columns
    bbw!(data_df, nboot; cluster_var=cluster_var, mydebug=mydebug)

    # prep only once
    _, xmatrix, yvec, u_comp = _prep_mlogit(data_df, formula, col_id, col_selected, nothing)

    return _boot_mlogit(nboot, xmatrix, data_df, col_id, yvec, u_comp, theta0, mydebug)    
end

