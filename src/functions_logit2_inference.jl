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

"""
    _check_boot_workers()

Verify that there are worker processes and that every one of them can see
`LogitTools`. Names the offending workers.

Shared by `boot_logit2` and `boot_logit2_rfx`.
"""
function _check_boot_workers()
    nprocs() > 1 || error(
        "parallel = true but nprocs() == 1: there are no worker processes. " *
        "Run addprocs(n) followed by @everywhere using LogitTools, or pass parallel = false.")

    bad = Int[]
    for w in workers()
        ok = try
            remotecall_fetch(() -> isdefined(Main, :LogitTools), w)
        catch
            false
        end
        ok || push!(bad, w)
    end

    isempty(bad) || error(
        "LogitTools is not loaded on worker(s) $bad. Run: @everywhere using LogitTools")

    return nothing
end

"""
    _boot_logit2(nboot, xmatrix, yvec, theta0, W; parallel = false, mydebug = false)

Core bootstrap loop. Takes the weights as a plain `Nobs × nboot` matrix rather
than reading `bw*` columns out of a DataFrame, so the parallel path can ship
numeric arrays to the workers instead of the whole DataFrame.

Both paths consume the same `W`, so `parallel = true` and `parallel = false`
produce identical results.
"""
function _boot_logit2(
            nboot::Int64,
            xmatrix::Matrix{Float64},
            yvec::Vector{Float64},
            theta0::Vector{Float64},
            W::Matrix{Float64};
            parallel::Bool=false,
            optim_options::Optim.Options=Optim.Options(),
            mydebug::Bool=false)

    # A fresh scratch buffer per replicate. u_comp is pure scratch (it is
    # overwritten by mul! at the top of every minus_ll / minus_grad call), so
    # this changes no result; it just avoids sharing a mutable buffer across
    # tasks running on the same worker.
    task = b -> _logit2(
        xmatrix = xmatrix,
        yvec    = yvec,
        u_comp  = similar(yvec),
        theta0  = theta0,
        wvec    = W[:, b],
        optim_options = optim_options)

    # The closure captures xmatrix / yvec / W. CachingPool serialises it once
    # per worker, not once per task; each task then transmits only an Int.
    all_boot_fits = if parallel
        pmap(task, CachingPool(workers()), 1:nboot)
    else
        map(1:nboot) do b
            mydebug && println("bootstrapping, column=", b)
            task(b)
        end
    end

    # store results
    theta_boot_table = zeros(nboot, length(theta0))
    for (b, boot_fit) in enumerate(all_boot_fits)
        theta_boot_table[b, :] .= boot_fit.theta_hat
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
Backwards-compatible method: pulls the `bw*` columns out of `data_df` and
delegates. Kept so any existing caller of the old internal signature keeps
working.
"""
function _boot_logit2(
            nboot::Int64,
            xmatrix::Matrix{Float64},
            yvec::Vector{Float64},
            u_comp::Vector{Float64},
            theta0::Vector{Float64},
            data_df::DataFrame,
            mydebug::Bool=false)

    W = _boot_weights_matrix(data_df, nboot)
    return _boot_logit2(nboot, xmatrix, yvec, theta0, W;
                        parallel=false, mydebug=mydebug)
end

"""Collect the `bw1 … bwN` columns that `bbw!` wrote into a plain matrix."""
function _boot_weights_matrix(data_df, nboot::Int)
    W = Matrix{Float64}(undef, nrow(data_df), nboot)
    for b in 1:nboot
        W[:, b] .= data_df[!, "bw" * string(b)]
    end
    return W
end

"""
    boot_logit2(data_df, formula, choice, theta0; kwargs...) -> MLEvcov

Bayesian bootstrap for [`logit2`](@ref).

# Keywords
- `nboot = 500`: number of bootstrap replicates.
- `cluster_var = nothing`: resample at this cluster level instead of by row.
- `parallel = false`: distribute replicates over `workers()`. Requires
  `addprocs(n)` and `@everywhere using LogitTools`. Defaults to `false` so that
  existing scripts are unaffected.
- `optim_options = Optim.Options()`: passed to every replicate's `optimize`
  call. Raise `iterations` for specifications with many regressors (e.g. a full
  set of fixed-effect dummies), where the default 1000-iteration cap can bind.
- `mydebug = false`: print per-replicate progress (serial path only).

`parallel = true` and `parallel = false` give **identical** results: the
Dirichlet weights are drawn once on the master by `bbw!` and both paths consume
the same weight matrix. Workers receive only numeric arrays, never the DataFrame.

As before, this appends `bw1 … bwN` weight columns to `data_df`.

# Example
```julia
using Distributed
addprocs(4)
@everywhere using LogitTools

fit.vcov = boot_logit2(df, myxs, :pick1, theta0;
                       cluster_var = :clusterid, nboot = 500, parallel = true)
```
"""
function boot_logit2(
    data_df,
    formula,
    choice,
    theta0;
    nboot=500,
    cluster_var=nothing,
    parallel::Bool=false,
    optim_options::Optim.Options=Optim.Options(),
    mydebug=false)

    # fail fast on a misconfigured cluster, before any work is done
    parallel && _check_boot_workers()

    # pre-compute a lot of weights columns
    bbw!(data_df, nboot; cluster_var=cluster_var, mydebug=mydebug)

    # prep only once
    _, xmatrix, yvec, _ = _prep_logit2(data_df, formula, choice, nothing)

    # hand the weights over as a numeric matrix so workers never see data_df
    W = _boot_weights_matrix(data_df, nboot)

    return _boot_logit2(nboot, xmatrix, yvec, theta0, W;
                        parallel=parallel, optim_options=optim_options,
                        mydebug=mydebug)
end




# function getcoefficient(all_boot_theta, idx; nruns = 100, nboot = 500)
#     coef = []
#     for i = 1:nruns
#         push!(coef, all_boot_theta[i][:,idx])
#     end
#     return mean(vcat(coef...))
# end