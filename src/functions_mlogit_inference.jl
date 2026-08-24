###############################################################################
# Bayesian bootstrap for mlogit.
#
# Mirrors `boot_logit2`: `bbw!` still runs first and still appends its `bw*`
# columns, so the RNG stream and the DataFrame side effects are unchanged; the
# weights are then collected into a plain matrix that both the serial and the
# parallel path consume, which is what keeps the two bit-for-bit identical and
# keeps the DataFrame off the workers.
###############################################################################

"""
    _boot_mlogit(nboot, xmatrix, yvec, G, theta0, W; kwargs...) -> MLEvcov

Core bootstrap loop. Takes the weights as a plain `Nobs x nboot` matrix rather
than reading `bw*` columns out of a DataFrame, so the parallel path can ship
numeric arrays to the workers instead of the whole DataFrame.
"""
function _boot_mlogit(
            nboot::Int64,
            xmatrix::Matrix{Float64},
            yvec::Vector{Float64},
            G::MlogitGroups,
            theta0::Vector{Float64},
            W::Matrix{Float64};
            parallel::Bool=false,
            optim_options::Optim.Options=Optim.Options(),
            mydebug::Bool=false)

    # A fresh scratch buffer per replicate. It is pure scratch (overwritten by
    # mul! at the top of every objective/gradient call), so this changes no
    # result; it just avoids sharing a mutable buffer across tasks running on the
    # same worker.
    task = b -> _mlogit(
        xmatrix = xmatrix,
        G       = G,
        yvec    = yvec,
        theta0  = theta0,
        wvec    = W[:, b],
        sc      = MlogitScratch(length(yvec), G.Tmax),
        optim_options = optim_options)

    # The closure captures xmatrix / yvec / G / W. CachingPool serialises it once
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
    boot_mlogit(data_df, formula, col_id, col_selected, theta0; kwargs...) -> MLEvcov

Bayesian bootstrap for [`mlogit`](@ref).

# Keywords
- `nboot = 500`: number of bootstrap replicates.
- `cluster_var = nothing`: resample at this cluster level instead of by row.
  With long-format data you almost always want this: the individual, not the row.
  Unclustered row-level weights are not constant within a choice set, and since
  only the chosen row of a set carries weight, each set then gets the weight that
  happened to land on its chosen option.
- `parallel = false`: distribute replicates over `workers()`. Requires
  `addprocs(n)` and `@everywhere using LogitTools`. Defaults to `false` so
  existing scripts are unaffected.
- `optim_options = Optim.Options()`: passed to every replicate's `optimize` call.
  Raise `iterations` for specifications with many regressors, where the default
  1000-iteration cap can bind silently.
- `mydebug = false`: print per-replicate progress (serial path only).

`parallel = true` and `parallel = false` give **identical** results: the Dirichlet
weights are drawn once on the master by `bbw!` and both paths consume the same
weight matrix. Workers receive only numeric arrays, never the DataFrame.

As before, this appends `bw1 ... bwN` weight columns to `data_df`.
"""
function boot_mlogit(
    data_df,
    formula,
    col_id,
    col_selected,
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
    _, xmatrix, yvec, G = _prep_mlogit(data_df, formula, col_id, col_selected, nothing)

    # hand the weights over as a numeric matrix so workers never see data_df
    W = _boot_weights_matrix(data_df, nboot)

    return _boot_mlogit(nboot, xmatrix, yvec, G, theta0, W;
                        parallel=parallel, optim_options=optim_options,
                        mydebug=mydebug)
end
