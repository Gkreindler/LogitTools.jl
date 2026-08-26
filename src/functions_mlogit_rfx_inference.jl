###############################################################################
# Bayesian bootstrap for mlogit_rfx.
#
# Group-level Dirichlet weights over col_group (the integration unit),
# parallelised over replicates with Distributed. Prep runs once on the master;
# workers never see the DataFrame.
#
# Reporting is deliberately NOT duplicated here: an mlogit_rfx fit is an ordinary
# MLEFit whose `extra` carries the same K / rfx / rfx_cols fields as a logit2_rfx
# fit, so `boot_report`, `regtable_rfx`, `rfx_level_moments` and `boot_vcov!` all
# work on it unchanged. The one thing that is genuinely new -- the per-term cell
# structure that decides whether a term is identified at all -- gets its own
# report below.
###############################################################################

"""
    boot_mlogit_rfx(data_df, formula, col_id, col_selected, theta0; kwargs...) -> MLEvcov

Bayesian bootstrap for [`mlogit_rfx`](@ref), resampling at the `col_group` level.

Resampling has to happen at the integration unit and nowhere else: a replicate
reweights whole groups, because the group log-likelihood is the independent unit.
Resampling choice sets would break the very factorisation the random effects rely
on.

# Keywords
- `col_group`, `rfx`, `rfx_corr`, `ndraws`, `seed`, `weights`, `optim_options`: as in
  [`mlogit_rfx`](@ref). Simulation draws are held fixed across replicates.
- `nboot = 500`: number of bootstrap replicates.
- `boot_seed = 12345`: seed for the Dirichlet weights. With the same `boot_seed`,
  `parallel = true` and `parallel = false` give identical results.
- `cluster_var = nothing`: if given, must equal `col_group`. It exists only so a
  call site can state the clustering explicitly and have it checked.
- `parallel = true`: distribute replicates over `workers()`. Requires
  `addprocs(n)` and `@everywhere using LogitTools`.
- `theta_start = nothing`: warm start; defaults to `theta0`. Passing the main
  fit's `theta_hat` is usually much faster. May also be an `nstarts x npar`
  **matrix**, in which case every replicate is itself multi-started and its best
  optimum kept -- `nstarts` times the runtime, and the right call when the
  specification has local optima, because a single warm start hands every
  replicate the point estimate's basin instead of the one its own resample
  prefers.
- `rethrow_errors = false`: by default a replicate that throws is captured as
  `errored` and the run continues. When replicates are failing, re-run with
  `parallel = false, nboot = 2, rethrow_errors = true` for a readable error.
- `mydebug = false`: print per-replicate progress (serial path only).

# Notes
Prep runs once on the master and rides along in the `pmap` closure, so workers
never receive the DataFrame -- only the numeric arrays in `MlogitRfxPrep`. A
`CachingPool` serialises that closure once per worker rather than once per task.

Replicates are dropped from `V` only on numerical failure. A replicate whose
`sigma` lands near zero is a legitimate draw from the sampling distribution;
dropping it would bias the standard error downward. `theta_boot_table` retains
**all** rows (`NaN` for errored replicates); `V` is computed on survivors.
"""
function boot_mlogit_rfx(
        data_df,
        formula,
        col_id,
        col_selected,
        theta0;
        col_group = nothing,
        rfx = [],
        rfx_corr = [],
        ndraws::Int = 1000,
        seed::Int = 20260808,
        weights::Union{Nothing, Symbol, String} = nothing,
        nboot::Int = 500,
        boot_seed::Int = 12345,
        cluster_var = nothing,
        parallel::Bool = true,
        theta_start = nothing,
        optim_options::Optim.Options = Optim.Options(),
        rethrow_errors::Bool = false,
        mydebug::Bool = false)

    cid = Symbol(col_id)
    cg  = isnothing(col_group) ? cid : Symbol(col_group)

    if !isnothing(cluster_var) && Symbol(cluster_var) != cg
        error("cluster_var = :$(Symbol(cluster_var)) must equal col_group = :$cg. " *
              "Clustering has to be at the integration unit -- the level the " *
              "likelihood factorises over -- or the group weighting is undefined." *
              (Symbol(cluster_var) == cid ?
               " You passed the choice-set column; pass col_group = :$(Symbol(cluster_var)) " *
               "if that is the individual." : ""))
    end

    nboot >= 2 || error("nboot must be at least 2; got $nboot")

    # check the cluster before prep, so a misconfigured parallel run fails
    # immediately rather than after prep has already been paid for
    parallel && _check_boot_workers()

    # prep once, on the master
    P, gw_user = _prep_mlogit_rfx(data_df, formula, cid, col_selected, cg, rfx,
                                  ndraws, seed, weights, rfx_corr)

    theta0s  = _mlogit_rfx_theta0_matrix(theta0, P)
    th_start = isnothing(theta_start) ? theta0s : _mlogit_rfx_theta0_matrix(theta_start, P)
    nstart   = size(th_start, 1)

    # group-level Dirichlet weights, and the user's weights on top of them
    W     = _rfx_boot_weights(P.N, nboot, boot_seed)
    Wfull = isnothing(gw_user) ? W : (W .* gw_user)

    # With several starts per replicate the inner fit must run SERIALLY: the outer
    # pmap already owns every worker, and nesting would deadlock on the same pool.
    task = if nstart == 1
        ths = vec(th_start)
        b -> _mlogit_rfx(P, ths, Vector{Float64}(view(Wfull, :, b)), optim_options;
                         rethrow_errors = rethrow_errors)
    else
        b -> _mlogit_rfx_multi(P, th_start, Vector{Float64}(view(Wfull, :, b)),
                               optim_options; parallel = false,
                               rethrow_errors = rethrow_errors,
                               warn_multi = false)
    end

    fits = if parallel
        pmap(task, CachingPool(workers()), 1:nboot)
    else
        map(1:nboot) do b
            mydebug && println("bootstrapping mlogit_rfx, replicate=", b)
            task(b)
        end
    end

    return _assemble_rfx_boot(fits, P.K + P.M + P.B, nboot)
end


# ----------------------------------------------------------------------------
# Cell-structure report
# ----------------------------------------------------------------------------

"""
    rfx_cell_report(fit) -> DataFrame

One row per random-coefficient term, describing the **cell structure** the term's
draws live on. This is the diagnostic that has no counterpart in `logit2_rfx`,
where every draw sits on the panel unit by construction.

Columns:

| column | meaning |
|---|---|
| `term`, `level`, `dist` | the term, its level column, its distribution |
| `n_cells` | distinct `(group, level)` cells actually observed |
| `n_cells_informative` | of those, how many have a nonzero loading on at least one row |
| `cells_per_group_*` | how many cells a group has **for this term** |
| `cells_per_group_informative_*` | the same, counting only cells with a nonzero loading |
| `rows_per_cell_*` | observations sharing one draw. 1 means nothing to identify from |
| `sets_per_cell_*` | choice sets a cell is seen in. 1 means the effect is only identified off within-set substitution |
| `cancel_share` | share of choice sets in which the term drops out of the softmax (constant loading and one cell across the set's options) |
| `within_cell_var_share` | share of the loading's variance that is within-cell |

Read `cells_per_group_informative_max` **summed over the terms** together with
`extra.ess_p10`: that sum is the dimension of the simulated integral, so a large
value with a small ESS says the draw set is too thin for the model, not that the
model is wrong. (The `ess_p10` warning `mlogit_rfx` raises already reports the
summed figure.)

Use the *informative* column, not the plain one. A cell whose loading is zero on
every one of its rows is allocated a draw that never multiplies anything, so it
never reaches the per-draw log-likelihood and cannot move the posterior weights or
the ESS -- it is inert, not costly. The gap between the two columns can be large:
a random coefficient on an interaction with a treatment arm is switched off for
every control individual's entire panel, so most of its cells are inert and
counting them would badly overstate how hard the integral is.
"""
function rfx_cell_report(fit::MLEFit)
    e = fit.extra
    (!isnothing(e) && hasproperty(e, :cell_stats)) || error(
        "this fit carries no cell statistics: rfx_cell_report is for mlogit_rfx fits " *
        "(a logit2_rfx fit has one cell per group by construction).")
    isempty(e.cell_stats) && return DataFrame()
    return DataFrame(collect(e.cell_stats))
end
