###############################################################################
# Multinomial (conditional) logit by MLE.
#
# The data is in long format: one row per (choice set, option), with `col_id`
# identifying the choice set and `col_selected` the 0/1 chosen indicator.
#
# The objective and gradient used to reach into the caller's DataFrame on every
# function evaluation -- `df.u_comp .= ...` followed by
# `transform!(groupby(df, col_id), ...)` -- which both mutated the caller's data
# and paid for a fresh grouping a few thousand times per fit. The grouping is now
# computed once, in prep, as a permutation plus contiguous ranges.
#
# That refactor is arithmetically inert *by construction*: every group's rows are
# visited in their original order, `logsumexp` / `softmax` see exactly the vectors
# they saw before, and the final reductions run over the same arrays in the same
# order. Results are bitwise identical, which `test_mlogit.jl` pins down against a
# verbatim copy of the old implementation.
###############################################################################

"""
Grouping of the long-format rows into choice sets, computed once per fit.

`perm` lists the rows sorted by choice set, ties broken by original position, and
`ranges` gives one contiguous range of `perm` per choice set. So
`view(perm, ranges[i])` is choice set `i`'s rows **in their original DataFrame
order** -- which is what `groupby` used to hand `logsumexp`, and is why swapping
one for the other changes no floating-point result.
"""
struct MlogitGroups
    perm::Vector{Int}
    ranges::Vector{UnitRange{Int}}
    Tmax::Int
end

function MlogitGroups(idcol)
    n = length(idcol)
    n > 0 || error("data_df has no rows")

    # The original position as the final key makes this a total order, so the
    # permutation does not depend on the sort algorithm being stable.
    perm = sortperm(1:n, by = t -> (idcol[t], t))

    ranges = UnitRange{Int}[]
    start = 1
    for t in 2:n
        if !isequal(idcol[perm[t]], idcol[perm[t-1]])
            push!(ranges, start:(t-1))
            start = t
        end
    end
    push!(ranges, start:n)

    return MlogitGroups(perm, ranges, maximum(length, ranges))
end

"""
Scratch space for one `mlogit` objective/gradient evaluation.

`aux` holds the per-row log-sum-exp in the objective and the per-row choice
probability in the gradient; the two are never live at the same time. `gbuf` is a
contiguous staging area for one choice set, so `softmax!` can write into it
instead of allocating a fresh vector per group per evaluation.

One of these per bootstrap replicate rather than one shared across replicates:
they are pure scratch (overwritten by `mul!` at the top of every call), so this
changes no result, it just stops tasks on the same worker sharing a buffer.
"""
struct MlogitScratch
    u::Vector{Float64}
    aux::Vector{Float64}
    gbuf::Vector{Float64}
end

MlogitScratch(nobs::Int, Tmax::Int) =
    MlogitScratch(Vector{Float64}(undef, nobs),
                  Vector{Float64}(undef, nobs),
                  Vector{Float64}(undef, Tmax))

"""
utility for each option (long format)
"""
compute_utility_mlogit(theta::Vector{Float64}, xmatrix::Matrix{Float64}) = xmatrix * theta

"""
minus log likelihood

    -sum_c  ( v_{c,sel} - logsumexp_{j in c} v_cj )

Weights multiply per-row contributions, and only the chosen row of a set carries
one, so the weight applied to choice set `c` is the weight sitting on its chosen
row. It must therefore be constant within `col_id` for the objective to mean what
it looks like it means.
"""
function mlogit_minus_ll(
        theta::Vector{Float64},
        yvec::Vector{Float64},
        xmatrix::Matrix{Float64},
        G::MlogitGroups,
        sc::MlogitScratch,
        weights::Union{Nothing, Vector{Float64}})

    u   = sc.u
    lse = sc.aux

    # Step 1) linear utility, in place
    mul!(u, xmatrix, theta)

    # Step 2) log(sum(exp())) per choice set, broadcast back to its rows
    @inbounds for rg in G.ranges
        idx = view(G.perm, rg)
        s   = logsumexp(view(u, idx))
        for t in idx
            lse[t] = s
        end
    end

    # Step 3) log likelihood per observation (nonzero only for the chosen option)
    @. u = yvec * (u - lse)

    # Step 4) weights (at the id level, so it is fine to apply them here)
    if !isnothing(weights)
        @. u *= weights
    end

    return -sum(u)
end

"""
Analytic gradient of the minus log likelihood for the multinomial logit model.

Returns a `1 x K` matrix, as it always has: `optimize(f, g, ...; inplace = false)`
is fed this directly and the shape is part of the calling convention here.
"""
function mlogit_minus_grad(
        theta::Vector{Float64},
        yvec::Vector{Float64},
        xmatrix::Matrix{Float64},
        G::MlogitGroups,
        sc::MlogitScratch,
        weights::Union{Nothing, Vector{Float64}})

    u  = sc.u
    pr = sc.aux

    # Step 1) linear utility, in place
    mul!(u, xmatrix, theta)

    # Step 2) choice probabilities within each choice set
    @inbounds for rg in G.ranges
        idx = view(G.perm, rg)
        dst = view(sc.gbuf, 1:length(idx))
        softmax!(dst, view(u, idx))
        for (k, t) in enumerate(idx)
            pr[t] = dst[k]
        end
    end

    # Step 3) the common term in the gradient, X_i' * (y - Pi)
    if isnothing(weights)
        return - sum((yvec .- pr) .* xmatrix, dims=1)
    else
        return - sum((yvec .- pr) .* weights .* xmatrix, dims=1)
    end
end

"""
    _prep_mlogit(data_df, formula, col_id, col_selected, weights) -> (wvec, xmatrix, yvec, G)

Build the numeric arrays and the choice-set grouping. Does **not** mutate
`data_df`: the in-place `convert.(Float64, ...)` writeback and the unused
`:__group_count` column that earlier versions appended are both gone. Nothing in
the package or in the replication scripts read either of them, and the estimates
are unaffected -- `Matrix{Float64}` produces the same numbers the converted
columns did.
"""
function _prep_mlogit(
    data_df,
    formula,
    col_id,
    col_selected,
    weights::Union{Nothing, Symbol, String}=nothing)

    # weights
    if !isnothing(weights)
        # Summing the group sums rather than the column directly: kept exactly as
        # it was, because the two differ in the last bits and existing tables were
        # produced with this one.
        temp_df = combine(groupby(data_df, col_id), weights => sum => :weights_sum)
        weights_sum = sum(temp_df.weights_sum)

        # weights should sum up to 1 at individual level
        wvec = Vector{Float64}(data_df[!, weights] ./ weights_sum)
    else
        wvec = nothing
    end

    # regressors
    xmatrix = Matrix{Float64}(data_df[:, formula])

    # outcome
    yvec = Vector{Float64}(data_df[:, col_selected])
    all((yvec .== 0.0) .| (yvec .== 1.0)) || error("choice column should have 0's and 1's only")

    # choice-set grouping (computed once; used by every function evaluation)
    G = MlogitGroups(data_df[:, col_id])

    return wvec, xmatrix, yvec, G
end

"""
    mlogit(data_df, formula, col_id, col_selected, theta0; kwargs...) -> MLEFit

Estimate a multinomial (conditional) logit by MLE, from long-format data.

# Arguments
- `data_df`: one row per (choice set, option). **Not mutated.**
- `formula`: `Vector{Symbol}` of regressors.
- `col_id`: choice-set identifier.
- `col_selected`: 0/1 column, the chosen option of each set.
- `theta0`: starting values, length `K`.

# Keywords
- `myweights = nothing`: column name; normalised to sum to one, as before.
- `optim_options = Optim.Options()`: passed to `optimize`. **Raise `iterations`
  for specifications with many regressors.** The default LBFGS cap is 1000, and a
  full set of alternative dummies can run past it and return `converged = false`
  -- which is easy to miss, because the fit still comes back and still tabulates.
  Check `fit.converged`.

See [`mlogit_rfx`](@ref) for the random-coefficient version, which takes the same
positional arguments.
"""
function mlogit(
    data_df,
    formula, # TODO: replace this with actual formula from StatsAPI
    col_id,
    col_selected,
    theta0;
    myweights::Union{Nothing, Symbol, String}=nothing,
    optim_options::Optim.Options=Optim.Options())

    wvec, xmatrix, yvec, G = _prep_mlogit(data_df, formula, col_id, col_selected, myweights)

    # estimate
    myfit = _mlogit(
            xmatrix=xmatrix,
            G=G,
            yvec=yvec,
            theta0=theta0,
            wvec=wvec,
            optim_options=optim_options)

    myfit.theta_names = string.(formula)

    return myfit
end

"""
    the inner function (to not repeat prep when bootstrapping)
"""
function _mlogit(;
    xmatrix::Matrix{Float64},
    G::MlogitGroups,
    yvec::Vector{Float64},
    theta0::Vector{Float64},
    wvec::Union{Nothing, Vector{Float64}}=nothing,
    sc::Union{Nothing, MlogitScratch}=nothing,
    optim_options::Optim.Options=Optim.Options())

    scratch = isnothing(sc) ? MlogitScratch(length(yvec), G.Tmax) : sc

    # define objective function (minus log likelihood)
        f = theta ->   mlogit_minus_ll(theta, yvec, xmatrix, G, scratch, wvec)
        g = theta -> mlogit_minus_grad(theta, yvec, xmatrix, G, scratch, wvec)

    # estimate
    time_it_took = @elapsed opt_results = optimize(f, g, theta0, LBFGS(), optim_options, inplace=false)

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
