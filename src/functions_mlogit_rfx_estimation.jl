###############################################################################
# Multinomial (conditional) logit with random coefficients, estimated by
# maximum simulated likelihood with an analytic gradient.
#
# What this adds over `logit2_rfx` is a THIRD nesting level. `logit2_rfx` knows
# one identifier, `col_id`, which is simultaneously the panel unit and the level
# at which every random draw lives. Here the three are separate:
#
#   col_group   integration unit (the individual). The likelihood factorises
#               here: one simulated integral per group.
#   col_id      choice set (the softmax group), nested inside col_group.
#   level_m     one per random-coefficient term: the identifier at which THAT
#               term's draw varies. Defaults to col_group.
#
# A term whose level column varies *within a choice set* is an option-level
# random effect: person i's idiosyncratic taste for the alternative sitting in
# row (c, j), shared by every other row of i with the same level value. Because
# the draw is indexed by (group, level) the integral still factorises over
# groups, which is the whole reason the level has to be read "within an
# individual" -- see the docstring of `mlogit_rfx`.
#
# Model (group i, choice set c, option j, draw r):
#
#   v_cjr = sum_k xlin_cjk*beta_k + sum_m Z_cjm * A_m(cell_m(cj), r)
#
#   :normal         A_m(g,r) =  sigma_m*eta_m[g,r]     (mu_m enters via X'beta)
#   :lognormal      A_m(g,r) =  exp(mu_m + sigma_m*eta_m[g,r])
#   :neg_lognormal  A_m(g,r) = -exp(mu_m + sigma_m*eta_m[g,r])
#
#   l_cr    = v_{c,sel(c),r} - logsumexp_{j in c} v_cjr
#   logL_i  = logsumexp_r( sum_{c in i} l_cr + logw_r )
#
# Objective (minimised):  Q(t) = - sum_i w_i * logL_i,  t = [mu (K); sigma (M)]
###############################################################################

# ----------------------------------------------------------------------------
# Terms
# ----------------------------------------------------------------------------

"""
One random-coefficient term of an [`mlogit_rfx`](@ref) model.

`var === nothing` is a random **intercept**: its loading is 1 on every row and
there is no matching formula coefficient, so its mean is fixed at zero. That is
the right normalisation, because the *mean* effect of a level value is either a
fixed effect in `formula` or not identified at all -- only the spread around it
is new information.

`col` is the formula position of `var`, or `0` for an intercept. `at_group`
records whether the level is the integration unit itself, which is the
`logit2_rfx` case.
"""
struct MlogitRfxTerm
    var::Union{Nothing, Symbol}
    level::Symbol
    dist::Symbol
    at_group::Bool
    col::Int
    name::String
end

"""
    rfx_term(var = nothing; level = nothing, dist = :normal)

Describe one random-coefficient term for [`mlogit_rfx`](@ref).

- `var`: the regressor whose coefficient is random. `nothing` (the default) means a
  random **intercept** at `level` -- a pure option-level random effect with mean zero.
- `level`: the column whose values index the draws. `nothing` means the model's
  integration unit (`col_group`), which reproduces `logit2_rfx`'s behaviour. Any
  other column gives a random effect that varies *within* an individual; draws are
  always individual-specific, i.e. the level is interacted with `col_group`.
- `dist`: `:normal`, `:lognormal` or `:neg_lognormal`, as in [`logit2_rfx`](@ref).
  The lognormal families need a `var` (their `mu` is an estimated formula
  coefficient), so they cannot be used for an intercept term.

Plain `Symbol` and `Pair` entries in `rfx` still mean what they mean in
`logit2_rfx` -- a random coefficient on that variable, at the group level -- so
`rfx_term` is only needed when you want a level other than the group.

# Example
```julia
rfx = [:any_fam,                              # agent-level normal coefficient
       :dur => :neg_lognormal,                # agent-level lognormal coefficient
       rfx_term(level = :nbh_code),           # option-level random intercept
       rfx_term(:salient; level = :nbh_code)] # option-level random slope
```
"""
rfx_term(var = nothing; level = nothing, dist = :normal) =
    (var   = isnothing(var)   ? nothing : Symbol(var),
     level = isnothing(level) ? nothing : Symbol(level),
     dist  = Symbol(dist))

"""
    _normalize_mlogit_rfx(rfx, formula_syms, col_group) -> Vector{MlogitRfxTerm}

Normalise every accepted `rfx` entry into an [`MlogitRfxTerm`](@ref) and validate
it. Accepted forms, in increasing generality:

    :x                                    normal random coefficient on :x, at col_group
    :x => :lognormal                      ditto with a distribution
    rfx_term(:x; level = :g, dist = :d)    full form; var = nothing for an intercept

The display name is `"x"` for a group-level coefficient -- so a model that only
uses the first two forms produces exactly the `theta_names` `logit2_rfx` would --
and `"x|level"` / `"1|level"` otherwise, following the `lme4` convention.
"""
function _normalize_mlogit_rfx(rfx, formula_syms, col_group::Symbol)

    terms = MlogitRfxTerm[]

    for (j, r) in enumerate(rfx)
        v, lv, d = if isa(r, Symbol)
            (r, nothing, :normal)
        elseif isa(r, AbstractString)
            (Symbol(r), nothing, :normal)
        elseif isa(r, Pair)
            (Symbol(first(r)), nothing, Symbol(last(r)))
        elseif isa(r, NamedTuple)
            bad = setdiff(collect(keys(r)), (:var, :level, :dist))
            isempty(bad) || error(
                "rfx entry $j has unknown field(s) $(bad); a random-coefficient term " *
                "accepts only (var, level, dist). Build it with rfx_term.")
            (haskey(r, :var)   && !isnothing(r.var)   ? Symbol(r.var)   : nothing,
             haskey(r, :level) && !isnothing(r.level) ? Symbol(r.level) : nothing,
             haskey(r, :dist)  && !isnothing(r.dist)  ? Symbol(r.dist)  : :normal)
        else
            error("rfx entry $j has type $(typeof(r)), which is not a random-coefficient " *
                  "term. Accepted: :x, :x => :lognormal, or rfx_term(:x; level = :g). " *
                  "Got: $(repr(r))")
        end

        d in _RFX_DISTS || error(
            "rfx distribution :$d is not supported (entry $j). " *
            "Supported: $(join(string.(":", _RFX_DISTS), ", ")).")

        level    = isnothing(lv) ? col_group : lv
        at_group = level === col_group

        col = 0
        if !isnothing(v)
            k = findfirst(==(v), formula_syms)
            isnothing(k) && error(
                "rfx variable :$v (entry $j) is not in formula. Available: $(formula_syms). " *
                "A random coefficient needs its mean in the formula; for a pure " *
                "random effect with no mean, use rfx_term(level = :$level) instead.")
            col = k
        end

        if _rfx_is_log(d) && isnothing(v)
            error("rfx entry $j is an intercept term with dist = :$d. The lognormal " *
                  "families are parameterised as +-exp(mu + sigma*eta), and an " *
                  "intercept term has no mu to estimate, so exp(sigma*eta) would only " *
                  "fix an arbitrary scale (its median is +-1). Give the term a formula " *
                  "variable, or use :normal.")
        end

        name = isnothing(v) ? "1|$(level)" :
               at_group     ? string(v)    : "$(v)|$(level)"

        push!(terms, MlogitRfxTerm(v, level, d, at_group, col, name))
    end

    # --- cross-term checks --------------------------------------------------
    tnames = [t.name for t in terms]
    if length(unique(tnames)) != length(tnames)
        dups = unique([n for n in tnames if count(==(n), tnames) > 1])
        error("rfx contains duplicate term(s) $(dups): the same (variable, level) pair " *
              "appears twice, which would make the two sigmas exchangeable and " *
              "unidentified.")
    end

    # A lognormal term owns its formula column outright: that column is dropped
    # from the linear part because exp(mu + sigma*eta) already carries the level.
    # A second term on the same variable would then be a deviation around a mean
    # that is no longer there.
    for t in terms
        _rfx_is_log(t.dist) || continue
        others = [u.name for u in terms if u.name != t.name && u.var === t.var]
        isempty(others) || error(
            "rfx variable :$(t.var) is :$(t.dist) and also carries the term(s) " *
            "$(others). A lognormal coefficient is the whole coefficient, not a " *
            "deviation from a formula mean, so :$(t.var) is dropped from the linear " *
            "part and no other term can hang off it. Make every term on :$(t.var) " *
            ":normal, or give the extra variance component its own variable.")
    end

    return terms
end


# ----------------------------------------------------------------------------
# Draws
# ----------------------------------------------------------------------------

"""
    _make_cell_draws(cells_per_group, R, seed) -> (eta, logw)

Antithetic standard normal draws, one row per (group, term, level) **cell**:
`eta` is `sum(cells_per_group) x R`, with the cells of group `i` occupying a
contiguous block, in group order.

Drawn group by group as a `C_i x R/2` block and then reflected. That ordering is
deliberate: when every term sits at the group level `C_i == M`, and the block is
then bit-for-bit the `randn(rng, M, R/2)` that [`_make_draws`](@ref) produces for
`logit2_rfx`. So the two models share an RNG stream in that case and can be
compared exactly rather than only up to simulation noise.

Antithetic reflection is mandatory, not optional: it is what makes the `sigma = 0`
saddle and the `sigma -> -sigma` mirror symmetry *exact*, because column `r` and
column `r + R/2` negate every cell together.
"""
function _make_cell_draws(cells_per_group::Vector{Int}, R::Int, seed::Int)
    iseven(R) || error("ndraws must be even (antithetic pairing); got $R")

    half   = R ÷ 2
    ncells = sum(cells_per_group)
    eta    = Matrix{Float64}(undef, ncells, R)
    rng    = MersenneTwister(seed)

    off = 0
    for Ci in cells_per_group
        if Ci > 0
            e = randn(rng, Ci, half)
            @views eta[off+1:off+Ci, 1:half]   .=   e
            @views eta[off+1:off+Ci, half+1:R] .= .-e
        end
        off += Ci
    end

    return eta, fill(-log(R), R)
end


# ----------------------------------------------------------------------------
# Prep
# ----------------------------------------------------------------------------

"""
Precomputed, worker-serialisable model data for [`mlogit_rfx`](@ref).

Rows are sorted by `(col_group, col_id, original position)`, so both the groups
and the choice sets inside them are contiguous ranges. Every field is concretely
typed: this struct is shipped to workers during the bootstrap, and abstract fields
would also deoptimise the inner loop.

`cellloc` is the workhorse. `cellloc[t, m]` is the index of row `t`'s cell for
term `m`, expressed **local to the row's group**, so the kernel can index the
group's slice of `eta` with no arithmetic in the hot loop.
"""
struct MlogitRfxPrep
    xmatrix::Matrix{Float64}              # Nobs x K, sorted
    xlin::Matrix{Float64}                 # xmatrix with lognormal-rfx columns zeroed
    zmatrix::Matrix{Float64}              # Nobs x M, loadings (1.0 for an intercept term)
    yvec::Vector{Float64}                 # Nobs, 0/1
    cellloc::Matrix{Int}                  # Nobs x M, cell index local to the group
    ranges::Vector{UnitRange{Int}}        # N, row range per group
    set_ranges::Vector{UnitRange{Int}}    # n_sets, row range per choice set
    set_of_group::Vector{UnitRange{Int}}  # N, range of SET indices per group
    sel_row::Vector{Int}                  # n_sets, absolute row of the selected option
    cell_ranges::Vector{UnitRange{Int}}   # N, cell range per group
    cell_term::Vector{Int}                # ncells, which term each cell belongs to
    eta::Array{Float64,2}                 # ncells x R
    logw::Vector{Float64}                 # R
    group_ids::Vector                     # N, unique col_group values in sorted order
    K::Int
    M::Int
    R::Int
    N::Int
    Tmax::Int                             # max rows per group
    Cmax::Int                             # max cells per group
    terms::Vector{MlogitRfxTerm}
    rfx_pairs::Vector{Pair{Symbol,Symbol}}  # display name => dist, for the shared reporting
    rfx_cols::Vector{Int}                   # formula index per term, 0 for an intercept
    rfx_islog::Vector{Bool}
    rfx_sgn::Vector{Float64}
    any_log::Bool
    theta_names::Vector{String}
    col_id::Symbol                        # choice set
    col_group::Symbol                     # integration unit
    seed::Int
    n_sets::Int
    cell_stats::Vector{NamedTuple}         # per-term identification diagnostics
end

"""
    _contiguous_blocks(v) -> Vector{UnitRange{Int}}

Ranges of equal consecutive values. `v` is already sorted on the relevant key, so
this is how the sorted group / choice-set structure becomes index arithmetic.
"""
function _contiguous_blocks(v)
    n = length(v)
    out = UnitRange{Int}[]
    n == 0 && return out
    start = 1
    for t in 2:n
        if !isequal(v[t], v[t-1])
            push!(out, start:(t-1))
            start = t
        end
    end
    push!(out, start:n)
    return out
end

"""
    _prep_mlogit_rfx(data_df, formula, col_id, col_selected, col_group, rfx,
                     ndraws, seed, weights) -> (P, gw)

Build an [`MlogitRfxPrep`](@ref). Does **not** mutate `data_df`.

`gw` is the vector of `N` group-level weights, or `nothing`.
"""
function _prep_mlogit_rfx(
        data_df,
        formula,
        col_id::Symbol,
        col_selected,
        col_group::Symbol,
        rfx,
        ndraws::Int,
        seed::Int,
        weights::Union{Nothing, Symbol, String})

    formula_syms = Symbol.(formula)
    K = length(formula_syms)
    K > 0 || error("formula is empty")

    terms = _normalize_mlogit_rfx(rfx, formula_syms, col_group)
    M     = length(terms)

    # --- column presence ----------------------------------------------------
    dfnames = Symbol.(names(data_df))
    needed  = vcat(formula_syms, Symbol(col_selected), col_id, col_group,
                   [t.level for t in terms])
    for c in unique(needed)
        c in dfnames || error("column :$c not found in data_df")
    end

    Nobs = nrow(data_df)
    Nobs > 0 || error("data_df has no rows")

    # --- sort by (group, set, original position) ----------------------------
    # The third key makes the order a total one, so the permutation does not
    # depend on the sort algorithm's stability and every run of every worker
    # sees the identical layout.
    gvals_raw = data_df[:, col_group]
    svals_raw = data_df[:, col_id]
    perm = sortperm(1:Nobs, by = t -> (gvals_raw[t], svals_raw[t], t))

    gvals = gvals_raw[perm]
    svals = svals_raw[perm]

    xmatrix = Float64.(Matrix(data_df[perm, formula_syms]))

    yvec = Float64.(data_df[perm, Symbol(col_selected)])
    all((yvec .== 0.0) .| (yvec .== 1.0)) ||
        error("selected column :$(col_selected) should have 0's and 1's only")

    # --- group and choice-set structure ------------------------------------
    ranges     = _contiguous_blocks(gvals)
    N          = length(ranges)
    group_ids  = [gvals[first(rg)] for rg in ranges]

    set_ranges   = UnitRange{Int}[]
    set_of_group = Vector{UnitRange{Int}}(undef, N)
    for (i, rg) in enumerate(ranges)
        s0 = length(set_ranges) + 1
        for b in _contiguous_blocks(view(svals, rg))
            push!(set_ranges, (first(rg) + first(b) - 1):(first(rg) + last(b) - 1))
        end
        set_of_group[i] = s0:length(set_ranges)
    end
    n_sets = length(set_ranges)

    # A choice set that showed up under two different groups would appear as two
    # separate blocks with the same id. The model is undefined in that case: the
    # set has to belong to exactly one integration unit.
    set_ids = [svals[first(sr)] for sr in set_ranges]
    if length(unique(set_ids)) != n_sets
        dup = first([s for s in set_ids if count(isequal(s), set_ids) > 1])
        gs  = unique([gvals[first(sr)] for sr in set_ranges if isequal(svals[first(sr)], dup)])
        error("choice set :$col_id = $(dup) appears under more than one :$col_group " *
              "(groups $(gs)). Choice sets must be nested inside the integration unit, " *
              "or the likelihood does not factorise over groups. Check that " *
              ":$col_id is unique across individuals -- an id like \"rank 1\" that " *
              "restarts for every person needs to be interacted with :$col_group first.")
    end

    # --- exactly one selected option per choice set -------------------------
    sel_row = Vector{Int}(undef, n_sets)
    for (s, sr) in enumerate(set_ranges)
        nsel = 0
        row  = 0
        for t in sr
            if yvec[t] == 1.0
                nsel += 1
                row = t
            end
        end
        nsel == 1 || error(
            "choice set :$col_id = $(svals[first(sr)]) has $nsel selected options " *
            "(:$(col_selected) sums to $nsel over its $(length(sr)) rows); exactly one " *
            "is required.")
        sel_row[s] = row
    end

    n_single = count(sr -> length(sr) == 1, set_ranges)
    n_single == n_sets && error(
        "every choice set has exactly one option, so every choice probability is 1 " *
        "and the log-likelihood is identically zero. Check :$col_id.")
    n_single > 0 && @warn "$n_single of $n_sets choice sets have a single option; they " *
                          "contribute nothing to the likelihood (their probability is 1 " *
                          "whatever theta is)."

    # --- cells: one per (group, term, level value) --------------------------
    # Ordered term-major inside each group, and by SORTED level value inside a
    # term. Sorted rather than first-appearance so that which draw a cell gets
    # does not depend on the order the rows arrived in: re-sorting the caller's
    # DataFrame would otherwise re-pair cells with draws and move every sigma by
    # simulation noise, for no reason a user could see.
    #
    # With every term at the group level each group has exactly M cells in term
    # order, which is what lets _make_cell_draws reproduce logit2_rfx's RNG
    # stream exactly (there is one level value per group, so the sort is a no-op).
    lvals = [t.at_group ? gvals : data_df[perm, t.level] for t in terms]

    cellloc         = zeros(Int, Nobs, M)
    cellidx         = zeros(Int, Nobs, M)     # global; diagnostics only
    cell_term       = Int[]
    cell_ranges     = Vector{UnitRange{Int}}(undef, N)
    cells_per_group = zeros(Int, N)

    for (i, rg) in enumerate(ranges)
        c0   = length(cell_term) + 1        # this group's first GLOBAL cell
        base = 0                            # cells already used by terms < m
        for m in 1:M
            vals = unique(view(lvals[m], rg))
            try
                sort!(vals)
            catch e
                error("the values of rfx level column :$(terms[m].level) cannot be " *
                      "sorted (eltype $(eltype(vals))), so cells cannot be given an " *
                      "order-independent numbering. Recode the column to something " *
                      "orderable, e.g. with denserank or by converting to String. " *
                      "Original error: $(sprint(showerror, e))")
            end
            loc_of = Dict(v => base + j for (j, v) in enumerate(vals))
            append!(cell_term, fill(m, length(vals)))
            for t in rg
                loc = loc_of[lvals[m][t]]   # index LOCAL to the group
                cellloc[t, m] = loc
                cellidx[t, m] = c0 + loc - 1
            end
            base += length(vals)
        end
        cell_ranges[i]     = c0:length(cell_term)
        cells_per_group[i] = base
    end

    ncells = length(cell_term)

    # --- loadings -----------------------------------------------------------
    zmatrix = Matrix{Float64}(undef, Nobs, M)
    for (m, t) in enumerate(terms)
        if isnothing(t.var)
            @views zmatrix[:, m] .= 1.0
        else
            @views zmatrix[:, m] .= xmatrix[:, t.col]
        end
    end

    # --- lognormal bookkeeping ---------------------------------------------
    rfx_islog = Bool[_rfx_is_log(t.dist)  for t in terms]
    rfx_sgn   = Float64[_rfx_sign(t.dist) for t in terms]
    rfx_cols  = Int[t.col for t in terms]
    any_log   = any(rfx_islog)

    xlin = xmatrix
    if any_log
        xlin = copy(xmatrix)
        for m in findall(rfx_islog)
            @views xlin[:, rfx_cols[m]] .= 0.0
        end
    end

    # --- identification diagnostics and guards -----------------------------
    cell_stats = _mlogit_rfx_cell_stats(terms, cellidx, zmatrix, ranges, set_ranges,
                                        ncells, col_id)

    Ti   = length.(ranges)
    Tmax = maximum(Ti)
    Cmax = M == 0 ? 0 : maximum(cells_per_group)

    # --- weights: permute, reduce to one per group, check constancy ---------
    gw = nothing
    if !isnothing(weights)
        wsym = Symbol(weights)
        wsym in dfnames || error("weights column :$wsym not found in data_df")
        wvec = Float64.(data_df[perm, wsym])

        gw = Vector{Float64}(undef, N)
        for (i, rg) in enumerate(ranges)
            w1 = wvec[first(rg)]
            for t in rg
                wvec[t] == w1 || error(
                    "weights column :$wsym is not constant within :$col_group " *
                    "(group $(group_ids[i]) has values $(w1) and $(wvec[t])). " *
                    "The weight multiplies the group log-likelihood, so it must be " *
                    "constant within group.")
            end
            gw[i] = w1
        end
    end

    # --- draws --------------------------------------------------------------
    eta, logw = _make_cell_draws(cells_per_group, ndraws, seed)

    theta_names = [string.(formula_syms); ["sd_" * t.name for t in terms]]
    rfx_pairs   = Pair{Symbol,Symbol}[Symbol(t.name) => t.dist for t in terms]

    P = MlogitRfxPrep(
        xmatrix, xlin, zmatrix, yvec, cellloc,
        ranges, set_ranges, set_of_group, sel_row,
        cell_ranges, cell_term, eta, logw, group_ids,
        K, M, ndraws, N, Tmax, Cmax,
        terms, rfx_pairs, rfx_cols, rfx_islog, rfx_sgn, any_log,
        theta_names, col_id, col_group, seed, n_sets, cell_stats)

    return P, gw
end

"""
    _mlogit_rfx_cell_stats(terms, cellidx, zmatrix, ranges, set_ranges, ncells, col_id)

Per-term identification diagnostics, plus the guards that depend on them.

Three things can go wrong with a random-coefficient term here, and only the first
is shared with `logit2_rfx`:

1. **Nothing to integrate against.** A cell seen in one single row contributes a
   normal perturbation to one utility. It is then not separably identified from
   the extreme-value noise already there, so a model in which *every* cell is a
   singleton is an error and a model in which the typical cell is a singleton gets
   a warning.

2. **The term cancels in the softmax.** Only *relative* utilities matter inside a
   choice set, so a term contributes nothing to a set whenever
   `Z_cj * A(cell(cj))` is constant across the options of that set -- which
   happens exactly when the set's rows share one cell and the loading is constant
   over them. A group-level random *intercept* is the leading example: it shifts
   every option equally and is not identified at all. That is a hard error when it
   holds for every set, and a warning when it holds for most of them.

3. **Collinearity with a random intercept at the same level.** A random slope
   whose loading barely varies inside its own cells is a rescaled random
   intercept, so the two variances trade off. Warned about only when an intercept
   term at that level is actually in the model, because otherwise there is nothing
   for it to be collinear with.
"""
function _mlogit_rfx_cell_stats(terms, cellidx, zmatrix, ranges, set_ranges,
                                ncells::Int, col_id::Symbol)

    M = length(terms)
    stats = NamedTuple[]
    M == 0 && return stats

    intercept_levels = Set(t.level for t in terms if isnothing(t.var))

    for (m, t) in enumerate(terms)

        # ---- rows and choice sets per cell ---------------------------------
        rows_per_cell = zeros(Int, ncells)
        for tt in axes(cellidx, 1)
            rows_per_cell[cellidx[tt, m]] += 1
        end
        used = findall(>(0), rows_per_cell)

        sets_per_cell = zeros(Int, ncells)
        for sr in set_ranges
            seen = Set{Int}()
            for tt in sr
                c = cellidx[tt, m]
                if !(c in seen)
                    push!(seen, c)
                    sets_per_cell[c] += 1
                end
            end
        end

        rpc = rows_per_cell[used]
        spc = sets_per_cell[used]
        cpg = [length(unique(view(cellidx, rg, m))) for rg in ranges]

        # A cell whose loading is zero on every one of its rows is allocated a
        # draw that never multiplies anything, so it never enters the per-draw
        # log-likelihood and cannot affect the posterior weights or the ESS. It
        # is inert, not costly. The INFORMATIVE count is therefore the honest
        # dimension of the simulated integral, and it can be far smaller: a
        # random coefficient on an interaction with a treatment arm is switched
        # off for every control individual's entire panel.
        nz_cell = falses(ncells)
        for tt in axes(cellidx, 1)
            zmatrix[tt, m] == 0.0 || (nz_cell[cellidx[tt, m]] = true)
        end
        cpg_inf = [count(c -> nz_cell[c], unique(view(cellidx, rg, m))) for rg in ranges]

        # ---- cancellation inside the choice set ----------------------------
        # The term contributes Z_cj * A(cell(cj), r) to option cj, and drops out
        # of the softmax whenever that is constant across the set's options for
        # EVERY draw. Since A varies freely across cells, that happens in exactly
        # two ways:
        #
        #   (a) every row of the set carries a zero loading -- the contribution
        #       is 0 whatever the draws are; or
        #   (b) every row sits in one cell AND shares one loading value.
        #
        # (a) is not a corner case. A random slope on a regressor that is
        # switched off for a whole subgroup -- an interaction with a treatment
        # arm, say -- contributes nothing for that subgroup's entire panel, and
        # its sigma is then identified off the remaining choice sets only.
        # Testing only (b) would report those sets as informative and understate
        # cancel_share, which is the number that decides whether the warning
        # below fires.
        n_cancel = 0
        for sr in set_ranges
            c1 = cellidx[first(sr), m]
            z1 = zmatrix[first(sr), m]
            all_zero = true
            same     = true
            for tt in sr
                zt = zmatrix[tt, m]
                zt == 0.0 || (all_zero = false)
                (cellidx[tt, m] == c1 && zt == z1) || (same = false)
                (all_zero || same) || break
            end
            (all_zero || same) && (n_cancel += 1)
        end
        cancel_share = n_cancel / length(set_ranges)

        # ---- within-cell share of the loading's variance -------------------
        z   = view(zmatrix, :, m)
        tot = sum(abs2, z .- mean(z))
        wth = 0.0
        if tot > 0
            sums  = zeros(Float64, ncells)
            for tt in eachindex(z)
                sums[cellidx[tt, m]] += z[tt]
            end
            for tt in eachindex(z)
                c = cellidx[tt, m]
                wth += abs2(z[tt] - sums[c] / rows_per_cell[c])
            end
        end
        within_share = tot > 0 ? wth / tot : 0.0

        push!(stats, (; term = t.name, level = t.level, dist = t.dist,
                        n_cells = length(used),
                        n_cells_informative = count(nz_cell),
                        cells_per_group_min = minimum(cpg),
                        cells_per_group_median = median(cpg),
                        cells_per_group_max = maximum(cpg),
                        cells_per_group_informative_median = median(cpg_inf),
                        cells_per_group_informative_max = maximum(cpg_inf),
                        rows_per_cell_min = minimum(rpc),
                        rows_per_cell_median = median(rpc),
                        rows_per_cell_max = maximum(rpc),
                        sets_per_cell_min = minimum(spc),
                        sets_per_cell_median = median(spc),
                        sets_per_cell_max = maximum(spc),
                        cancel_share = cancel_share,
                        within_cell_var_share = within_share))

        # ---- guards --------------------------------------------------------
        all(iszero, z) && error(
            "rfx term $(t.name) has a loading that is zero on every row, so sigma is " *
            "not identified (the term contributes nothing to any utility).")

        cancel_share == 1.0 && error(
            "rfx term $(t.name) cancels in every choice set: each set's rows share one " *
            "draw and one loading value, so the term shifts all of a set's options " *
            "equally and drops out of the softmax. " *
            (isnothing(t.var) ?
             "A random intercept at the integration unit is never identified in a " *
             "multinomial logit -- give it a level that varies WITHIN a choice set " *
             "(e.g. the option's identifier)." :
             "The loading :$(t.var) is constant within :$col_id, so its random " *
             "coefficient is not identified; use a regressor that varies across the " *
             "options of a choice set."))

        if cancel_share > 0.5
            @warn "rfx term $(t.name) cancels in the softmax for " *
                  "$(round(100 * cancel_share, digits = 1))% of choice sets " *
                  "(their rows share one draw and one loading value), so sigma is " *
                  "identified off the remaining " *
                  "$(length(set_ranges) - n_cancel) sets only."
        end

        maximum(rpc) == 1 && error(
            "rfx term $(t.name) has exactly one observation in every (group, " *
            ":$(t.level)) cell, so each draw perturbs a single utility and is not " *
            "separably identified from the logit error. Use a coarser level, or drop " *
            "the term.")

        if median(rpc) < 2
            @warn "rfx term $(t.name) has a median of $(median(rpc)) observation(s) per " *
                  "(group, :$(t.level)) cell: the random effect is weakly identified " *
                  "because most draws are seen only once."
        end

        if !isnothing(t.var) && t.level in intercept_levels && within_share < 0.01
            @warn "rfx term $(t.name) has $(round(100 * within_share, digits = 3))% of " *
                  "the variance of :$(t.var) within its own cells, and the model also " *
                  "has a random intercept at :$(t.level). A slope whose loading is " *
                  "constant within a cell is a rescaled intercept, so the two sigmas " *
                  "will trade off."
        end
    end

    return stats
end


# ----------------------------------------------------------------------------
# Buffers
# ----------------------------------------------------------------------------

"""
Scratch space for the fused objective/gradient kernel, allocated once per fit and
reused across groups via views.

`A` and `S` are indexed by (cell, draw) rather than (term, draw): that is the only
structural difference from [`RfxBuffers`](@ref), and it is what carries the extra
nesting level.
"""
struct MlogitRfxBuffers
    V::Matrix{Float64}       # Tmax x R    linear index
    E::Matrix{Float64}       # Tmax x R    residuals y - P
    A::Matrix{Float64}       # Cmax x R    per-cell coefficient contribution
    S::Matrix{Float64}       # Cmax x R    per-cell score, sum_t e_tr * Z_tm
    ll::Vector{Float64}      # R
    pw::Vector{Float64}      # R           posterior weights tau
    xb::Vector{Float64}      # Tmax
    ebar::Vector{Float64}    # Tmax
end

function MlogitRfxBuffers(P::MlogitRfxPrep)
    MlogitRfxBuffers(
        Matrix{Float64}(undef, P.Tmax, P.R),
        Matrix{Float64}(undef, P.Tmax, P.R),
        Matrix{Float64}(undef, P.Cmax, P.R),
        Matrix{Float64}(undef, P.Cmax, P.R),
        Vector{Float64}(undef, P.R),
        Vector{Float64}(undef, P.R),
        Vector{Float64}(undef, P.Tmax),
        Vector{Float64}(undef, P.Tmax),
    )
end


# ----------------------------------------------------------------------------
# Per-cell coefficients
# ----------------------------------------------------------------------------

"""
    _mlogit_rfx_fill_A!(Ai, beta, sigma, etai, ct, P)

Fill `Ai` (`C_i x R`) with the quantity that multiplies the loading in the linear
index, one row per cell of the current group:

    :normal          A[c,r] = sigma_m * eta[c,r]                (deviation from mu_m)
    :lognormal       A[c,r] = +-exp(mu_m + sigma_m * eta[c,r])  (the coefficient itself)

where `m = ct[c]` is the cell's term. `xlin` has the lognormal columns zeroed,
which is why the second line is the whole coefficient and not a deviation. The
sign of `:neg_lognormal` is folded in here, so nothing downstream knows about it.
"""
@inline function _mlogit_rfx_fill_A!(Ai, beta, sigma, etai, ct, P::MlogitRfxPrep)

    Ci = length(ct)

    if !P.any_log
        @inbounds for r in 1:P.R, c in 1:Ci
            Ai[c, r] = sigma[ct[c]] * etai[c, r]
        end
        return nothing
    end

    @inbounds for r in 1:P.R, c in 1:Ci
        m = ct[c]
        Ai[c, r] = P.rfx_islog[m] ?
            P.rfx_sgn[m] * exp(beta[P.rfx_cols[m]] + sigma[m] * etai[c, r]) :
            sigma[m] * etai[c, r]
    end

    # exp() overflows to Inf above an exponent of ~709, and Inf then propagates
    # into the gradient as Inf*0 = NaN, from which LBFGS cannot recover -- it
    # would report a converged fit at a garbage theta. Fail with the cause named
    # instead. O(C_i*R) with C_i small, so this is free next to the T_i x R work.
    if !all(isfinite, view(Ai, 1:Ci, :))
        lg = findall(P.rfx_islog)
        error("a lognormal random coefficient overflowed: exp(mu + sigma*eta) is not " *
              "finite at mu = $(round.([beta[P.rfx_cols[m]] for m in lg], digits = 3)), " *
              "sigma = $(round.([sigma[m] for m in lg], digits = 3)) " *
              "(terms $([P.terms[m].name for m in lg])). mu is on the LOG scale for a " *
              "lognormal coefficient: build theta0 with theta0_mlogit_rfx, which " *
              "converts a level b0 for you.")
    end

    return nothing
end


# ----------------------------------------------------------------------------
# Fused objective and gradient
# ----------------------------------------------------------------------------

"""
    _mlogit_rfx_fg!(F, G, theta, P, buf, gw) -> objective or nothing

Fused objective/gradient in the `Optim.only_fg!` convention: fills `G` when
`G !== nothing`, returns the objective when `F !== nothing`.

As in `logit2_rfx`, the draw dimension is collapsed *before* the design matrix is
touched:

    ebar_t = sum_r tau_r * e_tr     O(T_i*R)   -> a T_i-vector
    d/dbeta = sum_t ebar_t * X_t    O(T_i*K)   -> one gemv against Xi

Building an augmented per-draw design matrix instead would cost O(Nobs*R*K).

The sigma-gradient does need the draw dimension, but only through a per-cell
score, which is accumulated in the same pass that computes the softmax residuals:

    S[c,r] = sum_{t in cell c} e_tr * Z_tm

so the extra nesting level costs O(T_i*R*M) with M small, not a factor of R.
"""
function _mlogit_rfx_fg!(F, G, theta::Vector{Float64}, P::MlogitRfxPrep,
                         buf::MlogitRfxBuffers, gw::Union{Nothing, Vector{Float64}})

    K, M, R = P.K, P.M, P.R

    beta  = view(theta, 1:K)
    sigma = view(theta, K+1:K+M)

    need_g = G !== nothing
    need_g && fill!(G, 0.0)
    gb = need_g ? view(G, 1:K)     : nothing
    gs = need_g ? view(G, K+1:K+M) : nothing

    scatter = need_g && M > 0
    Q = 0.0

    @inbounds for i in 1:P.N
        rrng = P.ranges[i]
        Ti   = length(rrng)
        r0   = first(rrng) - 1
        crng = P.cell_ranges[i]
        Ci   = length(crng)
        om   = isnothing(gw) ? 1.0 : gw[i]

        Xi = view(P.xlin,    rrng, :)      # Ti x K (lognormal columns zeroed)
        Zi = view(P.zmatrix, rrng, :)      # Ti x M
        Li = view(P.cellloc, rrng, :)      # Ti x M, group-local cell index
        yi = view(P.yvec,    rrng)         # Ti

        etai = view(P.eta, crng, :)        # Ci x R
        ct   = view(P.cell_term, crng)     # Ci
        Ai   = view(buf.A, 1:Ci, :)
        Si   = view(buf.S, 1:Ci, :)
        Vi   = view(buf.V, 1:Ti, :)
        Ei   = view(buf.E, 1:Ti, :)
        xb   = view(buf.xb,   1:Ti)
        eb   = view(buf.ebar, 1:Ti)

        # --- 1. per-cell coefficients ---------------------------------------
        M > 0 && _mlogit_rfx_fill_A!(Ai, beta, sigma, etai, ct, P)

        # --- 2. linear index ------------------------------------------------
        mul!(xb, Xi, beta)
        if M > 0
            for r in 1:R, t in 1:Ti
                v = xb[t]
                for m in 1:M
                    v += Zi[t, m] * Ai[Li[t, m], r]
                end
                Vi[t, r] = v
            end
        else
            Vi .= xb
        end

        # --- 3. softmax per choice set: log-lik, residuals, per-cell score ---
        scatter && fill!(Si, 0.0)
        for r in 1:R
            acc = P.logw[r]
            for s in P.set_of_group[i]
                sr = P.set_ranges[s]
                lo = first(sr) - r0
                hi = last(sr)  - r0

                # Shift by the max before exponentiating: a utility of a few
                # hundred is perfectly reachable with fixed effects in the
                # formula, and exp() of it is Inf.
                mx = -Inf
                for t in lo:hi
                    Vi[t, r] > mx && (mx = Vi[t, r])
                end
                se = 0.0
                for t in lo:hi
                    se += exp(Vi[t, r] - mx)
                end
                lse = mx + log(se)

                if scatter
                    for t in lo:hi
                        e = yi[t] - exp(Vi[t, r] - lse)
                        Ei[t, r] = e
                        for m in 1:M
                            Si[Li[t, m], r] += e * Zi[t, m]
                        end
                    end
                else
                    for t in lo:hi
                        Ei[t, r] = yi[t] - exp(Vi[t, r] - lse)
                    end
                end

                acc += Vi[P.sel_row[s] - r0, r] - lse
            end
            buf.ll[r] = acc
        end

        # --- 4. group log-likelihood and posterior weights over draws --------
        lse_all = logsumexp(buf.ll)
        Q -= om * lse_all

        # --- 5. gradient: collapse draws FIRST -------------------------------
        if need_g
            buf.pw .= exp.(buf.ll .- lse_all)

            mul!(eb, Ei, buf.pw)                      # Ti   ebar_t = sum_r tau_r e_tr
            mul!(gb, Xi', eb, -om, 1.0)               # gb -= om * Xi' * ebar

            if M > 0
                # c outer / r inner would stride across columns of the Ci x R
                # column-major arrays; r outer keeps every access contiguous.
                for r in 1:R
                    tau = buf.pw[r]
                    for c in 1:Ci
                        m = ct[c]
                        co = om * tau * Si[c, r]
                        if P.rfx_islog[m]
                            gs[m]             -= co * etai[c, r] * Ai[c, r]
                            gb[P.rfx_cols[m]] -= co * Ai[c, r]
                        else
                            gs[m] -= co * etai[c, r]
                        end
                    end
                end
            end
        end
    end

    return F !== nothing ? Q : nothing
end


# ----------------------------------------------------------------------------
# ESS diagnostic
# ----------------------------------------------------------------------------

"""
    _mlogit_rfx_ess(theta, P, buf) -> Vector{Float64}

Effective number of draws per group at `theta`: `ESS_i = 1 / sum_r tau_ir^2`.

This matters more here than it does for `logit2_rfx`. The integral's dimension is
the number of *cells* in the group, not the number of terms, so an option-level
random effect over 20 alternatives is a 20-dimensional integral being done with
the same `R` draws. A collapsing ESS is how that shows up.
"""
function _mlogit_rfx_ess(theta::Vector{Float64}, P::MlogitRfxPrep,
                         buf::MlogitRfxBuffers)

    K, M, R = P.K, P.M, P.R
    beta  = view(theta, 1:K)
    sigma = view(theta, K+1:K+M)

    ess = Vector{Float64}(undef, P.N)

    @inbounds for i in 1:P.N
        rrng = P.ranges[i]
        Ti   = length(rrng)
        r0   = first(rrng) - 1
        crng = P.cell_ranges[i]
        Ci   = length(crng)

        Xi   = view(P.xlin,    rrng, :)
        Zi   = view(P.zmatrix, rrng, :)
        Li   = view(P.cellloc, rrng, :)
        etai = view(P.eta, crng, :)
        ct   = view(P.cell_term, crng)
        Ai   = view(buf.A, 1:Ci, :)
        Vi   = view(buf.V, 1:Ti, :)
        xb   = view(buf.xb, 1:Ti)

        M > 0 && _mlogit_rfx_fill_A!(Ai, beta, sigma, etai, ct, P)

        mul!(xb, Xi, beta)
        if M > 0
            for r in 1:R, t in 1:Ti
                v = xb[t]
                for m in 1:M
                    v += Zi[t, m] * Ai[Li[t, m], r]
                end
                Vi[t, r] = v
            end
        else
            Vi .= xb
        end

        for r in 1:R
            acc = P.logw[r]
            for s in P.set_of_group[i]
                sr = P.set_ranges[s]
                lo = first(sr) - r0
                hi = last(sr)  - r0
                mx = -Inf
                for t in lo:hi
                    Vi[t, r] > mx && (mx = Vi[t, r])
                end
                se = 0.0
                for t in lo:hi
                    se += exp(Vi[t, r] - mx)
                end
                acc += Vi[P.sel_row[s] - r0, r] - (mx + log(se))
            end
            buf.ll[r] = acc
        end

        lse = logsumexp(buf.ll)
        buf.pw .= exp.(buf.ll .- lse)
        ess[i] = 1.0 / sum(abs2, buf.pw)
    end

    return ess
end


# ----------------------------------------------------------------------------
# Starting values
# ----------------------------------------------------------------------------

"""
    theta0_mlogit_rfx(formula, rfx; b0, s0, col_group) -> Vector{Float64}

Assemble a starting vector `[mu; sigma]` for [`mlogit_rfx`](@ref).

The `mlogit_rfx` counterpart of [`theta0_rfx`](@ref): same contract, but it
accepts the `rfx_term` entries too, and so needs `col_group` in order to work out
which terms are at the group level.

Default `s0 = 0.5`, never `0.0`: `sigma = 0` is a stationary point (a saddle) of
the simulated likelihood, so an optimiser started there cannot move.

`b0` is given on the **level** scale for every variable -- the scale of a plain
`mlogit` coefficient -- including the lognormal ones. For a `:lognormal` or
`:neg_lognormal` term on formula position `k`, this converts it to the log scale
that `mlogit_rfx` actually estimates,

    mu_k = log|b0_k| - s0_m^2 / 2

which is the value whose implied mean coefficient `+-exp(mu + sigma^2/2)` equals
`b0_k`.
"""
function theta0_mlogit_rfx(formula, rfx;
                           b0 = zeros(length(formula)),
                           s0 = fill(0.5, length(rfx)),
                           col_group::Symbol = :__group__)

    K = length(formula)
    M = length(rfx)

    length(b0) == K || error("b0 has length $(length(b0)) but formula has $K variables")
    length(s0) == M || error("s0 has length $(length(s0)) but rfx has $M terms")

    th    = Float64[b0...; s0...]
    terms = _normalize_mlogit_rfx(rfx, Symbol.(formula), col_group)

    # Translate level starts into the log scale for the lognormal families, so
    # that b0 means one thing (a level coefficient) whatever the mix of
    # distributions, and a wrong sign is caught before any optimisation is paid
    # for.
    for (m, t) in enumerate(terms)
        _rfx_is_log(t.dist) || continue

        sgn = _rfx_sign(t.dist)
        b   = sgn * th[t.col]
        b > 0 || error(
            "rfx term $(t.name) is :$(t.dist), whose support is " *
            (sgn > 0 ? "(0, Inf)" : "(-Inf, 0)") * ", but b0[$(t.col)] = $(th[t.col]). " *
            "Pass a level starting value with the right sign -- e.g. the matching " *
            "coefficient from a plain mlogit fit. The default b0 = zeros(K) cannot be " *
            "used with a lognormal random coefficient.")

        th[t.col] = log(b) - s0[m]^2 / 2
    end

    return th
end

"""
    theta0_mlogit_rfx_multistart(formula, rfx; b0, nstarts, s_range, b_jitter,
                                 seed, col_group) -> Matrix

An `nstarts x (K + M)` matrix of starting values for [`mlogit_rfx`](@ref), **one
start per row**. The `mlogit_rfx` counterpart of
[`theta0_rfx_multistart`](@ref); see that docstring for why `sigma` is drawn
log-uniformly and why `b_jitter` is multiplicative.

Row 1 is exactly `theta0_mlogit_rfx(formula, rfx; b0 = b0)`, so a multi-start run
can never return a worse optimum than the corresponding single-start run.
"""
function theta0_mlogit_rfx_multistart(formula, rfx;
                                      b0 = zeros(length(formula)),
                                      nstarts::Int = 100,
                                      s_range = (0.05, 2.0),
                                      b_jitter::Real = 0.0,
                                      seed::Int = 20260808,
                                      col_group::Symbol = :__group__)

    nstarts >= 1 || error("nstarts must be at least 1; got $nstarts")
    lo, hi = float(s_range[1]), float(s_range[2])
    (0 < lo && lo <= hi) || error("s_range must satisfy 0 < first <= last; got $s_range")
    b_jitter >= 0 || error("b_jitter must be >= 0; got $b_jitter")

    K, M = length(formula), length(rfx)
    rng  = MersenneTwister(seed)

    out = Matrix{Float64}(undef, nstarts, K + M)
    out[1, :] .= theta0_mlogit_rfx(formula, rfx; b0 = b0, col_group = col_group)

    for r in 2:nstarts
        s = exp.(log(lo) .+ (log(hi) - log(lo)) .* rand(rng, M))
        b = b_jitter > 0 ? collect(Float64, b0) .* exp.(b_jitter .* randn(rng, K)) : b0
        out[r, :] .= theta0_mlogit_rfx(formula, rfx; b0 = b, s0 = s,
                                       col_group = col_group)
    end

    return out
end

"""Validate `theta0` against a prepped model."""
function _check_theta0_mlogit_rfx(theta0, P::MlogitRfxPrep)
    th   = Float64.(collect(theta0))
    npar = P.K + P.M

    length(th) == npar || error(
        "theta0 has length $(length(th)) but the model has K + M = $(P.K) + $(P.M) = " *
        "$npar parameters. Use theta0_mlogit_rfx(formula, rfx) to assemble it.")

    if P.M > 0 && any(th[P.K+1:end] .== 0)
        bad = findall(th[P.K+1:end] .== 0)
        error("theta0 has sigma = 0 for rfx term(s) " *
              "$([P.terms[b].name for b in bad]): sigma = 0 is a stationary point (a " *
              "saddle) of the simulated likelihood, so the optimiser cannot move away " *
              "from it. Use a nonzero start such as 0.5.")
    end

    return th
end

"""
    _mlogit_rfx_theta0_matrix(theta0, P) -> Matrix{Float64}

Normalise `theta0` to an `nstarts x npar` matrix and validate every row. Accepts a
plain vector (one start), a matrix with `npar` columns, or a vector of vectors.
"""
function _mlogit_rfx_theta0_matrix(theta0, P::MlogitRfxPrep)
    npar = P.K + P.M

    if theta0 isa AbstractMatrix
        size(theta0, 2) == npar || error(
            "theta0 has $(size(theta0, 2)) columns but the model has K + M = $npar " *
            "parameters. Multi-start theta0 is nstarts x npar: one start per ROW. " *
            "Build it with theta0_mlogit_rfx_multistart.")
        out = Matrix{Float64}(undef, size(theta0, 1), npar)
        for r in axes(theta0, 1)
            out[r, :] .= _check_theta0_mlogit_rfx(collect(view(theta0, r, :)), P)
        end
        return out
    end

    if theta0 isa AbstractVector && !isempty(theta0) && first(theta0) isa AbstractVector
        out = Matrix{Float64}(undef, length(theta0), npar)
        for (r, t) in enumerate(theta0)
            out[r, :] .= _check_theta0_mlogit_rfx(collect(t), P)
        end
        return out
    end

    return reshape(_check_theta0_mlogit_rfx(theta0, P), 1, npar)
end


# ----------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------

"""
    _mlogit_rfx(P, theta0, gw, optim_options; rethrow_errors = false) -> MLEFit

Inner estimation routine: takes a prepped [`MlogitRfxPrep`](@ref), so the
bootstrap can reuse prep and vary only the group weights `gw`.

Canonicalises `sigma >= 0` at the source, so that every path -- the main fit and
every bootstrap replicate -- returns a canonical sign. Without this,
`cov(theta_boot_table)` would mix the `2^M` mirror modes and be meaningless.

By default this never throws: on failure it returns an `MLEFit` with
`errored = true` and the message in `error_message`, so a single bad bootstrap
replicate cannot take the whole run down. Pass `rethrow_errors = true` to let the
original exception propagate with its stacktrace intact.
"""
function _mlogit_rfx(
        P::MlogitRfxPrep,
        theta0::Vector{Float64},
        gw::Union{Nothing, Vector{Float64}},
        optim_options::Optim.Options = Optim.Options();
        rethrow_errors::Bool = false)

    npar = P.K + P.M

    local myfit
    try
        buf = MlogitRfxBuffers(P)
        fg! = (F, G, th) -> _mlogit_rfx_fg!(F, G, th, P, buf, gw)

        time_it_took = @elapsed opt = optimize(
            Optim.only_fg!(fg!), copy(theta0), LBFGS(), optim_options)

        # mirror-mode canonicalisation, at the source
        th = copy(Optim.minimizer(opt))
        th[P.K+1:end] .= abs.(th[P.K+1:end])

        ess = P.M > 0 ? _mlogit_rfx_ess(th, P, buf) : fill(Float64(P.R), P.N)

        Ti = length.(P.ranges)
        Si = length.(P.set_of_group)

        myfit = MLEFit(
            theta0      = theta0,
            theta_hat   = th,
            theta_names = P.theta_names,
            n_obs       = size(P.xmatrix, 1),
            weights     = gw,
            obj_value   = Optim.minimum(opt),
            converged   = Optim.converged(opt),
            iterations  = Optim.iterations(opt),
            iteration_limit_reached = Optim.iteration_limit_reached(opt),
            time_it_took = time_it_took,
            extra = (; model = :mlogit_rfx,
                       n_groups = P.N, K = P.K, M = P.M, R = P.R,
                       col_id = P.col_group,          # the integration unit
                       col_set = P.col_id,            # the softmax group
                       n_sets = P.n_sets,
                       rfx = P.rfx_pairs, rfx_cols = P.rfx_cols,
                       rfx_terms = P.terms, cell_stats = P.cell_stats,
                       seed = P.seed,
                       ess_min = minimum(ess), ess_p10 = quantile(ess, 0.10),
                       ess_median = median(ess), ess_mean = mean(ess),
                       Ti_min = minimum(Ti), Ti_median = median(Ti),
                       Ti_max = maximum(Ti),
                       sets_min = minimum(Si), sets_median = median(Si),
                       sets_max = maximum(Si))
        )

    catch e
        rethrow_errors && rethrow(e)
        myfit = MLEFit(
            theta0      = theta0,
            theta_hat   = fill(NaN, npar),
            theta_names = P.theta_names,
            n_obs       = size(P.xmatrix, 1),
            weights     = gw,
            obj_value   = NaN,
            errored     = true,
            error_message = sprint(showerror, e),
            converged   = false,
            iterations  = missing,
            iteration_limit_reached = missing,
            time_it_took = missing
        )
    end

    return myfit
end

"""
    _mlogit_rfx_multi(P, theta0s, gw, optim_options; parallel, rethrow_errors) -> MLEFit

Fit from every row of `theta0s` and return the best usable fit, carrying the full
set of attempts in `fits_df`. Mirrors [`_logit2_rfx_multi`](@ref); see that
docstring for why a non-converged fit is never selected even when its objective is
lower, and for what the multi-optimum warning is telling you.
"""
function _mlogit_rfx_multi(P::MlogitRfxPrep, theta0s::AbstractMatrix{Float64},
                           gw::Union{Nothing, Vector{Float64}},
                           optim_options::Optim.Options = Optim.Options();
                           parallel::Bool = false,
                           rethrow_errors::Bool = false,
                           warn_multi::Bool = true,
                           obj_tol::Float64 = 1e-4)

    nstarts = size(theta0s, 1)

    task = r -> _mlogit_rfx(P, Vector{Float64}(view(theta0s, r, :)), gw, optim_options;
                            rethrow_errors = rethrow_errors)

    fits = parallel ? pmap(task, CachingPool(workers()), 1:nstarts) : map(task, 1:nstarts)

    objs = [f.obj_value for f in fits]
    ok   = [(!f.errored) && f.converged for f in fits]
    cand = findall(ok)

    if isempty(cand)
        nerr = count(f -> f.errored, fits)
        msg = "none of the $nstarts starts produced a usable fit " *
              "($nerr errored, $(nstarts - nerr) ran but did not converge)."
        if nerr > 0
            msg *= "\nFirst error was:\n    " *
                   replace(fits[findfirst(f -> f.errored, fits)].error_message,
                           "\n" => "\n    ")
        end
        rethrow_errors && error(msg)

        bad = fits[something(findfirst(f -> f.errored, fits), 1)]
        bad.errored = true
        bad.error_message = msg
        return bad
    end

    ibest = cand[argmin(objs[cand])]
    best  = fits[ibest]

    reps = Float64[]
    for q in sort(objs[cand])
        (isempty(reps) || q - reps[end] > obj_tol) && push!(reps, q)
    end
    ndistinct = length(reps)
    nbest = count(q -> q - reps[1] <= obj_tol, objs[cand])

    best.fits_df = DataFrame(
        start      = 1:nstarts,
        obj_value  = objs,
        converged  = [f.converged for f in fits],
        errored    = [f.errored   for f in fits],
        iterations = [f.iterations for f in fits],
        is_best    = (1:nstarts) .== ibest,
        theta0     = [Vector{Float64}(view(theta0s, r, :)) for r in 1:nstarts],
        theta_hat  = [copy(f.theta_hat) for f in fits],
    )

    best.extra = merge(best.extra, (;
        n_starts          = nstarts,
        n_usable_starts   = length(cand),
        n_distinct_optima = ndistinct,
        n_at_best         = nbest,
        obj_best          = reps[1],
        obj_worst         = reps[end],
        obj_second        = ndistinct > 1 ? reps[2] : NaN))

    if warn_multi && ndistinct > 1
        @warn "multi-start found $ndistinct distinct optima across $nstarts starts: " *
              "the best objective is $(round(reps[1], digits = 4)) and the worst is " *
              "$(round(reps[end], digits = 4)) (gap $(round(reps[end] - reps[1], digits = 4))). " *
              "Only $nbest of $(length(cand)) usable starts reached the best one, so a " *
              "single-start fit had a $(round(100 * (1 - nbest / length(cand)), digits = 1))% " *
              "chance of returning a non-maximum. The best fit is returned; see fits_df " *
              "for all attempts."
    end

    return best
end

"""
    mlogit_rfx(data_df, formula, col_id, col_selected, theta0; kwargs...) -> MLEFit

Multinomial (conditional) logit with independent random coefficients, estimated by
maximum simulated likelihood with an analytic gradient.

The positional arguments match [`mlogit`](@ref) exactly, so an existing `mlogit`
call becomes an `mlogit_rfx` call by adding keywords. `rfx` and `col_group` are
keywords, because three adjacent positional identifiers would be far too easy to
transpose silently.

# Three levels, and why the level has to sit inside an individual
`logit2_rfx` knows one identifier: `col_id` is at once the panel unit and the level
at which every draw lives. Here they come apart:

| | meaning |
|---|---|
| `col_group` | integration unit (the individual). One simulated integral per group. |
| `col_id` | choice set: the rows the softmax runs over. Nested in `col_group`. |
| a term's `level` | the identifier at which *that* term's draw varies. Defaults to `col_group`. |

A term whose `level` varies **within a choice set** is an *option-level* random
effect: `nu_{i,l}` is individual `i`'s idiosyncratic taste for the alternative
labelled `l`, shared by every row of `i` carrying that label -- across choice sets,
which is exactly what makes it more than noise. Draws are always indexed by
`(group, level)`, i.e. **the level column is interacted with `col_group`**, which
is what keeps the likelihood a sum of independent group terms. A random effect
genuinely common to all individuals (a `xi_l` with no `i`) cannot be fitted this
way at all -- nothing would factorise -- and is not what this estimates even if
the level column's values are shared across people.

Binary choice is the `J = 2` case: two rows per choice set, the alternatives'
attributes undifferenced. That is how you get an option-level random effect into a
binary model -- in the wide, differenced `logit2` layout there is no single column
holding "this row's alternative", so there is nothing for a level to point at.

# Arguments
- `data_df`: long format, one row per (choice set, option). **Not mutated.**
- `formula`: `Vector{Symbol}` of regressors, as in `mlogit`.
- `col_id`: choice-set identifier. Must be unique across `col_group`.
- `col_selected`: 0/1 column, exactly one 1 per choice set.
- `theta0`: length `K + M` (see [`theta0_mlogit_rfx`](@ref)), or an
  `nstarts x (K + M)` matrix -- one start per **row** -- for a multi-start fit.

# Keywords
- `col_group = nothing`: integration unit. Defaults to `col_id`, i.e. one choice
  set per individual (the textbook cross-sectional mixed logit).
- `rfx = []`: random-coefficient terms. `:x` and `:x => :lognormal` mean what they
  mean in `logit2_rfx`; [`rfx_term`](@ref) adds the level and the intercept form.
- `ndraws = 1000`: simulation draws, must be even (antithetic pairing).
- `seed = 20260808`: draw seed. Draws are generated once and reused for every
  function evaluation, so the objective is a deterministic function of theta.
- `weights = nothing`: column name; must be constant within `col_group`.
- `optim_options = Optim.Options()`.
- `parallel = false`: distribute a **multi-start** fit over `workers()`.
- `rethrow_errors = false`: by default a failed optimisation is captured into
  `errored`/`error_message`; `true` lets the exception propagate.

# Parameter ordering
`theta = [mu (K, formula order); sigma (M, rfx order)]`, named
`[formula...; "sd_" .* term names]`, where a term's name is `x` at the group level
and `x|level` / `1|level` otherwise. Returned `sigma` is always `>= 0`: the
likelihood satisfies `Q(mu, sigma) = Q(mu, -sigma)`, so there are `2^M` mirror
optima and the sign is not identified.

# What is *not* identified
Three failures are checked in prep rather than left to produce a plausible number:

- A **group-level random intercept** shifts every option of a choice set equally
  and cancels in the softmax. It is an error, and the message points at the fix
  (give it an option-level `level`). The same holds for a random coefficient on a
  regressor that is constant within the choice set.
- A term whose every `(group, level)` **cell holds one row** perturbs a single
  utility and is not separable from the logit error. Also an error; a merely
  low median gets a warning.
- A random **slope** whose loading barely varies inside its own cells, in a model
  that also has an intercept at that level, is a rescaled intercept. Warned about,
  since the two variances will trade off.

`fit.extra.cell_stats` carries the per-term cell counts these come from.

# Simulation error
The integral's dimension is the number of *cells* in a group, not the number of
terms, so an option-level effect over 20 alternatives is a 20-dimensional integral.
Watch `extra.ess_p10`, and check `sigma` across seeds before reporting it: because
draws are fixed across bootstrap replicates (they must be, or the objective is not
deterministic in theta), simulation error never enters `boot_se`, and `abs()`
canonicalisation makes noise around a true zero read as a positive number.

# Example
```julia
# ranked/exploded logit: person ranks neighborhoods, one choice set per rank level
th0 = theta0_mlogit_rfx(myxs, myrfx; b0 = plain.theta_hat, col_group = :uniqueid)
fit = mlogit_rfx(stacked_df, myxs, :uniqueid_choice, :selected, th0;
                 col_group = :uniqueid,
                 rfx = [rfx_term(level = :nbh_code)],   # taste for a neighborhood
                 ndraws = 400)
fit.vcov = boot_mlogit_rfx(stacked_df, myxs, :uniqueid_choice, :selected, th0;
                          col_group = :uniqueid, rfx = [rfx_term(level = :nbh_code)],
                          ndraws = 400, nboot = 500, theta_start = fit.theta_hat)
regtable_rfx(fit)
```
"""
function mlogit_rfx(
        data_df,
        formula,
        col_id,
        col_selected,
        theta0;
        col_group = nothing,
        rfx = [],
        ndraws::Int = 1000,
        seed::Int = 20260808,
        weights::Union{Nothing, Symbol, String} = nothing,
        optim_options::Optim.Options = Optim.Options(),
        parallel::Bool = false,
        rethrow_errors::Bool = false)

    cid = Symbol(col_id)
    cg  = isnothing(col_group) ? cid : Symbol(col_group)

    P, gw = _prep_mlogit_rfx(data_df, formula, cid, col_selected, cg, rfx,
                             ndraws, seed, weights)

    theta0s = _mlogit_rfx_theta0_matrix(theta0, P)

    myfit = if size(theta0s, 1) == 1
        _mlogit_rfx(P, vec(theta0s), gw, optim_options; rethrow_errors = rethrow_errors)
    else
        parallel && _check_boot_workers()
        _mlogit_rfx_multi(P, theta0s, gw, optim_options;
                          parallel = parallel, rethrow_errors = rethrow_errors)
    end

    if !myfit.errored && P.M > 0 && myfit.extra.ess_p10 < 30
        dim = sum(s.cells_per_group_informative_max for s in P.cell_stats)
        @warn "10th-percentile effective number of draws is " *
              "$(round(myfit.extra.ess_p10, digits=1)) (< 30): the posterior over the " *
              "random coefficients is concentrated relative to the draw set. " *
              "The integral's dimension here is the number of cells per group, not " *
              "the number of terms: up to $dim per group carry a nonzero loading " *
              "(of $(P.Cmax) allocated -- a cell whose loading is zero on every row " *
              "is inert and costs no accuracy). Consider increasing ndraws " *
              "(currently $(P.R))."
    end

    return myfit
end
