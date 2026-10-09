###############################################################################
# mlogit_rfx: multinomial (conditional) logit with random coefficients, by
# maximum simulated likelihood with an analytic gradient.
#
# Identifiers
#   col_group  integration unit (a person). The likelihood factorises here.
#   col_id     choice set: the rows one softmax runs over. Nested in col_group.
#   level      per random term: the column whose values index that term's draws
#              *within* a group. The default is the group itself, i.e. a
#              person-level coefficient. A column such as a neighborhood code
#              gives a person-by-neighborhood effect, shared by every row of the
#              person that carries that value.
#
# A cell is one (group, term, level value); each cell has its own draws.
#
# Utility of row t of group i at draw r:
#
#     v_tr = x_t' beta + sum_m z_tm * A_m(cell_m(t), r)
#
#     :normal, :uniform, :triangular   A = sigma_m * zeta     (zeta standardised)
#     :lognormal                       A = +exp(beta_m + sigma_m * eta)
#     :neg_lognormal                   A = -exp(beta_m + sigma_m * eta)
#
# For a term outside any correlation block zeta = eta, the cell's own draw. For
# a term inside a block, zeta_j = sum_k L[j,k] * eta_k, where L is the Cholesky
# factor of the block's correlation matrix and eta_k are the draws of the
# block's terms in the same group.
#
#     log L_i = log( (1/R) * sum_r prod_{choice sets of i} P(chosen | r) )
#     Q(theta) = - sum_i w_i * log L_i                              (minimised)
#     theta = [beta (K, formula order); sigma (M, rfx order); block coords (B)]
#
# Draws are antithetic: draw r + R/2 is the negative of draw r. Only the first
# half is stored (P.eta, H x ncells with H = R/2).
###############################################################################

const _MRFX_LINEAR_DISTS = (:normal, :uniform, :triangular)
const _MRFX_DISTS = (_MRFX_LINEAR_DISTS..., :lognormal, :neg_lognormal)

# In the optimiser's coordinates sigma = floor + softplus(phi) and each block
# coordinate is LIMIT * tanh(phi). The margin below one keeps sqrt(1 - z^2) and
# its derivative finite.
const _MRFX_SIGMA_FLOOR = eps(Float64)
const _MRFX_CORR_LIMIT  = 1.0 - sqrt(eps(Float64))

_mrfx_islog(d::Symbol) = d === :lognormal || d === :neg_lognormal


# ----------------------------------------------------------------------------
# Terms
# ----------------------------------------------------------------------------

"""
One random-coefficient term, after validation.

`var === nothing` is a random intercept (loading 1 on every row). `col` is the
formula position of `var` when its mean is estimated there, else 0. `at_group`
is true when the term's draws are indexed by the group alone.
"""
struct RfxTerm
    var::Union{Nothing, Symbol}
    level::Symbol
    dist::Symbol
    at_group::Bool
    col::Int
    name::String
end

"""
    rfx_term(var = nothing; level = nothing, dist = :normal, mean = true)

Describe one random-coefficient term for [`mlogit_rfx_prep`](@ref).

- `var`: the regressor whose coefficient is random. `nothing` is a random
  intercept at `level`, with mean zero.
- `level`: the column indexing the draws within a group. `nothing` (default)
  means the group itself: one draw per person. Any other column gives one draw
  per (person, value), e.g. a person-by-neighborhood effect.
- `dist`: `:normal` (default), `:uniform`, `:triangular`, `:lognormal` or
  `:neg_lognormal`. The first three are standardised, so `sigma` is the
  coefficient's standard deviation under each. The lognormal families model the
  whole coefficient as `+-exp(mu + sigma * eta)`; `mu` is the formula
  coefficient of `var`.
- `mean`: whether `var` is also in `formula`, carrying the coefficient's mean.
  With `mean = false` the term is a mean-zero random loading on `var`, and `var`
  need not be in `formula`.

A bare `Symbol` `:x` in `rfx` is shorthand for `rfx_term(:x)`, and `:x => :dist`
for `rfx_term(:x; dist = :dist)`.
"""
function rfx_term(var = nothing; level = nothing, dist = :normal, mean::Bool = true)
    return (var   = isnothing(var)   ? nothing : Symbol(var),
            level = isnothing(level) ? nothing : Symbol(level),
            dist  = Symbol(dist),
            mean  = mean)
end

function _mrfx_terms(rfx, formula_syms::Vector{Symbol}, col_group::Symbol)
    terms = RfxTerm[]
    for (j, r) in enumerate(rfx)
        v, lv, d, has_mean = if r isa Symbol || r isa AbstractString
            (Symbol(r), nothing, :normal, true)
        elseif r isa Pair
            (Symbol(first(r)), nothing, Symbol(last(r)), true)
        elseif r isa NamedTuple && haskey(r, :var)
            (r.var, get(r, :level, nothing), get(r, :dist, :normal), get(r, :mean, true))
        else
            error("rfx entry $j is $(repr(r)). Use :x, :x => :lognormal, or rfx_term(...).")
        end

        d in _MRFX_DISTS || error(
            "rfx entry $j: distribution :$d is not supported. Use one of $(_MRFX_DISTS).")

        level    = isnothing(lv) ? col_group : lv
        at_group = level === col_group

        col = 0
        if !isnothing(v)
            k = findfirst(==(v), formula_syms)
            if has_mean
                isnothing(k) && error(
                    "rfx variable :$v is not in formula. Add it to formula, or use " *
                    "rfx_term(:$v; mean = false) for a mean-zero random loading.")
                col = k
            end
        end

        _mrfx_islog(d) && col == 0 && error(
            "rfx entry $j is :$d but has no mean in formula. A lognormal coefficient " *
            "is exp(mu + sigma*eta) and needs its variable in formula with mean = true.")

        name = isnothing(v) ? "1|$(level)" : at_group ? string(v) : "$(v)|$(level)"
        push!(terms, RfxTerm(v, level, d, at_group, col, name))
    end

    names = [t.name for t in terms]
    allunique(names) || error(
        "rfx repeats a term: $(unique(filter(n -> count(==(n), names) > 1, names))). " *
        "Two terms on the same variable and level are not separately identified.")

    for t in terms
        _mrfx_islog(t.dist) || continue
        any(u -> u !== t && u.var === t.var, terms) && error(
            "rfx variable :$(t.var) is :$(t.dist) and also has another term. A " *
            "lognormal term is the whole coefficient, so no other term can share it.")
    end
    return terms
end


# ----------------------------------------------------------------------------
# Correlation blocks
# ----------------------------------------------------------------------------

"""
    rfx_corr_block(terms...; zero = [])

A block of correlated person-level random coefficients, for the `rfx_corr`
keyword of [`mlogit_rfx_prep`](@ref). Terms are named as in `theta_names`
without the `sd_` prefix (or given by their position in `rfx`).

The block's correlation matrix is `C = L * L'`, where `L` is lower triangular
with rows of unit length. With `zero = []` (the default) `C` is unrestricted, so
`rfx_corr_block(:a, :b)` is an ordinary correlated pair. `zero` lists entries of
`L` fixed at zero, as pairs of terms `(a, b)`; the term that comes later in the
block gives the row. Zeros encode conditional independence. For example, with

    rfx_corr_block(:F, :V, :AV, :Y, :W; zero = [(:Y, :V), (:Y, :AV), (:W, :V), (:W, :AV)])

the correlation matrices of `(F, V, AV)` and of `(F, Y, W)` are both
unrestricted, and `(V, AV)` and `(Y, W)` are independent conditional on `F`.

Each free entry of `L` adds one parameter, a coordinate in `(-1, 1)`. In row
`j`, with free columns `c_1 < ... < c_n`,

    L[j, c_l] = z_l * prod_{q < l} sqrt(1 - z_q^2),   L[j, j] = prod_q sqrt(1 - z_q^2).

In an unrestricted block `z_l` is the partial correlation of term `j` with term
`c_l` given terms `c_1 ... c_{l-1}`, hence the `pcor_` prefix in `theta_names`.
Use [`mlogit_rfx_correlation`](@ref) or [`mlogit_rfx_covariance`](@ref) to
report the implied correlations and covariances.

All terms in a block must be person-level (no `level`) slopes of the same linear
family, and blocks must be disjoint. Terms outside every block, including all
person-by-level terms, are independent of everything else. The order of the
terms matters only for the meaning of `zero`.
"""
function rfx_corr_block(terms...; zero = [])
    length(terms) >= 2 || error("rfx_corr_block needs at least two terms")
    return (terms = collect(terms), zero = collect(zero))
end

"""
A validated correlation block. `free[j]` lists the block columns `c < j` whose
`L[j, c]` is estimated. Coordinates are ordered row by row, by column within a
row; coordinate `h` sits at `L[coord_row[h], coord_col[h]]` and is
`theta[K + M + off + h]`. The block's terms occupy columns
`first_col : first_col + k - 1` of `P.block_cells`.
"""
struct RfxBlock
    ids::Vector{Int}
    free::Vector{Vector{Int}}
    coord_row::Vector{Int}
    coord_col::Vector{Int}
    off::Int
    first_col::Int
end

function _mrfx_blocks(rfx_corr, terms::Vector{RfxTerm})
    function resolve(x)
        if x isa Integer
            1 <= x <= length(terms) || error("rfx_corr term index $x is out of range")
            return Int(x)
        end
        hits = findall(t -> t.name == string(x), terms)
        length(hits) == 1 || error(
            "rfx_corr term $(repr(x)) matches no term. Term names: $([t.name for t in terms])")
        return only(hits)
    end

    blocks = RfxBlock[]
    used   = Set{Int}()
    off    = 0
    col    = 1
    for (b, entry) in enumerate(rfx_corr)
        members, zero = if entry isa NamedTuple && haskey(entry, :terms)
            (collect(entry.terms), collect(entry.zero))
        elseif entry isa Pair
            (Any[first(entry), last(entry)], [])
        elseif entry isa Tuple || entry isa AbstractVector
            (collect(entry), [])
        else
            error("rfx_corr entry $b is $(repr(entry)); use a tuple of terms or rfx_corr_block.")
        end

        ids = Int[resolve(x) for x in members]
        k   = length(ids)
        k >= 2 || error("rfx_corr entry $b has fewer than two terms")
        allunique(ids) || error("rfx_corr entry $b repeats a term")
        for m in ids
            t = terms[m]
            m in used && error("rfx_corr blocks must be disjoint; $(t.name) is in two blocks")
            t.at_group || error(
                "rfx_corr term $(t.name) has a level. Correlation blocks take person-level terms only.")
            isnothing(t.var) && error("rfx_corr term $(t.name) is an intercept")
            t.dist in _MRFX_LINEAR_DISTS || error(
                "rfx_corr term $(t.name) is :$(t.dist); lognormal terms cannot be correlated")
            t.dist === terms[ids[1]].dist || error(
                "rfx_corr entry $b mixes distributions; all terms in a block need the same one")
            push!(used, m)
        end

        position(x) = begin
            p = x isa Integer ? Int(x) : findfirst(==(resolve(x)), ids)
            (isnothing(p) || !(1 <= p <= k)) && error(
                "rfx_corr entry $b: zero entry $(repr(x)) is not a term of the block")
            p
        end
        zeros_ = Set{Tuple{Int,Int}}()
        for z in zero
            length(z) == 2 || error("rfx_corr entry $b: each zero must be a pair of terms")
            p, q = position(z[1]), position(z[2])
            p == q && error("rfx_corr entry $b: a zero pairs a term with itself")
            push!(zeros_, (max(p, q), min(p, q)))
        end

        free = [Int[c for c in 1:j-1 if !((j, c) in zeros_)] for j in 1:k]
        coord_row = Int[]; coord_col = Int[]
        for j in 2:k, c in free[j]
            push!(coord_row, j); push!(coord_col, c)
        end
        push!(blocks, RfxBlock(ids, free, coord_row, coord_col, off, col))
        off += length(coord_row)
        col += k
    end
    return blocks
end

_mrfx_ncoords(blocks) = isempty(blocks) ? 0 : sum(b -> length(b.coord_row), blocks)

"""
    _mrfx_block_factor!(L, J, blk, z)

Fill the block's Cholesky factor `L` (k x k) and its derivatives `J[:, :, h]`
with respect to coordinate `h` (only row `coord_row[h]` is nonzero), at block
coordinates `z`.
"""
function _mrfx_block_factor!(L::Matrix{Float64}, J::Array{Float64,3}, blk::RfxBlock, z)
    k = length(blk.ids)
    fill!(L, 0.0)
    fill!(J, 0.0)
    L[1, 1] = 1.0
    h0 = 0
    @inbounds for j in 2:k
        cols = blk.free[j]
        n = length(cols)
        p = 1.0
        for l in 1:n
            L[j, cols[l]] = z[h0 + l] * p
            p *= sqrt(1.0 - z[h0 + l]^2)
        end
        L[j, j] = p

        for l in 1:n
            h  = h0 + l
            zl = z[h]
            dl = -zl / sqrt(1.0 - zl^2)       # d sqrt(1 - z_l^2) / d z_l, over itself
            before = 1.0                       # prod_{q < l} sqrt(1 - z_q^2)
            for q in 1:l-1
                before *= sqrt(1.0 - z[h0 + q]^2)
            end
            J[j, cols[l], h] = before
            run = before                       # prod over q' < q, q' != l
            for q in l+1:n
                J[j, cols[q], h] = z[h0 + q] * run * dl
                run *= sqrt(1.0 - z[h0 + q]^2)
            end
            J[j, j, h] = run * dl
        end
        h0 += n
    end
    return nothing
end


# ----------------------------------------------------------------------------
# Draws
# ----------------------------------------------------------------------------

"""Inverse CDF of the symmetric triangular distribution with unit variance."""
@inline _mrfx_std_triangular(u::Float64) =
    u < 0.5 ? sqrt(6.0) * (sqrt(2.0 * u) - 1.0) : sqrt(6.0) * (1.0 - sqrt(2.0 * (1.0 - u)))

"""
First antithetic half of the standardised draws, `H x ncells`, cell by cell.
Lognormal cells draw a standard normal (their randomness is normal on the log
scale). All families are symmetric about zero, so `-eta` is a valid draw too.
"""
function _mrfx_draws(cell_dist::Vector{Symbol}, H::Int, seed::Int)
    rng = MersenneTwister(seed)
    eta = Matrix{Float64}(undef, H, length(cell_dist))
    for c in eachindex(cell_dist)
        d = cell_dist[c]
        for r in 1:H
            eta[r, c] = d === :uniform    ? sqrt(3.0) * (2.0 * rand(rng) - 1.0) :
                        d === :triangular ? _mrfx_std_triangular(rand(rng)) :
                                            randn(rng)
        end
    end
    return eta
end


# ----------------------------------------------------------------------------
# Prepared model
# ----------------------------------------------------------------------------

"""
A prepared `mlogit_rfx` model: data in numeric form, cells and draws. Built by
[`mlogit_rfx_prep`](@ref), then passed to [`mlogit_rfx_fit`](@ref),
[`boot_mlogit_rfx`](@ref) and the other `mlogit_rfx_*` functions. Never holds
the DataFrame, so it can be shipped to worker processes.

Rows are sorted by (group, choice set, original position), so groups and
choice sets are contiguous ranges. `cellloc[t, m]` is the cell of row `t` for
term `m`, numbered within the row's group.
"""
struct MlogitRfxPrep
    X::Matrix{Float64}                    # Nobs x K mean regressors; lognormal columns zeroed
    Z::Matrix{Float64}                    # Nobs x M loadings
    y::Vector{Bool}                       # Nobs, selected
    cellloc::Matrix{Int}                  # Nobs x M, cell within the group
    ranges::Vector{UnitRange{Int}}        # N, rows of each group
    set_ranges::Vector{UnitRange{Int}}    # rows of each choice set
    set_of_group::Vector{UnitRange{Int}}  # N, choice sets of each group
    sel_row::Vector{Int}                  # selected row of each choice set
    cell_ranges::Vector{UnitRange{Int}}   # N, global cells of each group
    cell_term::Vector{Int}                # term of each global cell
    block_cells::Matrix{Int}              # N x (terms in blocks), cell within the group
    eta::Matrix{Float64}                  # H x ncells, first antithetic half of the draws
    group_ids::Vector                     # N, values of col_group, sorted
    group_weights::Union{Nothing, Vector{Float64}}
    K::Int
    M::Int
    B::Int                                # number of block coordinates
    R::Int                                # draws (even)
    H::Int                                # R / 2
    N::Int                                # groups
    Tmax::Int                             # max rows per group
    Cmax::Int                             # max cells per group
    Smax::Int                             # max choice sets per group
    terms::Vector{RfxTerm}
    blocks::Vector{RfxBlock}
    term_islog::Vector{Bool}
    term_sign::Vector{Float64}
    term_col::Vector{Int}
    any_log::Bool
    theta_names::Vector{String}
    col_id::Symbol
    col_group::Symbol
    seed::Int
    kernel::Symbol                        # :general or :binary
    cell_stats::Vector{NamedTuple}
    # --- :binary kernel only (empty otherwise) ---------------------------------
    set_ptr::Vector{Int}                  # n_sets + 1, pointers into set_loc / set_dz
    set_loc::Vector{Int}                  # cell (within group) of each signed loading
    set_dz::Vector{Float64}               # loading of row 1 minus loading of row 2
    dX::Matrix{Float64}                   # n_sets x K, X(row 1) - X(row 2)
    ysel::Vector{Bool}                    # n_sets, is row 1 the selected one
end

function _mrfx_blocks_of_equal(v)
    out = UnitRange{Int}[]
    isempty(v) && return out
    start = 1
    for t in 2:length(v)
        if !isequal(v[t], v[t-1])
            push!(out, start:(t-1))
            start = t
        end
    end
    push!(out, start:length(v))
    return out
end

"""
    mlogit_rfx_prep(data_df, formula, col_id, col_selected; kwargs...) -> MlogitRfxPrep

Prepare a random-coefficient logit for estimation. `data_df` is in long format,
one row per (choice set, option), and is not modified.

# Arguments
- `formula`: vector of regressor names; their coefficients are the means.
- `col_id`: choice-set identifier, unique across groups.
- `col_selected`: 0/1 column, exactly one 1 per choice set.

# Keywords
- `col_group = nothing`: integration unit (person). Defaults to `col_id`.
- `rfx = []`: random terms; see [`rfx_term`](@ref). With no terms the model is a
  plain conditional logit.
- `rfx_corr = []`: correlation blocks; see [`rfx_corr_block`](@ref). A tuple of
  term names is an unrestricted block.
- `ndraws = 1000`: simulation draws per cell, even. Draws are antithetic and
  fixed, so the objective is a smooth deterministic function of `theta`.
- `seed = 20260808`: draw seed.
- `weights = nothing`: optional weight column, constant within each group.
- `kernel = :auto`: likelihood kernel. `:general` handles any choice-set size
  and all distributions. `:binary` requires two options per choice set and no
  lognormal term; it computes each antithetic pair of draws at once and is
  several times faster. `:auto` uses `:binary` when it applies.
"""
function mlogit_rfx_prep(data_df, formula, col_id, col_selected;
                         col_group = nothing,
                         rfx = [],
                         rfx_corr = [],
                         ndraws::Int = 1000,
                         seed::Int = 20260808,
                         weights = nothing,
                         kernel::Symbol = :auto)

    kernel in (:auto, :general, :binary) || error("kernel must be :auto, :general or :binary")
    col_id    = Symbol(col_id)
    col_group = isnothing(col_group) ? col_id : Symbol(col_group)
    col_sel   = Symbol(col_selected)

    formula = Symbol.(collect(formula))
    K = length(formula)
    K > 0 || error("formula is empty")
    terms  = _mrfx_terms(rfx, formula, col_group)
    M      = length(terms)
    blocks = _mrfx_blocks(rfx_corr, terms)
    B      = _mrfx_ncoords(blocks)

    # With no random terms the draws are irrelevant: use the minimum.
    R = M == 0 ? 2 : ndraws
    (R >= 2 && iseven(R)) || error("ndraws must be even and at least 2; got $ndraws")
    H = R ÷ 2

    # --- columns ------------------------------------------------------------
    dfnames = Symbol.(names(data_df))
    needed  = vcat(formula, col_sel, col_id, col_group, [t.level for t in terms],
                   Symbol[t.var for t in terms if !isnothing(t.var)])
    isnothing(weights) || push!(needed, Symbol(weights))
    for c in unique(needed)
        c in dfnames || error("column :$c not found in data_df")
    end
    Nobs = nrow(data_df)
    Nobs > 0 || error("data_df has no rows")

    # --- sort by (group, set, original position) -----------------------------
    graw = data_df[!, col_group]
    sraw = data_df[!, col_id]
    perm = sortperm(1:Nobs, by = t -> (graw[t], sraw[t], t))
    gv   = graw[perm]
    sv   = sraw[perm]

    X = Matrix{Float64}(undef, Nobs, K)
    for (k, x) in enumerate(formula)
        X[:, k] .= Float64.(data_df[perm, x])
    end
    ysel_raw = data_df[perm, col_sel]
    all(v -> v == 0 || v == 1, ysel_raw) || error("column :$col_sel must be 0/1")
    y = Bool.(ysel_raw .== 1)

    # --- groups and choice sets ----------------------------------------------
    ranges    = _mrfx_blocks_of_equal(gv)
    N         = length(ranges)
    group_ids = [gv[first(rg)] for rg in ranges]

    set_ranges   = UnitRange{Int}[]
    set_of_group = Vector{UnitRange{Int}}(undef, N)
    for (i, rg) in enumerate(ranges)
        s0 = length(set_ranges) + 1
        for b in _mrfx_blocks_of_equal(view(sv, rg))
            push!(set_ranges, (first(rg) + first(b) - 1):(first(rg) + last(b) - 1))
        end
        set_of_group[i] = s0:length(set_ranges)
    end
    n_sets = length(set_ranges)
    allunique(sv[first(sr)] for sr in set_ranges) || error(
        "a value of :$col_id appears under more than one :$col_group. Choice sets " *
        "must be nested in groups; make :$col_id unique across groups.")

    sel_row = Vector{Int}(undef, n_sets)
    for (s, sr) in enumerate(set_ranges)
        nsel = count(t -> y[t], sr)
        nsel == 1 || error("choice set :$col_id = $(sv[first(sr)]) has $nsel selected options; " *
                           "exactly one is required")
        sel_row[s] = sr[findfirst(t -> y[t], sr)]
    end
    n_single = count(sr -> length(sr) == 1, set_ranges)
    n_single == n_sets && error("every choice set has a single option")
    n_single > 0 && @warn "$n_single of $n_sets choice sets have a single option and " *
                          "contribute nothing to the likelihood."

    # --- cells: one per (group, term, level value) ---------------------------
    # Term-major within each group, sorted level values within a term, so the
    # draw a cell receives does not depend on the input row order.
    lvals = [t.at_group ? gv : data_df[perm, t.level] for t in terms]
    cellloc     = zeros(Int, Nobs, M)
    cellidx     = zeros(Int, Nobs, M)          # global, for the diagnostics
    cell_term   = Int[]
    cell_ranges = Vector{UnitRange{Int}}(undef, N)
    for (i, rg) in enumerate(ranges)
        c0   = length(cell_term) + 1
        base = 0
        for m in 1:M
            vals = sort!(unique(view(lvals[m], rg)))
            loc  = Dict(v => base + j for (j, v) in enumerate(vals))
            append!(cell_term, fill(m, length(vals)))
            for t in rg
                cellloc[t, m] = loc[lvals[m][t]]
                cellidx[t, m] = c0 + cellloc[t, m] - 1
            end
            base += length(vals)
        end
        cell_ranges[i] = c0:length(cell_term)
    end

    nblockterms = isempty(blocks) ? 0 : sum(b -> length(b.ids), blocks)
    block_cells = zeros(Int, N, nblockterms)
    for (i, rg) in enumerate(ranges), blk in blocks, (j, m) in enumerate(blk.ids)
        block_cells[i, blk.first_col + j - 1] = cellloc[first(rg), m]
    end

    # --- loadings; lognormal columns leave the linear index -------------------
    Z = Matrix{Float64}(undef, Nobs, M)
    for (m, t) in enumerate(terms)
        Z[:, m] .= isnothing(t.var) ? 1.0 :
                   t.col > 0        ? X[:, t.col] : Float64.(data_df[perm, t.var])
    end
    term_islog = [_mrfx_islog(t.dist) for t in terms]
    term_sign  = [t.dist === :neg_lognormal ? -1.0 : 1.0 for t in terms]
    term_col   = [t.col for t in terms]
    for m in findall(term_islog)
        X[:, term_col[m]] .= 0.0
    end

    cell_stats = _mrfx_cell_stats(terms, cellidx, Z, ranges, set_ranges,
                                  length(cell_term), col_id)

    # --- weights --------------------------------------------------------------
    group_weights = nothing
    if !isnothing(weights)
        wv = Float64.(data_df[perm, Symbol(weights)])
        group_weights = [wv[first(rg)] for rg in ranges]
        for (i, rg) in enumerate(ranges)
            all(==(group_weights[i]), view(wv, rg)) || error(
                "weights must be constant within :$col_group (group $(group_ids[i]))")
        end
    end

    eta = _mrfx_draws(Symbol[terms[m].dist for m in cell_term], H, seed)

    theta_names = vcat(string.(formula), ["sd_" * t.name for t in terms])
    for blk in blocks, h in eachindex(blk.coord_row)
        a = terms[blk.ids[blk.coord_row[h]]].name
        b = terms[blk.ids[blk.coord_col[h]]].name
        push!(theta_names, "pcor_$(a)__$(b)")
    end

    # --- kernel ----------------------------------------------------------------
    two_rows = all(sr -> length(sr) == 2, set_ranges)
    binary_ok = two_rows && !any(term_islog)
    mode = kernel === :auto ? (binary_ok ? :binary : :general) : kernel
    mode === :binary && !binary_ok && error(
        "kernel = :binary needs exactly two options per choice set and no lognormal term")

    set_ptr = ones(Int, n_sets + 1)
    set_loc = Int[]
    set_dz  = Float64[]
    dX      = Matrix{Float64}(undef, 0, K)
    ysel    = Bool[]
    if mode === :binary
        # Merge the two rows' loadings per cell into one signed list, dropping
        # exact cancellations (e.g. both options familiar).
        acc = Dict{Int,Float64}()
        for s in 1:n_sets
            lo = first(set_ranges[s]); hi = lo + 1
            empty!(acc)
            for m in 1:M
                Z[lo, m] != 0 && (acc[cellloc[lo, m]] = get(acc, cellloc[lo, m], 0.0) + Z[lo, m])
                Z[hi, m] != 0 && (acc[cellloc[hi, m]] = get(acc, cellloc[hi, m], 0.0) - Z[hi, m])
            end
            for c in sort!(collect(keys(acc)))
                acc[c] == 0.0 && continue
                push!(set_loc, c)
                push!(set_dz, acc[c])
            end
            set_ptr[s + 1] = length(set_loc) + 1
        end
        dX   = X[first.(set_ranges), :] .- X[first.(set_ranges) .+ 1, :]
        ysel = y[first.(set_ranges)]
    end

    return MlogitRfxPrep(
        X, Z, y, cellloc, ranges, set_ranges, set_of_group, sel_row,
        cell_ranges, cell_term, block_cells, eta, group_ids, group_weights,
        K, M, B, R, H, N,
        maximum(length, ranges),
        M == 0 ? 0 : maximum(length, cell_ranges),
        maximum(length, set_of_group),
        terms, blocks, term_islog, term_sign, term_col, any(term_islog),
        theta_names, col_id, col_group, seed, mode, cell_stats,
        set_ptr, set_loc, set_dz, dX, ysel)
end

"""
Per-term cell diagnostics, plus the identification guards that use them:

1. A loading that is zero everywhere, or a term that cancels in every choice set
   (all options of the set share one cell and one loading, e.g. a person-level
   random intercept), is not identified: error. Cancelling in most sets: warning.
2. A term whose every cell holds a single row perturbs one utility and is not
   separable from the logit error: error. A median below two rows: warning.
3. A slope whose loading barely varies within its cells, next to a random
   intercept at the same level, is a rescaled intercept: warning.
"""
function _mrfx_cell_stats(terms, cellidx, Z, ranges, set_ranges, ncells::Int, col_id)
    stats = NamedTuple[]
    intercept_levels = Set(t.level for t in terms if isnothing(t.var))
    for (m, t) in enumerate(terms)
        rows_per_cell = zeros(Int, ncells)
        for tt in axes(cellidx, 1)
            rows_per_cell[cellidx[tt, m]] += 1
        end
        rpc = rows_per_cell[rows_per_cell .> 0]
        informative = falses(ncells)
        for tt in axes(cellidx, 1)
            Z[tt, m] == 0.0 || (informative[cellidx[tt, m]] = true)
        end
        cpg = [count(c -> informative[c], unique(view(cellidx, rg, m))) for rg in ranges]

        n_cancel = 0
        for sr in set_ranges
            c1, z1 = cellidx[first(sr), m], Z[first(sr), m]
            all_zero = all(tt -> Z[tt, m] == 0.0, sr)
            same     = all(tt -> cellidx[tt, m] == c1 && Z[tt, m] == z1, sr)
            (all_zero || same) && (n_cancel += 1)
        end
        cancel_share = n_cancel / length(set_ranges)

        z = view(Z, :, m)
        tot = sum(abs2, z .- mean(z))
        within_share = 0.0
        if tot > 0
            sums = zeros(ncells)
            for tt in eachindex(z)
                sums[cellidx[tt, m]] += z[tt]
            end
            within_share = sum(abs2(z[tt] - sums[cellidx[tt, m]] / rows_per_cell[cellidx[tt, m]])
                               for tt in eachindex(z)) / tot
        end

        push!(stats, (; term = t.name, level = t.level, dist = t.dist,
                        n_cells = length(rpc), n_cells_informative = count(informative),
                        informative_cells_per_group_median = median(cpg),
                        informative_cells_per_group_max = maximum(cpg),
                        rows_per_cell_median = median(rpc), rows_per_cell_max = maximum(rpc),
                        cancel_share = cancel_share, within_cell_var_share = within_share))

        all(iszero, z) && error("rfx term $(t.name) has a loading that is zero on every row")
        cancel_share == 1.0 && error(
            "rfx term $(t.name) cancels in every choice set: all options of each set share " *
            "one draw and one loading, so it drops out of the softmax. " *
            (isnothing(t.var) ? "A random intercept needs a level that varies within " *
                                "choice sets (e.g. the option's identifier)." :
                                "Its loading must vary across the options of a choice set."))
        cancel_share > 0.5 && @warn "rfx term $(t.name) cancels in " *
            "$(round(100 * cancel_share, digits = 1))% of choice sets; its sigma is " *
            "identified from the rest only."
        maximum(rpc) == 1 && error(
            "rfx term $(t.name) has one row per (group, :$(t.level)) cell, so it is not " *
            "separable from the logit error")
        median(rpc) < 2 && @warn "rfx term $(t.name) has a median of $(median(rpc)) row(s) " *
            "per cell: weakly identified."
        if !isnothing(t.var) && t.level in intercept_levels && within_share < 0.01
            @warn "rfx term $(t.name) barely varies within its cells and the model has a " *
                  "random intercept at :$(t.level); the two sigmas will trade off."
        end
    end
    return stats
end

"""
    rfx_cell_report(P) -> DataFrame

One row per random term: cell counts (overall and with a nonzero loading),
informative cells per group (summed over terms, the dimension of the simulated
integral), rows per cell, the share of choice sets in which the term cancels,
and the share of its loading's variance that is within cells.
"""
rfx_cell_report(P::MlogitRfxPrep) = isempty(P.cell_stats) ? DataFrame() : DataFrame(P.cell_stats)


# ----------------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------------

"""
Scratch space for one likelihood evaluation, reused across groups. Draw-major
(draws in rows), so the inner loops over draws are contiguous.
"""
struct MlogitRfxBuffers
    zeta::Matrix{Float64}        # H x Cmax   standardised draws after block mixing
    A::Matrix{Float64}           # R x Cmax   per-cell coefficient
    S::Matrix{Float64}           # R x Cmax   per-cell score: sum over rows of e_t * z_t
    E::Matrix{Float64}           # R x max(Tmax, Smax) residuals (rows, or sets for :binary)
    v::Vector{Float64}           # Tmax       utilities at one draw
    xb::Vector{Float64}          # max(Tmax, Smax)
    ebar::Vector{Float64}        # max(Tmax, Smax)
    g::Vector{Float64}           # H          random part of one set's utility difference
    ll::Vector{Float64}          # R          log-likelihood of the group at each draw
    tau::Vector{Float64}         # R          posterior weights of the draws
    L::Vector{Matrix{Float64}}   # Cholesky factor of each block
    J::Vector{Array{Float64,3}}  # its derivatives
end

function MlogitRfxBuffers(P::MlogitRfxPrep)
    TS = max(P.Tmax, P.Smax)
    MlogitRfxBuffers(
        zeros(P.H, P.Cmax), zeros(P.R, P.Cmax), zeros(P.R, P.Cmax), zeros(P.R, TS),
        zeros(P.Tmax), zeros(TS), zeros(TS), zeros(P.H), zeros(P.R), zeros(P.R),
        [zeros(length(b.ids), length(b.ids)) for b in P.blocks],
        [zeros(length(b.ids), length(b.ids), length(b.coord_row)) for b in P.blocks])
end

function _mrfx_block_factors!(buf::MlogitRfxBuffers, theta, P::MlogitRfxPrep)
    for (b, blk) in enumerate(P.blocks)
        z = view(theta, P.K + P.M + blk.off .+ (1:length(blk.coord_row)))
        _mrfx_block_factor!(buf.L[b], buf.J[b], blk, z)
    end
    return nothing
end

"""Fill `buf.zeta` (first half) and `buf.A` (all draws) for group `i`."""
function _mrfx_fill_A!(buf::MlogitRfxBuffers, theta, P::MlogitRfxPrep, i::Int)
    K, H = P.K, P.H
    cr = P.cell_ranges[i]
    c0 = first(cr) - 1
    Ci = length(cr)
    zeta, A, eta = buf.zeta, buf.A, P.eta

    @inbounds for c in 1:Ci, r in 1:H
        zeta[r, c] = eta[r, c0 + c]
    end
    @inbounds for (b, blk) in enumerate(P.blocks)
        L = buf.L[b]
        cells = view(P.block_cells, i, blk.first_col:(blk.first_col + length(blk.ids) - 1))
        for j in 2:length(blk.ids)
            cj = cells[j]
            for r in 1:H
                s = 0.0
                for q in 1:j
                    s += L[j, q] * eta[r, c0 + cells[q]]
                end
                zeta[r, cj] = s
            end
        end
    end

    @inbounds for c in 1:Ci
        m = P.cell_term[c0 + c]
        s = theta[K + m]
        if P.term_islog[m]
            mu = theta[P.term_col[m]]
            sg = P.term_sign[m]
            for r in 1:H
                A[r, c]     = sg * exp(mu + s * zeta[r, c])
                A[r + H, c] = sg * exp(mu - s * zeta[r, c])
            end
        else
            for r in 1:H
                a = s * zeta[r, c]
                A[r, c]     = a
                A[r + H, c] = -a
            end
        end
    end

    if P.any_log && !all(isfinite, view(A, :, 1:Ci))
        error("a lognormal coefficient overflowed: exp(mu + sigma*eta) is not finite. " *
              "mu is on the log scale; build starting values with theta0_mlogit_rfx.")
    end
    return nothing
end

"""
Likelihood pass for group `i`, any choice-set size: fills `buf.ll` and, when
`need_g`, the row residuals `buf.E` and per-cell scores `buf.S`.
"""
function _mrfx_pass_general!(buf::MlogitRfxBuffers, theta, P::MlogitRfxPrep, i::Int,
                             need_g::Bool)
    M, R = P.M, P.R
    rows = P.ranges[i]
    r0   = first(rows) - 1
    xb   = view(buf.xb, 1:length(rows))
    mul!(xb, view(P.X, rows, :), view(theta, 1:P.K))
    v, A, S, E, ll = buf.v, buf.A, buf.S, buf.E, buf.ll
    need_g && fill!(view(S, :, 1:length(P.cell_ranges[i])), 0.0)

    @inbounds for r in 1:R
        acc = 0.0
        for s in P.set_of_group[i]
            sr = P.set_ranges[s]
            mx = -Inf
            for t in sr
                u = xb[t - r0]
                for m in 1:M
                    u += P.Z[t, m] * A[r, P.cellloc[t, m]]
                end
                v[t - r0] = u
                mx = max(mx, u)
            end
            se = 0.0
            for t in sr
                se += exp(v[t - r0] - mx)
            end
            lse = mx + log(se)
            acc += v[P.sel_row[s] - r0] - lse
            if need_g
                for t in sr
                    e = (P.y[t] ? 1.0 : 0.0) - exp(v[t - r0] - lse)
                    E[r, t - r0] = e
                    for m in 1:M
                        S[r, P.cellloc[t, m]] += e * P.Z[t, m]
                    end
                end
            end
        end
        ll[r] = acc
    end
    return nothing
end

"""Log probability of the chosen option and residual `y1 - P(1)`, for `d = v1 - v2`."""
@inline function _mrfx_binary_logit(d::Float64, y1::Bool)
    if d >= 0.0
        ed = exp(-d)
        lp1 = -log1p(ed)                       # log P(1)
        return (y1 ? lp1 : lp1 - d), (y1 ? 1.0 : 0.0) - 1.0 / (1.0 + ed)
    else
        ed = exp(d)
        lp2 = -log1p(ed)                       # log P(2)
        return (y1 ? lp2 + d : lp2), (y1 ? 1.0 : 0.0) - ed / (1.0 + ed)
    end
end

"""
Likelihood pass for group `i` when every choice set has two options and every
term is linear in its draw. The random part `g` of the utility difference
flips sign between antithetic draws, so it is computed once per pair and both
`dx + g` and `dx - g` are evaluated. Fills `buf.ll` and, when `need_g`, the set
residuals `buf.E` (row 1 of each set) and per-cell scores `buf.S`.
"""
function _mrfx_pass_binary!(buf::MlogitRfxBuffers, theta, P::MlogitRfxPrep, i::Int,
                            need_g::Bool)
    H = P.H
    sets = P.set_of_group[i]
    dx = view(buf.xb, 1:length(sets))
    mul!(dx, view(P.dX, sets, :), view(theta, 1:P.K))
    A, S, E, ll, g = buf.A, buf.S, buf.E, buf.ll, buf.g
    fill!(ll, 0.0)
    need_g && fill!(view(S, :, 1:length(P.cell_ranges[i])), 0.0)

    @inbounds for (js, s) in enumerate(sets)
        k1, k2 = P.set_ptr[s], P.set_ptr[s + 1] - 1
        fill!(g, 0.0)
        for k in k1:k2
            z, c = P.set_dz[k], P.set_loc[k]
            @simd for r in 1:H
                g[r] += z * A[r, c]
            end
        end
        y1, d0 = P.ysel[s], dx[js]
        for r in 1:H
            lp, e = _mrfx_binary_logit(d0 + g[r], y1)
            ll[r] += lp
            need_g && (E[r, js] = e)
            lp, e = _mrfx_binary_logit(d0 - g[r], y1)
            ll[r + H] += lp
            need_g && (E[r + H, js] = e)
        end
        if need_g
            for k in k1:k2
                z, c = P.set_dz[k], P.set_loc[k]
                @simd for r in 1:P.R
                    S[r, c] += E[r, js] * z
                end
            end
        end
    end
    return nothing
end

_mrfx_pass!(buf, theta, P, i, need_g) =
    P.kernel === :binary ? _mrfx_pass_binary!(buf, theta, P, i, need_g) :
                           _mrfx_pass_general!(buf, theta, P, i, need_g)

"""Add `-om * d log L_i / d theta` to `G`, given the pass for group `i` with log L_i = lse - log R."""
function _mrfx_grad_group!(G, theta, P::MlogitRfxPrep, buf::MlogitRfxBuffers, i::Int,
                           om::Float64, lse::Float64)
    K, M, R, H = P.K, P.M, P.R, P.H
    tau = buf.tau
    @inbounds for r in 1:R
        tau[r] = exp(buf.ll[r] - lse)
    end

    # means: collapse the draws first, then one product with the design
    gb = view(G, 1:K)
    if P.kernel === :binary
        sets = P.set_of_group[i]
        eb = view(buf.ebar, 1:length(sets))
        mul!(eb, transpose(view(buf.E, :, 1:length(sets))), tau)
        mul!(gb, transpose(view(P.dX, sets, :)), eb, -om, 1.0)
    else
        rows = P.ranges[i]
        eb = view(buf.ebar, 1:length(rows))
        mul!(eb, transpose(view(buf.E, :, 1:length(rows))), tau)
        mul!(gb, transpose(view(P.X, rows, :)), eb, -om, 1.0)
    end
    M == 0 && return nothing

    cr = P.cell_ranges[i]
    c0 = first(cr) - 1
    S, zeta, A = buf.S, buf.zeta, buf.A

    # standard deviations (and lognormal means). Draw r + H is minus draw r.
    @inbounds for c in 1:length(cr)
        m = P.cell_term[c0 + c]
        if P.term_islog[m]
            as = 0.0; ab = 0.0
            for r in 1:H
                cp = tau[r] * S[r, c] * A[r, c]
                cm = tau[r + H] * S[r + H, c] * A[r + H, c]
                as += (cp - cm) * zeta[r, c]
                ab += cp + cm
            end
            G[K + m] -= om * as
            G[P.term_col[m]] -= om * ab
        else
            acc = 0.0
            @simd for r in 1:H
                acc += (tau[r] * S[r, c] - tau[r + H] * S[r + H, c]) * zeta[r, c]
            end
            G[K + m] -= om * acc
        end
    end

    # block coordinates: d zeta_j / d z_h = sum_q J[j, q, h] * eta_q
    @inbounds for (b, blk) in enumerate(P.blocks)
        J = buf.J[b]
        cells = view(P.block_cells, i, blk.first_col:(blk.first_col + length(blk.ids) - 1))
        for h in eachindex(blk.coord_row)
            j  = blk.coord_row[h]
            cj = cells[j]
            acc = 0.0
            for r in 1:H
                d = 0.0
                for q in 1:j
                    d += J[j, q, h] * P.eta[r, c0 + cells[q]]
                end
                acc += (tau[r] * S[r, cj] - tau[r + H] * S[r + H, cj]) * d
            end
            G[K + M + blk.off + h] -= om * theta[K + blk.ids[j]] * acc
        end
    end
    return nothing
end

"""
    _mrfx_fg!(F, G, theta, P, buf, w) -> objective or nothing

Objective `Q(theta) = -sum_i w_i log L_i` and, when `G !== nothing`, its
gradient in `G`, in the `Optim.only_fg!` convention. `w = nothing` means unit
weights; groups with zero weight are skipped.
"""
function _mrfx_fg!(F, G, theta::AbstractVector{Float64}, P::MlogitRfxPrep,
                   buf::MlogitRfxBuffers, w::Union{Nothing, AbstractVector{Float64}})
    _mrfx_block_factors!(buf, theta, P)
    need_g = G !== nothing
    need_g && fill!(G, 0.0)
    logR = log(P.R)
    Q = 0.0
    for i in 1:P.N
        om = isnothing(w) ? 1.0 : w[i]
        om == 0.0 && continue
        P.M > 0 && _mrfx_fill_A!(buf, theta, P, i)
        _mrfx_pass!(buf, theta, P, i, need_g)
        lse = logsumexp(buf.ll)
        Q -= om * (lse - logR)
        need_g && _mrfx_grad_group!(G, theta, P, buf, i, om, lse)
    end
    return F === nothing ? nothing : Q
end

"""
    mlogit_rfx_ess(theta, P) -> Vector{Float64}

Effective number of draws for each group at `theta`, `1 / sum_r tau_ir^2`,
where `tau_ir` is the posterior weight of draw `r`. A small value means the
simulated integral rests on few draws; the dimension of that integral is the
number of informative cells per group (see [`rfx_cell_report`](@ref)).
"""
function mlogit_rfx_ess(theta::AbstractVector, P::MlogitRfxPrep)
    th  = Vector{Float64}(theta)
    buf = MlogitRfxBuffers(P)
    _mrfx_block_factors!(buf, th, P)
    ess = Vector{Float64}(undef, P.N)
    for i in 1:P.N
        P.M > 0 && _mrfx_fill_A!(buf, th, P, i)
        _mrfx_pass!(buf, th, P, i, false)
        lse = logsumexp(buf.ll)
        ess[i] = 1.0 / sum(r -> exp(2.0 * (buf.ll[r] - lse)), 1:P.R)
    end
    return ess
end

"""
    mlogit_rfx_correlation(theta, P) -> Matrix

Correlation matrix of the `M` random terms implied by the block coordinates in
`theta` (identity outside the blocks).
"""
function mlogit_rfx_correlation(theta::AbstractVector, P::MlogitRfxPrep)
    C = Matrix{Float64}(I, P.M, P.M)
    buf = MlogitRfxBuffers(P)
    _mrfx_block_factors!(buf, theta, P)
    for (b, blk) in enumerate(P.blocks)
        C[blk.ids, blk.ids] .= buf.L[b] * buf.L[b]'
    end
    return C
end

"""
    mlogit_rfx_covariance(theta, P) -> Matrix

Covariance matrix of the `M` random terms, `D * C * D` with `D = diag(sigma)`.
For a lognormal term the entry refers to the coefficient's logarithm.
"""
function mlogit_rfx_covariance(theta::AbstractVector, P::MlogitRfxPrep)
    s = theta[P.K+1:P.K+P.M]
    return mlogit_rfx_correlation(theta, P) .* (s * s')
end


# ----------------------------------------------------------------------------
# Optimiser coordinates
# ----------------------------------------------------------------------------

function _mrfx_to_theta!(theta, phi, K::Int, M::Int, B::Int)
    @inbounds for j in 1:K
        theta[j] = phi[j]
    end
    @inbounds for m in K+1:K+M
        theta[m] = _MRFX_SIGMA_FLOOR + log1pexp(phi[m])
    end
    @inbounds for b in K+M+1:K+M+B
        theta[b] = _MRFX_CORR_LIMIT * tanh(phi[b])
    end
    return theta
end

"""
Unconstrained coordinates of `theta`. Starting standard deviations below
`sd_min` are raised to it and block coordinates are clipped to
`[-corr_max, corr_max]`: at a boundary value the softplus and tanh maps have a
vanishing derivative, and the optimiser could not move away.
"""
function _mrfx_to_phi(theta, K::Int, M::Int, B::Int; sd_min::Float64, corr_max::Float64)
    0.0 < corr_max < _MRFX_CORR_LIMIT || error("corr_max must lie in (0, 1)")
    phi = Vector{Float64}(theta)
    for m in K+1:K+M
        s = max(theta[m], sd_min)
        s > 2 * _MRFX_SIGMA_FLOOR || error(
            "a starting standard deviation is zero; pass a positive start or sd_min > 0")
        phi[m] = logexpm1(s - _MRFX_SIGMA_FLOOR)
    end
    for b in K+M+1:K+M+B
        abs(theta[b]) < 1 || error("block coordinates must lie strictly inside (-1, 1)")
        phi[b] = atanh(clamp(theta[b], -corr_max, corr_max) / _MRFX_CORR_LIMIT)
    end
    return phi
end

function _mrfx_fg_phi!(F, G, phi, P::MlogitRfxPrep, buf, w, theta, gtheta)
    K, M, B = P.K, P.M, P.B
    _mrfx_to_theta!(theta, phi, K, M, B)
    val = _mrfx_fg!(F, G === nothing ? nothing : gtheta, theta, P, buf, w)
    if G !== nothing
        @inbounds for j in 1:K
            G[j] = gtheta[j]
        end
        @inbounds for m in K+1:K+M
            G[m] = gtheta[m] * logistic(phi[m])
        end
        @inbounds for b in K+M+1:K+M+B
            t = tanh(phi[b])
            G[b] = gtheta[b] * _MRFX_CORR_LIMIT * (1.0 - t * t)
        end
    end
    return val
end


# ----------------------------------------------------------------------------
# Starting values
# ----------------------------------------------------------------------------

function _mrfx_theta0(K, terms, B; b0, s0, corr0)
    M = length(terms)
    length(b0) == K || error("b0 has length $(length(b0)); formula has $K variables")
    s = s0 isa Real ? fill(Float64(s0), M) : Float64.(collect(s0))
    c = corr0 isa Real ? fill(Float64(corr0), B) : Float64.(collect(corr0))
    length(s) == M || error("s0 has length $(length(s)); there are $M random terms")
    length(c) == B || error("corr0 has length $(length(c)); there are $B block coordinates")
    all(>(0), s) || error("s0 must be positive")
    all(x -> abs(x) < 1, c) || error("corr0 must lie strictly inside (-1, 1)")

    th = vcat(Float64.(collect(b0)), s, c)
    # A lognormal mean is on the log scale: convert the level start b0 so that
    # E[coefficient] = b0 at the starting sigma.
    for (m, t) in enumerate(terms)
        _mrfx_islog(t.dist) || continue
        b = (t.dist === :neg_lognormal ? -1.0 : 1.0) * th[t.col]
        b > 0 || error("b0 for lognormal term $(t.name) has the wrong sign (or is zero)")
        th[t.col] = log(b) - s[m]^2 / 2
    end
    return th
end

"""
    theta0_mlogit_rfx(P; b0 = zeros(K), s0 = 0.5, corr0 = 0.0) -> Vector
    theta0_mlogit_rfx(formula, rfx; rfx_corr = [], b0, s0, corr0) -> Vector

Starting values `[b0; s0; corr0]`. `b0` are coefficient levels (e.g. from a
plain logit); for a lognormal term it is converted to the log scale. `s0` and
`corr0` may be scalars or vectors.
"""
theta0_mlogit_rfx(P::MlogitRfxPrep; b0 = zeros(P.K), s0 = 0.5, corr0 = 0.0) =
    _mrfx_theta0(P.K, P.terms, P.B; b0, s0, corr0)

function theta0_mlogit_rfx(formula, rfx; rfx_corr = [], b0 = zeros(length(formula)),
                           s0 = 0.5, corr0 = 0.0)
    terms = _mrfx_terms(rfx, Symbol.(collect(formula)), :__group__)
    B = _mrfx_ncoords(_mrfx_blocks(rfx_corr, terms))
    return _mrfx_theta0(length(formula), terms, B; b0, s0, corr0)
end

"""
    theta0_mlogit_rfx_multistart(P; b0, nstarts = 10, s_range = (0.05, 2.0),
                                 corr_range = (-0.5, 0.5), b_jitter = 0.0,
                                 seed = 20260808) -> Matrix

`nstarts x npar` starting values, one per row. Row 1 is
`theta0_mlogit_rfx(P; b0)`. Other rows draw standard deviations log-uniformly
on `s_range`, block coordinates uniformly on `corr_range`, and (if
`b_jitter > 0`) multiply `b0` by `exp(b_jitter * N(0,1))`.
"""
function theta0_mlogit_rfx_multistart(P::MlogitRfxPrep; b0 = zeros(P.K), nstarts::Int = 10,
                                      s_range = (0.05, 2.0), corr_range = (-0.5, 0.5),
                                      b_jitter::Real = 0.0, seed::Int = 20260808)
    nstarts >= 1 || error("nstarts must be at least 1")
    lo, hi = s_range
    0 < lo <= hi || error("s_range must satisfy 0 < lo <= hi")
    -1 < corr_range[1] <= corr_range[2] < 1 || error("corr_range must lie inside (-1, 1)")
    rng = MersenneTwister(seed)
    out = Matrix{Float64}(undef, nstarts, P.K + P.M + P.B)
    out[1, :] .= theta0_mlogit_rfx(P; b0)
    for r in 2:nstarts
        s = exp.(log(lo) .+ (log(hi) - log(lo)) .* rand(rng, P.M))
        c = corr_range[1] .+ (corr_range[2] - corr_range[1]) .* rand(rng, P.B)
        b = b_jitter > 0 ? b0 .* exp.(b_jitter .* randn(rng, P.K)) : b0
        out[r, :] .= theta0_mlogit_rfx(P; b0 = b, s0 = s, corr0 = c)
    end
    return out
end

function _mrfx_starts(theta0, P::MlogitRfxPrep)
    npar = P.K + P.M + P.B
    S = theta0 isa AbstractMatrix ? Matrix{Float64}(theta0) :
                                    reshape(Vector{Float64}(theta0), 1, :)
    size(S, 2) == npar || error(
        "starting values have $(size(S, 2)) parameters; the model has $npar " *
        "(K = $(P.K) means, M = $(P.M) standard deviations, B = $(P.B) block coordinates). " *
        "Multiple starts go one per row.")
    all(isfinite, S) || error("starting values must be finite")
    return S
end


# ----------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------

function _mrfx_weights(P::MlogitRfxPrep, weights)
    isnothing(weights) && return P.group_weights
    w = Vector{Float64}(weights)
    length(w) == P.N || error("weights must have one entry per group ($(P.N))")
    all(>=(0), w) || error("weights must be nonnegative")
    return isnothing(P.group_weights) ? w : w .* P.group_weights
end

function _mrfx_fit_one(P::MlogitRfxPrep, theta0::Vector{Float64}, w, optim_options,
                       optimizer, sd_min, corr_max, rethrow_errors)
    npar = length(theta0)
    try
        buf = MlogitRfxBuffers(P)
        th  = similar(theta0)
        gt  = similar(theta0)
        phi0 = _mrfx_to_phi(theta0, P.K, P.M, P.B; sd_min, corr_max)
        fg! = (F, G, phi) -> _mrfx_fg_phi!(F, G, phi, P, buf, w, th, gt)
        time = @elapsed res = optimize(Optim.only_fg!(fg!), phi0, optimizer, optim_options)
        _mrfx_to_theta!(th, Optim.minimizer(res), P.K, P.M, P.B)
        return (theta = copy(th), obj = Optim.minimum(res), converged = Optim.converged(res),
                iterations = Optim.iterations(res),
                limit = Optim.iteration_limit_reached(res), time = time,
                errored = false, message = "")
    catch e
        rethrow_errors && rethrow()
        return (theta = fill(NaN, npar), obj = NaN, converged = false, iterations = 0,
                limit = false, time = 0.0, errored = true, message = sprint(showerror, e))
    end
end

function _mrfx_check_workers()
    nprocs() > 1 || error("parallel = true needs worker processes: addprocs(n) and " *
                          "@everywhere using LogitTools")
    for wk in workers()
        remotecall_fetch(() -> isdefined(Main, :LogitTools), wk) || error(
            "LogitTools is not loaded on worker $wk; run @everywhere using LogitTools")
    end
    return nothing
end

"""Map `task` over `ids` on all workers, shipping its closure once per worker."""
function _mrfx_pmap(task, ids)
    _mrfx_check_workers()
    pool = CachingPool(workers())
    try
        return pmap(task, pool, ids)
    finally
        clear!(pool)
    end
end

function _mrfx_assemble(P::MlogitRfxPrep, starts, results, w; obj_tol::Float64 = 1e-4)
    ok   = [!r.errored for r in results]
    conv = [!r.errored && r.converged for r in results]
    pool = any(conv) ? findall(conv) : findall(ok)
    df = DataFrame(start = 1:length(results), obj_value = [r.obj for r in results],
                   converged = [r.converged for r in results],
                   errored = [r.errored for r in results],
                   iterations = [r.iterations for r in results],
                   theta_hat = [r.theta for r in results])
    if isempty(pool)
        return MLEFit(theta0 = starts[1, :], theta_hat = fill(NaN, size(starts, 2)),
                      theta_names = P.theta_names, n_obs = size(P.X, 1), weights = w,
                      obj_value = NaN, errored = true,
                      error_message = results[1].message, converged = false,
                      iterations = missing, iteration_limit_reached = missing,
                      time_it_took = sum(r.time for r in results), fits_df = df)
    end
    best = pool[argmin([results[s].obj for s in pool])]
    r = results[best]
    objs = [results[s].obj for s in pool]
    n_at_best = count(o -> o - r.obj <= obj_tol, objs)
    others = filter(o -> o - r.obj > obj_tol, objs)
    df.is_best = (1:length(results)) .== best
    return MLEFit(theta0 = starts[best, :], theta_hat = r.theta,
                  theta_names = P.theta_names, n_obs = size(P.X, 1), weights = w,
                  obj_value = r.obj, converged = r.converged, iterations = r.iterations,
                  iteration_limit_reached = r.limit,
                  time_it_took = sum(x.time for x in results), fits_df = df,
                  extra = (; model = :mlogit_rfx, K = P.K, M = P.M, B = P.B, R = P.R,
                             N = P.N, kernel = P.kernel,
                             n_starts = length(results), n_converged = count(conv),
                             n_errored = count(!, ok), n_at_best = n_at_best,
                             obj_gap = isempty(others) ? 0.0 : minimum(others) - r.obj))
end

"""
    mlogit_rfx_fit(P, theta0; kwargs...) -> MLEFit

Estimate a prepared model by maximum simulated likelihood.

`theta0` is a vector, or a matrix with one start per row (see
[`theta0_mlogit_rfx_multistart`](@ref)); with several starts the best converged
fit is returned and all attempts are in `fit.fits_df`. `fit.extra` records the
number of starts that converged and reached the best objective (within 1e-4),
the gap to the next optimum, and the effective number of draws.

# Keywords
- `weights = nothing`: group weights, multiplied with the prep's weights.
- `optimizer = BFGS()`, `optim_options = Optim.Options()`: passed to Optim.
- `parallel = false`: distribute the starts over `workers()`.
- `sd_start_min = 0.01`, `corr_start_max = 0.99`: starting standard deviations
  are raised to at least `sd_start_min`, and block coordinates are clipped to
  `[-corr_start_max, corr_start_max]`. At an exact boundary the optimiser could
  not move (see the README). Only starting values are affected.
- `rethrow_errors = false`: by default a failed start is recorded in
  `fits_df` rather than thrown.
- `ess = true`: compute effective draws at the estimate (one extra pass).
"""
function mlogit_rfx_fit(P::MlogitRfxPrep, theta0;
                        weights = nothing,
                        optimizer = BFGS(),
                        optim_options::Optim.Options = Optim.Options(),
                        parallel::Bool = false,
                        sd_start_min::Real = 0.01,
                        corr_start_max::Real = 0.99,
                        rethrow_errors::Bool = false,
                        ess::Bool = true)
    starts = _mrfx_starts(theta0, P)
    w = _mrfx_weights(P, weights)
    task = s -> _mrfx_fit_one(P, starts[s, :], w, optim_options, optimizer,
                              Float64(sd_start_min), Float64(corr_start_max), rethrow_errors)
    ids = 1:size(starts, 1)
    results = parallel ? _mrfx_pmap(task, ids) : map(task, ids)
    fit = _mrfx_assemble(P, starts, results, w)

    if ess && !fit.errored && P.M > 0
        e = mlogit_rfx_ess(fit.theta_hat, P)
        fit.extra = merge(fit.extra, (; ess_min = minimum(e), ess_p10 = quantile(e, 0.1),
                                        ess_median = median(e)))
        quantile(e, 0.1) < 30 && @warn "10th percentile of effective draws per group is " *
            "$(round(quantile(e, 0.1), digits = 1)) (< 30). Consider more draws (ndraws = $(P.R))."
    end
    return fit
end

"""
    mlogit_rfx(data_df, formula, col_id, col_selected, theta0; kwargs...) -> MLEFit

[`mlogit_rfx_prep`](@ref) followed by [`mlogit_rfx_fit`](@ref). Keywords of
both are accepted. Use the two steps directly to bootstrap, compute standard
errors, or reuse the prepared model.
"""
function mlogit_rfx(data_df, formula, col_id, col_selected, theta0;
                    col_group = nothing, rfx = [], rfx_corr = [], ndraws::Int = 1000,
                    seed::Int = 20260808, weights = nothing, kernel::Symbol = :auto,
                    fit_kwargs...)
    P = mlogit_rfx_prep(data_df, formula, col_id, col_selected; col_group, rfx, rfx_corr,
                        ndraws, seed, weights, kernel)
    return mlogit_rfx_fit(P, theta0; fit_kwargs...)
end
