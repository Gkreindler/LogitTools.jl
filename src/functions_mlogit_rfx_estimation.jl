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
#   :uniform        same, with eta_m ~ U(-sqrt(3), sqrt(3))       (unit variance)
#   :triangular     same, with eta_m symmetric triangular on +-sqrt(6), mode 0
#   :lognormal      A_m(g,r) =  exp(mu_m + sigma_m*eta_m[g,r])
#   :neg_lognormal  A_m(g,r) = -exp(mu_m + sigma_m*eta_m[g,r])
#
# The three LINEAR families differ only in the draw generator (see
# _make_cell_draws); the kernel, gradient and reporting are shared, and because
# every eta is standardised, sigma_m is the coefficient's standard deviation
# under each of them.
#
#   l_cr    = v_{c,sel(c),r} - logsumexp_{j in c} v_cjr
#   logL_i  = logsumexp_r( sum_{c in i} l_cr + logw_r )
#
# Objective (minimised):  Q(t) = - sum_i w_i * logL_i,  t = [mu (K); sigma (M)]
###############################################################################

# ----------------------------------------------------------------------------
# Terms
# ----------------------------------------------------------------------------

# Keep the public correlation strictly inside (-1, 1). The small margin prevents
# `sqrt(1-rho^2)` and its derivative from becoming singular after `tanh` rounds
# to exactly one at a large optimiser coordinate.
const _MLOGIT_RFX_CORR_LIMIT = 1.0 - sqrt(eps(Float64))

"""
Distributions accepted in an `mlogit_rfx` term. The three LINEAR families enter
the utility as `sigma * eta` with `eta` standardised to mean zero and unit
variance, so `sigma` is the coefficient's standard deviation whatever the family
and the `sd_` rows of a table are comparable across families:

    :normal      eta ~ N(0, 1)
    :uniform     eta ~ U(-sqrt(3), sqrt(3))
    :triangular  eta ~ symmetric triangular on (-sqrt(6), sqrt(6)) with mode 0

The lognormal families are as in `logit2_rfx`. `logit2_rfx` itself still accepts
only `_RFX_DISTS`; the bounded families are an `mlogit_rfx` extension.
"""
const _MLOGIT_RFX_LINEAR_DISTS = (:normal, :uniform, :triangular)
const _MLOGIT_RFX_DISTS = (_MLOGIT_RFX_LINEAR_DISTS..., :lognormal, :neg_lognormal)

"""Is `d` a linear family, i.e. `A = sigma * eta` with a standardised draw?"""
_rfx_is_linear(d::Symbol) = d in _MLOGIT_RFX_LINEAR_DISTS

"""
One random-coefficient term of an [`mlogit_rfx`](@ref) model.

`var === nothing` is a random **intercept**: its loading is 1 on every row and
there is no matching formula coefficient, so its mean is fixed at zero. That is
the right normalisation, because the *mean* effect of a level value is either a
fixed effect in `formula` or not identified at all -- only the spread around it
is new information.

`col` is the formula position of `var`, or `0` when the loading has no fixed
mean in `formula` (including an intercept). `at_group` records whether the level
is the integration unit itself, which is the `logit2_rfx` case.
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
    rfx_term(var = nothing; level = nothing, dist = :normal, mean = true)

Describe one random-coefficient term for [`mlogit_rfx`](@ref).

- `var`: the regressor whose coefficient is random. `nothing` (the default) means a
  random **intercept** at `level` -- a pure option-level random effect with mean zero.
- `level`: the column whose values index the draws. `nothing` means the model's
  integration unit (`col_group`), which reproduces `logit2_rfx`'s behaviour. Any
  other column gives a random effect that varies *within* an individual; draws are
  always individual-specific, i.e. the level is interacted with `col_group`.
- `dist`: `:normal` (default), `:uniform`, `:triangular`, `:lognormal` or
  `:neg_lognormal`. The first three are linear in a standardised draw, so their
  `sigma` is the coefficient's standard deviation under every one of them; a
  bounded family only changes the shape (uniform on `mu +- sqrt(3) sigma`,
  symmetric triangular on `mu +- sqrt(6) sigma`). The lognormal families are as
  in [`logit2_rfx`](@ref); they need a `var` (their `mu` is an estimated formula
  coefficient), so they cannot be used for an intercept term.
- `mean`: whether `var` must also appear in `formula` as the coefficient's fixed
  mean. The default is `true`, preserving the existing random-slope contract.
  Set `mean = false` for a centered normal factor loading whose variable is used
  only by the random part. This is useful when one latent factor loads on a sum
  of fixed-effect regressors without introducing a collinear fixed coefficient.

Plain `Symbol` and `Pair` entries in `rfx` still mean what they mean in
`logit2_rfx` -- a random coefficient on that variable, at the group level -- so
`rfx_term` is only needed when you want a level other than the group.

# Example
```julia
rfx = [:any_fam,                              # agent-level normal coefficient
       :dur => :neg_lognormal,                # agent-level lognormal coefficient
       rfx_term(:net_value; mean = false),    # centered agent-level factor loading
       rfx_term(level = :nbh_code),           # option-level random intercept
       rfx_term(:salient; level = :nbh_code)] # option-level random slope
```
"""
function rfx_term(var = nothing; level = nothing, dist = :normal, mean::Bool = true)
    base = (var   = isnothing(var)   ? nothing : Symbol(var),
            level = isnothing(level) ? nothing : Symbol(level),
            dist  = Symbol(dist))
    # Preserve the exact pre-extension NamedTuple for every existing call. The
    # extra field is emitted only when the caller requests the new behavior.
    return mean ? base : merge(base, (mean = false,))
end

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
        v, lv, d, has_mean = if isa(r, Symbol)
            (r, nothing, :normal, true)
        elseif isa(r, AbstractString)
            (Symbol(r), nothing, :normal, true)
        elseif isa(r, Pair)
            (Symbol(first(r)), nothing, Symbol(last(r)), true)
        elseif isa(r, NamedTuple)
            bad = setdiff(collect(keys(r)), (:var, :level, :dist, :mean))
            isempty(bad) || error(
                "rfx entry $j has unknown field(s) $(bad); a random-coefficient term " *
                "accepts only (var, level, dist, mean). Build it with rfx_term.")
            (haskey(r, :var)   && !isnothing(r.var)   ? Symbol(r.var)   : nothing,
             haskey(r, :level) && !isnothing(r.level) ? Symbol(r.level) : nothing,
             haskey(r, :dist)  && !isnothing(r.dist)  ? Symbol(r.dist)  : :normal,
             haskey(r, :mean)  && !isnothing(r.mean)  ? Bool(r.mean)    : true)
        else
            error("rfx entry $j has type $(typeof(r)), which is not a random-coefficient " *
                  "term. Accepted: :x, :x => :lognormal, or rfx_term(:x; level = :g). " *
                  "Got: $(repr(r))")
        end

        d in _MLOGIT_RFX_DISTS || error(
            "rfx distribution :$d is not supported (entry $j). " *
            "Supported: $(join(string.(":", _MLOGIT_RFX_DISTS), ", ")).")

        level    = isnothing(lv) ? col_group : lv
        at_group = level === col_group

        col = 0
        if !isnothing(v)
            k = findfirst(==(v), formula_syms)
            has_mean && isnothing(k) && error(
                "rfx variable :$v (entry $j) is not in formula. Available: $(formula_syms). " *
                "A random coefficient needs its mean in the formula by default. For a " *
                "centered random loading with no fixed mean, use " *
                "rfx_term(:$v; level = :$level, mean = false).")
            isnothing(k) || (col = k)
        end

        if _rfx_is_log(d) && (isnothing(v) || col == 0)
            error("rfx entry $j has dist = :$d but no fixed formula mean. The lognormal " *
                  "families are parameterised as +-exp(mu + sigma*eta), and an " *
                  "intercept or mean-free loading has no mu to estimate. Put the variable " *
                  "in formula with mean = true, or use :normal.")
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



# Ordered normal block: the first two coefficients have zero MARGINAL
# covariance. The other five entries are unrestricted subject to PD.
function _normalize_mlogit_rfx_corr4(blocks, terms, pairs)
    used = Set{Int}(Iterators.flatten(pairs))
    out = NTuple{4,Int}[]
    for entry in blocks
        entry isa Tuple && length(entry) == 4 || error("rfx_corr4 entries must be ordered four-tuples")
        ids = map(entry) do x
            hits = x isa Integer ? (1 <= x <= length(terms) ? [Int(x)] : Int[]) :
                   findall(t -> t.name == string(x), terms)
            length(hits) == 1 || error("rfx_corr4 term $(repr(x)) is missing or ambiguous")
            only(hits)
        end
        length(unique(ids)) == 4 || error("rfx_corr4 repeats a term")
        for m in ids
            m in used && error("rfx_corr and rfx_corr4 blocks must be disjoint")
            t = terms[m]
            t.at_group && t.dist === :normal && !isnothing(t.var) ||
                error("rfx_corr4 requires group-level normal slopes")
            push!(used, m)
        end
        push!(out, ids)
    end
    out
end

# Coordinates (x,y,u,w,t) are partial correlations, NOT five marginal
# correlations. Scaling every row by its marginal SD preserves SD semantics.
function _mlogit_rfx_corr4_factors(q)
    x,y,u,w,t = q
    hx,hy,hu,hw,ht = sqrt.(1 .- (x,y,u,w,t).^2)
    dx,dy,du,dw,dt = (-x/hx,-y/hy,-u/hu,-w/hw,-t/ht)
    L = [1.0 0.0 0.0 0.0;
         0.0 1.0 0.0 0.0;
         x hx*y hx*hy 0.0;
         u hu*w hu*hw*t hu*hw*ht]
    J = zeros(4,4,5)
    J[3,:,1] .= (1.0,dx*y,dx*hy,0.0)
    J[3,:,2] .= (0.0,hx,hx*dy,0.0)
    J[4,:,3] .= (1.0,du*w,du*hw*t,du*hw*ht)
    J[4,:,4] .= (0.0,hu,hu*dw*t,hu*dw*ht)
    J[4,:,5] .= (0.0,0.0,hu*hw,hu*hw*dt)
    L,J
end

"""
    mlogit_rfx_correlation_matrix(fit)
    mlogit_rfx_correlation_matrix(theta, prep)

Return marginal correlations in random-term order. Four-term block parameters
in `theta` are partial correlations; this converts them to marginal correlations.
Transform each bootstrap replicate separately before calculating uncertainty.
"""
function mlogit_rfx_correlation_matrix(theta, P)
    _mlogit_rfx_correlation_matrix(theta, P.K, P.M, P.corr_pairs, P.corr4_blocks)
end
function mlogit_rfx_correlation_matrix(fit::MLEFit)
    e = fit.extra
    blocks = hasproperty(e, :rfx_corr4) ? e.rfx_corr4 : NTuple{4,Int}[]
    _mlogit_rfx_correlation_matrix(fit.theta_hat, e.K, e.M, e.rfx_corr, blocks)
end
function _mlogit_rfx_correlation_matrix(theta, K, M, pairs, blocks)
    C = Matrix{Float64}(I, M, M)
    for (b,(p,q)) in enumerate(pairs)
        C[p,q] = C[q,p] = theta[K+M+b]
    end
    for (b,ids) in enumerate(blocks)
        off = K+M+length(pairs)+5*(b-1)
        L,_ = _mlogit_rfx_corr4_factors(view(theta,off+1:off+5))
        C[collect(ids),collect(ids)] .= L*L'
    end
    C
end

"""
    _normalize_mlogit_rfx_corr(rfx_corr, terms) -> Vector{NTuple{2,Int}}

Resolve bivariate correlation blocks against the normalised random-effect terms.
Each entry is a `Pair` or two-tuple of term names (or indices). Correlation is
currently deliberately limited to disjoint pairs of group-level terms of one
linear family: that is the covariance structure needed for correlated person
effects, and the restriction prevents a partial or non-positive-definite
covariance matrix from being specified accidentally.

Under a bounded family the block is built exactly as under the normal, as
`A_p = sigma_p eta_p` and `A_q = sigma_q (rho eta_p + sqrt(1 - rho^2) eta_q)`
from two independent standardised draws of that family. `sigma_p`, `sigma_q`
and `rho` keep their meaning (marginal standard deviations and correlation),
and the first coefficient keeps the family's exact shape, but the second is a
weighted sum of two such draws -- a trapezoid for `:uniform` -- rather than a
member of the family itself. Since the two draws are exchangeable in the
likelihood only up to that shape, a caller who cares which coefficient keeps the
exact marginal should list it first.

The returned indices are ordered by term position, so writing `(a, v)` or `(v,
a)` produces the identical finite-draw likelihood.
"""
function _normalize_mlogit_rfx_corr(rfx_corr, terms::Vector{MlogitRfxTerm})
    resolve(x) = if x isa Integer
        1 <= x <= length(terms) || error(
            "rfx_corr term index $x is outside 1:$(length(terms))")
        Int(x)
    else
        name = string(x)
        hits = findall(t -> t.name == name, terms)
        length(hits) == 1 || error(
            "rfx_corr term $(repr(x)) matched $(length(hits)) terms; available term " *
            "names are $([t.name for t in terms])")
        only(hits)
    end

    out  = NTuple{2,Int}[]
    used = Set{Int}()
    for (b, entry) in enumerate(rfx_corr)
        x, y = if entry isa Pair
            (first(entry), last(entry))
        elseif entry isa Tuple && length(entry) == 2
            entry
        else
            error("rfx_corr entry $b must be a Pair or two-tuple of term names/indices; " *
                  "got $(repr(entry))")
        end
        p, q = sort((resolve(x), resolve(y)))
        p == q && error("rfx_corr entry $b names the same term twice: $(terms[p].name)")
        (!terms[p].at_group || !terms[q].at_group) && error(
            "rfx_corr entry $b ($(terms[p].name), $(terms[q].name)) is not a pair of " *
            "group-level terms. Correlated blocks currently support person-level " *
            "normal coefficients only.")
        (_rfx_is_linear(terms[p].dist) && terms[p].dist === terms[q].dist) || error(
            "rfx_corr entry $b ($(terms[p].name), $(terms[q].name)) pairs a " *
            ":$(terms[p].dist) term with a :$(terms[q].dist) term. A correlation " *
            "block needs two terms of the same linear family (:normal, :uniform or " *
            ":triangular); lognormal terms cannot be correlated.")
        (!isnothing(terms[p].var) && !isnothing(terms[q].var)) || error(
            "rfx_corr entry $b contains a random intercept. A group-level intercept " *
            "cancels from every choice set and cannot be correlated meaningfully.")
        (p in used || q in used) && error(
            "rfx_corr blocks must be disjoint; term $(p in used ? terms[p].name : terms[q].name) " *
            "appears in more than one block.")
        push!(out, (p, q))
        push!(used, p); push!(used, q)
    end

    length(unique(out)) == length(out) || error("rfx_corr contains a duplicate block")
    return out
end


# ----------------------------------------------------------------------------
# Draws
# ----------------------------------------------------------------------------

"""
    _rfx_standard_triangular(u) -> Float64

Inverse CDF of the symmetric triangular distribution with mode 0, scaled to unit
variance: support `(-sqrt(6), sqrt(6))`, since the triangular on `(-1, 1)` has
variance `1/6`. `u` is a `U(0, 1)` draw.
"""
@inline function _rfx_standard_triangular(u::Float64)
    return u < 0.5 ? sqrt(6.0) * (sqrt(2.0 * u) - 1.0) :
                     sqrt(6.0) * (1.0 - sqrt(2.0 * (1.0 - u)))
end

"""
    _make_cell_draws(cells_per_group, R, seed, cell_dist = nothing) -> (eta, logw)

Antithetic standardised draws, one row per (group, term, level) **cell**: `eta`
is `sum(cells_per_group) x R`, with the cells of group `i` occupying a contiguous
block, in group order.

`cell_dist`, when given, names the family of every cell (`:normal`, `:uniform`,
`:triangular`; a lognormal cell draws a standard normal, because its randomness
is normal on the log scale). Every family is standardised to mean zero and unit
variance, so `sigma * eta` has standard deviation `sigma` under each.

When every cell is normal (the default, and any lognormal-only extension) the
draws are made group by group as a `C_i x R/2` block and then reflected. That
ordering is deliberate: when every term sits at the group level `C_i == M`, and
the block is then bit-for-bit the `randn(rng, M, R/2)` that [`_make_draws`](@ref)
produces for `logit2_rfx`. So the two models share an RNG stream in that case
and can be compared exactly rather than only up to simulation noise -- and every
existing all-normal fit reproduces exactly. With a bounded family present the
draws are made cell by cell instead, so the RNG stream of such a model is its
own; nothing is shared with the all-normal stream.

Antithetic reflection is mandatory, not optional: it is what makes the `sigma = 0`
saddle and the `sigma -> -sigma` mirror symmetry *exact*, because column `r` and
column `r + R/2` negate every cell together. It is valid for the bounded families
because both are symmetric about zero, so `-eta` has the same distribution as
`eta`; that same symmetry is what lets the antithetic SIMD kernel handle them.
"""
function _make_cell_draws(cells_per_group::Vector{Int}, R::Int, seed::Int,
                          cell_dist::Union{Nothing, AbstractVector{Symbol}} = nothing)
    iseven(R) || error("ndraws must be even (antithetic pairing); got $R")

    half   = R ÷ 2
    ncells = sum(cells_per_group)
    eta    = Matrix{Float64}(undef, ncells, R)
    rng    = MersenneTwister(seed)

    if !isnothing(cell_dist)
        length(cell_dist) == ncells || error(
            "cell_dist has length $(length(cell_dist)) but there are $ncells cells")
        for d in cell_dist
            (_rfx_is_linear(d) || _rfx_is_log(d)) || error(
                "unsupported draw family :$d; expected one of $(_MLOGIT_RFX_DISTS)")
        end
    end
    all_normal = isnothing(cell_dist) ||
                 all(d -> d === :normal || _rfx_is_log(d), cell_dist)

    off = 0
    for Ci in cells_per_group
        if Ci > 0 && all_normal
            e = randn(rng, Ci, half)
            @views eta[off+1:off+Ci, 1:half]   .=   e
            @views eta[off+1:off+Ci, half+1:R] .= .-e
        elseif Ci > 0
            for c in (off + 1):(off + Ci)
                d = cell_dist[c]
                e = if d === :uniform
                    sqrt(3.0) .* (2.0 .* rand(rng, half) .- 1.0)
                elseif d === :triangular
                    _rfx_standard_triangular.(rand(rng, half))
                else
                    randn(rng, half)
                end
                @views eta[c, 1:half]   .=   e
                @views eta[c, half+1:R] .= .-e
            end
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
    B::Int                                # total correlation coordinates (pairs + 5 per four-block)
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
    corr_pairs::Vector{NTuple{2,Int}}     # term indices (first is the draw anchor)
    corr4_blocks::Vector{NTuple{4,Int}}   # ordered (independent1, independent2, third, fourth)
    corr4_dependent::Vector{Bool}         # term is row 3 or 4 of a four-block
    corr_second::Vector{Int}              # term -> block index, zero unless second
    corr_cells::Matrix{Int}               # group-local cells: 2 per pair, then 4 per four-block
    corr_names::Vector{String}
    theta_names::Vector{String}
    col_id::Symbol                        # choice set
    col_group::Symbol                     # integration unit
    seed::Int
    n_sets::Int
    cell_stats::Vector{NamedTuple}         # per-term identification diagnostics
    kernel::Symbol                         # resolved kernel mode, see _MLOGIT_RFX_KERNELS
    kernel_requested::Symbol               # what the caller asked for (:auto or a mode)
    binary::Bool                           # every choice set has exactly two rows
    nz_ptr::Vector{Int}                    # Nobs+1, CSR row pointers into nz_loc / nz_z
    nz_loc::Vector{Int}                    # group-local cell of each nonzero loading
    nz_z::Vector{Float64}                  # the nonzero loading values, in term order
    # --- :binary_antithetic_simd only (empty otherwise) ---------------------
    H::Int                                 # R/2, number of antithetic pairs
    set_ptr::Vector{Int}                   # n_sets+1, pointers into set_loc / set_dz
    set_loc::Vector{Int}                   # group-local cell of each merged signed loading
    set_dz::Vector{Float64}                # z(first row) - z(second row), merged per cell
    dX::Matrix{Float64}                    # n_sets x K, xlin(first row) - xlin(second row)
    ysel::Vector{Bool}                     # n_sets, is the FIRST row the selected one
    etaT::Matrix{Float64}                  # H x ncells, first half of eta transposed
    Smax::Int                              # max choice sets per group
end

"""Kernel modes accepted by `mlogit_rfx`'s `kernel` keyword."""
const _MLOGIT_RFX_KERNELS = (:auto, :general, :binary, :binary_antithetic_simd)

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
                     ndraws, seed, weights, rfx_corr = []) -> (P, gw)

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
        weights::Union{Nothing, Symbol, String},
        rfx_corr = [];
        rfx_corr4 = [],
        kernel::Symbol = :general)

    kernel in _MLOGIT_RFX_KERNELS || error(
        "kernel must be one of $(_MLOGIT_RFX_KERNELS); got :$kernel")

    formula_syms = Symbol.(formula)
    K = length(formula_syms)
    K > 0 || error("formula is empty")

    terms = _normalize_mlogit_rfx(rfx, formula_syms, col_group)
    M     = length(terms)
    corr_pairs = _normalize_mlogit_rfx_corr(rfx_corr, terms)
    corr4_blocks = _normalize_mlogit_rfx_corr4(rfx_corr4, terms, corr_pairs)
    B          = length(corr_pairs) + 5length(corr4_blocks)

    # --- column presence ----------------------------------------------------
    dfnames = Symbol.(names(data_df))
    loading_vars = Symbol[t.var for t in terms if !isnothing(t.var)]
    needed  = vcat(formula_syms, Symbol(col_selected), col_id, col_group,
                   [t.level for t in terms], loading_vars)
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

    # A correlation block is restricted to two group-level terms, hence one
    # cell per term and group. Cache those local cell indices once rather than
    # searching `cell_term` inside every likelihood evaluation.
    corr_cells = Matrix{Int}(undef, N, 2length(corr_pairs) + 4length(corr4_blocks))
    for (i, rg) in enumerate(ranges), (b, (p, q)) in enumerate(corr_pairs)
        corr_cells[i, 2b-1] = cellloc[first(rg), p]
        corr_cells[i, 2b]   = cellloc[first(rg), q]
    end

    for (i, rg) in enumerate(ranges), (b, ids) in enumerate(corr4_blocks), j in 1:4
        corr_cells[i, 2length(corr_pairs)+4(b-1)+j] = cellloc[first(rg), ids[j]]
    end

    # --- loadings -----------------------------------------------------------
    zmatrix = Matrix{Float64}(undef, Nobs, M)
    for (m, t) in enumerate(terms)
        if isnothing(t.var)
            @views zmatrix[:, m] .= 1.0
        elseif t.col > 0
            @views zmatrix[:, m] .= xmatrix[:, t.col]
        else
            @views zmatrix[:, m] .= Float64.(data_df[perm, t.var])
        end
    end

    # --- lognormal bookkeeping ---------------------------------------------
    rfx_islog = Bool[_rfx_is_log(t.dist)  for t in terms]
    rfx_sgn   = Float64[_rfx_sign(t.dist) for t in terms]
    rfx_cols  = Int[t.col for t in terms]
    any_log   = any(rfx_islog)
    corr4_dependent = falses(M) |> Vector{Bool}
    for ids in corr4_blocks, j in 3:4
        corr4_dependent[ids[j]] = true
    end
    corr_second = zeros(Int, M)
    for (b, (_, q)) in enumerate(corr_pairs)
        corr_second[q] = b
    end
    corr_names = ["cor_$(terms[p].name)__$(terms[q].name)" for (p, q) in corr_pairs]

    for ids in corr4_blocks
        n = [terms[m].name for m in ids]
        append!(corr_names, ["pcor_$(n[3])__$(n[1])", "pcor_$(n[3])__$(n[2])_given_$(n[1])",
            "pcor_$(n[4])__$(n[1])", "pcor_$(n[4])__$(n[2])_given_$(n[1])",
            "pcor_$(n[4])__$(n[3])_given_$(n[1])_$(n[2])"])
    end
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
    # One family per cell, read off the cell's term. An all-normal model takes
    # the block path inside _make_cell_draws and reproduces the pre-extension
    # draws bit for bit.
    cell_dist = Symbol[terms[m].dist for m in cell_term]
    eta, logw = _make_cell_draws(cells_per_group, ndraws, seed, cell_dist)

    theta_names = [string.(formula_syms); ["sd_" * t.name for t in terms]; corr_names]
    rfx_pairs   = Pair{Symbol,Symbol}[Symbol(t.name) => t.dist for t in terms]

    # --- kernel mode ---------------------------------------------------------
    # :general                 any choice-set size, any distribution (reference)
    # :binary                  every set has two rows: logistic softmax, and a
    #                          per-row list of the NONZERO loadings; normal and
    #                          lognormal terms
    # :binary_antithetic_simd  two rows AND only linear terms (normal, uniform,
    #                          triangular): the second half of the antithetic
    #                          draws is the negative of the first and every
    #                          coefficient is sigma * eta, so the random part
    #                          of the utility difference is computed once per
    #                          pair; draws are stored contiguously so the inner
    #                          loops vectorise. A lognormal coefficient is not
    #                          odd in its draw, which is why it is excluded.
    # :auto picks the most specialised eligible mode. Asking for an ineligible
    # mode is an error rather than a silent fallback.
    binary  = n_sets > 0 && all(sr -> length(sr) == 2, set_ranges)
    simd_ok = binary && M > 0 && !any_log
    mode = if kernel === :auto
        simd_ok ? :binary_antithetic_simd : binary ? :binary : :general
    else
        kernel === :binary && !binary && error(
            "kernel = :binary needs every choice set to have exactly two rows")
        kernel === :binary_antithetic_simd && !simd_ok && error(
            "kernel = :binary_antithetic_simd needs two-row choice sets, at least one " *
            "random term, and no lognormal term (its sign symmetry is what the " *
            "antithetic pairing exploits)")
        kernel
    end

    nz_ptr = ones(Int, Nobs + 1)
    nz_loc = Int[]
    nz_z   = Float64[]
    if mode !== :general
        for t in 1:Nobs
            for m in 1:M
                z = zmatrix[t, m]
                z == 0.0 && continue
                push!(nz_loc, cellloc[t, m])
                push!(nz_z, z)
            end
            nz_ptr[t + 1] = length(nz_loc) + 1
        end
    end

    H = ndraws ÷ 2
    set_ptr = ones(Int, n_sets + 1)
    set_loc = Int[]
    set_dz  = Float64[]
    dX      = Matrix{Float64}(undef, 0, K)
    ysel    = Bool[]
    etaT    = Matrix{Float64}(undef, 0, 0)
    Smax    = maximum(length.(set_of_group))
    if mode === :binary_antithetic_simd
        # Merge the two rows' nonzero loadings per cell into one signed list,
        # dropping exact cancellations (both options in the same cell with the
        # same loading, e.g. both familiar): that set carries no random part.
        acc = Dict{Int,Float64}()
        for s in 1:n_sets
            lo = first(set_ranges[s]); hi = lo + 1
            empty!(acc)
            for k in nz_ptr[lo]:(nz_ptr[lo + 1] - 1)
                acc[nz_loc[k]] = get(acc, nz_loc[k], 0.0) + nz_z[k]
            end
            for k in nz_ptr[hi]:(nz_ptr[hi + 1] - 1)
                acc[nz_loc[k]] = get(acc, nz_loc[k], 0.0) - nz_z[k]
            end
            for c in sort!(collect(keys(acc)))
                acc[c] == 0.0 && continue
                push!(set_loc, c); push!(set_dz, acc[c])
            end
            set_ptr[s + 1] = length(set_loc) + 1
        end
        dX   = Matrix{Float64}(undef, n_sets, K)
        ysel = Vector{Bool}(undef, n_sets)
        for s in 1:n_sets
            lo = first(set_ranges[s])
            @views dX[s, :] .= xlin[lo, :] .- xlin[lo + 1, :]
            ysel[s] = yvec[lo] == 1.0
        end
        # The SIMD kernel reads only the first half of the draws, transposed, and
        # infers the antithetic second half. Keep just that copy: a prepared
        # column-2 model is serialised to every bootstrap worker, and the full
        # `eta` would add 50% to that traffic and to per-worker memory. The
        # general kernel's `eta` is left empty in this mode.
        etaT = Matrix{Float64}(transpose(view(eta, :, 1:H)))
        eta  = Matrix{Float64}(undef, 0, 0)
    end

    P = MlogitRfxPrep(
        xmatrix, xlin, zmatrix, yvec, cellloc,
        ranges, set_ranges, set_of_group, sel_row,
        cell_ranges, cell_term, eta, logw, group_ids,
        K, M, B, ndraws, N, Tmax, Cmax,
        terms, rfx_pairs, rfx_cols, rfx_islog, rfx_sgn, any_log,
        corr_pairs, corr4_blocks, corr4_dependent, corr_second, corr_cells, corr_names,
        theta_names, col_id, col_group, seed, n_sets, cell_stats,
        mode, kernel, binary, nz_ptr, nz_loc, nz_z,
        H, set_ptr, set_loc, set_dz, dX, ysel, etaT, Smax)

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
    # --- :binary_antithetic_simd only (0 x 0 otherwise): draws contiguous ---
    AT::Matrix{Float64}      # H x Cmax    per-cell coefficient, first half
    STp::Matrix{Float64}     # H x Cmax    per-cell score, +g half
    STm::Matrix{Float64}     # H x Cmax    per-cell score, -g half
    Ep::Matrix{Float64}      # H x Smax    residual of the first row, +g half
    Em::Matrix{Float64}      # H x Smax    residual of the first row, -g half
    g::Vector{Float64}       # H           random part of one set's utility difference
end

function MlogitRfxBuffers(P::MlogitRfxPrep)
    simd = P.kernel === :binary_antithetic_simd
    Hs   = simd ? P.H : 0
    MlogitRfxBuffers(
        Matrix{Float64}(undef, P.Tmax, simd ? 0 : P.R),
        Matrix{Float64}(undef, P.Tmax, simd ? 0 : P.R),
        Matrix{Float64}(undef, P.Cmax, simd ? 0 : P.R),
        Matrix{Float64}(undef, P.Cmax, simd ? 0 : P.R),
        Vector{Float64}(undef, P.R),
        Vector{Float64}(undef, P.R),
        Vector{Float64}(undef, P.Tmax),
        Vector{Float64}(undef, P.Tmax),
        Matrix{Float64}(undef, Hs, simd ? P.Cmax : 0),
        Matrix{Float64}(undef, Hs, simd ? P.Cmax : 0),
        Matrix{Float64}(undef, Hs, simd ? P.Cmax : 0),
        Matrix{Float64}(undef, Hs, simd ? P.Smax : 0),
        Matrix{Float64}(undef, Hs, simd ? P.Smax : 0),
        Vector{Float64}(undef, Hs),
    )
end


# ----------------------------------------------------------------------------
# Per-cell coefficients
# ----------------------------------------------------------------------------

"""
    _mlogit_rfx_fill_A!(Ai, beta, sigma, corr, etai, ct, corr_cells, P)

Fill `Ai` (`C_i x R`) with the quantity that multiplies the loading in the linear
index, one row per cell of the current group:

    linear families  A[c,r] = sigma_m * eta[c,r]                (deviation from mu_m)
    :lognormal       A[c,r] = +-exp(mu_m + sigma_m * eta[c,r])  (the coefficient itself)

The linear families (`:normal`, `:uniform`, `:triangular`) differ only in how
`eta` was drawn, so they share the first line.

where `m = ct[c]` is the cell's term. `xlin` has the lognormal columns zeroed,
which is why the second line is the whole coefficient and not a deviation. The
sign of `:neg_lognormal` is folded in here, so nothing downstream knows about it.

For correlation block `(p,q)`, the two group-level normal coefficients use

    A_p = sigma_p * eta_p
    A_q = sigma_q * (rho * eta_p + sqrt(1-rho^2) * eta_q),

so `sigma_p` and `sigma_q` remain marginal standard deviations and `rho` is
their correlation. `corr_cells` supplies the two group-local draw rows.
"""
@inline function _mlogit_rfx_fill_A!(Ai, beta, sigma, corr, etai, ct, corr_cells,
                                     P::MlogitRfxPrep)

    Ci = length(ct)

    if !P.any_log
        @inbounds for r in 1:P.R, c in 1:Ci
            Ai[c, r] = sigma[ct[c]] * etai[c, r]
        end
    else
        @inbounds for r in 1:P.R, c in 1:Ci
            m = ct[c]
            Ai[c, r] = P.rfx_islog[m] ?
                P.rfx_sgn[m] * exp(beta[P.rfx_cols[m]] + sigma[m] * etai[c, r]) :
                sigma[m] * etai[c, r]
        end
    end

    # Override the second coefficient in each block with the correlated normal
    # combination. The first coefficient already has sigma_p * eta_p above.
    @inbounds for b in eachindex(P.corr_pairs)
        cp, cq = corr_cells[2b-1], corr_cells[2b]
        rho    = corr[b]
        root   = sqrt(1.0 - rho * rho)
        q      = P.corr_pairs[b][2]
        for r in 1:P.R
            Ai[cq, r] = sigma[q] * (rho * etai[cp, r] + root * etai[cq, r])
        end
    end

    for (b, ids) in enumerate(P.corr4_blocks)
        off = length(P.corr_pairs)+5(b-1)
        cells = view(corr_cells, 2length(P.corr_pairs)+4(b-1)+1:2length(P.corr_pairs)+4b)
        L,_ = _mlogit_rfx_corr4_factors(view(corr,off+1:off+5))
        for j in 3:4, r in 1:P.R
            z = 0.0
            for k in 1:4
                z += L[j,k]*etai[cells[k],r]
            end
            Ai[cells[j],r] = sigma[ids[j]]*z
        end
    end

    # exp() overflows to Inf above an exponent of ~709, and Inf then propagates
    # into the gradient as Inf*0 = NaN, from which LBFGS cannot recover -- it
    # would report a converged fit at a garbage theta. Fail with the cause named
    # instead. O(C_i*R) with C_i small, so this is free next to the T_i x R work.
    if P.any_log && !all(isfinite, view(Ai, 1:Ci, :))
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
# Two-option kernel
# ----------------------------------------------------------------------------

"""
    _mlogit_rfx_binary_pass!(P, i, r0, Ti, Ai, Vi, Ei, Si, ll, xb, yi, want_e, scatter)

Steps 2 and 3 of the fused kernel for a group whose choice sets all have exactly
two rows: fill the linear index `Vi`, the per-draw log-likelihood `ll`, and (when
`want_e`) the residuals `Ei` and (when `scatter`) the per-cell scores `Si`.

Same likelihood as the general path, cheaper arithmetic:

- The linear index adds only the nonzero loadings of each row (`P.nz_*`), in
  term order; a zero loading contributes exactly nothing.
- With two options the softmax is a binary logit: `P(1) = logistic(V1 - V2)`
  and `log P(chosen) = -log1p(exp(-|d|))` (minus `|d|` on the losing side), so
  one `exp` and one `log1p` per set and draw replace four `exp` and one `log`.
  The residual of the second row is the negative of the first, because exactly
  one row is selected.
- The per-cell score accumulates the same terms as the general path.

The two paths agree to roundoff (about 1e-12 relative on production data), not
bit for bit, because the logistic form rounds differently from the max-shifted
softmax. `test_mlogit_rfx.jl` checks the agreement.
"""
@inline function _mlogit_rfx_binary_pass!(P::MlogitRfxPrep, i::Int, r0::Int, Ti::Int,
                                          Ai, Vi, Ei, Si, ll, xb, yi,
                                          want_e::Bool, scatter::Bool)
    R   = P.R
    ptr = P.nz_ptr
    loc = P.nz_loc
    zv  = P.nz_z

    @inbounds begin
        for r in 1:R, t in 1:Ti
            v  = xb[t]
            tt = t + r0
            for k in ptr[tt]:(ptr[tt + 1] - 1)
                v += zv[k] * Ai[loc[k], r]
            end
            Vi[t, r] = v
        end

        scatter && fill!(Si, 0.0)
        for r in 1:R
            acc = P.logw[r]
            for s in P.set_of_group[i]
                lo = first(P.set_ranges[s]) - r0
                hi = lo + 1
                # Binary logit: P(1) = logistic(d), d = V1 - V2. One exp and one
                # log1p per set and draw, on the side of d that cannot overflow.
                d = Vi[lo, r] - Vi[hi, r]
                if d >= 0.0
                    ed = exp(-d)
                    p1 = 1.0 / (1.0 + ed)
                    lp = log1p(ed)                       # lse - V1
                    acc += yi[lo] == 1.0 ? -lp : -(lp + d)
                else
                    ed = exp(d)
                    p1 = ed / (1.0 + ed)
                    lp = log1p(ed)                       # lse - V2
                    acc += yi[lo] == 1.0 ? -(lp - d) : -lp
                end

                if want_e
                    e1 = yi[lo] - p1                     # e2 = -e1 since y1 + y2 = 1
                    Ei[lo, r] = e1
                    Ei[hi, r] = -e1
                    if scatter
                        for k in ptr[lo + r0]:(ptr[lo + r0 + 1] - 1)
                            Si[loc[k], r] += e1 * zv[k]
                        end
                        for k in ptr[hi + r0]:(ptr[hi + r0 + 1] - 1)
                            Si[loc[k], r] -= e1 * zv[k]
                        end
                    end
                end
            end
            ll[r] = acc
        end
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

    P.kernel === :binary_antithetic_simd &&
        return _mlogit_rfx_fg_simd!(F, G, theta, P, buf, gw)

    K, M, B, R = P.K, P.M, P.B, P.R

    beta  = view(theta, 1:K)
    sigma = view(theta, K+1:K+M)
    corr  = view(theta, K+M+1:K+M+B)

    need_g = G !== nothing
    need_g && fill!(G, 0.0)
    gb = need_g ? view(G, 1:K)     : nothing
    gs = need_g ? view(G, K+1:K+M) : nothing
    gc = need_g ? view(G, K+M+1:K+M+B) : nothing

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
        cc   = view(P.corr_cells, i, :)    # 2B group-local correlated cells
        Ai   = view(buf.A, 1:Ci, :)
        Si   = view(buf.S, 1:Ci, :)
        Vi   = view(buf.V, 1:Ti, :)
        Ei   = view(buf.E, 1:Ti, :)
        xb   = view(buf.xb,   1:Ti)
        eb   = view(buf.ebar, 1:Ti)

        # --- 1. per-cell coefficients ---------------------------------------
        M > 0 && _mlogit_rfx_fill_A!(Ai, beta, sigma, corr, etai, ct, cc, P)

        mul!(xb, Xi, beta)
        if P.kernel === :binary
            # --- 2+3. two-option kernel ---------------------------------------
            _mlogit_rfx_binary_pass!(P, i, r0, Ti, Ai, Vi, Ei, Si, buf.ll, xb, yi,
                                     need_g, scatter)
        else
        # --- 2. linear index ------------------------------------------------
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
        end # kernel

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
                        P.corr4_dependent[m] && continue
                        co = om * tau * Si[c, r]
                        if P.rfx_islog[m]
                            gs[m]             -= co * etai[c, r] * Ai[c, r]
                            gb[P.rfx_cols[m]] -= co * Ai[c, r]
                        else
                            b = P.corr_second[m]
                            if b == 0
                                gs[m] -= co * etai[c, r]
                            else
                                cp   = cc[2b-1]
                                rho  = corr[b]
                                root = sqrt(1.0 - rho * rho)
                                zeta = rho * etai[cp, r] + root * etai[c, r]
                                gs[m] -= co * zeta
                                gc[b] -= co * sigma[m] *
                                         (etai[cp, r] - (rho / root) * etai[c, r])
                            end
                        end
                    end
                end
            end
            for (b, ids) in enumerate(P.corr4_blocks)
                off = length(P.corr_pairs)+5(b-1)
                cells = view(cc, 2length(P.corr_pairs)+4(b-1)+1:2length(P.corr_pairs)+4b)
                L,J = _mlogit_rfx_corr4_factors(view(corr,off+1:off+5))
                for j in 3:4, r in 1:R
                    co = om*buf.pw[r]*Si[cells[j],r]
                    z = sum(L[j,k]*etai[cells[k],r] for k in 1:4)
                    gs[ids[j]] -= co*z
                    for h in 1:5
                        dz = sum(J[j,k,h]*etai[cells[k],r] for k in 1:4)
                        gc[off+h] -= co*sigma[ids[j]]*dz
                    end
                end
            end
        end
    end

    return F !== nothing ? Q : nothing
end


# ----------------------------------------------------------------------------
# Two-option, linear-terms-only, antithetic, draw-contiguous kernel
# ----------------------------------------------------------------------------

"""
    _mlogit_rfx_simd_pass!(P, i, theta, buf, need_g) -> nothing

Likelihood pass for one group under `:binary_antithetic_simd`; fills `buf.ll`
(all `R` draws) and, when `need_g`, the residuals `Ep`/`Em` and per-cell scores
`STp`/`STm` for the two antithetic halves.

Three facts do the work:

1. **Two rows.** The softmax of a two-row set is a binary logit in the utility
   difference `d = V1 - V2`, so each set costs one `exp` and one `log1p`, and
   the second row's residual is minus the first's.
2. **Antithetic draws.** `_make_cell_draws` makes draw `r + R/2` the negative of
   draw `r`. Every term is linear in its draw (`:normal`, `:uniform` or
   `:triangular`; no lognormal), so each cell coefficient flips sign with
   its draw, and the random part `g_r` of `d` satisfies `g_{r+R/2} = -g_r`. Only
   `g` for the first half is computed; both `d = dx + g` and `d = dx - g` are
   then evaluated. The per-cell scores must be kept separately for the two
   halves because the residuals differ; in the sigma/rho gradient they combine
   as `tau_plus * S_plus - tau_minus * S_minus` since the draw itself is negated.
3. **Draws contiguous.** `etaT`, `AT`, `ST*` and `E*` are stored with the draw
   index first, so `g` accumulates over a contiguous vector per signed loading,
   and the score scatter and the gradient reductions are straight loops over
   draws that vectorise.

Per set the random part of `d` uses the merged signed loadings
`P.set_loc`/`P.set_dz` (mean below 3 entries here), and the fixed part comes from
`P.dX * beta`, computed once per group. `dX` is first-minus-second, and `P.ysel`
records whether the first row is the selected one.
"""
@inline function _mlogit_rfx_simd_pass!(P::MlogitRfxPrep, i::Int, theta::Vector{Float64},
                                        buf::MlogitRfxBuffers, need_g::Bool)
    K, M, B, H = P.K, P.M, P.B, P.H
    beta  = view(theta, 1:K)
    sigma = view(theta, K+1:K+M)
    corr  = view(theta, K+M+1:K+M+B)

    crng = P.cell_ranges[i]
    Ci   = length(crng)
    c0   = first(crng) - 1
    sets = P.set_of_group[i]
    ns   = length(sets)
    ct   = view(P.cell_term, crng)
    cc   = view(P.corr_cells, i, :)
    AT   = buf.AT
    etaT = P.etaT
    g    = buf.g
    ll   = buf.ll

    @inbounds begin
        # --- per-cell coefficients for the first half of the draws ----------
        for c in 1:Ci
            sg = sigma[ct[c]]
            @simd for r in 1:H
                AT[r, c] = sg * etaT[r, c0 + c]
            end
        end
        for b in eachindex(P.corr_pairs)
            cp, cq = cc[2b-1], cc[2b]
            rho  = corr[b]
            root = sqrt(1.0 - rho * rho)
            sq   = sigma[P.corr_pairs[b][2]]
            @simd for r in 1:H
                AT[r, cq] = sq * (rho * etaT[r, c0 + cp] + root * etaT[r, c0 + cq])
            end
        end

        for (b, ids) in enumerate(P.corr4_blocks)
            off = length(P.corr_pairs)+5(b-1)
            cells = view(cc, 2length(P.corr_pairs)+4(b-1)+1:2length(P.corr_pairs)+4b)
            L,_ = _mlogit_rfx_corr4_factors(view(corr,off+1:off+5))
            for j in 3:4
                c1,c2,c3,c4 = (c0+cells[k] for k in 1:4)
                l1,l2,l3,l4 = (L[j,k] for k in 1:4)
                sg = sigma[ids[j]]
                @simd for r in 1:H
                    AT[r,cells[j]] = sg*(l1*etaT[r,c1]+l2*etaT[r,c2]+l3*etaT[r,c3]+l4*etaT[r,c4])
                end
            end
        end

        # --- fixed part of the utility difference, one gemv per group --------
        dx = view(buf.xb, 1:ns)
        mul!(dx, view(P.dX, sets, :), beta)

        if need_g
            fill!(view(buf.STp, :, 1:Ci), 0.0)
            fill!(view(buf.STm, :, 1:Ci), 0.0)
        end
        for r in 1:H
            ll[r]     = P.logw[r]
            ll[r + H] = P.logw[r + H]
        end

        for (js, s) in enumerate(sets)
            k1 = P.set_ptr[s]
            k2 = P.set_ptr[s + 1] - 1
            if k1 <= k2
                z = P.set_dz[k1]; c = P.set_loc[k1]
                @simd for r in 1:H
                    g[r] = z * AT[r, c]
                end
                for k in (k1 + 1):k2
                    z = P.set_dz[k]; c = P.set_loc[k]
                    @simd for r in 1:H
                        g[r] += z * AT[r, c]
                    end
                end
            else
                fill!(g, 0.0)
            end

            y1  = P.ysel[s]
            dxs = dx[js]
            for r in 1:H
                d = dxs + g[r]
                if d >= 0.0
                    ed = exp(-d); p1 = 1.0 / (1.0 + ed); lp = log1p(ed)
                    ll[r] += y1 ? -lp : -(lp + d)
                else
                    ed = exp(d);  p1 = ed / (1.0 + ed);  lp = log1p(ed)
                    ll[r] += y1 ? -(lp - d) : -lp
                end
                need_g && (buf.Ep[r, js] = (y1 ? 1.0 : 0.0) - p1)

                d = dxs - g[r]
                if d >= 0.0
                    ed = exp(-d); p1 = 1.0 / (1.0 + ed); lp = log1p(ed)
                    ll[r + H] += y1 ? -lp : -(lp + d)
                else
                    ed = exp(d);  p1 = ed / (1.0 + ed);  lp = log1p(ed)
                    ll[r + H] += y1 ? -(lp - d) : -lp
                end
                need_g && (buf.Em[r, js] = (y1 ? 1.0 : 0.0) - p1)
            end

            if need_g
                for k in k1:k2
                    z = P.set_dz[k]; c = P.set_loc[k]
                    @simd for r in 1:H
                        buf.STp[r, c] += buf.Ep[r, js] * z
                        buf.STm[r, c] += buf.Em[r, js] * z
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _mlogit_rfx_fg_simd!(F, G, theta, P, buf, gw)

`_mlogit_rfx_fg!` for `P.kernel === :binary_antithetic_simd`. See
[`_mlogit_rfx_simd_pass!`](@ref) for the likelihood pass; the gradient collapses
the draw dimension first, as in the general kernel, with the beta gradient
written as `-om * dX' * ebar` over the group's choice sets.
"""
function _mlogit_rfx_fg_simd!(F, G, theta::Vector{Float64}, P::MlogitRfxPrep,
                              buf::MlogitRfxBuffers, gw::Union{Nothing, Vector{Float64}})
    K, M, B, H, R = P.K, P.M, P.B, P.H, P.R
    sigma = view(theta, K+1:K+M)
    corr  = view(theta, K+M+1:K+M+B)

    need_g = G !== nothing
    need_g && fill!(G, 0.0)
    gb = need_g ? view(G, 1:K)         : nothing
    gs = need_g ? view(G, K+1:K+M)     : nothing
    gc = need_g ? view(G, K+M+1:K+M+B) : nothing

    Q = 0.0
    @inbounds for i in 1:P.N
        crng = P.cell_ranges[i]
        Ci   = length(crng)
        c0   = first(crng) - 1
        sets = P.set_of_group[i]
        ns   = length(sets)
        om   = isnothing(gw) ? 1.0 : gw[i]
        ct   = view(P.cell_term, crng)
        cc   = view(P.corr_cells, i, :)

        _mlogit_rfx_simd_pass!(P, i, theta, buf, need_g)

        lse_all = logsumexp(buf.ll)
        Q -= om * lse_all

        if need_g
            buf.pw .= exp.(buf.ll .- lse_all)
            pwp = view(buf.pw, 1:H)
            pwm = view(buf.pw, H+1:R)

            # beta: the fixed part does not flip between halves, so the two
            # halves' residuals ADD.
            eb = view(buf.ebar, 1:ns)
            mul!(eb, transpose(view(buf.Ep, :, 1:ns)), pwp)
            mul!(eb, transpose(view(buf.Em, :, 1:ns)), pwm, 1.0, 1.0)
            mul!(gb, transpose(view(P.dX, sets, :)), eb, -om, 1.0)

            # sigma / rho: the draw flips sign in the second half, so the
            # halves' weighted scores SUBTRACT.
            etaT = P.etaT
            for c in 1:Ci
                m = ct[c]
                P.corr4_dependent[m] && continue
                b = P.corr_second[m]
                if b == 0
                    acc = 0.0
                    @simd for r in 1:H
                        acc += (pwp[r] * buf.STp[r, c] - pwm[r] * buf.STm[r, c]) *
                               etaT[r, c0 + c]
                    end
                    gs[m] -= om * acc
                else
                    cp   = cc[2b-1]
                    rho  = corr[b]
                    root = sqrt(1.0 - rho * rho)
                    rr   = rho / root
                    acc1 = 0.0
                    acc2 = 0.0
                    @simd for r in 1:H
                        co = pwp[r] * buf.STp[r, c] - pwm[r] * buf.STm[r, c]
                        ep = etaT[r, c0 + cp]
                        eq = etaT[r, c0 + c]
                        acc1 += co * (rho * ep + root * eq)
                        acc2 += co * (ep - rr * eq)
                    end
                    gs[m] -= om * acc1
                    gc[b] -= om * sigma[m] * acc2
                end
            end
            for (b, ids) in enumerate(P.corr4_blocks)
                off = length(P.corr_pairs)+5(b-1)
                cells = view(cc, 2length(P.corr_pairs)+4(b-1)+1:2length(P.corr_pairs)+4b)
                L,J = _mlogit_rfx_corr4_factors(view(corr,off+1:off+5))
                for j in 3:4
                    c = cells[j]
                    c1,c2,c3,c4 = (c0+cells[k] for k in 1:4)
                    l1,l2,l3,l4 = (L[j,k] for k in 1:4)
                    acc = 0.0
                    @simd for r in 1:H
                        co = pwp[r]*buf.STp[r,c]-pwm[r]*buf.STm[r,c]
                        acc += co*(l1*etaT[r,c1]+l2*etaT[r,c2]+l3*etaT[r,c3]+l4*etaT[r,c4])
                    end
                    gs[ids[j]] -= om*acc
                    for h in 1:5
                        d1,d2,d3,d4 = (J[j,k,h] for k in 1:4)
                        acc = 0.0
                        @simd for r in 1:H
                            co = pwp[r]*buf.STp[r,c]-pwm[r]*buf.STm[r,c]
                            acc += co*(d1*etaT[r,c1]+d2*etaT[r,c2]+d3*etaT[r,c3]+d4*etaT[r,c4])
                        end
                        gc[off+h] -= om*sigma[ids[j]]*acc
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

    K, M, B, R = P.K, P.M, P.B, P.R
    beta  = view(theta, 1:K)
    sigma = view(theta, K+1:K+M)
    corr  = view(theta, K+M+1:K+M+B)

    ess = Vector{Float64}(undef, P.N)

    if P.kernel === :binary_antithetic_simd
        @inbounds for i in 1:P.N
            _mlogit_rfx_simd_pass!(P, i, theta, buf, false)
            lse = logsumexp(buf.ll)
            buf.pw .= exp.(buf.ll .- lse)
            ess[i] = 1.0 / sum(abs2, buf.pw)
        end
        return ess
    end

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
        cc   = view(P.corr_cells, i, :)
        Ai   = view(buf.A, 1:Ci, :)
        Vi   = view(buf.V, 1:Ti, :)
        xb   = view(buf.xb, 1:Ti)

        M > 0 && _mlogit_rfx_fill_A!(Ai, beta, sigma, corr, etai, ct, cc, P)

        mul!(xb, Xi, beta)
        if P.kernel === :binary
            _mlogit_rfx_binary_pass!(P, i, r0, Ti, Ai, Vi, view(buf.E, 1:Ti, :),
                                     view(buf.S, 1:Ci, :), buf.ll, xb, view(P.yvec, rrng),
                                     false, false)
        else
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
        end # kernel

        lse = logsumexp(buf.ll)
        buf.pw .= exp.(buf.ll .- lse)
        ess[i] = 1.0 / sum(abs2, buf.pw)
    end

    return ess
end


# ----------------------------------------------------------------------------
# Starting values
# ----------------------------------------------------------------------------

"""Map public `[beta; sigma; corr]` parameters to unconstrained coordinates."""
function _mlogit_rfx_unconstrained_start(theta0::Vector{Float64}, K::Int, M::Int,
                                         B::Int)
    phi = _rfx_positive_start(theta0, K, M)
    @inbounds for b in 1:B
        phi[K + M + b] = atanh(theta0[K + M + b] / _MLOGIT_RFX_CORR_LIMIT)
    end
    return phi
end

"""Map unconstrained coordinates to public `[beta; sigma; corr]` in place."""
@inline function _mlogit_rfx_public_theta!(theta, phi, K::Int, M::Int, B::Int)
    _rfx_positive_theta!(theta, phi, K, M)
    @inbounds for b in 1:B
        theta[K + M + b] = _MLOGIT_RFX_CORR_LIMIT * tanh(phi[K + M + b])
    end
    return theta
end

"""Optim adapter applying softplus to standard deviations and tanh to correlations."""
function _mlogit_rfx_unconstrained_fg!(F, G, phi::Vector{Float64}, K::Int, M::Int,
                                       B::Int, theta::Vector{Float64},
                                       grad_theta::Vector{Float64}, kernel)
    _mlogit_rfx_public_theta!(theta, phi, K, M, B)
    value = kernel(F, isnothing(G) ? nothing : grad_theta, theta)

    if !isnothing(G)
        @views G[1:K] .= grad_theta[1:K]
        @inbounds for m in 1:M
            G[K + m] = grad_theta[K + m] * logistic(phi[K + m])
        end
        @inbounds for b in 1:B
            z = tanh(phi[K + M + b])
            G[K + M + b] = grad_theta[K + M + b] *
                             _MLOGIT_RFX_CORR_LIMIT * (1.0 - z * z)
        end
    end

    return value
end

"""
    theta0_mlogit_rfx(formula, rfx; b0, s0, rfx_corr, corr0, col_group)
        -> Vector{Float64}

which is the value whose implied mean coefficient `+-exp(mu + sigma^2/2)` equals
`b0_k`.
"""
function theta0_mlogit_rfx(formula, rfx;
                           b0 = zeros(length(formula)),
                           s0 = fill(0.5, length(rfx)),
                           rfx_corr = [],
                           rfx_corr4 = [],
                           corr0 = fill(0.0, length(rfx_corr)+5length(rfx_corr4)),
                           col_group::Symbol = :__group__)

    K = length(formula)
    M = length(rfx)
    terms = _normalize_mlogit_rfx(rfx, Symbol.(formula), col_group)
    pairs = _normalize_mlogit_rfx_corr(rfx_corr, terms)
    B = length(pairs)+5length(_normalize_mlogit_rfx_corr4(rfx_corr4, terms, pairs))

    length(b0) == K || error("b0 has length $(length(b0)) but formula has $K variables")
    length(s0) == M || error("s0 has length $(length(s0)) but rfx has $M terms")
    length(corr0) == B || error(
        "corr0 has length $(length(corr0)) but the correlation specification has $B coordinates")
    all(>(_RFX_SIGMA_FLOOR), s0) || error(
        "all s0 values must exceed the numerical sigma floor " *
        "$(_RFX_SIGMA_FLOOR)")
    all(c -> isfinite(c) && abs(c) < _MLOGIT_RFX_CORR_LIMIT, corr0) || error(
        "all corr0 values must be finite and strictly between " *
        "$(-_MLOGIT_RFX_CORR_LIMIT) and $(_MLOGIT_RFX_CORR_LIMIT)")

    th = Float64[b0...; s0...; corr0...]

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
    theta0_mlogit_rfx_multistart(formula, rfx; b0, nstarts, s_range, corr_range,
                                 b_jitter, seed, rfx_corr, col_group) -> Matrix

An `nstarts x (K + M + B)` matrix of starting values for [`mlogit_rfx`](@ref), **one
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
                                      rfx_corr = [],
                           rfx_corr4 = [],
                                      corr0 = fill(0.0, length(rfx_corr)+5length(rfx_corr4)),
                                      corr_range = (-0.8, 0.8),
                                      b_jitter::Real = 0.0,
                                      seed::Int = 20260808,
                                      col_group::Symbol = :__group__)

    nstarts >= 1 || error("nstarts must be at least 1; got $nstarts")
    lo, hi = float(s_range[1]), float(s_range[2])
    (0 < lo && lo <= hi) || error("s_range must satisfy 0 < first <= last; got $s_range")
    clo, chi = float(corr_range[1]), float(corr_range[2])
    (-_MLOGIT_RFX_CORR_LIMIT < clo <= chi < _MLOGIT_RFX_CORR_LIMIT) || error(
        "corr_range must lie strictly inside (-1, 1); got $corr_range")
    b_jitter >= 0 || error("b_jitter must be >= 0; got $b_jitter")

    terms = _normalize_mlogit_rfx(rfx, Symbol.(formula), col_group)
    pairs = _normalize_mlogit_rfx_corr(rfx_corr, terms)
    B = length(pairs)+5length(_normalize_mlogit_rfx_corr4(rfx_corr4, terms, pairs))
    length(corr0) == B || error(
        "corr0 has length $(length(corr0)) but the correlation specification has $B coordinates")
    K, M = length(formula), length(rfx)
    rng  = MersenneTwister(seed)

    out = Matrix{Float64}(undef, nstarts, K + M + B)
    out[1, :] .= theta0_mlogit_rfx(formula, rfx; b0 = b0, rfx_corr = rfx_corr, rfx_corr4 = rfx_corr4,
                                   corr0 = corr0, col_group = col_group)

    for r in 2:nstarts
        s = exp.(log(lo) .+ (log(hi) - log(lo)) .* rand(rng, M))
        c = B == 0 ? Float64[] : clo .+ (chi - clo) .* rand(rng, B)
        b = b_jitter > 0 ? collect(Float64, b0) .* exp.(b_jitter .* randn(rng, K)) : b0
        out[r, :] .= theta0_mlogit_rfx(formula, rfx; b0 = b, s0 = s,
                                       rfx_corr = rfx_corr, rfx_corr4 = rfx_corr4, corr0 = c,
                                       col_group = col_group)
    end

    return out
end

"""Validate `theta0` against a prepped model."""
function _check_theta0_mlogit_rfx(theta0, P::MlogitRfxPrep)
    th   = Float64.(collect(theta0))
    npar = P.K + P.M + P.B

    length(th) == npar || error(
        "theta0 has length $(length(th)) but the model has K + M + B = $(P.K) + " *
        "$(P.M) + $(P.B) = $npar parameters. Use " *
        "theta0_mlogit_rfx(formula, rfx; rfx_corr = ...) to assemble it.")

    if P.M > 0 && any(th[P.K+1:P.K+P.M] .< _RFX_SIGMA_FLOOR)
        bad = findall(th[P.K+1:P.K+P.M] .< _RFX_SIGMA_FLOOR)
        error("theta0 must have strictly positive sigma for rfx term(s) " *
              "$([P.terms[b].name for b in bad]); got $(th[P.K .+ bad]). Standard " *
              "deviations are constrained at or above the numerical floor " *
              "$(_RFX_SIGMA_FLOOR) during optimisation. Use a start " *
              "such as 0.5.")
    end

    if P.B > 0
        c = view(th, P.K+P.M+1:npar)
        all(x -> isfinite(x) && abs(x) < _MLOGIT_RFX_CORR_LIMIT, c) || error(
            "theta0 correlations must be finite and strictly inside (-1, 1); got " *
            "$(collect(c))")
    end

    return th
end

"""
    _mlogit_rfx_theta0_matrix(theta0, P) -> Matrix{Float64}

Normalise `theta0` to an `nstarts x npar` matrix and validate every row. Accepts a
plain vector (one start), a matrix with `npar` columns, or a vector of vectors.
"""
function _mlogit_rfx_theta0_matrix(theta0, P::MlogitRfxPrep)
    npar = P.K + P.M + P.B

    if theta0 isa AbstractMatrix
        size(theta0, 2) == npar || error(
            "theta0 has $(size(theta0, 2)) columns but the model has K + M + B = $npar " *
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

"""
    _mlogit_rfx_theta0_cube(theta0, P, nboot) -> Array{Float64,3}

Validate replicate-specific bootstrap starts. The dimensions are
`nstarts × npar × nboot`: every `theta0[:, :, b]` is the ordinary start matrix
for bootstrap replicate `b`. Keeping the existing matrix dimensions first makes
each replicate's slice contiguous and preserves the row-per-start convention of
[`_mlogit_rfx_theta0_matrix`](@ref).
"""
function _mlogit_rfx_theta0_cube(theta0::AbstractArray{<:Real,3},
                                 P::MlogitRfxPrep, nboot::Int)
    size(theta0, 1) >= 1 || error(
        "replicate-specific theta_start must contain at least one start")
    size(theta0, 2) == P.K + P.M + P.B || error(
        "replicate-specific theta_start has $(size(theta0, 2)) parameters but " *
        "the model has K + M + B = $(P.K + P.M + P.B)")
    size(theta0, 3) == nboot || error(
        "replicate-specific theta_start has $(size(theta0, 3)) bootstrap slices " *
        "but nboot=$nboot")

    out = Array{Float64,3}(undef, size(theta0))
    for b in 1:nboot
        out[:, :, b] .= _mlogit_rfx_theta0_matrix(view(theta0, :, :, b), P)
    end
    return out
end


# ----------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------

"""Optimizer settings worth recording for provenance: LBFGS memory, line search."""
function _mlogit_rfx_optimizer_config(opt)
    cfg = Pair{Symbol,Any}[]
    hasproperty(opt, :m) && push!(cfg, :m => getproperty(opt, :m))
    if hasproperty(opt, :linesearch!)
        push!(cfg, :linesearch => string(typeof(getproperty(opt, :linesearch!))))
    end
    if hasproperty(opt, :alphaguess!)
        push!(cfg, :alphaguess => string(typeof(getproperty(opt, :alphaguess!))))
    end
    return (; cfg...)
end

"""
    _mlogit_rfx(P, theta0, gw, optim_options; rethrow_errors = false) -> MLEFit

Inner estimation routine: takes a prepped [`MlogitRfxPrep`](@ref), so the
bootstrap can reuse prep and vary only the group weights `gw`.

Enforces `sigma > 0` during optimisation with the shared softplus transformation.
With finite antithetic draws, applying `abs()` to separate sigma components after
an unconstrained fit can change the simulated objective; the transformation keeps
the optimiser and the returned parameters in the same identified orthant.

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
        rethrow_errors::Bool = false,
        optimizer::Optim.AbstractOptimizer = LBFGS())

    npar = P.K + P.M + P.B

    local myfit
    try
        buf = MlogitRfxBuffers(P)
        theta_work = similar(theta0)
        grad_work  = similar(theta0)
        kernel = (F, G, theta) -> _mlogit_rfx_fg!(F, G, theta, P, buf, gw)
        fg! = (F, G, phi) -> _mlogit_rfx_unconstrained_fg!(
            F, G, phi, P.K, P.M, P.B, theta_work, grad_work, kernel)

        phi0 = _mlogit_rfx_unconstrained_start(theta0, P.K, P.M, P.B)

        time_it_took = @elapsed opt = optimize(
            Optim.only_fg!(fg!), phi0, optimizer, optim_options)

        th = similar(theta0)
        _mlogit_rfx_public_theta!(th, Optim.minimizer(opt), P.K, P.M, P.B)

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
                       n_groups = P.N, K = P.K, M = P.M, B = P.B, R = P.R,
                       col_id = P.col_group,          # the integration unit
                       col_set = P.col_id,            # the softmax group
                       n_sets = P.n_sets,
                       rfx = P.rfx_pairs, rfx_cols = P.rfx_cols,
                       rfx_terms = P.terms, cell_stats = P.cell_stats,
                       rfx_corr = P.corr_pairs, rfx_corr4 = P.corr4_blocks, corr_names = P.corr_names,
                       seed = P.seed,
                       kernel = P.kernel, kernel_requested = P.kernel_requested,
                       optimizer = Optim.summary(optimizer),
                       optimizer_type = string(typeof(optimizer)),
                       optimizer_config = _mlogit_rfx_optimizer_config(optimizer),
                       sigma_parameterization = RFX_SIGMA_PARAMETERIZATION,
                       corr_parameterization = MLOGIT_RFX_CORR_PARAMETERIZATION,
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
                           obj_tol::Float64 = 1e-4,
                           optimizer::Optim.AbstractOptimizer = LBFGS())

    nstarts = size(theta0s, 1)

    task = r -> _mlogit_rfx(P, Vector{Float64}(view(theta0s, r, :)), gw, optim_options;
                            rethrow_errors = rethrow_errors, optimizer = optimizer)

    fits = if parallel
        # The prepped simulated likelihood can be hundreds of MB at large R.
        # A CachingPool keeps the task closure (and therefore P) resident on
        # every worker until it is explicitly cleared. Repeated multi-start
        # calls at different draw seeds otherwise accumulate old P objects and
        # can eventually fail while serialising the next one. Scope the cache to
        # this fit and release it deterministically when pmap returns or throws.
        pool = CachingPool(workers())
        try
            pmap(task, pool, 1:nstarts)
        finally
            clear!(pool)
        end
    else
        map(task, 1:nstarts)
    end

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

Multinomial (conditional) logit with random coefficients, estimated by maximum
simulated likelihood with an analytic gradient. Coefficients are independent by
default; `rfx_corr` adds disjoint bivariate blocks, while `rfx_corr4` adds
ordered four-term normal blocks with zero marginal covariance between the first
two terms. The latter permits all five remaining correlations subject to positive
definiteness (approaching the semidefinite boundary).

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
- `theta0`: length `K + M + B` (see [`theta0_mlogit_rfx`](@ref)), or an
  `nstarts x (K + M + B)` matrix -- one start per **row** -- for a multi-start fit.

# Keywords
- `col_group = nothing`: integration unit. Defaults to `col_id`, i.e. one choice
  set per individual (the textbook cross-sectional mixed logit).
- `rfx = []`: random-coefficient terms. `:x` and `:x => :lognormal` mean what they
  mean in `logit2_rfx`; [`rfx_term`](@ref) adds the level, the intercept form,
  and the bounded families `:uniform` and `:triangular` (standardised, so their
  `sigma` is the coefficient's standard deviation exactly as under `:normal`).
- `rfx_corr4 = []`: ordered four-tuples of group-level normal slopes, disjoint
  from all other blocks. For example `[(:anticipated, :cost, :familiarity, :realized)]`
  fixes `Cov(anticipated,cost)=0`. Each block appends five bounded partial
  correlation coordinates in order `(3,1), (3,2 | 1), (4,1), (4,2 | 1), (4,3 | 1,2)`.
  These are labelled `pcor_`; they are not all marginal correlations. Use
  [`mlogit_rfx_correlation_matrix`](@ref) to recover marginal correlations, and
  transform each bootstrap replicate before reporting their uncertainty.
- `rfx_corr = []`: disjoint pairs of group-level term names, of one linear
  family, whose coefficients are correlated, e.g. `[(:net_value, :training)]`.
  Each pair adds one correlation parameter. Pair order is canonicalised to
  formula-term order. See [`_normalize_mlogit_rfx_corr`](@ref) for what the
  block means under a bounded family.
- `ndraws = 1000`: simulation draws, must be even (antithetic pairing).
- `seed = 20260808`: draw seed. Draws are generated once and reused for every
  function evaluation, so the objective is a deterministic function of theta.
- `weights = nothing`: column name; must be constant within `col_group`.
- `optim_options = Optim.Options()`.
- `parallel = false`: distribute a **multi-start** fit over `workers()`.
- `rethrow_errors = false`: by default a failed optimisation is captured into
  `errored`/`error_message`; `true` lets the exception propagate.
- `kernel = :general`: likelihood kernel. `:general` handles any choice-set
  size and distribution and is the reference. `:binary` uses the logistic form
  when every choice set has two rows; `:binary_antithetic_simd` additionally
  requires linear terms only (no lognormal) and computes each antithetic pair of draws once with
  draw-contiguous storage (about 4x faster than `:general` per evaluation on the
  two-option Table 2 panel). `:auto` picks the most specialised eligible kernel;
  naming an ineligible one is an error. All kernels compute the same likelihood
  and agree to roundoff (about 1e-12 relative on that panel), not bit for bit,
  so an optimiser can follow a slightly different path; the default therefore
  stays `:general` until a run has been validated under the fast kernel. The
  requested and resolved kernels are recorded in `extra.kernel_requested` and
  `extra.kernel`.
- `optimizer = LBFGS()`: any first-order `Optim` algorithm. With about a hundred
  parameters full `BFGS()` keeps far better curvature than the limited-memory
  default: on the Table 2 fixed-effect specification it reached the same
  optimum in 284 evaluations where LBFGS took 1,413 iterations, and its dense
  matrix costs nothing next to one likelihood evaluation. That is a benchmark
  result on one model, not a guarantee. Recorded in `extra.optimizer`,
  `extra.optimizer_type` and `extra.optimizer_config`.

# Parameter ordering
`theta = [mu (K, formula order); sigma (M, rfx order); corr (B coordinates)]`,
where correlation coordinates list all bivariate correlations first, followed by
five partial correlations for each ordered four-term block. Means use formula
names, standard deviations use `sd_`, bivariate correlations use `cor_`, and
four-term coordinates use `pcor_`. A term's name is `x` at the group level and `x|level` / `1|level` otherwise. Returned
`sigma` is always `> 0`, enforced through softplus; correlations stay strictly
inside `(-1,1)` through a scaled `tanh`. This keeps the stored objective and
public parameters on exactly the same constrained parameterisation.

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
deterministic in theta), simulation error never enters `boot_se`, and the positive
constraint makes noise around a true zero read as a positive number.

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
        rfx_corr = [],
        rfx_corr4 = [],
        ndraws::Int = 1000,
        seed::Int = 20260808,
        weights::Union{Nothing, Symbol, String} = nothing,
        optim_options::Optim.Options = Optim.Options(),
        parallel::Bool = false,
        rethrow_errors::Bool = false,
        kernel::Symbol = :general,
        optimizer::Optim.AbstractOptimizer = LBFGS())

    cid = Symbol(col_id)
    cg  = isnothing(col_group) ? cid : Symbol(col_group)

    P, gw = _prep_mlogit_rfx(data_df, formula, cid, col_selected, cg, rfx,
                             ndraws, seed, weights, rfx_corr; rfx_corr4 = rfx_corr4, kernel = kernel)

    theta0s = _mlogit_rfx_theta0_matrix(theta0, P)

    myfit = if size(theta0s, 1) == 1
        _mlogit_rfx(P, vec(theta0s), gw, optim_options;
                    rethrow_errors = rethrow_errors, optimizer = optimizer)
    else
        parallel && _check_boot_workers()
        _mlogit_rfx_multi(P, theta0s, gw, optim_options;
                          parallel = parallel, rethrow_errors = rethrow_errors,
                          optimizer = optimizer)
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
