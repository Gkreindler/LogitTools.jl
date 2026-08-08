
###############################################################################
# Binary logit with independent normal random coefficients, estimated by
# maximum simulated likelihood (MSL) with an analytic gradient.
#
# Model (group i, observation t, draw r, q_it = 2*y_it - 1):
#
#   v_itr = X_it'β + Σ_m σ_m · Z_itm · η_irm        η_ir ~ N(0, I_M)
#   λ_itr = Λ(q_it · v_itr)
#   ℓ_ir  = Σ_t log λ_itr
#   log L̂_i = logsumexp_r(ℓ_ir + logw_r)
#
# Objective (minimised):  Q(θ) = - Σ_i ω_i · log L̂_i,  θ = [β (K); σ (M)]
###############################################################################

# ----------------------------------------------------------------------------
# Draws
# ----------------------------------------------------------------------------

"""
    _make_draws(M, R, N, seed; scheme = :mc)

Generate individual-specific antithetic standard normal draws.

Returns `(eta, logw)` where `eta` is `M × R × N` and `logw` is length `R`.

`(nodes, log_weights)` is the integration-rule abstraction: Monte Carlo sets
`logw = -log(R)`; a Halton or Gauss-Hermite scheme would swap only the node
generator (and, for quadrature, supply real weights). The likelihood loop and
gradient are identical in all three cases, so do not inline `-log(R)`.

Antithetic reflection is mandatory, not optional: it halves simulation variance
and it is what makes the `σ → -σ` symmetry and the `σ = 0` saddle point *exact*.
"""
function _make_draws(M::Int, R::Int, N::Int, seed::Int; scheme::Symbol = :mc)
    scheme === :mc || error("only :mc is implemented; :halton and :gauss_hermite are hooks. Got scheme=:$scheme")
    iseven(R) || error("ndraws must be even (antithetic pairing); got $R")

    rng  = MersenneTwister(seed)
    half = R ÷ 2
    eta  = Array{Float64,3}(undef, M, R, N)

    for i in 1:N
        e = randn(rng, M, half)
        @views eta[:, 1:half,   i] .=   e
        @views eta[:, half+1:R, i] .= .-e
    end

    logw = fill(-log(R), R)

    return eta, logw
end


# ----------------------------------------------------------------------------
# Prep
# ----------------------------------------------------------------------------

"""
Precomputed, worker-serialisable model data for `logit2_rfx`.

Every field is concretely typed: this struct is shipped to workers during the
bootstrap, and abstract fields would also deoptimise the inner loop.
"""
struct RfxPrep
    xmatrix::Matrix{Float64}          # Nobs × K, sorted by col_id
    zmatrix::Matrix{Float64}          # Nobs × M, sorted by col_id
    yvec::Vector{Float64}             # Nobs
    q::Vector{Float64}                # Nobs, = 2y - 1
    ranges::Vector{UnitRange{Int}}    # length N, contiguous row range per group
    eta::Array{Float64,3}             # M × R × N
    logw::Vector{Float64}             # R
    group_ids::Vector                 # length N, unique col_id values in sorted order
    K::Int
    M::Int
    R::Int
    N::Int
    Tmax::Int
    rfx_pairs::Vector{Pair{Symbol,Symbol}}
    theta_names::Vector{String}
    col_id::Symbol
    seed::Int
end

"""
    _normalize_rfx(rfx) -> Vector{Pair{Symbol,Symbol}}

Accept `[:dur, :dist]` or `[:dur => :normal, ...]` and normalise to pairs.
Only `:normal` is supported today; this is the extension point for lognormal
coefficients later.
"""
function _normalize_rfx(rfx)
    pairs = Pair{Symbol,Symbol}[]
    for r in rfx
        if isa(r, Pair)
            push!(pairs, Symbol(first(r)) => Symbol(last(r)))
        else
            push!(pairs, Symbol(r) => :normal)
        end
    end

    for (v, d) in pairs
        d === :normal || error(
            "rfx distribution :$d is not supported for variable :$v. Only :normal is implemented.")
    end

    return pairs
end

"""
    theta0_rfx(formula, rfx; b0, s0) -> Vector{Float64}

Assemble a starting vector `[β; σ]` for `logit2_rfx`.

Default `s0 = 0.5`, never `0.0`: σ = 0 is a stationary point (a saddle) of the
simulated likelihood, so an optimiser started there cannot move.
"""
function theta0_rfx(formula, rfx;
                    b0 = zeros(length(formula)),
                    s0 = fill(0.5, length(rfx)))

    K = length(formula)
    M = length(rfx)

    length(b0) == K || error("b0 has length $(length(b0)) but formula has $K variables")
    length(s0) == M || error("s0 has length $(length(s0)) but rfx has $M variables")

    return Float64[b0...; s0...]
end

"""
    _prep_logit2_rfx(data_df, formula, choice, col_id, rfx, ndraws, seed, weights)

Build an `RfxPrep`. Does **not** mutate `data_df`.

Returns `(P, gw)` where `gw` is the vector of `N` group-level weights (or
`nothing` when no weights column was given).
"""
function _prep_logit2_rfx(
        data_df,
        formula,
        choice,
        col_id::Symbol,
        rfx,
        ndraws::Int,
        seed::Int,
        weights::Union{Nothing, Symbol, String})

    formula_syms = Symbol.(formula)
    K = length(formula_syms)

    # --- rfx validation -----------------------------------------------------
    rfx_pairs = _normalize_rfx(rfx)
    rfx_syms  = first.(rfx_pairs)
    M = length(rfx_syms)

    missing_rfx = setdiff(rfx_syms, formula_syms)
    isempty(missing_rfx) || error(
        "rfx variable(s) $(missing_rfx) are not in formula. Available: $(formula_syms)")

    if length(unique(rfx_syms)) != M
        dups = unique([s for s in rfx_syms if count(==(s), rfx_syms) > 1])
        error("rfx contains duplicate variable(s) $(dups)")
    end

    # --- column presence ----------------------------------------------------
    dfnames = Symbol.(names(data_df))
    for c in vcat(formula_syms, Symbol(choice), col_id)
        c in dfnames || error("column :$c not found in data_df")
    end

    # --- sort by group ------------------------------------------------------
    idcol = data_df[:, col_id]
    perm  = sortperm(idcol)

    xmatrix = Float64.(Matrix(data_df[perm, formula_syms]))
    ids_sorted = idcol[perm]

    # zmatrix: materialised copy (not a view) so the inner loop gets contiguous
    # columns for BLAS
    zcols = [findfirst(==(s), formula_syms) for s in rfx_syms]
    zmatrix = M == 0 ? Matrix{Float64}(undef, size(xmatrix, 1), 0) :
                       Matrix{Float64}(xmatrix[:, zcols])

    # --- outcome ------------------------------------------------------------
    yvec = Float64.(data_df[perm, Symbol(choice)])
    all((yvec .== 0.0) .| (yvec .== 1.0)) ||
        error("choice column should have 0's and 1's only")
    q = 2.0 .* yvec .- 1.0

    # --- group ranges (groups are contiguous after sorting) -----------------
    Nobs = length(ids_sorted)
    Nobs > 0 || error("data_df has no rows")

    ranges    = UnitRange{Int}[]
    group_ids = similar(ids_sorted, 0)
    start = 1
    for t in 2:Nobs
        if ids_sorted[t] != ids_sorted[t-1]
            push!(ranges, start:(t-1))
            push!(group_ids, ids_sorted[start])
            start = t
        end
    end
    push!(ranges, start:Nobs)
    push!(group_ids, ids_sorted[start])

    N    = length(ranges)
    Ti   = length.(ranges)
    Tmax = maximum(Ti)

    # --- panel-shape guards -------------------------------------------------
    if M > 0
        Tmax == 1 && error(
            "every group in :$col_id is a singleton (all T_i == 1); " *
            "σ is not identified without repeated observations per group")

        if median(Ti) < 3
            @warn "median group size is $(median(Ti)) (< 3): random coefficients are " *
                  "weakly identified with very short panels"
        end

        # within-group variation of each rfx variable
        for (j, s) in enumerate(rfx_syms)
            z = view(zmatrix, :, j)
            tot = sum(abs2, z .- mean(z))
            wth = 0.0
            for rg in ranges
                zg = view(z, rg)
                length(zg) < 2 && continue
                wth += sum(abs2, zg .- mean(zg))
            end
            if tot > 0 && wth < 0.01 * tot
                @warn "rfx variable :$s has within-group variance $(round(wth/tot*100, digits=3))% " *
                      "of its total variance: a random coefficient on a (near) group-invariant " *
                      "regressor is near-collinear with a random intercept"
            end
        end
    end

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
                    "weights column :$wsym is not constant within :$col_id " *
                    "(group $(group_ids[i]) has values $(w1) and $(wvec[t])). " *
                    "The weight multiplies the group log-likelihood, so it must be " *
                    "constant within group.")
            end
            gw[i] = w1
        end
    end

    # --- draws --------------------------------------------------------------
    eta, logw = _make_draws(M, ndraws, N, seed)

    theta_names = [string.(formula_syms); "sd_" .* string.(rfx_syms)]

    P = RfxPrep(
        xmatrix, zmatrix, yvec, q, ranges, eta, logw, group_ids,
        K, M, ndraws, N, Tmax, rfx_pairs, theta_names, col_id, seed)

    return P, gw
end


# ----------------------------------------------------------------------------
# Buffers
# ----------------------------------------------------------------------------

"""
Scratch space for the fused objective/gradient kernel, allocated once per fit
and reused across groups via views.
"""
struct RfxBuffers
    V::Matrix{Float64}       # Tmax × R    linear index
    E::Matrix{Float64}       # Tmax × R    residuals e_itr
    A::Matrix{Float64}       # M × R       σ .* η_i
    S::Matrix{Float64}       # M × R       Zi' * E
    ll::Vector{Float64}      # R
    pw::Vector{Float64}      # R           posterior weights τ
    xb::Vector{Float64}      # Tmax
    ebar::Vector{Float64}    # Tmax
end

function RfxBuffers(P::RfxPrep)
    RfxBuffers(
        Matrix{Float64}(undef, P.Tmax, P.R),
        Matrix{Float64}(undef, P.Tmax, P.R),
        Matrix{Float64}(undef, P.M, P.R),
        Matrix{Float64}(undef, P.M, P.R),
        Vector{Float64}(undef, P.R),
        Vector{Float64}(undef, P.R),
        Vector{Float64}(undef, P.Tmax),
        Vector{Float64}(undef, P.Tmax),
    )
end


# ----------------------------------------------------------------------------
# Fused objective and gradient
# ----------------------------------------------------------------------------

"""
    _rfx_fg!(F, G, θ, P, buf, gw) -> objective or nothing

Fused objective/gradient in the `Optim.only_fg!` convention: fills `G` when
`G !== nothing`, returns the objective when `F !== nothing`.

The draw dimension is collapsed *before* the design matrix is touched:

    ē_it = Σ_r τ_ir · e_itr      O(T_i · R)   → a T_i-vector
    ∂/∂β = Σ_t ē_it · X_it       O(T_i · K)   → one gemv against Xi

Building an augmented per-draw design matrix instead would cost O(Nobs·R·K),
a ~90× penalty at the target dimensions. Only the σ-gradient needs the draw
dimension, and M is small.
"""
function _rfx_fg!(F, G, θ::Vector{Float64}, P::RfxPrep, buf::RfxBuffers,
                  gw::Union{Nothing, Vector{Float64}})

    K, M, R = P.K, P.M, P.R

    β = view(θ, 1:K)
    σ = view(θ, K+1:K+M)

    need_g = G !== nothing
    if need_g
        fill!(G, 0.0)
    end
    gβ = need_g ? view(G, 1:K)     : nothing
    gσ = need_g ? view(G, K+1:K+M) : nothing

    Q = 0.0

    @inbounds for i in 1:P.N
        rng = P.ranges[i]
        Ti  = length(rng)
        ω_i = isnothing(gw) ? 1.0 : gw[i]

        Xi = view(P.xmatrix, rng, :)      # Ti × K
        Zi = view(P.zmatrix, rng, :)      # Ti × M
        qi = view(P.q,       rng)         # Ti
        ηi = view(P.eta, :, :, i)         # M × R (contiguous)

        Vi = view(buf.V, 1:Ti, :)         # Ti × R
        Ei = view(buf.E, 1:Ti, :)         # Ti × R
        xb = view(buf.xb,   1:Ti)
        eb = view(buf.ebar, 1:Ti)

        # --- 1. linear index ------------------------------------------------
        mul!(xb, Xi, β)                   # Ti
        if M > 0
            buf.A .= σ .* ηi              # M × R
            mul!(Vi, Zi, buf.A)           # Ti × R
            Vi .+= xb                     # broadcast down columns
        else
            Vi .= xb
        end

        # --- 2. log-lik per draw and residuals, in ONE pass ------------------
        # Never form ∏_t λ_it directly: with T_i = 21 and λ ≈ 0.5 that is ~1e-7
        # and longer panels underflow. Accumulate ℓ_ir in logs.
        buf.ll .= P.logw
        for r in 1:R, t in 1:Ti
            a      = qi[t] * Vi[t, r]
            loglam = -log1pexp(-a)                # stable log λ
            buf.ll[r] += loglam
            Ei[t, r]   = -qi[t] * expm1(loglam)   # = q*(1-λ) = y - Λ(v), exact
        end

        # --- 3. group log-likelihood and posterior weights -------------------
        lse   = logsumexp(buf.ll)
        logLi = lse
        buf.pw .= exp.(buf.ll .- lse)

        # --- 4. gradient: collapse draws FIRST -------------------------------
        if need_g
            mul!(eb, Ei, buf.pw)                       # Ti   ē_it = Σ_r τ_ir e_itr
            mul!(gβ, Xi', eb, -ω_i, 1.0)               # gβ -= ω_i · Xi' ē
            if M > 0
                mul!(buf.S, Zi', Ei)                   # M × R  S_mr = Σ_t Z_itm e_itr
                # r outer / m inner: buf.S and ηi are M × R and column-major
                for r in 1:R, m in 1:M
                    gσ[m] -= ω_i * buf.pw[r] * ηi[m, r] * buf.S[m, r]
                end
            end
        end

        # --- 5. objective ----------------------------------------------------
        Q -= ω_i * logLi
    end

    return F !== nothing ? Q : nothing
end
