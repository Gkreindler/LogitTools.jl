
###############################################################################
# Binary logit with independent normal random coefficients, estimated by
# maximum simulated likelihood (MSL) with an analytic gradient.
#
# Model (group i, observation t, draw r, q_it = 2*y_it - 1):
#
#   v_itr = Σ_{k not lognormal-rfx} β_k·X_itk + Σ_m β_irm · Z_itm
#   λ_itr = Λ(q_it · v_itr)
#
# with η_ir ~ N(0, I_M) and, per rfx variable m,
#
#   :normal         β_irm =  μ_m + σ_m·η_irm       (μ_m enters through X'β)
#   :lognormal      β_irm =  exp(μ_m + σ_m·η_irm)
#   :neg_lognormal  β_irm = -exp(μ_m + σ_m·η_irm)
#
# For the lognormal families μ_m is the mean of log|β_irm| and that variable's X
# column is dropped from the linear part (the exp() already carries the level),
# so θ = [μ (K); σ (M)] in every case. The σ = 0 saddle and the σ -> -σ mirror
# symmetry hold for all three: with antithetic draws {η_r} = {-η_r} as a set.
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
    xlin::Matrix{Float64}             # xmatrix with lognormal-rfx columns zeroed;
                                      # === xmatrix when any_log is false
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
    rfx_cols::Vector{Int}             # formula index of each rfx variable
    rfx_islog::Vector{Bool}           # lognormal family?
    rfx_sgn::Vector{Float64}          # +1, or -1 for :neg_lognormal
    any_log::Bool                     # gates every lognormal code path
    theta_names::Vector{String}
    col_id::Symbol
    seed::Int
end

"""Distributions accepted in `rfx`. A bare symbol in `rfx` means `:normal`."""
const _RFX_DISTS = (:normal, :lognormal, :neg_lognormal)

"""Is `d` a lognormal family, i.e. parameterised on the log scale?"""
_rfx_is_log(d::Symbol) = (d === :lognormal) || (d === :neg_lognormal)

"""Sign of the coefficient's support: `-1.0` for `:neg_lognormal`, else `+1.0`."""
_rfx_sign(d::Symbol) = d === :neg_lognormal ? -1.0 : 1.0

"""
    _normalize_rfx(rfx) -> Vector{Pair{Symbol,Symbol}}

Accept `[:dur, :dist]` or `[:dur => :normal, :tfx => :lognormal, ...]` and
normalise to pairs. A bare symbol means `:normal`, so every call written before
the lognormal families existed keeps its meaning exactly.

Supported distributions, for the coefficient `β_im` on rfx variable `m`:

    :normal         β_im =  μ_m + σ_m·η_im          support ℝ
    :lognormal      β_im =  exp(μ_m + σ_m·η_im)     support (0, ∞)
    :neg_lognormal  β_im = -exp(μ_m + σ_m·η_im)     support (-∞, 0)

In all three cases `θ = [μ (K); σ (M)]`, but for the lognormal families `μ_m`
and `σ_m` are the mean and standard deviation of `log|β_im|`, **not** of `β_im`
itself. Use [`rfx_level_moments`](@ref) to get the level moments.
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
        d in _RFX_DISTS || error(
            "rfx distribution :$d is not supported for variable :$v. " *
            "Supported: $(join(string.(":", _RFX_DISTS), ", ")).")
    end

    return pairs
end

"""
    theta0_rfx(formula, rfx; b0, s0) -> Vector{Float64}

Assemble a starting vector `[μ; σ]` for `logit2_rfx`.

Default `s0 = 0.5`, never `0.0`: σ = 0 is a stationary point (a saddle) of the
simulated likelihood, so an optimiser started there cannot move.

`b0` is given on the **level** scale for every variable — the scale of a plain
`logit2` coefficient — including the lognormal ones. For a `:lognormal` or
`:neg_lognormal` rfx variable at formula position `k`, this function converts it
into the log scale that `logit2_rfx` actually estimates:

    μ_k = log|b0_k| - s0_m^2 / 2

which is the value whose implied mean coefficient `±exp(μ + σ²/2)` equals `b0_k`.
Passing `log(b0_k)` yourself would instead start at a mean of `b0_k·exp(σ²/2)`.

For a lognormal variable `b0_k` must therefore be nonzero and carry the sign its
support allows (`> 0` for `:lognormal`, `< 0` for `:neg_lognormal`), so the
default `b0 = zeros(K)` cannot be used there — pass a plain `logit2` `theta_hat`.
"""
function theta0_rfx(formula, rfx;
                    b0 = zeros(length(formula)),
                    s0 = fill(0.5, length(rfx)))

    K = length(formula)
    M = length(rfx)

    length(b0) == K || error("b0 has length $(length(b0)) but formula has $K variables")
    length(s0) == M || error("s0 has length $(length(s0)) but rfx has $M variables")

    th = Float64[b0...; s0...]

    # Translate level starts into the log scale for the lognormal families. Done
    # here rather than inside the fit so that `b0` means one thing (a level
    # coefficient) whatever the mix of distributions, and so a wrong sign is
    # caught before any optimisation is paid for.
    rfx_pairs    = _normalize_rfx(rfx)
    formula_syms = Symbol.(formula)
    for (m, (v, d)) in enumerate(rfx_pairs)
        _rfx_is_log(d) || continue

        k = findfirst(==(v), formula_syms)
        isnothing(k) && error(
            "rfx variable :$v is not in formula. Available: $(formula_syms)")

        sgn = _rfx_sign(d)
        b   = sgn * th[k]
        b > 0 || error(
            "rfx variable :$v is :$d, whose support is " *
            (sgn > 0 ? "(0, ∞)" : "(-∞, 0)") * ", but b0[$k] = $(th[k]). " *
            "Pass a level starting value with the right sign — e.g. the matching " *
            "coefficient from a plain logit2 fit. The default b0 = zeros(K) cannot be " *
            "used with a lognormal random coefficient.")

        th[k] = log(b) - s0[m]^2 / 2
    end

    return th
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

    # --- lognormal bookkeeping ---------------------------------------------
    # `any_log` gates every lognormal branch, so a model with only normal
    # coefficients runs the identical arithmetic on the identical memory it ran
    # before the lognormal families existed.
    rfx_islog = Bool[_rfx_is_log(d)  for (_, d) in rfx_pairs]
    rfx_sgn   = Float64[_rfx_sign(d) for (_, d) in rfx_pairs]
    any_log   = any(rfx_islog)

    # A lognormal coefficient carries its own level inside exp(μ + σ·η), so that
    # variable's column must NOT also enter the linear X'β term: it would be
    # counted twice and μ would not be identified. Alias when there is nothing to
    # zero out.
    xlin = xmatrix
    if any_log
        xlin = copy(xmatrix)
        for m in findall(rfx_islog)
            @views xlin[:, zcols[m]] .= 0.0
        end
    end

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
        xmatrix, xlin, zmatrix, yvec, q, ranges, eta, logw, group_ids,
        K, M, ndraws, N, Tmax, rfx_pairs, zcols, rfx_islog, rfx_sgn, any_log,
        theta_names, col_id, seed)

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
# Draw-specific coefficients
# ----------------------------------------------------------------------------

"""
    _rfx_fill_A!(buf, β, σ, ηi, P)

Fill `buf.A` (M × R) with the quantity that multiplies `Zi` in the linear index,

    :normal          A[m,r] = σ_m · η_mr                      (deviation from μ_m)
    :lognormal       A[m,r] = ±exp(μ_m + σ_m · η_mr)          (the coefficient itself)

so that `Vi = Zi * A .+ Xlin * β` holds for any mix of distributions. `Xlin` has
the lognormal columns zeroed, which is why the second row is the whole
coefficient and not a deviation. The sign of `:neg_lognormal` is folded into `A`,
so nothing downstream needs to know about it.

The `any_log == false` branch is the original one-line broadcast, unchanged, and
is what runs whenever `rfx` contains no lognormal entry.
"""
@inline function _rfx_fill_A!(buf::RfxBuffers, β, σ, ηi, P::RfxPrep)

    if !P.any_log
        buf.A .= σ .* ηi
        return nothing
    end

    @inbounds for r in 1:P.R, m in 1:P.M
        buf.A[m, r] = P.rfx_islog[m] ?
            P.rfx_sgn[m] * exp(β[P.rfx_cols[m]] + σ[m] * ηi[m, r]) :
            σ[m] * ηi[m, r]
    end

    # exp() overflows to Inf above an exponent of ~709, and Inf then propagates
    # into the gradient as Inf·0 = NaN, from which LBFGS cannot recover -- it
    # would report a converged fit at a garbage θ. Fail with the cause named
    # instead. O(M·R) with M small, so this is free next to the T_i × R work.
    if !all(isfinite, buf.A)
        lg = findall(P.rfx_islog)
        error("a lognormal random coefficient overflowed: exp(μ + σ·η) is not finite " *
              "at μ = $(round.([β[P.rfx_cols[m]] for m in lg], digits = 3)), " *
              "σ = $(round.([σ[m] for m in lg], digits = 3)) " *
              "(variables $(first.(P.rfx_pairs)[lg])). μ is on the LOG scale for a " *
              "lognormal coefficient: build theta0 with theta0_rfx, which converts a " *
              "level b0 for you.")
    end

    return nothing
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

        Xi = view(P.xlin,    rng, :)      # Ti × K (lognormal columns zeroed)
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
            _rfx_fill_A!(buf, β, σ, ηi, P)   # M × R
            mul!(Vi, Zi, buf.A)              # Ti × R
            Vi .+= xb                        # broadcast down columns
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
                if !P.any_log
                    for r in 1:R, m in 1:M
                        gσ[m] -= ω_i * buf.pw[r] * ηi[m, r] * buf.S[m, r]
                    end
                else
                    # Both lognormal derivatives reuse the same S. With
                    # A_mr = ±exp(μ_m + σ_m·η_mr):
                    #   ∂v_itr/∂σ_m = η_mr · A_mr · Z_itm      (:normal: η_mr · Z_itm)
                    #   ∂v_itr/∂μ_m =         A_mr · Z_itm
                    # μ_m lives at β position rfx_cols[m], where the Xi'ē term
                    # above contributed nothing because that column is zeroed.
                    for r in 1:R, m in 1:M
                        c = ω_i * buf.pw[r] * buf.S[m, r]
                        if P.rfx_islog[m]
                            gσ[m]             -= c * ηi[m, r] * buf.A[m, r]
                            gβ[P.rfx_cols[m]] -= c * buf.A[m, r]
                        else
                            gσ[m] -= c * ηi[m, r]
                        end
                    end
                end
            end
        end

        # --- 5. objective ----------------------------------------------------
        Q -= ω_i * logLi
    end

    return F !== nothing ? Q : nothing
end


# ----------------------------------------------------------------------------
# ESS diagnostic
# ----------------------------------------------------------------------------

"""
    _rfx_ess(θ, P, buf) -> Vector{Float64}

Effective number of draws per group at `θ`:  `ESS_i = 1 / Σ_r τ_ir²`.

A long panel concentrates the posterior over `η_i`, so many draws contribute
nothing. This is the real risk at large `T_i`.
"""
function _rfx_ess(θ::Vector{Float64}, P::RfxPrep, buf::RfxBuffers)

    K, M, R = P.K, P.M, P.R
    β = view(θ, 1:K)
    σ = view(θ, K+1:K+M)

    ess = Vector{Float64}(undef, P.N)

    @inbounds for i in 1:P.N
        rng = P.ranges[i]
        Ti  = length(rng)

        Xi = view(P.xlin,    rng, :)
        Zi = view(P.zmatrix, rng, :)
        qi = view(P.q,       rng)
        ηi = view(P.eta, :, :, i)

        Vi = view(buf.V, 1:Ti, :)
        xb = view(buf.xb, 1:Ti)

        mul!(xb, Xi, β)
        if M > 0
            _rfx_fill_A!(buf, β, σ, ηi, P)
            mul!(Vi, Zi, buf.A)
            Vi .+= xb
        else
            Vi .= xb
        end

        buf.ll .= P.logw
        for r in 1:R, t in 1:Ti
            buf.ll[r] += -log1pexp(-(qi[t] * Vi[t, r]))
        end

        lse = logsumexp(buf.ll)
        buf.pw .= exp.(buf.ll .- lse)
        ess[i] = 1.0 / sum(abs2, buf.pw)
    end

    return ess
end


# ----------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------

"""
    _logit2_rfx(P, theta0, gw, optim_options; rethrow_errors = false) -> MLEFit

Inner estimation routine: takes a prepped `RfxPrep`, so the bootstrap can reuse
prep and vary only the group weights `gw`.

Canonicalises `σ ≥ 0` at the source, so that every path — the main fit and every
bootstrap replicate — returns a canonical sign. Without this, `cov(theta_boot_table)`
would mix the `2^M` mirror modes and be meaningless.

By default this never throws: on failure it returns an `MLEFit` with
`errored = true` and the message in `error_message`, so a single bad bootstrap
replicate cannot take the whole run down.

Pass `rethrow_errors = true` to disable that handler and let the original
exception propagate with its stacktrace intact. This is the debugging switch to
reach for when replicates fail and you need to see *why*.
"""
function _logit2_rfx(
        P::RfxPrep,
        theta0::Vector{Float64},
        gw::Union{Nothing, Vector{Float64}},
        optim_options::Optim.Options = Optim.Options();
        rethrow_errors::Bool = false)

    npar = P.K + P.M

    local myfit
    try
        buf = RfxBuffers(P)

        fg! = (F, G, θ) -> _rfx_fg!(F, G, θ, P, buf, gw)

        time_it_took = @elapsed opt = optimize(
            Optim.only_fg!(fg!), copy(theta0), LBFGS(), optim_options)

        # mirror-mode canonicalisation, at the source
        th = copy(Optim.minimizer(opt))
        th[P.K+1:end] .= abs.(th[P.K+1:end])

        # ESS diagnostic at the fitted parameter
        ess = P.M > 0 ? _rfx_ess(th, P, buf) : fill(Float64(P.R), P.N)
        ess_min    = minimum(ess)
        ess_p10    = quantile(ess, 0.10)
        ess_median = median(ess)
        ess_mean   = mean(ess)

        Ti = length.(P.ranges)

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
            extra = (; n_groups = P.N, K = P.K, M = P.M, R = P.R,
                       col_id = P.col_id, rfx = P.rfx_pairs,
                       rfx_cols = P.rfx_cols, seed = P.seed,
                       ess_min = ess_min, ess_p10 = ess_p10,
                       ess_median = ess_median, ess_mean = ess_mean,
                       Ti_min = minimum(Ti), Ti_median = median(Ti),
                       Ti_max = maximum(Ti))
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
    logit2_rfx(data_df, formula, choice, col_id, theta0; kwargs...) -> MLEFit

Binary logit with independent normal random coefficients, estimated by maximum
simulated likelihood with an analytic gradient.

`col_id` is positional because the model is undefined without it (mirrors `mlogit`).
`rfx` is a keyword because two adjacent positional `Vector{Symbol}` arguments are
too easy to transpose silently.

# Arguments
- `data_df`: a `DataFrame`. **Not mutated.**
- `formula`: `Vector{Symbol}` of regressors, as in `logit2`.
- `choice`: the 0/1 outcome column.
- `col_id`: group (individual) identifier for the random coefficients.
- `theta0`: starting values, length `K + M`. See [`theta0_rfx`](@ref).

# Keywords
- `rfx = Symbol[]`: subset of `formula` carrying random coefficients. A bare
  symbol means `:normal`; `[:dur => :lognormal]` form selects a distribution per
  variable from `:normal`, `:lognormal` (support `(0, ∞)`) and `:neg_lognormal`
  (support `(-∞, 0)`). See "Lognormal coefficients" below.
- `ndraws = 1000`: number of simulation draws, must be even (antithetic pairing).
- `seed = 20260808`: draw seed. Draws are generated once and reused for every
  function evaluation, so the objective is a deterministic function of θ.
- `weights = nothing`: column name; must be constant within `col_id`.
- `optim_options = Optim.Options()`.
- `rethrow_errors = false`: by default a failed optimisation is captured into
  `errored`/`error_message`. Set `true` to let the exception propagate instead,
  which is what you want when debugging a fit that will not run.

# Parameter ordering
`θ = [μ (K, in formula order); σ (M, in the order listed in rfx)]`, named
`[formula...; "sd_" .* rfx...]`. For a `:normal` coefficient `μ` is the mean of
the coefficient, which is the usual `β`.

Returned `σ` is always `≥ 0`: the likelihood satisfies `Q(μ, σ) = Q(μ, -σ)`, so
there are `2^M` mirror optima and the sign is not identified. This holds for the
lognormal families too, because antithetic draws make `{η_r} = {-η_r}` as a set.

# Lognormal coefficients
With `:lognormal` the coefficient is `β_im = exp(μ_m + σ_m·η_im) > 0`, and with
`:neg_lognormal` it is `-exp(μ_m + σ_m·η_im) < 0`. Use these when a coefficient
is sign-constrained on economic grounds and a normal random coefficient with a
large `σ` would put an implausible share of the population on the wrong side of
zero.

**`μ_m` and `σ_m` are then the mean and standard deviation of `log|β_im|`, not of
`β_im`.** `theta_names` is unchanged (`x` and `sd_x`), so `theta_hat` alone does
not tell you which scale a row is on — read `extra.rfx` for that, or use
[`rfx_level_moments`](@ref) / `boot_report`, which report both scales explicitly.
The level moments are

    E[β_im]  = ±exp(μ_m + σ_m²/2)
    median   = ±exp(μ_m)
    SD[β_im] =  exp(μ_m + σ_m²/2)·sqrt(exp(σ_m²) - 1)

`regtable_rfx` prints these level moments for a lognormal row, so that a table
mixing normal and lognormal specifications is comparable row by row.

`fit.extra` carries `n_groups`, `K`, `M`, `R`, `col_id`, `rfx`, `seed`, the ESS
diagnostic (`ess_min`, `ess_p10`, `ess_median`, `ess_mean`) and the panel shape
(`Ti_min`, `Ti_median`, `Ti_max`).

# Example
```julia
theta0 = theta0_rfx(myxs, [:dur, :dist]; b0 = logit_fit.theta_hat)
fit = logit2_rfx(df, myxs, :pick1, :personid, theta0; rfx = [:dur, :dist])
```
"""
function logit2_rfx(
        data_df,
        formula,
        choice,
        col_id::Symbol,
        theta0;
        rfx = Symbol[],
        ndraws::Int = 1000,
        seed::Int = 20260808,
        weights::Union{Nothing, Symbol, String} = nothing,
        optim_options::Optim.Options = Optim.Options(),
        rethrow_errors::Bool = false)

    P, gw = _prep_logit2_rfx(data_df, formula, choice, col_id, rfx,
                             ndraws, seed, weights)

    theta0 = _check_theta0_rfx(theta0, P)

    myfit = _logit2_rfx(P, theta0, gw, optim_options; rethrow_errors = rethrow_errors)

    if !myfit.errored && P.M > 0 && myfit.extra.ess_p10 < 30
        @warn "10th-percentile effective number of draws is " *
              "$(round(myfit.extra.ess_p10, digits=1)) (< 30): the posterior over the " *
              "random coefficients is concentrated relative to the draw set. " *
              "Consider increasing ndraws (currently $(P.R))."
    end

    return myfit
end

"""Validate `theta0` against a prepped model."""
function _check_theta0_rfx(theta0, P::RfxPrep)
    th = Float64.(collect(theta0))
    npar = P.K + P.M

    length(th) == npar || error(
        "theta0 has length $(length(th)) but the model has K + M = $(P.K) + $(P.M) = " *
        "$npar parameters. Use theta0_rfx(formula, rfx) to assemble it.")

    if P.M > 0 && any(th[P.K+1:end] .== 0)
        bad = findall(th[P.K+1:end] .== 0)
        error("theta0 has σ = 0 for rfx variable(s) " *
              "$(first.(P.rfx_pairs)[bad]): σ = 0 is a stationary point (a saddle) of " *
              "the simulated likelihood, so the optimiser cannot move away from it. " *
              "Use a nonzero start such as 0.5.")
    end

    return th
end
