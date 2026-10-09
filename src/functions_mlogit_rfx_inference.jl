###############################################################################
# mlogit_rfx: objective and scores, clustered sandwich, Bayesian bootstrap.
#
# Every function takes a prepared model (mlogit_rfx_prep) and a parameter
# vector theta = [beta; sigma; block coordinates], in theta_names order.
###############################################################################

"""
    mlogit_rfx_objective(theta, P; weights = nothing) -> Float64

Simulated negative log-likelihood, `-sum_i w_i log L_i`.
"""
function mlogit_rfx_objective(theta::AbstractVector, P::MlogitRfxPrep; weights = nothing)
    return _mrfx_fg!(true, nothing, Vector{Float64}(theta), P, MlogitRfxBuffers(P),
                     _mrfx_weights(P, weights))
end

"""
    mlogit_rfx_gradient(theta, P; weights = nothing) -> Vector{Float64}

Gradient of [`mlogit_rfx_objective`](@ref) with respect to `theta` (in
`theta_names` order, i.e. with respect to the standard deviations and block
coordinates themselves).
"""
function mlogit_rfx_gradient(theta::AbstractVector, P::MlogitRfxPrep; weights = nothing)
    G = zeros(length(theta))
    _mrfx_fg!(nothing, G, Vector{Float64}(theta), P, MlogitRfxBuffers(P),
              _mrfx_weights(P, weights))
    return G
end

"""
    mlogit_rfx_scores(theta, P) -> Matrix{Float64}

`N x npar` matrix whose row `i` is `w_i * d log L_i / d theta`, the score of group
`i` (`w_i` = prep weight, 1 if none). Rows sum to minus the gradient.
"""
function mlogit_rfx_scores(theta::AbstractVector, P::MlogitRfxPrep)
    th  = Vector{Float64}(theta)
    buf = MlogitRfxBuffers(P)
    w   = zeros(P.N)
    g   = zeros(length(th))
    out = zeros(P.N, length(th))
    for i in 1:P.N
        w[i] = isnothing(P.group_weights) ? 1.0 : P.group_weights[i]
        _mrfx_fg!(nothing, g, th, P, buf, w)
        out[i, :] .= .-g
        w[i] = 0.0
    end
    return out
end

"""
    mlogit_rfx_boundary(theta, P; sd_tol = 1e-3, corr_tol = 1e-4) -> BitVector

Flags parameters at the boundary of the parameter space: standard deviations
below `sd_tol` and block coordinates within `corr_tol` of +-1. Means are never
flagged. Intended as the `fixed` argument of [`mlogit_rfx_vcov`](@ref).
"""
function mlogit_rfx_boundary(theta::AbstractVector, P::MlogitRfxPrep;
                             sd_tol::Real = 1e-3, corr_tol::Real = 1e-4)
    K, M = P.K, P.M
    return BitVector([j <= K ? false :
                      j <= K + M ? theta[j] < sd_tol : abs(theta[j]) > 1 - corr_tol
                      for j in eachindex(theta)])
end

"""
    mlogit_rfx_vcov(theta, P; fixed = nothing, step = 1e-4) -> Matrix{Float64}

Person-clustered sandwich covariance `H^-1 J H^-1 * G/(G-1)` at `theta`, where
`H` is the Hessian of the objective and `J = sum_i s_i s_i'` sums the group
scores (clusters = groups, `G` = groups with positive weight). The covariance
includes uncertainty in the standard deviations and block coordinates.

`H` is computed by central differences of the analytic gradient with step
`step * max(1, |theta_j|)`, kept inside the parameter space.

`fixed` (a Bool vector, e.g. from [`mlogit_rfx_boundary`](@ref)) holds parameters
at their estimated values: their rows and columns are zero and the remaining
block is the sandwich of the model with them fixed. A standard deviation at
zero or a correlation at +-1 has no regular curvature, so this is the usable
version there; it does not account for the boundary itself.
"""
function mlogit_rfx_vcov(theta::AbstractVector, P::MlogitRfxPrep;
                         fixed = nothing, step::Real = 1e-4)
    th = Vector{Float64}(theta)
    npar = length(th)
    free = isnothing(fixed) ? collect(1:npar) : findall(.!collect(Bool, fixed))
    K, M = P.K, P.M
    w = P.group_weights
    buf = MlogitRfxBuffers(P)
    grad(x) = (G = zeros(npar); _mrfx_fg!(nothing, G, x, P, buf, w); G)

    n = length(free)
    Hs = zeros(n, n)
    for (a, j) in enumerate(free)
        h = step * max(1.0, abs(th[j]))
        if K < j <= K + M
            h = min(h, (th[j] - _MRFX_SIGMA_FLOOR) / 2)
        elseif j > K + M
            h = min(h, (_MRFX_CORR_LIMIT - abs(th[j])) / 2)
        end
        h > 0 || error("parameter $(P.theta_names[j]) is at its boundary; add it to `fixed`")
        xp = copy(th); xp[j] += h
        xm = copy(th); xm[j] -= h
        Hs[:, a] .= (grad(xp)[free] .- grad(xm)[free]) ./ (2h)
    end
    Hs = (Hs + Hs') / 2

    S = mlogit_rfx_scores(th, P)[:, free]
    G = isnothing(w) ? P.N : count(>(0), w)
    Jm = (S' * S) * (G / (G - 1))

    ev = eigvals(Symmetric(Hs))
    minimum(ev) > 0 || @warn "the Hessian is not positive definite (smallest eigenvalue " *
        "$(minimum(ev))); the sandwich is not reliable. Check convergence, or fix " *
        "boundary parameters."
    Hinv = inv(Symmetric(Hs))
    V = zeros(npar, npar)
    V[free, free] .= Hinv * Jm * Hinv
    return Symmetric(V) |> Matrix
end


# ----------------------------------------------------------------------------
# Bayesian bootstrap
# ----------------------------------------------------------------------------

"""
    mlogit_rfx_boot_weights(N, nboot; seed = 12345) -> Matrix{Float64}

`N x nboot` Bayesian-bootstrap weights: column `b` is a Dirichlet(1, ..., 1)
draw over the `N` groups, scaled to mean one.
"""
function mlogit_rfx_boot_weights(N::Int, nboot::Int; seed::Int = 12345)
    rng = MersenneTwister(seed)
    W = Matrix{Float64}(undef, N, nboot)
    for b in 1:nboot
        w = rand(rng, Dirichlet(N, 1.0))
        W[:, b] .= w .* (N / sum(w))
    end
    return W
end

"""
    boot_mlogit_rfx(P, starts; nboot = 500, seed = 12345, W = nothing,
                    parallel = false, fit_kwargs...) -> MLEvcov

Bayesian bootstrap at the group (person) level. Replicate `b` re-estimates the
model with group weights `W[:, b]` (times any prep weights), holding the
simulation draws fixed.

- `starts`: a vector (typically the full-sample estimate) used as the single
  start of every replicate, or a matrix with one start per row, in which case
  every replicate is a multi-start fit and keeps its best optimum.
- `W`: weight matrix, `N x nboot`. By default
  `mlogit_rfx_boot_weights(P.N, nboot; seed)`. Passing columns of a saved `W`
  runs a subset of replicates, e.g. in chunks.
- `parallel = true`: distribute replicates over `workers()`; each worker
  receives `P` once. Starts within a replicate run serially.
- Other keywords (`optimizer`, `optim_options`, `sd_start_min`,
  `corr_start_max`) are passed to [`mlogit_rfx_fit`](@ref). With a single
  start, `sd_start_min` and `corr_start_max` move boundary estimates slightly
  inside so that each replicate can leave the boundary.

Returns an `MLEvcov` with `theta_boot_table` (`nboot x npar`, `NaN` rows for
replicates where every start errored), `boot_fits` (one `MLEFit` per
replicate, with convergence information) and `V`, the covariance across the
converged replicates.
"""
function boot_mlogit_rfx(P::MlogitRfxPrep, starts;
                         nboot::Int = 500, seed::Int = 12345, W = nothing,
                         parallel::Bool = false, fit_kwargs...)
    W = isnothing(W) ? mlogit_rfx_boot_weights(P.N, nboot; seed) : Matrix{Float64}(W)
    size(W, 1) == P.N || error("W must have one row per group ($(P.N))")
    S = _mrfx_starts(starts, P)

    fit_replicate = (d, b) -> mlogit_rfx_fit(d.P, d.S; weights = d.W[:, b], parallel = false,
                                             ess = false, fit_kwargs...)
    data = (; P, S, W)
    ids = 1:size(W, 2)
    fits = parallel ? _mrfx_pmap(fit_replicate, data, ids) : [fit_replicate(data, b) for b in ids]

    table = permutedims(reduce(hcat, [f.theta_hat for f in fits]))
    good  = [!f.errored && f.converged for f in fits]
    nbad  = count(!, good)
    nbad > 0 && @warn "$nbad of $(length(fits)) bootstrap replicates did not converge or errored"
    return MLEvcov(method = :bayesian_bootstrap,
                   V = count(good) >= 2 ? cov(table[good, :]) : nothing,
                   theta_boot_table = table, boot_fits = fits)
end
