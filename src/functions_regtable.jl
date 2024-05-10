
struct bayesian_bootstrap <: CovarianceEstimator
end

Base.@kwdef struct LogitRegModel <: RegressionModel
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix
    vcov_type::CovarianceEstimator
    nclusters::Union{NamedTuple, Nothing} = nothing

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    residuals::Union{AbstractVector, Nothing} = nothing
    fe::DataFrame
    fekeys::Vector{Symbol}


    coefnames::Vector       # Name of coefficients
    responsename::Union{String, Symbol} # Name of dependent variable
    # formula::FormulaTerm        # Original formula
    # formula_schema::FormulaTerm # Schema for predict
    contrasts::Dict

    nobs::Int64             # Number of observations
    dof::Int64              # Number parameters estimated - has_intercept. Used for p-value of F-stat.
    dof_fes::Int64          # Number of fixed effects
    dof_residual::Int64     # dof used for t-test and p-value of F-stat. nobs - degrees of freedoms with simple std
    rss::Float64            # Sum of squared residuals
    tss::Float64            # Total sum of squares

    F::Float64              # F statistics
    p::Float64              # p value for the F statistics

    # for FE
    iterations::Int         # Number of iterations
    converged::Bool         # Has the demeaning algorithm converged?
    r2_within::Union{Float64, Nothing} = nothing      # within r2 (with fixed effect

    # for IV
    F_kp::Union{Float64, Nothing} = nothing           # First Stage F statistics KP
    p_kp::Union{Float64, Nothing} = nothing           # First Stage p value KP
end


has_iv(m::LogitRegModel) = m.F_kp !== nothing
has_fe(m::LogitRegModel) = false

# RegressionTables.get_coefname(x::Tuple{Vararg{Term}}) = RegressionTables.get_coefname.(x)
# RegressionTables.replace_name(x::Tuple{Vararg{Any}}, a::Dict{String, String}, b::Dict{String, String}) = [RegressionTables.replace_name(x[i], a, b) for i=1:length(x)]
# RegressionTables.formula(m::GMMModel) = term(m.responsename) ~ sum(term.(String.(m.coefnames)))

RegressionTables._responsename(x::LogitRegModel) = RegressionTables.CoefName(string(responsename(x)))
RegressionTables._coefnames(x::LogitRegModel) = RegressionTables.CoefName.(string.(coefnames(x)))

StatsAPI.coef(m::LogitRegModel) = m.coef
StatsAPI.coefnames(m::LogitRegModel) = m.coefnames
StatsAPI.responsename(m::LogitRegModel) = m.responsename
StatsAPI.vcov(m::LogitRegModel) = m.vcov
StatsAPI.nobs(m::LogitRegModel) = m.nobs
StatsAPI.dof(m::LogitRegModel) = m.dof
StatsAPI.dof_residual(m::LogitRegModel) = m.dof_residual
StatsAPI.r2(m::LogitRegModel) = r2(m, :devianceratio)
StatsAPI.islinear(m::LogitRegModel) = true
StatsAPI.deviance(m::LogitRegModel) = rss(m)
StatsAPI.nulldeviance(m::LogitRegModel) = m.tss
StatsAPI.rss(m::LogitRegModel) = m.rss
StatsAPI.mss(m::LogitRegModel) = nulldeviance(m) - rss(m)
# StatsModels.formula(m::GMMResultTable) = m.formula_schema
dof_fes(m::LogitRegModel) = m.dof_fes


function vcov(r::MLEFit)
    if isnothing(r.vcov)
        nparams = length(r.theta_hat)
        return zeros(nparams, nparams)
    else
        # print warnings
        # (r.vcov.method == :bayesian_bootstrap) && (r.vcov.boot_fits.n_errored > 0) && @warn string(r.vcov.boot_fits.n_errored) * " out of " * string(length(r.vcov.boot_fits.errored)) * " bootstrap runs errored completely (no estimation results). Dropping."
        return r.vcov.V
    end
end

function vcov_method(r::MLEFit)
    if isnothing(r.vcov) || (r.vcov.method == :simple)
        return Vcov.simple()
        
    elseif r.vcov.method == :bayesian_bootstrap
        return bayesian_bootstrap()
    end
end


function LogitRegModel(r::MLEFit)
    
    nobs = r.n_obs

    if isnothing(r.vcov)
        @error "No vcov estimated yet. Using zeros for vcov matrix."
        # @error "Cannot print table. No vcov estimated yet"
        # error("Cannot print table. No vcov estimated yet")
    end

    if isnothing(r.theta_names)
        r.theta_names = ["theta_$i" for i=1:length(r.theta_hat)]
    end

    LogitRegModel(
        coef = r.theta_hat,
        vcov = vcov(r),
        vcov_type=Vcov.simple(), # ! update with clustering!
        esample=[],
        fe=DataFrame(),
        fekeys=[],
        coefnames=r.theta_names,
        responsename="", # no column header
        # formula::FormulaTerm        # Original formula
        # formula_schema::FormulaTerm # Schema for predict
        contrasts=Dict(),
        nobs=nobs,
        dof=nobs,
        dof_fes=1,
        dof_residual=nobs, # TODO: needs adjustment for parameters (!?)
        rss=0.00,
        tss=0.00,
        F=0.0,
        p=0.0,
        iterations=5, 
        converged=true)         
end

# TODO: integrate better with RegressionModels, allow mixed inputs etc. Should be easy.
RegressionTables.regtable(r::MLEFit) = RegressionTables.regtable(LogitRegModel(r), render = AsciiTable())

# function gmm_regtable  
function RegressionTables.regtable(rrs::Vararg{Union{RegressionModel, MLEFit}}; kwargs...)

    rrs_converted = [isa(r, MLEFit) ? LogitRegModel(r) : r for r=rrs]

    return RegressionTables.regtable(rrs_converted...; kwargs...)
end



# Bootstrap confidence intervals (95%)
cis(myfit::MLEFit; ci_levels=[2.5, 97.5]) = cis(myfit.vcov, ci_levels=ci_levels)

# function cis(myvcov::GMMvcov; ci_levels=[2.5, 97.5]) 
    
#     if myvcov.method == :simple
#         error("CI not implemented yet for simple vcov")
#     elseif myvcov.method == :bayesian_bootstrap
#         return cis(myvcov.boot_fits, ci_levels=ci_levels)
#     else
#         error("Unknown vcov method ", myvcov.method)
#     end
# end

# function cis(b::GMMBootFits; ci_levels=[2.5, 97.5])

#     all(b.errored) && error("All bootstrap runs errored completely (no estimation results). Cannot compute CIs.")

#     theta_hat_boot = boot_table(b)
#     nparams = size(theta_hat_boot, 2)

#     cis = []
#     for i=1:nparams
#         cil, cih = percentile(theta_hat_boot[:, i], ci_levels)
#         push!(cis, (cil, cih))
#     end
    
#     return cis
# end

