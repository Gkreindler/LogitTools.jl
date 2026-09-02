module LogitTools

# Write your package code here.

using Optim
using LinearAlgebra
using StatsBase
using Distributions
using LogExpFunctions

using DataFrames

using IntervalSets
using Printf

using RegressionTables, StatsAPI, Vcov

using Random          # MersenneTwister
using Statistics      # median, quantile
using Distributed     # pmap, CachingPool, workers, nprocs, remotecall_fetch

# Public implementation marker for downstream replication code that needs the
# finite-draw-safe positive-sigma parameterisation introduced for the RFX fits.
const RFX_SIGMA_PARAMETERIZATION = :softplus
const MLOGIT_RFX_CORR_PARAMETERIZATION = :scaled_tanh

export mlogit, boot_mlogit,
       logit2, boot_logit2,
       regtable

export logit2_rfx, boot_logit2_rfx, theta0_rfx, boot_report, boot_vcov!, regtable_rfx,
       rfx_level_moments, theta0_rfx_multistart

export mlogit_rfx, boot_mlogit_rfx, fit_mlogit_rfx_bootstrap_replicate,
       theta0_mlogit_rfx, theta0_mlogit_rfx_multistart,
       rfx_term, rfx_cell_report, RFX_SIGMA_PARAMETERIZATION,
       MLOGIT_RFX_CORR_PARAMETERIZATION

include("MLE_objects.jl")

include("functions_mlogit_estimation.jl")
include("functions_mlogit_inference.jl")

include("functions_logit2_estimation.jl")
include("functions_logit2_inference.jl")

include("functions_logit2_rfx_estimation.jl")
include("functions_logit2_rfx_inference.jl")

# after logit2_rfx: reuses its distribution helpers (_normalize_rfx internals,
# _rfx_is_log, _rfx_sign) and its reporting layer (boot_report, regtable_rfx,
# rfx_level_moments, _rfx_boot_weights, _assemble_rfx_boot)
include("functions_mlogit_rfx_estimation.jl")
include("functions_mlogit_rfx_inference.jl")

# include("optimization_backends.jl")
# include("functions_inference.jl")

include("functions_regtable.jl")

# include("utilities.jl")
# include("io.jl")

end
