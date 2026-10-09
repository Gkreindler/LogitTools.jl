module LogitTools

# Write your package code here.

using Optim
using LinearAlgebra
using StatsBase
using Distributions
using LogExpFunctions

using DataFrames

using Printf

using RegressionTables, StatsAPI, Vcov

using Random          # MersenneTwister
using Statistics      # median, quantile
using Distributed     # pmap, CachingPool, workers

export mlogit, boot_mlogit,
       logit2, boot_logit2,
       regtable

export rfx_term, rfx_corr_block,
       mlogit_rfx_prep, mlogit_rfx_fit, mlogit_rfx,
       theta0_mlogit_rfx, theta0_mlogit_rfx_multistart,
       mlogit_rfx_objective, mlogit_rfx_gradient, mlogit_rfx_scores,
       mlogit_rfx_ess, mlogit_rfx_correlation, mlogit_rfx_covariance,
       mlogit_rfx_boundary, mlogit_rfx_vcov,
       mlogit_rfx_boot_weights, boot_mlogit_rfx, rfx_cell_report

include("MLE_objects.jl")

include("functions_mlogit_estimation.jl")
include("functions_mlogit_inference.jl")

include("functions_logit2_estimation.jl")
include("functions_logit2_inference.jl")

include("functions_mlogit_rfx_estimation.jl")
include("functions_mlogit_rfx_inference.jl")

# include("optimization_backends.jl")
# include("functions_inference.jl")

include("functions_regtable.jl")

# include("utilities.jl")
# include("io.jl")

end
