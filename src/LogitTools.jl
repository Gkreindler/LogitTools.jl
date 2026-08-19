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

export mlogit, boot_mlogit,
       logit2, boot_logit2,
       regtable

export logit2_rfx, boot_logit2_rfx, theta0_rfx, boot_report, boot_vcov!, regtable_rfx,
       rfx_level_moments, theta0_rfx_multistart

include("MLE_objects.jl")

include("functions_mlogit_estimation.jl")
include("functions_mlogit_inference.jl")

include("functions_logit2_estimation.jl")
include("functions_logit2_inference.jl")

include("functions_logit2_rfx_estimation.jl")
include("functions_logit2_rfx_inference.jl")

# include("optimization_backends.jl")
# include("functions_inference.jl")

include("functions_regtable.jl")

# include("utilities.jl")
# include("io.jl")

end
