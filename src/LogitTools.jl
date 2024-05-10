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

export mlogit, boot_mlogit, 
       logit2, boot_logit2, 
       regtable

include("MLE_objects.jl")

include("functions_mlogit_estimation.jl")
include("functions_mlogit_inference.jl")

include("functions_logit2_estimation.jl")
include("functions_logit2_inference.jl")

# include("optimization_backends.jl")
# include("functions_inference.jl")

include("functions_regtable.jl")

# include("utilities.jl")
# include("io.jl")

end
