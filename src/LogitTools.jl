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

export logit2, boot_logit2, regtable

include("functions_logit2.jl")
# include("optimization_backends.jl")
# include("functions_inference.jl")
include("functions_regtable.jl")
# include("utilities.jl")
# include("io.jl")

end
