using LogitTools
using Test

@testset "LogitTools.jl" begin
    include("test_boot_logit2_parallel.jl")
    include("test_logit2_rfx.jl")
end
