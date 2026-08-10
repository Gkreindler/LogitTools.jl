using LogitTools, DataFrames, Random, RegressionTables, Serialization
rng = MersenneTwister(21); N,T = 60,8; nobs=N*T
df = DataFrame(personid=repeat(1:N,inner=T), clusterid=repeat(1:N,inner=T))
for k in 1:3; df[!,Symbol("x",k)] = randn(rng,nobs); end
d1 = repeat(0.7.*randn(rng,N),inner=T); d3 = repeat(0.4.*randn(rng,N),inner=T)
v = Matrix(df[:,[:x1,:x2,:x3]])*[0.6,-0.9,0.4] .+ d1.*df.x1 .+ d3.*df.x3
df.pick1 = Float64.(rand(rng,nobs) .< 1 ./(1 .+exp.(-v)))
myxs=[:x1,:x2,:x3]

g = logit2(copy(df), myxs, :pick1, zeros(3))
g.vcov = boot_logit2(copy(df), myxs, :pick1, zeros(3); nboot=20, cluster_var=:clusterid)

th0=theta0_rfx(myxs,[:x1,:x3])
f = logit2_rfx(df,myxs,:pick1,:personid,th0; rfx=[:x1,:x3], ndraws=100, seed=7)
f.vcov = boot_logit2_rfx(df,myxs,:pick1,:personid,th0; rfx=[:x1,:x3], ndraws=100,
                         seed=7, nboot=20, boot_seed=777, parallel=false, theta_start=f.theta_hat)

println("### mixed logit2 + rfx in one regtable_rfx ###")
regtable_rfx(g, f, g, f) |> display

println("\n### round-trip through Serialization ###")
tmp = tempname()*".jls"
serialize(tmp, Dict("plain"=>g, "rfx"=>f))
d = deserialize(tmp)
println("  loaded: ", sort(collect(keys(d))))
println("  rfx extra.M = ", d["rfx"].extra.M, "  boot_fits kept = ", length(d["rfx"].vcov.boot_fits))
regtable_rfx(d["plain"], d["rfx"]) |> display
println("  file size = ", round(filesize(tmp)/2^20, digits=2), " MB")
rm(tmp)

println("\n### renaming theta_names to separate a row ###")
f2 = deepcopy(f)
f2.theta_names = [n=="x3" ? "x3_alt" : (n=="sd_x3" ? "sd_x3_alt" : n) for n in f2.theta_names]
regtable_rfx(f, f2) |> display
