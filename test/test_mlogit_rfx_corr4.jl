using LogitTools, Test, DataFrames, Random, LinearAlgebra, FiniteDiff, Optim
const LT4 = LogitTools
function corr4_data(J=2)
    rng=MersenneTwister(411)
    df=DataFrame(id=Int[],set=Int[],alt=Int[],selected=Int[],x1=Float64[],x2=Float64[],x3=Float64[],x4=Float64[],x5=Float64[],x6=Float64[])
    for i in 1:12, s in 1:5
        chosen=rand(rng,1:J)
        for j in 1:J
            push!(df,(i,5(i-1)+s,j,Int(j==chosen),randn(rng,6)...))
        end
    end
    df
end
@testset "Normal four-term covariance block" begin
    f=[:x1,:x2,:x3,:x4,:x5,:x6]
    r=Any[rfx_term(x) for x in f]
    push!(r,rfx_term(:x1;level=:alt,mean=false))
    b4=[(:x3,:x4,:x1,:x2)]
    pairs=[(:x5,:x6)]
    q=[0.2,-0.3,0.4,0.15,-0.45]
    for qq in (q,zeros(5),fill(0.999,5),fill(-0.999,5))
        L,J=LT4._mlogit_rfx_corr4_factors(qq)
        C=L*L'
        @test diag(C) ≈ ones(4)
        @test C[1,2] == 0
        @test minimum(eigvals(Symmetric(C))) > -1e-14
        fd=FiniteDiff.finite_difference_jacobian(z->vec(first(LT4._mlogit_rfx_corr4_factors(z))),qq)
        @test reshape(J,16,5) ≈ fd atol=2e-5 rtol=2e-4
    end
    th=theta0_mlogit_rfx(f,r;rfx_corr=pairs,rfx_corr4=b4,corr0=[0.25;q],col_group=:id)
    @test length(th)==19
    starts=theta0_mlogit_rfx_multistart(f,r;rfx_corr=pairs,rfx_corr4=b4,nstarts=3,col_group=:id)
    @test size(starts)==(3,19)
    objs=Float64[];grads=Vector{Float64}[]
    for kernel in (:general,:binary,:binary_antithetic_simd)
        P,gw=LT4._prep_mlogit_rfx(corr4_data(),f,:set,:selected,:id,r,64,83,nothing,pairs;rfx_corr4=b4,kernel=kernel)
        buf=LT4.MlogitRfxBuffers(P)
        gw=Float64[0;ones(P.N-1)]
        g=zeros(length(th))
        fun=z->LT4._mlogit_rfx_fg!(true,nothing,z,P,buf,gw)
        obj=LT4._mlogit_rfx_fg!(true,g,th,P,buf,gw)
        fd=FiniteDiff.finite_difference_gradient(fun,th)
        @test g ≈ fd atol=2e-6 rtol=2e-5
        push!(objs,obj);push!(grads,g)
        C=mlogit_rfx_correlation_matrix(th,P)
        @test C[3,4]==0
        @test C[1,3]≈q[1]
        @test C[1,4]≈sqrt(1-q[1]^2)*q[2]
        @test C[5,6]≈0.25
        @test C[7,1]==0
        # Test optimizer-space chain rule as well as public-coordinate gradient.
        phi=LT4._mlogit_rfx_unconstrained_start(th,P.K,P.M,P.B)
        tw=similar(th);gg=similar(th);gp=similar(th)
        kfun=(F,G,z)->LT4._mlogit_rfx_fg!(F,G,z,P,buf,gw)
        pfun=z->LT4._mlogit_rfx_unconstrained_fg!(true,nothing,z,P.K,P.M,P.B,tw,gg,kfun)
        LT4._mlogit_rfx_unconstrained_fg!(true,gp,phi,P.K,P.M,P.B,tw,gg,kfun)
        @test gp ≈ FiniteDiff.finite_difference_gradient(pfun,phi) atol=2e-6 rtol=2e-5
    end
    @test maximum(objs)-minimum(objs)<1e-10
    @test grads[1]≈grads[2] atol=1e-10
    @test grads[1]≈grads[3] atol=1e-10
    # Multinomial reference path and public estimation metadata.
    fit=mlogit_rfx(corr4_data(3),f,:set,:selected,th;col_group=:id,rfx=r,rfx_corr=pairs,rfx_corr4=b4,ndraws=16,optim_options=Optim.Options(iterations=2),rethrow_errors=true)
    @test !fit.errored
    @test size(mlogit_rfx_correlation_matrix(fit))==(7,7)
    @test fit.extra.rfx_corr4==[(3,4,1,2)]
    @test any(startswith("pcor_"),fit.theta_names)
    for bad in ([(:x1,:x1,:x2,:x3)],[(:x1,:x2,:x3,:x5)],[(:missing,:x2,:x3,:x4)])
        @test_throws ErrorException theta0_mlogit_rfx(f,r;rfx_corr=pairs,rfx_corr4=bad,col_group=:id)
    end
    rb=copy(r);rb[1]=rfx_term(:x1;dist=:uniform)
    @test_throws ErrorException theta0_mlogit_rfx(f,rb;rfx_corr4=b4,col_group=:id)
end
@testset "Four-term bootstrap forwarding and zero-block compatibility" begin
    f=[:x1,:x2,:x3,:x4];r=Any[rfx_term(x) for x in f];b4=[(:x3,:x4,:x1,:x2)]
    df=corr4_data()
    # This checks API forwarding and replicate bookkeeping, not optimizer
    # convergence on a deliberately tiny, underidentified synthetic sample.
    opts=Optim.Options(iterations=2,g_abstol=1e9)
    th=theta0_mlogit_rfx(f,r;rfx_corr4=b4,col_group=:id)
    boot=boot_mlogit_rfx(df,f,:set,:selected,th;col_group=:id,rfx=r,rfx_corr4=b4,ndraws=16,nboot=2,parallel=false,kernel=:auto,optim_options=opts)
    @test size(boot.theta_boot_table)==(2,13)
    rep=fit_mlogit_rfx_bootstrap_replicate(df,f,:set,:selected,th,2;col_group=:id,rfx=r,rfx_corr4=b4,ndraws=16,nboot=2,kernel=:auto,optim_options=opts)
    @test rep.theta_hat == boot.boot_fits[2].theta_hat
    @test rep.obj_value == boot.boot_fits[2].obj_value
    @test mlogit_rfx_correlation_matrix(rep)[3,4]==0
    # All-zero block coordinates exactly reproduce independent random effects
    # under the same draws (including SD and mean derivatives).
    for kernel in (:general,:binary,:binary_antithetic_simd)
        P,_=LT4._prep_mlogit_rfx(df,f,:set,:selected,:id,r,32,17,nothing;rfx_corr4=b4,kernel=kernel)
        P0,_=LT4._prep_mlogit_rfx(df,f,:set,:selected,:id,r,32,17,nothing;kernel=kernel)
        g=zeros(13);g0=zeros(8)
        v=LT4._mlogit_rfx_fg!(true,g,th,P,LT4.MlogitRfxBuffers(P),nothing)
        v0=LT4._mlogit_rfx_fg!(true,g0,th[1:8],P0,LT4.MlogitRfxBuffers(P0),nothing)
        @test v≈v0 atol=1e-12
        @test g[1:8]≈g0 atol=1e-12
    end
end
