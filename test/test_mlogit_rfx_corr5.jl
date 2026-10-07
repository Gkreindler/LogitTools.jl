using LogitTools, Test, DataFrames, Random, LinearAlgebra, FiniteDiff, Optim
const LT5=LogitTools
function corr5_data()
    rng=MersenneTwister(731)
    d=DataFrame(id=Int[],set=Int[],alt=Int[],selected=Int[],x1=Float64[],x2=Float64[],x3=Float64[],x4=Float64[],x5=Float64[])
    for i in 1:12, j in 1:5
        chosen=rand(rng,1:2)
        for a in 1:2
            push!(d,(i,5(i-1)+j,a,Int(a==chosen),randn(rng,5)...))
        end
    end
    d
end
@testset "Two observable Gaussian blocks sharing one person term" begin
    q=[.2,-.3,.4,-.15,.35,-.25]
    for z in (q,zeros(6),fill(.999,6),fill(-.999,6))
        L,J=LT5._mlogit_rfx_corr5_factors(z); C=L*L'
        @test diag(C)≈ones(5)
        @test minimum(eigvals(Symmetric(C)))>-1e-13
        @test C[2:3,4:5]≈C[2:3,1]*C[1,4:5]'
        fd=FiniteDiff.finite_difference_jacobian(v->vec(first(LT5._mlogit_rfx_corr5_factors(v))),z)
        @test reshape(J,25,6)≈fd atol=2e-5 rtol=2e-4
    end
    # The observable moment map has 11 independent coordinates.
    function moments(z)
        L,_=LT5._mlogit_rfx_corr5_factors(z[6:11]); C=Diagonal(z[1:5])*L*L'*Diagonal(z[1:5])
        [C[1,1],C[2,2],C[3,3],C[1,2],C[1,3],C[2,3],C[4,4],C[5,5],C[1,4],C[1,5],C[4,5]]
    end
    @test rank(FiniteDiff.finite_difference_jacobian(moments,[.8,1.1,.7,.9,.6,q...]))==11
    f=[:x1,:x2,:x3,:x4,:x5]
    r=Any[rfx_term(x) for x in f]
    append!(r,[rfx_term(:x1;level=:alt,mean=false),rfx_term(:x2;level=:alt,mean=false)])
    b5=[(:x1,:x2,:x3,:x4,:x5)]
    th=theta0_mlogit_rfx(f,r;rfx_corr5=b5,corr0=q,col_group=:id)
    @test length(th)==18
    @test size(theta0_mlogit_rfx_multistart(f,r;rfx_corr5=b5,nstarts=3,col_group=:id))==(3,18)
    vals=[]; grads=[]
    for kernel in (:general,:binary,:binary_antithetic_simd)
        P,_=LT5._prep_mlogit_rfx(corr5_data(),f,:set,:selected,:id,r,64,52,nothing;rfx_corr5=b5,kernel)
        buf=LT5.MlogitRfxBuffers(P); g=zeros(length(th)); gw=[0.;ones(P.N-1)]
        fun=z->LT5._mlogit_rfx_fg!(true,nothing,z,P,buf,gw)
        val=LT5._mlogit_rfx_fg!(true,g,th,P,buf,gw)
        @test g≈FiniteDiff.finite_difference_gradient(fun,th) atol=2e-6 rtol=2e-5
        push!(vals,val); push!(grads,g)
        C=mlogit_rfx_correlation_matrix(th,P)
        @test C[2:3,4:5]≈C[2:3,1]*C[1,4:5]'
        @test C[6,1]==0
        phi=LT5._mlogit_rfx_unconstrained_start(th,P.K,P.M,P.B)
        tw=similar(th); gg=similar(th); gp=similar(th)
        kfun=(F,G,z)->LT5._mlogit_rfx_fg!(F,G,z,P,buf,gw)
        pfun=z->LT5._mlogit_rfx_unconstrained_fg!(true,nothing,z,P.K,P.M,P.B,tw,gg,kfun)
        LT5._mlogit_rfx_unconstrained_fg!(true,gp,phi,P.K,P.M,P.B,tw,gg,kfun)
        @test gp≈FiniteDiff.finite_difference_gradient(pfun,phi) atol=2e-6 rtol=2e-5
        # Zero coordinates exactly reproduce independent effects under the same draws.
        P0,_=LT5._prep_mlogit_rfx(corr5_data(),f,:set,:selected,:id,r,64,52,nothing;kernel)
        z=copy(th);z[end-5:end].=0;g0=zeros(12);gz=zeros(18)
        v0=LT5._mlogit_rfx_fg!(true,g0,z[1:12],P0,LT5.MlogitRfxBuffers(P0),gw)
        vz=LT5._mlogit_rfx_fg!(true,gz,z,P,buf,gw)
        @test v0≈vz atol=1e-12
        @test g0≈gz[1:12] atol=1e-12
        # Person scores must add to the joint objective and whole-vector score.
        sumg=zeros(18);sumv=0.0;one=zeros(P.N)
        for i in 2:P.N
            fill!(one,0);one[i]=1
            sumv+=LT5._mlogit_rfx_fg!(true,gz,th,P,buf,one);sumg.+=gz
        end
        @test sumv≈val atol=1e-12
        @test sumg≈g atol=1e-12
    end
    @test maximum(vals)-minimum(vals)<1e-10
    @test grads[1]≈grads[2] atol=1e-10
    @test grads[1]≈grads[3] atol=1e-10
    fit=mlogit_rfx(corr5_data(),f,:set,:selected,th;col_group=:id,rfx=r,rfx_corr5=b5,ndraws=16,optim_options=Optim.Options(iterations=2),rethrow_errors=true)
    @test !fit.errored
    @test fit.extra.rfx_corr5==[(1,2,3,4,5)]
    @test size(mlogit_rfx_correlation_matrix(fit))==(7,7)
    for bad in ([(:x1,:x1,:x3,:x4,:x5)],[(:missing,:x2,:x3,:x4,:x5)])
        @test_throws ErrorException theta0_mlogit_rfx(f,r;rfx_corr5=bad,col_group=:id)
    end
    @test_throws ErrorException theta0_mlogit_rfx(f,r;rfx_corr=[(:x1,:x2)],rfx_corr5=b5,col_group=:id)
    rb=copy(r);rb[1]=rfx_term(:x1;dist=:uniform)
    @test_throws ErrorException theta0_mlogit_rfx(f,rb;rfx_corr5=b5,col_group=:id)
end
