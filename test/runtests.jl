using Test
using LinearAlgebra
using GfPEPS

@testset "ABD" begin
    Nv = 1
    Jm = J(Nv)
    @test size(Jm) == (2 * (4 * Nv + 2), 2 * (4 * Nv + 2))
    A, B, D = getABD(Jm)
    @test size(A) == (4, 4)
    @test size(B, 1) == 4
    @test size(D) == (size(Jm, 1) - 4, size(Jm, 2) - 4)
end

@testset "Gin" begin
    k = 0.123
    G = SingleGammaIn(k)
    @test size(G) == (4, 4)
    @test G[1, 4] ≈ cis(k)
    @test G[4, 1] ≈ -conj(cis(k))

    Nv = 2
    K = BatchK(5, 7)
    @test size(K) == (5 * 7, 2)

    BG = BatchGammaIn(3, 4, Nv)
    @test size(BG) == (3 * 4, 8 * Nv, 8 * Nv)
end

@testset "Gaussian map + energy" begin
    Nv = 1
    T = Matrix{Float64}(LinearAlgebra.I, 2 * (4 * Nv + 2), 2 * (4 * Nv + 2))
    Glocal = getGammaProjector(T, Nv)
    Gin = GammaIn([0.1, 0.2], Nv)
    Gout = GaussianLinearMap(Glocal, Gin)
    @test size(Gout) == (4, 4)

    BGin = BatchGammaIn(2, 2, Nv)
    BGout = GaussianLinearMap(Glocal, BGin)
    @test size(BGout) == (size(BGin, 1), 4, 4)

    energy = energy_function(hoping=1.0, DeltaX=0.5, DeltaY=0.0, Mu=0.1, Lx=2, Ly=2)
    e = energy(BGout)
    @test isfinite(e)
end

@testset "exact + solve_mu" begin
    e = eg(2, 3, 1.0, 0.4, 0.3, 0.2)
    @test isfinite(e)

    mu = solve_mu(0.5, 0.1)
    @test isfinite(mu)
end

@testset "gaussian_fpeps (mock cfg)" begin
    # Tiny config types
    Base.@kwdef mutable struct Params; Nv::Int=1; seed::Int=1; end
    Base.@kwdef mutable struct Lattice; Lx::Int=2; Ly::Int=2; end
    Base.@kwdef mutable struct FileCfg; LoadFile::Union{Nothing,String}=nothing; WriteFile::Union{Nothing,String}=nothing; SaveEachSteps::Bool=false; end
    Base.@kwdef mutable struct Ham; ht::Float64=1.0; DeltaX::Float64=0.2; DeltaY::Float64=0.1; delta::Float64=0.1; Mu::Float64=0.5; solve_mu_from_delta::Bool=false; end
    Base.@kwdef mutable struct Opt; MaxIter::Int=3; lr::Float64=0.05; tol::Float64=1e-5; end
    Base.@kwdef mutable struct Cfg; params::Params=Params(); lattice::Lattice=Lattice(); file::FileCfg=FileCfg(); hamiltonian::Ham=Ham(); optimizer::Opt=Opt(); backend::Symbol=:cpu; end
    cfg = Cfg()
    T = gaussian_fpeps(cfg)
    @test size(T) == (8 * cfg.params.Nv + 4, 8 * cfg.params.Nv + 4)
end
