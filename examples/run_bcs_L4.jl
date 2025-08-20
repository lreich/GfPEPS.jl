using Revise
using GfPEPS
using LinearAlgebra

# Minimal cfg types (compatible with gaussian_fpeps)
Base.@kwdef mutable struct Params; Nv::Int=1; seed::Int=42; end
Base.@kwdef mutable struct Lattice; Lx::Int=4; Ly::Int=4; end
Base.@kwdef mutable struct FileCfg; LoadFile::Union{Nothing,String}=nothing; WriteFile::Union{Nothing,String}=nothing; SaveEachSteps::Bool=false; end
Base.@kwdef mutable struct Ham; ht::Float64=1.0; DeltaX::Float64=0.5; DeltaY::Float64=0.5; delta::Float64=0.1; Mu::Float64=0.5; solve_mu_from_delta::Bool=true; end
Base.@kwdef mutable struct Opt; MaxIter::Int=400; lr::Float64=0.2; tol::Float64=1e-7; end
Base.@kwdef mutable struct Cfg; params::Params=Params(); lattice::Lattice=Lattice(); file::FileCfg=FileCfg(); hamiltonian::Ham=Ham(); optimizer::Opt=Opt(); backend::Symbol=:cpu; end

cfg = Cfg(params=Params(Nv=4), optimizer=Opt(MaxIter=50, lr=0.25, tol=1e-7), file=FileCfg(WriteFile="X_opt.jld2"))

# Run optimization
Topt = gaussian_fpeps(cfg)

# Build the same loss closure to compute optimized energy
Nv = cfg.params.Nv
Lx, Ly = cfg.lattice.Lx, cfg.lattice.Ly
ht = cfg.hamiltonian.ht
DeltaX, DeltaY = cfg.hamiltonian.DeltaX, cfg.hamiltonian.DeltaY
Mu = cfg.hamiltonian.solve_mu_from_delta ? solve_mu(DeltaX, cfg.hamiltonian.delta) : cfg.hamiltonian.Mu

lossT = optimize_runtime_loss(; Nv=Nv, Lx=Lx, Ly=Ly, hoping=ht, DeltaX=DeltaX, DeltaY=DeltaY, Mu=Mu)
E_opt = lossT(Topt)
E_exact = eg(Lx, Ly, ht, DeltaX, DeltaY, Mu)

println("Lx=Ly=$(Lx), Nv=$(Nv)")
println("Optimized energy density: ", E_opt)
println("Exact energy density:      ", E_exact)
println("Absolute error:            ", abs(E_opt - E_exact))
