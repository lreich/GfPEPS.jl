using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using LinearAlgebra
using BlockDiagonals
using Random 

Random.seed!(32178046)

Nf = 2
Nv = 2
N = (Nf + 4*Nv)

Γ_fiducial, X = GfPEPS.rand_CM(Nf,Nv)

# peps = GfPEPS.translate_naive(X, Nf, Nv)
pepsMy = GfPEPS.translate(X, Nf, Nv);

# Espace = Vect[FermionParity](0 => 4, 1 => 4)
# env = CTMRGEnv(randn, ComplexF64, peps, Espace)
# # env = CTMRGEnv(randn, ComplexF64, peps)
# for χenv in [8, 16, 32]
#     trscheme = truncdim(χenv)
#     env, = leading_boundary(
#         env, peps; tol = 1.0e-11, maxiter = 200, trscheme,
#         alg = :sequential, projector_alg = :fullinfinite
#     )
# end

Espace = Vect[FermionParity](0 => 4, 1 => 4)
envMy = CTMRGEnv(randn, ComplexF64, pepsMy, Espace)
# for χenv in [8, 16, 32, 40]
for χenv in [8, 16, 32]
    trscheme = truncdim(χenv)
    envMy, = leading_boundary(
        envMy, pepsMy; tol = 1.0e-12, maxiter = 100, trscheme,
        alg = :simultaneous, projector_alg = :fullinfinite
    )
end

t = 1.0
μ = 1.0
Δ_0 = 1.0
Lx = 128
Ly = 128

ham = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(1, 1); t=t, Δ_0 = Δ_0, μ = μ)
# energy1 = real(expectation_value(peps, ham, env))
energy2 = real(expectation_value(pepsMy, ham, envMy))

bz = BrillouinZone2D(Lx, Ly, (:APBC, :PBC))
energy3 = GfPEPS.energy_CM(Γ_fiducial, bz, Nf; t=t, mu=μ, Δ_0=Δ_0)

# @info "Energy per site (PEPS)" energy1
@info "Energy per site (PEPS My)" energy2
@info "Energy per site (CM)" energy3