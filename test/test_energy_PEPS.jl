using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using Random 

Random.seed!(1234)

Nf = 2
Nv = 3
N = (Nf + 4*Nv)

Γ_fiducial, X = GfPEPS.rand_CM(Nf,Nv)

# peps = GfPEPS.translate_naive(X, Nf, Nv)
peps = GfPEPS.translate(X, Nf, Nv);

χenv_max = 8
boundary_alg = (; tol = 1e-8, maxiter=100, alg = :simultaneous, trscheme = FixedSpaceTruncation())
Espace = Vect[FermionParity](0 => χenv_max / 2, 1 => χenv_max / 2)
env0 = CTMRGEnv(peps, oneunit(space(peps.A[1],2)))
env1, = leading_boundary(env0, peps; alg = :sequential, trscheme = truncspace(Espace), maxiter = 5)
env, = leading_boundary(env1, peps; boundary_alg...)

t = 1.0
pairing_type = "d_wave"
Δ_0 = 1.0
μ = 1.0
params = GfPEPS.BCS(
    t,
    μ,
    pairing_type,
    Δ_0,
)

Lx = 128
Ly = 128

ham = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(1, 1); t=t, Δ_0 = Δ_0, μ = μ)
energy1 = real(expectation_value(peps, ham, env))

bz = BrillouinZone2D(Lx, Ly, (:APBC, :PBC))
energy2 = GfPEPS.energy_CM(Γ_fiducial, bz, Nf, params)

@test energy1 ≈ energy2
