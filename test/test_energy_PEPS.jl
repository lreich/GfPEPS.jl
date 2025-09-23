using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using LinearAlgebra
using BlockDiagonals

res = Gaussian_fPEPS()

Nf = res.Nf
Nv = res.Nv
N = (Nf + 4*Nv)
Lx = res.Lx
Ly = res.Ly

X = res.X_opt
peps = GfPEPS.translate(X, Nf, Nv)

Espace = Vect[FermionParity](0 => 4, 1 => 4)
env = CTMRGEnv(randn, ComplexF64, peps, Espace)
# env = CTMRGEnv(randn, ComplexF64, peps)
for χenv in [8, 16]
    trscheme = truncdim(χenv)
    env, = leading_boundary(
        env, peps; tol = 1.0e-10, maxiter = 200, trscheme,
        alg = :sequential, projector_alg = :fullinfinite
    )
end

Δ_x = res.Δ_options["Δ_x"]
Δ_y = res.Δ_options["Δ_y"]
Lx = 101
Ly = 101
bz = BrillouinZone2D(Lx, Ly, (:APBC, :PBC))

ham = GfPEPS.hamiltonian(ComplexF64, InfiniteSquare(1, 1); t=res.t, Δx = Δ_x, Δy = Δ_y, mu = res.μ)
energy1 = expectation_value(peps, ham, env)
# energy2 = BCS.energy_peps(G, bz, Np; Δx, Δy, t, mu)

energy2 = GfPEPS.energy_CM(GfPEPS.Γ_fiducial(X, Nv, Nf), bz, Nf; t=res.t, mu=res.μ, Δx=Δ_x, Δy=Δ_y)

# G_in = GfPEPS.G_in_Fourier(bz, Nv)
# G_out = GaussianMap(Γ_out, G_in, Nf, Nv)
# energy2 = GfPEPS.energy_loss(res.t, res.μ, bz, res.Δ_options["pairing_type"], Δ_x, Δ_y)(G_out)

@info "Energy per site (PEPS)" energy1
@info "Energy per site (CM)" energy2