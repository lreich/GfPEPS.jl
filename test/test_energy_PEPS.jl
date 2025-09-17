using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using LinearAlgebra

res = Gaussian_fPEPS()

Nf = res.Nf
Nv = res.Nv
N = (Nf + 4*Nv)
Lx = res.Lx
Ly = res.Ly

X = res.X_opt
Γ_out = GfPEPS.Γ_fiducial(X, Nv, Nf)

# Nf = 2
# Nv = 2
# N = (Nf + 4*Nv)
# Lx = 5
# Ly = 5

# Γ_out, X = GfPEPS.rand_CM(Nf, Nv)
Ω, Ωdag = GfPEPS.get_Dirac_to_Majorana_transformation(N)
Γ_out_dirac = 1/2 .* (transpose(Ω) * Γ_out * conj(Ω))

Rconj = Γ_out_dirac[1:N, 1:N]
Qconj = Γ_out_dirac[1:N, N+1:end]
Q = Γ_out_dirac[N+1:end, 1:N]
R = Γ_out_dirac[N+1:end, N+1:end]
@test Rconj ≈ conj(R)
@test Qconj ≈ conj(Q)
@test R' ≈ -R
@test transpose(Q) ≈ -Q

H = GfPEPS.get_parent_hamiltonian(Γ_out)
E, M = GfPEPS.bogoliubov(H)

_, M =  eigen(H)
U,V = GfPEPS.get_bogoliubov_blocks(M)
@test U'U + V'V ≈ I
transpose(U) * V - transpose(V) * U

@test transpose(U) * V ≈ - transpose(V) * U

M'*H*M

M'M ≈ I


peps = GfPEPS.translate(X, Nf, Nv)

bz = GfPEPS.BrillouinZone2D(Lx,Ly,(:APBC,:PBC))
G_in = GfPEPS.G_in_Fourier(bz, Nv)
G_f = GfPEPS.GaussianMap(Γ_out, G_in, Nf, Nv)

t, Δx, Δy, mu = 1.0, 0.3, -0.7, -0.4
energy = GfPEPS.energy_loss(t, mu, bz, "d_wave", Δx, Δy)
E_exact = energy(G_f)