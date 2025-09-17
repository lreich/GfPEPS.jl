using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit

Nf = 2
Nv = 2
N = (Nf + 4*Nv)
Lx = 5
Ly = 5

Γ_out = GfPEPS.rand_CM(Nf, Nv)

bz = GfPEPS.BrillouinZone2D(Lx,Ly,(:APBC,:PBC))
G_in = GfPEPS.G_in_Fourier(bz, Nv)
G_f = GfPEPS.GaussianMap(Γ_out, G_in, Nf, Nv)

t, Δx, Δy, mu = 1.0, 0.3, -0.7, -0.4
energy = GfPEPS.energy_loss(t, mu, bz, "d_wave", Δx, Δy)
E_exact = energy(G_f)