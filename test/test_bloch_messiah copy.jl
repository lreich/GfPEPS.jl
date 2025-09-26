using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using LinearAlgebra
using BlockDiagonals
using Random 
using MatrixFactorizations
using SkewLinearAlgebra: pfaffian

Random.seed!(32178046)

Nf = 2
Nv = 2
N = (Nf + 4*Nv)

# res = Gaussian_fPEPS();
# X_opt = res.X_opt
# Γ_fiducial = GfPEPS.Γ_fiducial(X_opt, Nv, Nf)
Γ_fiducial, X = GfPEPS.rand_CM(Nf,Nv)
H = GfPEPS.get_parent_hamiltonian(Γ_fiducial)
_, M = GfPEPS.bogoliubov(H)

Dmat,UVmat,Cmat = GfPEPS.bloch_messiah_decomposition(M)

Ubar = UVmat[1:N, 1:N]
Vbar = UVmat[N+1:end, 1:N]

v_els = [Vbar[i-1, i] for i in 2:2:N]

D = Dmat[1:N, 1:N]
R_mat = D*Vbar
Q_mat = Ubar*Vbar

pfaff_mat = [zeros(N,N) R_mat; -transpose(R_mat) Q_mat]
pfaff_mat = (pfaff_mat - transpose(pfaff_mat)) / 2 # enforce exact skew-symmetry

(-1)^(1/2 * N*(N-1)) / prod(v_els) * pfaffian(pfaff_mat)