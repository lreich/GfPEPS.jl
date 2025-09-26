using Revise
using Test
using GfPEPS
using JSON: parsefile
using LinearAlgebra

using TensorOperations
using PEPSKit
using TensorKit

GfPEPS.virtual_bond_state(8)

obj = Gaussian_fPEPS();
X_opt = obj.X_opt
G = GfPEPS.Γ_fiducial(X_opt, obj.Nv, obj.Nf)

# test if even parity
H_parent = GfPEPS.get_parent_hamiltonian(G)
E, W = GfPEPS.bogoliubov(H_parent)
det(W)
@assert det(W) ≈ 1

trans_h = GfPEPS.corr_matrix_to_dirac(-G)
N = div(size(G,1),2)
h_rho = -1im .* trans_h[1:N, N+1:end]'
h_kappa = 1im .* trans_h[1:N,1:N]
local_h = GfPEPS.fiducial_hamiltonian(h_rho, h_kappa)
E, M = eigen(local_h)
A_fiducial = reshape(M[:,1], (2^obj.Nv, 2^obj.Nv, 2^obj.Nf, 2^obj.Nv, 2^obj.Nv))
A_fiducial = permutedims(A_fiducial, (5,4,3,2,1))
A_fiducial_final = GfPEPS.add_gates(A_fiducial, obj.Nv) # order ulfdr

# Move physical leg first: (f, u, l, d, r)
Af = permutedims(A_fiducial_final, (3, 1, 2, 4, 5))

Vp = ℂ^(2^obj.Nf)         # physical space
Vv = ℂ^(2^obj.Nv)         # virtual space

# Create a TensorMap with codomain = Vp and domain = Vv ⊗ Vv ⊗ Vv ⊗ Vv
A = TensorMap(Af, Vp ← Vv ⊗ Vv ⊗ Vv ⊗ Vv)
ipeps = InfinitePEPS(A; unitcell = (1, 1))



# @assert det(M) ≈ 1

# U, G = GfPEPS.generate_cormat(obj.Nf, obj.Nv)
# peps = GfPEPS.translate(X_opt, obj.Nf, obj.Nv)

# Γ_opt_majoranas = -GfPEPS.Γ_fiducial(X_opt, obj.Nv, obj.Nf)

# bz = obj.bz

# ε_arr = Array{Float64,1}(undef, size(bz.kvals,2))
# for (i,k) in enumerate(eachcol(bz.kvals))
#     ε_arr[i] = -GfPEPS.E(k, obj.t, obj.μ, obj.Δ_options["pairing_type"], obj.Δ_options["Δ_x"], obj.Δ_options["Δ_y"])
# end
# ε_arr = sort(ε_arr)

# J = GfPEPS.build_J(0, size(bz.kvals,2))


# # convert from majorana basis to complex fermion basis
# Ω = [I(4*obj.Nv + obj.Nf) I(4*obj.Nv + obj.Nf);
#      im*I(4*obj.Nv + obj.Nf) -im*I(4*obj.Nv + obj.Nf)]

# Γ_opt_fermions = -0.5im .* Ω' * Γ_opt_majoranas * Ω 

# E, V = eigen(Γ_opt_fermions)
