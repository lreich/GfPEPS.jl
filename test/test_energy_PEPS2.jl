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

# @assert Γ_fiducial ≈ -transpose(Γ_fiducial)
# @assert Γ_fiducial^2 ≈ -I
# A = Γ_fiducial[1:2Nf, 1:2Nf]
# D = Γ_fiducial[2Nf+1:end, 2Nf+1:end]
# B = Γ_fiducial[1:2Nf, 2Nf+1:end]
# @assert A ≈ -transpose(A)
# @assert D ≈ -transpose(D)
# @assert all(isreal.(B))

# # #= Test transformation to Dirac =#
# # F = GfPEPS.qp_to_qq_ordering_transformation(Nf)
# # P_full = BlockDiagonal([F, Matrix(I(8*Nv))])
# # Γ_fiducial_qq = P_full * Γ_fiducial * P_full'
# # @assert Γ_fiducial_qq ≈ -transpose(Γ_fiducial_qq)
# # @assert Γ_fiducial_qq^2 ≈ -I

# S0 = [1  1; im  -im]
# S = kron(I(N), S0)

# # transpose(S) * Γ_fiducial * conj.(S)
# # Γ_fiducial_dirac = inv(conj.(S)) * Γ_fiducial * inv(transpose(S))
# Γ_fiducial_dirac = 1/4 .* S' * Γ_fiducial * S
# perm = vcat(1:2:(2N), 2:2:(2N))
# Γ_fiducial_dirac = Γ_fiducial_dirac[perm, perm]
# @assert Γ_fiducial_dirac' ≈ -Γ_fiducial_dirac # anti hermitian
# Γ_fiducial_dirac*Γ_fiducial_dirac'
# @assert Γ_fiducial_dirac*Γ_fiducial_dirac' ≈ I / 4
# R_conj = Γ_fiducial_dirac[1:N, 1:N]
# Q_conj = Γ_fiducial_dirac[1:N, N+1:end]
# Q = Γ_fiducial_dirac[N+1:end, 1:N]
# R = Γ_fiducial_dirac[N+1:end, N+1:end]
# @test R_conj ≈ conj(R)
# @test Q_conj ≈ conj(Q)
# @test R' ≈ -R
# @test transpose(Q) ≈ -Q

# H2 = (S' * (-0.5im * Γ_fiducial) * S)
# perm = vcat(1:2:(2N), 2:2:(2N))
# H2 = Hermitian(H2[perm, perm])

# H = Hermitian(-2im .* Γ_fiducial_dirac)
# E, M = GfPEPS.bogoliubov(H)
# U,V = GfPEPS.get_bogoliubov_blocks(M)

# H ≈ H2

# H = GfPEPS.get_parent_hamiltonian(Γ_fiducial)
# _, M = GfPEPS.bogoliubov(H)
# U,V = GfPEPS.get_bogoliubov_blocks(M)
# Z = V * inv(U) # pairing matrix 
# Z = (Z - transpose(Z)) / 2  # ensure exact antisymmetry

# # #=  =#
# E, W0 = eigen(H; sortby = (x -> -real(x)))
# n = div(2N, 2)
# Wpos = W0[:, 1:n]
# U2 = Wpos[1:n, :]
# V2 = conj(Wpos[(n + 1):end, :])
# W = similar(W0)
# W[1:n, 1:n] = U2
# W[1:n, (n + 1):(2n)] = V2
# W[(n + 1):(2n), 1:n] = conj.(V2)
# W[(n + 1):(2n), (n + 1):(2n)] = conj.(U2)
# W = Matrix(W')

# U2 = W[1:N, 1:N]
# V2 = W[1:N, N+1:end]

# U' ≈ U2
# V' ≈ V2

# Z = -inv(U')*V'
# Z2 = -inv(U2) * V2
# Z3 = V * inv(U)

# Z ≈ Z2
# Z2 ≈ Z3

# #=  =#

# @test U'U + V'V ≈ I
# @test transpose(U) * V ≈ - transpose(V) * U

# Z = U * inv(V)
# @test Z ≈ -transpose(Z)
# # Z = -U \ V # pairing matrix
# # Z = -conj(U) \ V # pairing matrix
# # Z = -U \ conj(V) # pairing matrix
# # Z = -inv(U')*transpose(V)

# # Z = -V \ conj(U) # pairing matrix
# # Z = -conj(V) \ U # pairing matrix
# Z = (Z - transpose(Z)) / 2  # ensure exact antisymmetry
# Z = (Z2 - transpose(Z2)) / 2
# Z = (Z3 - transpose(Z3)) / 2

# peps = GfPEPS.translate(X, Nf, Nv)

# ω = GfPEPS.virtual_bond_state(Nv)
# F = GfPEPS.fiducial_state(Nf, Nv, Z)
# peps = GfPEPS.get_peps(ω, F)

peps = GfPEPS.translate(X, Nf, Nv)

Espace = Vect[FermionParity](0 => 4, 1 => 4)
env = CTMRGEnv(randn, ComplexF64, peps, Espace)
# env = CTMRGEnv(randn, ComplexF64, peps)
for χenv in [8, 16]
    trscheme = truncdim(χenv)
    env, = leading_boundary(
        env, peps; tol = 1.0e-11, maxiter = 200, trscheme,
        alg = :sequential, projector_alg = :fullinfinite
    )
end

t = 1.0
μ = 1.0
Δ_0 = 1.0
Lx = 128
Ly = 128

ham = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(1, 1); t=t, Δ_0 = Δ_0, μ = μ)
energy1 = real(expectation_value(peps, ham, env))
# energy2 = BCS.energy_peps(G, bz, Np; Δx, Δy, t, mu)

bz = BrillouinZone2D(Lx, Ly, (:APBC, :PBC))

# G_in = GfPEPS.G_in_Fourier(bz, Nv)
# G_out = GaussianMap(Γ_fiducial, G_in, Nf, Nv)
# energy2 = GfPEPS.energy_loss(t, μ, bz, res.Δ_options["pairing_type"], Δ_x, Δ_y)(G_out)

energy2 = GfPEPS.energy_CM(Γ_fiducial, bz, Nf; t=t, mu=μ, Δ_0=Δ_0)

@info "Energy per site (PEPS)" energy1
@info "Energy per site (CM)" energy2