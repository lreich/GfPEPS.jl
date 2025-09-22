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

Γ_fiducial,_ = GfPEPS.rand_CM(Nf,Nv)

@assert Γ_fiducial ≈ -transpose(Γ_fiducial)
@assert Γ_fiducial^2 ≈ -I
A = Γ_fiducial[1:2Nf, 1:2Nf]
D = Γ_fiducial[2Nf+1:end, 2Nf+1:end]
B = Γ_fiducial[1:2Nf, 2Nf+1:end]
@assert A ≈ -transpose(A)
@assert D ≈ -transpose(D)
@assert all(isreal.(B))

# #= Test transformation to Dirac =#
# F = GfPEPS.qp_to_qq_ordering_transformation(Nf)
# P_full = BlockDiagonal([F, Matrix(I(8*Nv))])
# Γ_fiducial_qq = P_full * Γ_fiducial * P_full'
# @assert Γ_fiducial_qq ≈ -transpose(Γ_fiducial_qq)
# @assert Γ_fiducial_qq^2 ≈ -I

S0 = [1  1; im  -im]
S = kron(I(N), S0)

# Γ_fiducial_dirac_qq = inv(conj.(S)) * Γ_fiducial_qq * inv(transpose(S))
# # Γ_fiducial_dirac_qq = inv(conj.(S)) * Γ_fiducial_qq * inv(S')
# perm = vcat(1:2:(2N), 2:2:(2N))
# Γ_fiducial_dirac = Γ_fiducial_dirac_qq[perm, perm]
# @assert Γ_fiducial_dirac' ≈ -Γ_fiducial_dirac # anti hermitian
# @assert 4 .* Γ_fiducial_dirac*Γ_fiducial_dirac' ≈ I
# R_conj = Γ_fiducial_dirac[1:N, 1:N]
# Q_conj = Γ_fiducial_dirac[1:N, N+1:end]
# Q = Γ_fiducial_dirac[N+1:end, 1:N]
# R = Γ_fiducial_dirac[N+1:end, N+1:end]
# @test R_conj ≈ conj(R)
# @test Q_conj ≈ conj(Q)
# @test R' ≈ -R
# @test transpose(Q) ≈ -Q

H = (S' * (-0.5im * Γ_fiducial) * S)
perm = vcat(1:2:(2N), 2:2:(2N))
H = Hermitian(H[perm, perm])

# H = Hermitian(-im .* Γ_fiducial_dirac)

E, M = GfPEPS.bogoliubov(H)
U,V = GfPEPS.get_bogoliubov_blocks(M)

# #=  =#
# E, W0 = eigen(H; sortby = (x -> -real(x)))
# n = div(2N, 2)
# Wpos = W0[:, 1:n]
# U = Wpos[1:n, :]
# V = conj(Wpos[(n + 1):end, :])
# W = similar(W0)
# W[1:n, 1:n] = U
# W[1:n, (n + 1):(2n)] = V
# W[(n + 1):(2n), 1:n] = conj.(V)
# W[(n + 1):(2n), (n + 1):(2n)] = conj.(U)
# W = Matrix(W')

# U2 = W[1:N, 1:N]

# V
# V2 = W[1:N, N+1:end]

# U' ≈ U2
# V' ≈ V2

Z = -inv(U')*V'
# Z2 = -inv(U2) * V2

#=  =#

@test U'U + V'V ≈ I
@test transpose(U) * V ≈ - transpose(V) * U

# Z = -U \ V # pairing matrix
# Z = -conj(U) \ V # pairing matrix
# Z = -U \ conj(V) # pairing matrix
# Z = -inv(U')*transpose(V)

# Z = -V \ conj(U) # pairing matrix
# Z = -conj(V) \ U # pairing matrix
Z = (Z - transpose(Z)) / 2  # ensure exact antisymmetry

# peps = GfPEPS.translate(X, Nf, Nv)

ω = GfPEPS.virtual_bond_state(Nv)
A = GfPEPS.fiducial_state(Nf, Nv, Z)
peps = GfPEPS.get_peps(ω, A)
# peps = peps / norm(peps)

Espace = Vect[FermionParity](0 => 4, 1 => 4)
env = CTMRGEnv(randn, ComplexF64, peps, Espace)
# env = CTMRGEnv(randn, ComplexF64, peps)
for χenv in [8, 16]
    trscheme = truncdim(χenv)
    env, = leading_boundary(
        env, peps; tol = 1.0e-9, maxiter = 200, trscheme,
        alg = :sequential, projector_alg = :fullinfinite
    )
end

t = 1.0
μ = 1.0
Δ_x = 1.0
Δ_y = 1.0
Lx = 201
Ly = 201

ham = GfPEPS.hamiltonian(ComplexF64, InfiniteSquare(1, 1); t=t, Δx = Δ_x, Δy = Δ_y, mu = μ)
energy1 = expectation_value(peps, ham, env)
# energy2 = BCS.energy_peps(G, bz, Np; Δx, Δy, t, mu)

bz = BrillouinZone2D(Lx, Ly, (:PBC, :APBC))

# G_in = GfPEPS.G_in_Fourier(bz, Nv)
# G_out = GaussianMap(Γ_fiducial, G_in, Nf, Nv)
# energy2 = GfPEPS.energy_loss(t, μ, bz, res.Δ_options["pairing_type"], Δ_x, Δ_y)(G_out)

energy2 = GfPEPS.energy_CM(Γ_fiducial, bz, Nf; t=t, mu=μ, Δx=Δ_x, Δy=Δ_y)

@info "Energy per site (PEPS)" energy1
@info "Energy per site (CM)" energy2