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
Γ_out = GfPEPS.Γ_fiducial(X, Nv, Nf)
@assert Γ_out ≈ -transpose(Γ_out)
@assert Γ_out^2 ≈ -I
A = Γ_out[1:2Nf, 1:2Nf]
D = Γ_out[2Nf+1:end, 2Nf+1:end]
B = Γ_out[1:2Nf, 2Nf+1:end]
@assert A ≈ -transpose(A)
@assert D ≈ -transpose(D)
@assert all(isreal.(B))

#= Test transformation to Dirac =#
F = GfPEPS.qp_to_qq_ordering_transformation(Nf)
P_full = BlockDiagonal([F, Matrix(I(8*Nv))])
Γ_out_qq = P_full * Γ_out * P_full'
@assert Γ_out_qq ≈ -transpose(Γ_out_qq)
@assert Γ_out_qq^2 ≈ -I

eigen(Γ_out_qq).values

#= Bis hier GUT =#

S0 = [1  1; im  -im]
S = kron(I(N), S0)

# Γ_out_dirac_qq = S' * Γ_out_qq * S
Γ_out_dirac_qq = inv(conj.(S)) * Γ_out_qq * inv(transpose(S))
# Γ_out_dirac_qq = inv(conj.(S)) * Γ_out_qq * inv(S')
perm = vcat(1:2:(2N), 2:2:(2N))
Γ_out_dirac = Γ_out_dirac_qq[perm, perm]

# # Majorana(qq) <- Nambu(interleaved per mode) map for one mode
# S0 = [1  1; im  -im]  # [c1; c2] = S0 * [a; a†]
# # Lift to all modes in your mode order (f..., l1,r1,..., u1,d1,...)
# S_pair = kron(I(N), S0)  # maps ψ_interleaved -> r
# # Permutation K: ψ_interleaved = K * ψ_block
# p = reduce(vcat, ([i, N_modes + i] for i in 1:N_modes))  # [1,N+1,2,N+2,...,N,2N]
# Id = Matrix(I, 2N_modes, 2N_modes)
# K  = Id[p, :]                      # so K * x == x[p]
# # Full map r = S * ψ_block
# S = S_pair * K

# Γ_out_dirac = S \ (Γ_out_qq / transpose(S))
# Γ_out_dirac = ( Γ_out_dirac - transpose(Γ_out_dirac) ) / 2  # ensure exact antisymmetry


# S = GfPEPS.get_Dirac_to_Majorana_qq_transformation(N)

# # Γ_out_dirac = 1/4 .* (S' * Γ_out_qq * conj.(S))

# Γ_out_dirac = inv(conj.(S)) * Γ_out_qq * inv(transpose(S))

Γ_out_dirac*Γ_out_dirac'


@assert Γ_out_dirac' ≈ -Γ_out_dirac # anti hermitian
@assert 4 .* Γ_out_dirac*Γ_out_dirac' ≈ I

R_conj = Γ_out_dirac[1:N, 1:N]
Q_conj = Γ_out_dirac[1:N, N+1:end]
Q = Γ_out_dirac[N+1:end, 1:N]
R = Γ_out_dirac[N+1:end, N+1:end]

# Q = Γ_out_dirac[1:N, 1:N]
# R = Γ_out_dirac[1:N, N+1:end]
# R_conj = Γ_out_dirac[N+1:end, 1:N]
# Q_conj = Γ_out_dirac[N+1:end, N+1:end]
@test R_conj ≈ conj(R)
@test Q_conj ≈ conj(Q)
@test R' ≈ -R
@test transpose(Q) ≈ -Q

abs.(eigen(Γ_out_dirac).values)

# H = GfPEPS.get_parent_hamiltonian(Γ_out, Nf, Nv)

H = Hermitian(-im .* Γ_out_dirac)

E, M = GfPEPS.bogoliubov(H)

_, M =  eigen(H; sortby = (x -> -real(x)))
U,V = GfPEPS.get_bogoliubov_blocks(M)
V = conj.(V)
@test U'U + V'V ≈ I
@test transpose(U) * V ≈ - transpose(V) * U

M'*H*M
M'M ≈ I

Z = -conj(U) \ V # pairing matrix
# Z = -conj(V) \ U # pairing matrix
Z = (Z - transpose(Z)) / 2  # ensure exact antisymmetry

peps = GfPEPS.translate(X, Nf, Nv)

ω = GfPEPS.virtual_bond_state(Nv)
A = GfPEPS.fiducial_state(Nf, Nv, Z)
peps = GfPEPS.get_peps(ω, A)
peps = peps / norm(peps)

peps

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

Δ_x = res.Δ_options["Δ_x"]
Δ_y = res.Δ_options["Δ_y"]

ham = GfPEPS.hamiltonian(ComplexF64, InfiniteSquare(1, 1); t=res.t, Δx = Δ_x, Δy = Δ_y, mu = res.μ)
energy1 = expectation_value(peps, ham, env)
# energy2 = BCS.energy_peps(G, bz, Np; Δx, Δy, t, mu)
@info "PEPS energy per site" energy1