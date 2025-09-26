using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using LinearAlgebra
using BlockDiagonals
using Random 
using MatrixFactorizations

Random.seed!(32178046)

Nf = 2
Nv = 2
N = (Nf + 4*Nv)

Γ_fiducial, X = GfPEPS.rand_CM(Nf,Nv)
H = GfPEPS.get_parent_hamiltonian(Γ_fiducial)
_, M = GfPEPS.bogoliubov(H)
U,V = GfPEPS.get_bogoliubov_blocks(M)

Q = conj.(V) * transpose(V)
@assert Q' ≈ Q

P = conj.(V) * transpose(U)
# P = (transpose(P) - P) / 2 # enforce exact skew-symmetry
@assert transpose(P) ≈ -P
@assert Q*P ≈ P*conj.(Q)

_, T = eigen(P'P)

transpose(T)*P*T

_, B = eigen(Q; sortby = (x -> -real(x)))
Q_bar = real(B'*Q*B)
P_bar1 = B'*P*conj.(B)
P_bar1[abs.(P_bar1) .< 1e-12] .= 0.0
P_bar1

S = GfPEPS.canonical_skew_permutation(P_bar1)
D = B*S

tess = D' * Q * D
tess[abs.(tess) .< 1e-12] .= 0.0
tess
tess = D' * P * conj(D)
tess[abs.(tess) .< 1e-12] .= 0.0
tess

A = D' * U
# F = MatrixFactorizations.rq(A)
# Ubar = real(Matrix(F.R))
# # Ubar[abs.(Ubar) .< 1e-12] .= 0.0
# C = Matrix(F.Q)


# # build reversal matrix J (antidiagonal identity)
# n = size(A,1)
# J = zeros(eltype(A), n, n)
# for i in 1:n
#     J[i, n+1-i] = one(eltype(A))
# end

# A_tilde = J*A
# F = qr(transpose(A_tilde))
# Q_tilde = Matrix(F.Q)
# R_tilde = Matrix(F.R)

# C = J*transpose(Q_tilde) 
# U_bar = J*transpose(R_tilde)*J
# U_bar ≈ D'U*C'

# QR of the reversed-transpose matrix
B = J * adjoint(A) * J
F = qr(B)                  # economy QR
Qb = Matrix(F.Q)
Rb = Matrix(F.R)

# form RQ from the QR factors of B
Ubar = real(J * adjoint(Rb) * J)   # upper triangular factor (Ū)
C    = J * adjoint(Qb) * J   # unitary factor (C)

# checks
@assert norm(Ubar * C - A) < 1e-10
@assert norm(adjoint(C) * C - I) < 1e-10   # C is unitary

# now C is the unitary in A = Ubar * C
C

Vbar = real(transpose(D) * V * C')
Vbar[abs.(Vbar) .< 1e-12] .= 0.0
Vbar
# Vbar = D' * V * transpose(C)

U ≈ D*Ubar*C
V ≈ conj.(D)*Vbar*C

Vbar
Ubar


Dmats = [D zeros(N,N); zeros(N,N) conj.(D)]
UV_mats = [Ubar Vbar; Vbar Ubar]
Cmats = [C zeros(N,N); zeros(N,N) conj.(C)]



Dmats * UV_mats * Cmats
M
M ≈ Dmats * UV_mats * Cmats