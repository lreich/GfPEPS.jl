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

X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt()
Γ_fiducial = GfPEPS.Γ_fiducial(X_opt, Nv, Nf)
# Γ_fiducial, X = GfPEPS.rand_CM(Nf,Nv)
H = GfPEPS.get_parent_hamiltonian(Γ_fiducial)
_, M = GfPEPS.bogoliubov(H)

#= =#
U,V = GfPEPS.get_bogoliubov_blocks(M)
Q = conj.(V) * transpose(V)
P = conj.(V) * transpose(U)
_, B = eigen(Q; sortby = (x -> -real(x)))
P_bar = B'*P*conj.(B)

W = Hermitian(P_bar' * P_bar)
E, Φ = eigen(Hermitian(W); sortby = (x -> -real(x)))
E
alphas = sqrt.(abs.(E))

Φ_prime = similar(Φ)
# build orthogonal eigenvectors
for j in eachindex(alphas)
    Φ_prime[:, j] = (P_bar'*conj.(Φ[:, j])) / alphas[j]
end

# build S
S = similar(P)
n_zeros = 0
for j in eachindex(alphas)
    if alphas[j] ≈ 0.0
        S[:, end-n_zeros] = Φ[:, j]
        n_zeros += 1
    else
        if isodd(j)
            S[:, j] = Φ[:, j]
        else
            S[:, j] = Φ_prime[:, j-1]
        end
    end
end

P_bar * Φ_prime[:,1]

S' * P_bar * conj.(S)

dot(Φ[:,2], Φ_prime[:,2])

S, X = GfPEPS.skew_canonical_form2(P_bar)

#=  =#

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