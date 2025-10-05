using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using LinearAlgebra
using BlockDiagonals
using Random 
using MatrixFactorizations
using SkewLinearAlgebra
using Base.Threads

Random.seed!(32178046)

Nf = 2
Nv = 2
N = (Nf + 4*Nv)

X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt()
Γ_fiducial = GfPEPS.Γ_fiducial(X_opt, Nv, Nf)
peps = translate(X_opt, Nv, Nf)
# Γ_fiducial, X = GfPEPS.rand_CM(Nf,Nv)
H = GfPEPS.get_parent_hamiltonian(Γ_fiducial, Nf, Nv)
_, M = GfPEPS.bogoliubov(H)

Dmat,UVmat,Cmat = GfPEPS.bloch_messiah_decomposition(M)
Dmat_prime,UVmat_prime,Cmat_prime = GfPEPS.truncated_bloch_messiah(Dmat, UVmat, Cmat)

D, Ubar, Vbar, C = GfPEPS.get_mats_from_bloch_messiah(Dmat_prime, UVmat_prime, Cmat_prime)
M_A = size(Vbar, 2)

v_prod = prod([abs(Vbar[i-1, i]) for i in 2:2:M_A])

R_mat_full = D*Vbar # has the same ordering as H
Q_mat = Ubar*Vbar # has the same ordering as H
@assert Q_mat ≈ - transpose(Q_mat)
Q_mat = (Q_mat - transpose(Q_mat)) / 2 # enforce exact skew-symmetry

T, physical_spaces, virtual_spaces = GfPEPS.get_empty_peps_tensor2(Nf, Nv)

#= overlap calc =#

# parity = mod(M_A,2)
parity = mod(size(Vbar, 1), 2)  # Row count = number of original modes kept

N_states = 0:(2^N - 1)
states = digits.(N_states, base=2, pad=N) # (f,u,r,d,l)

# swap_adjacent_pairs!(v) = ( @assert iseven(length(v)); for i in 1:2:length(v); v[i], v[i+1] = v[i+1], v[i]; end; v )
# foreach(swap_adjacent_pairs!, states)

# Threads.@threads for state in states
for i in eachindex(states)
    state = states[i]
    state_ind = state .+ 1
    occ_bool = state .== 1
    M_prime = sum(occ_bool)

    parity_f = mod(sum(state[1:Nf]), 2)
    parity_v = mod(sum(state[Nf+1:end]), 2)

    if mod(M_prime,2) != parity || parity_f != parity_v # skip if parity doesn't match
        continue
    end

    if M_prime!=0  
        # build R_mat
        R_mat = R_mat_full[occ_bool,:]
        fsign = isodd((M_prime * (M_prime - 1)) ÷ 2) ? -1 : 1 # fermionic sign from reordering
        pf = pfaffian([zeros(M_prime,M_prime) R_mat; -transpose(R_mat) Q_mat])
        T[state_ind...] = fsign * pf / v_prod
    else # all unoccupied
        T[state_ind...] = pfaffian(Q_mat) / v_prod
    end
end

# # filter zero elements out
# T = T[T .!= 0]

pepsTM = TensorMap(T, physical_spaces ← virtual_spaces)
ω = GfPEPS.virtual_bond_state(Nv)

pepsT = GfPEPS.get_peps(ω, pepsTM)

# pepsT2
# # pepsT = PEPSKit.peps_normalize(pepsT)

# pepsT3, physical_spaces2, virtual_spaces2 = GfPEPS.get_empty_peps_tensor2(Nf, Nv)

# pepsT3[1,1,1,1,1,1,1,1,1,1] = 1.0

# pepsT3[[1,1,1,1,1,1,1,1,1,1]...]


# TensorMap(pepsT3, physical_spaces2 ← virtual_spaces2)

# U,V = GfPEPS.get_bogoliubov_blocks(M)
# pepsT2 = GfPEPS.fiducial_state(Nf, Nv, V * inv(U))


# pepsT2 = pepsT2 / norm(pepsT2)

# pepsT2[(FermionParity(1), FermionParity(1), FermionParity(0), FermionParity(0), FermionParity(0), FermionParity(0), FermionParity(0), FermionParity(0), FermionParity(0), FermionParity(0))]

# space(pepsT.A[1])
# space(pepsT2).codomain === physical_spaces2
# space(pepsT2).domain === virtual_spaces2

sector_data = pepsT.A[1][(FermionParity(0), FermionParity(0), FermionParity(0), FermionParity(0), FermionParity(0))]
sector_data2 = peps.A[1][(FermionParity(0), FermionParity(0), FermionParity(0), FermionParity(0), FermionParity(0))]
abs.(sector_data) ≈ abs.(sector_data2)

#=  =#
t = 1.0
# μ = 1.0
Δ_0 = 1.0
Lx = 128
Ly = 128
bz = BrillouinZone2D(Lx, Ly, (:APBC, :PBC))
μ = GfPEPS.solve_for_mu(bz,0.13,t,"d_wave",Δ_0)

ham = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(1, 1); t=t, Δ_0 = Δ_0, μ = μ)

Espace = Vect[FermionParity](0 => 4, 1 => 4)
envT = CTMRGEnv(randn, ComplexF64, pepsT, Espace)
# env = CTMRGEnv(randn, ComplexF64, peps)
for χenv in [8, 16, 32]
    trscheme = truncdim(χenv)
    envT, = leading_boundary(
        envT, pepsT; tol = 1.0e-9, maxiter = 500, trscheme,
        alg = :simultaneous, projector_alg = :fullinfinite
    )
end
energy1 = real(expectation_value(pepsT, ham, envT))
@show energy1

Espace = Vect[FermionParity](0 => 4, 1 => 4)
env = CTMRGEnv(randn, ComplexF64, peps, Espace)
# env = CTMRGEnv(randn, ComplexF64, peps)
for χenv in [8, 16, 32]
    trscheme = truncdim(χenv)
    env, = leading_boundary(
        env, peps; tol = 1.0e-9, maxiter = 500, trscheme,
        alg = :simultaneous, projector_alg = :fullinfinite
    )
end
energy2 = real(expectation_value(peps, ham, env))
@show energy2

energy3 = GfPEPS.energy_CM(Γ_fiducial, bz, Nf; t=t, pairing_type="d_wave", Δ_0=Δ_0, mu=μ)
@show energy3

# use computed energy for faster tests
@test energy1 ≈ energy2 atol=1e-6

# ind_f_dict2 = GfPEPS.translate_occ_to_TM_dict(Nf)
# ind_v_dict2 = GfPEPS.translate_occ_to_TM_dict(Nv)
# state = (1,1,0,0,0)
# f_occ, l_occ, r_occ, d_occ, u_occ = state
# # convert occ to bitstrings
# f = (digits(f_occ, base=2, pad=Nf))
# l = (digits(l_occ, base=2, pad=Nv))
# r = (digits(r_occ, base=2, pad=Nv))
# d = (digits(d_occ, base=2, pad=Nv))
# u = (digits(u_occ, base=2, pad=Nv))
# TTest, physical_spaces, virtual_spaces = GfPEPS.get_empty_peps_tensor(Nf, Nv)
# TTest[ind_f_dict2[f], ind_v_dict2[l], ind_v_dict2[r], ind_v_dict2[d], ind_v_dict2[u]] = 8.88888888888888
# TTestT = TensorMap(TTest, physical_spaces ← virtual_spaces)
# TTestT[(FermionParity(1), FermionParity(1), FermionParity(0), FermionParity(0), FermionParity(0))]