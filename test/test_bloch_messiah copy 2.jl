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

T, physical_spaces, virtual_spaces = GfPEPS.get_empty_peps_tensor(Nf, Nv)

#= overlap calc =#

# parity = mod(M_A,2)
parity = mod(size(Vbar, 1), 2)  # Row count = number of original modes kept

states_f = 0:(2^Nf - 1)
states_v = 0:(2^Nv - 1)

# Cartesian product; store as tuples
# states = [(f,l,r,d,u) for f in states_f for l in states_v for r in states_v
#                                    for d in states_v for u in states_v]

# states = [(r,d,l,u,f) for r in states_v for d in states_v for l in states_v
#                                    for u in states_v for f in states_f]

states = [(f,u,r,d,l) for f in states_f for u in states_v for r in states_v
                                   for d in states_v for l in states_v]

ind_f_dict = GfPEPS.translate_occ_to_TM_dict(Nf)
ind_v_dict = GfPEPS.translate_occ_to_TM_dict(Nv)

# Threads.@threads for state in states
for state in states
    # f_occ, l_occ, r_occ, d_occ, u_occ = state
    f_occ, u_occ, r_occ, d_occ, l_occ = state

    # convert occ to bitstrings
    f = (digits(f_occ, base=2, pad=Nf))
    u = (digits(u_occ, base=2, pad=Nv))
    l = (digits(l_occ, base=2, pad=Nv))
    d = (digits(d_occ, base=2, pad=Nv))
    r = (digits(r_occ, base=2, pad=Nv))

    # Boolean occupation vector to select rows from R_mat_full (true if occupied)
    # occ_bool = map(==(1), reverse(vcat(f,l,r,d,u)))
    # occ_bool = (map(==(1), vcat(f,u,r,d,l)))
    # occ_bool = (map(==(1), vcat(u,d,r,l,f)))
    # occ_bool = (map(==(1), vcat(u,d,l,r,f)))
    # occ_bool = (map(==(1), vcat(f,u,r,d,l)))
    occ_bool = vcat(f, u, r, d, l) .== 1
    # occ_bool = (map(==(1), reverse(vcat(f,l,r,u,d))))
    M_prime = sum(occ_bool)

    parity_f = mod(sum(f), 2)
    parity_v = mod(sum(l) + sum(u) + sum(r) + sum(d), 2)

    if mod(M_prime,2) != parity || parity_f != parity_v # skip if parity doesn't match
        continue
    end

    # nL = sum(l); nR = sum(r); nD = sum(d); nU = sum(u)
    # gauge_exponent = nL*nR + nL*nD + nL*nU + nR*nU + nD*nU
    # gauge_sign = isodd(gauge_exponent) ? -1 : 1

    # gauge_exponent1 = sum(l) + sum(r)
    # gauge_sign1 = isodd(gauge_exponent1) ? -1 : 1
    # gauge_exponent2 = sum(u) + sum(d)
    # gauge_sign2 = isodd(gauge_exponent2) ? -1 : 1
    # gauge_sign = gauge_sign1 * gauge_sign2

    # gauge_exponent = sum(u)* (sum(l) + sum(r) + sum(d) + sum(f)) +
    #              sum(l)*(sum(d) +sum(r))
    # parity_gate = isodd(gauge_exponent) ? -1 : 1
    # parity_gate = 1

    if M_prime!=0  
        # build R_mat
        R_mat = R_mat_full[occ_bool,:]

        # fsign = (-1)^(0.5 * M_prime*(M_prime-1)) # fermionic sign from reordering
        fsign = isodd((M_prime * (M_prime - 1)) ÷ 2) ? -1 : 1 # fermionic sign from reordering
        # fsign = 1 
        
        pf_mat = [zeros(M_prime,M_prime) R_mat; -transpose(R_mat) Q_mat]
        # pf_mat = [zeros(M_prime,M_prime) R_mat; -R_mat' Q_mat]
        pf = pfaffian(pf_mat)

        # T[ind_f_dict[f], ind_v_dict[u], ind_v_dict[r], ind_v_dict[d], ind_v_dict[l]] = gauge_sign * fsign * pf / v_prod
        # T[ind_f_dict[f], ind_v_dict[l], ind_v_dict[r], ind_v_dict[d], ind_v_dict[u]] = fsign * parity_gate * pf / v_prod
        T[ind_f_dict[f], ind_v_dict[u], ind_v_dict[r], ind_v_dict[d], ind_v_dict[l]] = fsign * pf / v_prod
        # T[ind_f_dict[f], ind_v_dict[l], ind_v_dict[r], ind_v_dict[u], ind_v_dict[d]] = fsign * pf / v_prod
        # T[ind_f_dict[f], ind_v_dict[d], ind_v_dict[u], ind_v_dict[r], ind_v_dict[l]] = fsign * pf / v_prod
    else # all unoccupied
        # T[ind_f_dict[f], ind_v_dict[u], ind_v_dict[r], ind_v_dict[d], ind_v_dict[l]] = pfaffian(Q_mat) / v_prod
        T[1,1,1,1,1] = pfaffian(Q_mat) / v_prod
    end
end

pepsT = InfinitePEPS(TensorMap(T, physical_spaces ← virtual_spaces))
# pepsT = PEPSKit.peps_normalize(pepsT)

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
for χenv in [8, 16]
    trscheme = truncdim(χenv)
    envT, = leading_boundary(
        envT, pepsT; tol = 1.0e-11, maxiter = 200, trscheme,
        alg = :sequential, projector_alg = :fullinfinite
    )
end
energy1 = real(expectation_value(pepsT, ham, envT))
@show energy1

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