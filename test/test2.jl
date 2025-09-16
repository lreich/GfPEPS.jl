using Revise
using Test
using GfPEPS
using JSON: parsefile
using LinearAlgebra

using TensorOperations
using PEPSKit
using TensorKit

using Random

import TensorKitTensors.HubbardOperators as hub


Random.seed!(32178046)

obj = Gaussian_fPEPS();
X_opt = obj.X_opt
G = GfPEPS.Γ_fiducial(X_opt, obj.Nv, obj.Nf)

Nf = obj.Nf
Nv = obj.Nv
# Nf = 2
# Nv = 2

# U, G1 = GfPEPS.generate_cormat(Nf, Nv)
# G = GfPEPS.fiducial_cormat(U)

# test if even parity
H_parent = GfPEPS.parent_Hamiltonian_BdG(G)
E, W = GfPEPS.bogoliubov(H_parent)
det(W)
@assert det(W) ≈ 1

A, B = GfPEPS.bogoliubov_blocks(W)

ω = GfPEPS.virtual_state(Nv)
F = GfPEPS.fiducial_state(Nf, Nv, -inv(A) * B)
peps = GfPEPS.get_peps(ω, F)
lattice = collect(space(t, 1) for t in peps.A)

Espace = Vect[FermionParity](0 => 4, 1 => 4)
env = CTMRGEnv(randn, ComplexF64, peps, Espace)
for χenv in [8, 16]
    trscheme = truncdim(χenv) & truncerr(1.0e-12)
    env, = leading_boundary(
        env, peps; tol = 1.0e-10, maxiter = 100, trscheme,
        alg = :sequential, projector_alg = :fullinfinite
    )
end

bz = GfPEPS.BrillouinZone((128,128), (false, true))

O = LocalOperator(lattice, ((1, 1),) => hub.e_num(Trivial, Trivial))
doping1 = 1 - real(expectation_value(peps, O, env))
doping2 = GfPEPS.doping_peps(G, bz, Nf)
@info "Doping" doping1 doping2

mags1 = map([hub.S_x, hub.S_y, hub.S_z]) do func
    O = LocalOperator(lattice, ((1, 1),) => func(Trivial, Trivial))
    real(expectation_value(peps, O, env))
end
mags2 = GfPEPS.mags_peps(G, bz, Nf)
@info "Magnetization" mags1 mags2

singlets1 = map([(1, 2), (2, 1)]) do site2
    O = LocalOperator(lattice, ((1, 1), site2) => -hub.singlet_min(Trivial, Trivial))
    expectation_value(peps, O, env)
end
singlets2 = map([[1, 0], [0, -1]]) do v
    GfPEPS.singlet_peps(G, bz, Nf, v)
end
@info "NN singlet pairing" singlets1 singlets2

hoppings1 = map([(1, 2), (2, 1)]) do site2
    O = LocalOperator(lattice, ((1, 1), site2) => hub.e_hopping(Trivial, Trivial))
    real(expectation_value(peps, O, env))
end
hoppings2 = map([[1, 0], [0, -1]]) do v
    GfPEPS.hopping_peps(G, bz, Nf, v)
end
@info "NN hopping energy" hoppings1 hoppings2

t, Δx, Δy, mu = obj.t, obj.Δ_options["Δ_x"], obj.Δ_options["Δ_y"], obj.μ
ham = GfPEPS.hamiltonian(ComplexF64, InfiniteSquare(1, 1); t, Δx, Δy, mu)
energy1 = expectation_value(peps, ham, env)
energy2 = GfPEPS.energy_peps(G, bz, Nf; Δx, Δy, t, mu)
@info "PEPS energy per site" energy1 energy2