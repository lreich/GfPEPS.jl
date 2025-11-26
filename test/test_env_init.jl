using Revise
using Test
using GfPEPS
using JSON: parsefile
using Random
using TensorKit
using PEPSKit
Random.seed!(1234)

config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_s_wave.json"))
config["params"]["N_virtual_fermions_on_bond"] = 2
config["system_params"]["Lx"] = 24
config["system_params"]["Ly"] = 24
X_opt, optim_energy, E_exact, _ = GfPEPS.get_X_opt(;conf=config)

peps = GfPEPS.translate(X_opt, config["params"]["N_physical_fermions_on_site"], config["params"]["N_virtual_fermions_on_bond"])

# initialize env Eriks methode:

V_id = ProductSpace{GradedSpace{FermionParity, Tuple{Int64, Int64}}, 0}()

corner_space = ProductSpace{GradedSpace{FermionParity, Tuple{Int64, Int64}}, 0}() ⊗ ProductSpace{GradedSpace{FermionParity, Tuple{Int64, Int64}}, 0}()'
edge_space = ProductSpace{GradedSpace{FermionParity, Tuple{Int64, Int64}}, 0}()

env0 = CTMRGEnv(peps, oneunit(space(peps.A[1],2)))
env0.corners
env0.edges
# init PEPSKit heuristics:


bz = BrillouinZone2D(24, 24, (Symbol(config["system_params"]["x_bc"]), Symbol(config["system_params"]["y_bc"])))
μ_from_δ = GfPEPS.solve_for_mu(bz, config["hamiltonian"]["hole_density"], config["hamiltonian"]["t"], config["hamiltonian"]["pairing_type"], config["hamiltonian"]["Δ_0"])

χenv_max = 12
boundary_alg = (; tol = 1e-8, maxiter=100, alg = :simultaneous, trscheme = FixedSpaceTruncation())
Espace = Vect[FermionParity](0 => χenv_max / 2, 1 => χenv_max / 2)
env0 = CTMRGEnv(peps, oneunit(space(peps.A[1],2)))
env1, = leading_boundary(env0, peps; alg = :sequential, trscheme = truncspace(Espace), maxiter = 5)
env, = leading_boundary(env1, peps; boundary_alg...)

H = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(1, 1); pairing_type = config["hamiltonian"]["pairing_type"], t=config["hamiltonian"]["t"], Δ_0 = config["hamiltonian"]["Δ_0"], μ = μ_from_δ)
energy_peps = real(expectation_value(peps, H, env))
@test energy_peps ≈ optim_energy atol=1e-2