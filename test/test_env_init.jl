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

bz = BrillouinZone2D(24, 24, (Symbol(config["system_params"]["x_bc"]), Symbol(config["system_params"]["y_bc"])))
μ_from_δ = GfPEPS.solve_for_mu(bz, config["hamiltonian"]["hole_density"], config["hamiltonian"]["t"], config["hamiltonian"]["pairing_type"], config["hamiltonian"]["Δ_0"])
H = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(1, 1); pairing_type = config["hamiltonian"]["pairing_type"], t=config["hamiltonian"]["t"], Δ_0 = config["hamiltonian"]["Δ_0"], μ = μ_from_δ)
boundary_alg = (; tol = 1e-8, maxiter=500, alg = :simultaneous)

χenv_max = 24

#= My init =#
env = GfPEPS.init_ctmrg_env(peps);
env, _ = GfPEPS.grow_env(peps, env, 6, χenv_max; boundary_alg...);
energy_peps_my = real(expectation_value(peps, H, env))
@show energy_peps_my

#= PEPSKit heuristics =#
# env0 = CTMRGEnv(peps, oneunit(space(peps.A[1],2)))
env, _ = GfPEPS.initialize_ctmrg_env_old(peps,6,χenv_max; boundary_alg...);
energy_peps_heuristics = real(expectation_value(peps, H, env))
@show energy_peps_heuristics

@show optim_energy

@test energy_peps_my ≈ energy_peps_heuristics atol=1e-8
@test energy_peps_my ≈ optim_energy atol=1e-2
@test energy_peps_heuristics ≈ optim_energy atol=1e-2