using Revise
using Test
using GfPEPS
using JSON: parsefile
using Random
using TensorKit
using PEPSKit

config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_kitaev.json"))
X_opt, optim_energy, exact_energy, _ = GfPEPS.get_X_opt(;conf=config)

@test optim_energy ≈ exact_energy atol=1e-4

params_Kitaev = GfPEPS.Kitaev(
    config["hamiltonian"]["Jx"],
    config["hamiltonian"]["Jy"],
    config["hamiltonian"]["Jz"],
)

peps = GfPEPS.translate(X_opt, config["params"]["N_physical_fermions_on_site"], config["params"]["N_virtual_fermions_on_bond"]; tol = 1e-8)

χenv_max = 8
boundary_alg = (; tol = 1e-8, maxiter=1000, alg = :simultaneous)
env, = GfPEPS.initialize_ctmrg_env(peps, 4, χenv_max; boundary_alg...);

ham = GfPEPS.Kitaev_Hamiltonian(ComplexF64, InfiniteSquare(1, 1); Jx=params_Kitaev.Jx, Jy=params_Kitaev.Jy, Jz=params_Kitaev.Jz)
energy1 = real(expectation_value(peps, ham, env))

corrh, corrv, expect_f_dag, expect_f = GfPEPS.kitaev_two_point_correlator(5, peps, env)

@show energy1
@show optim_energy
@test energy1 ≈ optim_energy atol=2e-2 # depends on Nv

# exact = -1.5746 #per unit cell
# exact_site = exact / 2 #per site