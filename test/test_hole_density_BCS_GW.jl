#= 
Test if the hole density matches the desired value AFTER the Gutzwiller projection
 =#

using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using JSON: parsefile

@testset "Hole density BCS after Gutzwiller projection" begin
    config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_Gutzwiller_doping.json"))
    X_opt, optim_energy, E_exact = GfPEPS.get_X_opt(;conf=config)
    peps = GfPEPS.translate(X_opt, config["params"]["N_physical_fermions_on_site"], config["params"]["N_virtual_fermions_on_bond"]);

    bz = GfPEPS.BrillouinZone2D(config["system_params"]["Lx"], config["system_params"]["Ly"], (Symbol(config["system_params"]["x_bc"]), Symbol(config["system_params"]["y_bc"])))

    GfPEPS.doping_bcs(X_opt, bz, config["params"]["N_physical_fermions_on_site"], config["params"]["N_virtual_fermions_on_bond"])

    χ_env_max = config["PEPSKit"]["χ_env_max"]
    Espace = Vect[FermionParity](0 => χ_env_max / 2, 1 => χ_env_max / 2)
    boundary_alg = (; tol = config["PEPSKit"]["tol"], maxiter = config["PEPSKit"]["maxiter"], alg = :simultaneous, trscheme = FixedSpaceTruncation())

    env0 = CTMRGEnv(peps, oneunit(space(peps.A[1],2)))
    env1, = leading_boundary(env0, peps; alg = :sequential, trscheme = truncspace(Espace), maxiter = 5)
    env, = leading_boundary(env1, peps; boundary_alg...);

    δ_PEPS = GfPEPS.doping_peps(peps,env)

    # find fugacity z such that doping after projection matches target doping
    δ = config["hamiltonian"]["hole_density"]
    δ_atol = 1e-8
    z, env_projected = GfPEPS.solve_for_fugacity(peps, δ_PEPS; atol=δ_atol)
    PG = GfPEPS.gutzwiller_projector(z)
    peps_projected = GfPEPS.gutzwiller_project(z,peps)

    δ_PEPS_projected = GfPEPS.doping_pepsGW(peps_projected,env_projected)

    @test δ_PEPS_projected ≈ δ_PEPS atol=δ_atol
end;