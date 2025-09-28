using Revise
using Test
using GfPEPS
using JSON: parsefile

@testset "Hole density BCS" begin
    conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json"))
    Nf = conf["params"]["N_physical_fermions_on_site"]
    Nv = conf["params"]["N_virtual_fermions_on_bond"]
    Lx = conf["system_params"]["Lx"]
    Ly = conf["system_params"]["Ly"]

    X_opt, optim_energy, exact_energy = get_X_opt(;conf=conf)
    Γ_opt = Γ_fiducial(X_opt, Nv, Nf)

    bc = (Symbol(conf["system_params"]["x_bc"]), Symbol(conf["system_params"]["y_bc"]))
    bz = BrillouinZone2D(Lx, Ly, bc)

    @test GfPEPS.doping_bcs(Γ_opt, bz, Nf) ≈ conf["hamiltonian"]["hole_density"] atol=1e-6
end;