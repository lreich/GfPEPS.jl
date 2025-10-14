using Revise
using Test
using GfPEPS
using JSON: parsefile

config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_kitaev.json"))
X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)

@test optim_energy ≈ exact_energy

params_Kitaev = GfPEPS.Kitaev(
    config["hamiltonian"]["Jx"],
    config["hamiltonian"]["Jy"],
    config["hamiltonian"]["Jz"],
)

# bz = GfPEPS.BrillouinZone2D(config["system_params"]["Lx"], config["system_params"]["Ly"], (:PBC, :APBC))

# gammarnd, _ = GfPEPS.rand_CM(1,2)
# G_in = GfPEPS.G_in_Fourier(bz, 2)
# CMout = GaussianMap(GfPEPS.get_Γ_blocks(gammarnd,1)..., G_in)
# GfPEPS.energy_loss(params_Kitaev,bz)(CMout)

# GfPEPS.exact_energy(params_Kitaev, bz)

# @testset "Energy Kitaev HC (vortex free)" begin
#         config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_kitaev.json"))
#         X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)
#         # test energy from CM
#         @test optim_energy ≈ exact_energy

# end;

# exact = -1.5746 #per unit cell
# exact_site = exact / 2 #per site