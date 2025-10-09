using Revise
using Test
using GfPEPS
using JSON: parsefile

config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_kitaev.json"))
X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)

@test optim_energy ≈ exact_energy

# @testset "Energy Kitaev HC (vortex free)" begin
#         config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_kitaev.json"))
#         X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)
#         # test energy from CM
#         @test optim_energy ≈ exact_energy

# end;

# exact = -1.5746 #per unit cell
# exact_site = exact / 2 #per site

-0.1968234 * 4 * 2