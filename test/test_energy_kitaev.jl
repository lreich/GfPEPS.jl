using Revise
using Test
using GfPEPS
using JSON: parsefile
using Random
using TensorKit
using PEPSKit

config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_kitaev.json"))
X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)

@test optim_energy ≈ exact_energy

params_Kitaev = GfPEPS.Kitaev(
    config["hamiltonian"]["Jx"],
    config["hamiltonian"]["Jy"],
    config["hamiltonian"]["Jz"],
)

peps = GfPEPS.translate(X_opt, config["params"]["N_physical_fermions_on_site"], config["params"]["N_virtual_fermions_on_bond"])

χenv_max = 8
boundary_alg = (; tol = 1e-8, maxiter=500, alg = :simultaneous, trscheme = FixedSpaceTruncation())
Espace = Vect[FermionParity](0 => χenv_max / 2, 1 => χenv_max / 2)
env0 = CTMRGEnv(peps, oneunit(space(peps.A[1],2)))
env1, = leading_boundary(env0, peps; alg = :sequential, trscheme = truncspace(Espace), maxiter = 5);
env, = leading_boundary(env1, peps; boundary_alg...);

ham = GfPEPS.Kitaev_Hamiltonian(ComplexF64, InfiniteSquare(1, 1); Jx=params_Kitaev.Jx, Jy=params_Kitaev.Jy, Jz=params_Kitaev.Jz)
energy1 = real(expectation_value(peps, ham, env))

@test energy1 ≈ optim_energy atol=2e-2 # depends on Nv

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