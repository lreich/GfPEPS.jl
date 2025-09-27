using Revise
using Test
using GfPEPS
using JSON: parsefile

@testset "Energy BCS" begin
    @testset "d_wave" begin
        config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json"))
        X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)
        # test energy from CM
        @test optim_energy ≈ exact_energy

        # peps = GfPEPS.translate(X_opt, config["params"]["N_physical_fermions_on_site"], config["params"]["N_virtual_fermions_on_bond"])

        # # test energy from PEPS
        # # peps = res.peps
        # Espace = Vect[FermionParity](0 => 4, 1 => 4)
        # env = CTMRGEnv(randn, ComplexF64, peps, Espace)
        # for χenv in [8, 16, 32]
        #     trscheme = truncdim(χenv) & truncerr(1.0e-12)
        #     env, = leading_boundary(
        #         env, peps; tol = 1.0e-11, maxiter = 200, trscheme,
        #         alg = :sequential, projector_alg = :fullinfinite
        #     )
        # end

        # ham = GfPEPS.hamiltonian(ComplexF64, InfiniteSquare(1, 1); 
        #     t= config["hamiltonian"]["t"], 
        #     Δx = config["hamiltonian"]["Δ_options"]["Δ_x"], 
        #     Δy = -config["hamiltonian"]["Δ_options"]["Δ_y"], 
        #     mu = config["hamiltonian"]["μ"])

        # bz = BrillouinZone2D(121, 121, (:APBC, :PBC))
        # energy_exact = GfPEPS.exact_energy_BCS_k(
        #     bz, config["hamiltonian"]["t"], 
        #     config["hamiltonian"]["μ"], 
        #     "default", 
        #     config["hamiltonian"]["Δ_options"]["Δ_x"], 
        #     -config["hamiltonian"]["Δ_options"]["Δ_y"]
        # )

        # @test real(expectation_value(peps, ham, env)) ≈ energy_exact
    end;

    @testset "p_wave" begin
        config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_p_wave.json"))
        X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)
        # test energy from CM
        @test optim_energy ≈ exact_energy
    end;

    @testset "s_wave" begin
        config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_s_wave.json"))
        X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)
        # test energy from CM
        @test optim_energy ≈ exact_energy
    end;
end;