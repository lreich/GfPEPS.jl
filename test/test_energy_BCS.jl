using Revise
using Test
using GfPEPS
using JSON: parsefile
using TensorKit
using PEPSKit

@time res = Gaussian_fPEPS();


@testset "Energy BCS" begin
    @testset "d_wave" begin
        config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json"))
        res = Gaussian_fPEPS(conf = config)

        # test energy from CM
        @test res.optim_energy ≈ res.exact_energy

        # test energy from PEPS
        peps = res.peps
        Espace = Vect[FermionParity](0 => 4, 1 => 4)
        env = CTMRGEnv(randn, ComplexF64, peps, Espace)
        for χenv in [8, 16]
            trscheme = truncdim(χenv) & truncerr(1.0e-12)
            env, = leading_boundary(
                env, peps; tol = 1.0e-10, maxiter = 200, trscheme,
                alg = :sequential, projector_alg = :fullinfinite
            )
        end

        ham = GfPEPS.hamiltonian(ComplexF64, InfiniteSquare(1, 1); 
            t= config["hamiltonian"]["t"], 
            Δx = config["hamiltonian"]["Δ_options"]["Δ_x"], 
            Δy = -config["hamiltonian"]["Δ_options"]["Δ_y"], 
            mu = config["hamiltonian"]["μ"])

        @test real(expectation_value(peps, ham, env)) ≈ res.optim_energy
    end;

    # @testset "p_wave" begin
    #     res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_p_wave.json")))
    #     @test res.optim_res ≈ res.exact_energy
    # end;

    # @testset "s_wave" begin
    #     res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_s_wave.json")))
    #     @test res.optim_res ≈ res.exact_energy
    # end;
end;