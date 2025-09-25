using Revise
using Test
using GfPEPS
using JSON: parsefile

@time res = Gaussian_fPEPS();


@testset "Energy BCS" begin
    @testset "d_wave" begin
        res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json")))
        @test res.optim_res ≈ res.exact_energy

        peps = res.peps
        Espace = Vect[FermionParity](0 => 4, 1 => 4)
        env = CTMRGEnv(randn, ComplexF64, peps, Espace)
        # env = CTMRGEnv(randn, ComplexF64, peps)
        for χenv in [8, 16]
            trscheme = truncdim(χenv) & truncerr(1.0e-12)
            env, = leading_boundary(
                env, peps; tol = 1.0e-10, maxiter = 200, trscheme,
                alg = :sequential, projector_alg = :fullinfinite
            )
        end

    end;

    @testset "p_wave" begin
        res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_p_wave.json")))
        @test res.optim_res ≈ res.exact_energy
    end;

    @testset "s_wave" begin
        res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_s_wave.json")))
        @test res.optim_res ≈ res.exact_energy
    end;
end;