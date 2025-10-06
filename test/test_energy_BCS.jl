using Revise
using Test
using GfPEPS
using JSON: parsefile

@time X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt();

@testset "Energy BCS" begin
    @testset "d_wave" begin
        config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json"))
        X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)
        # test energy from CM
        @test optim_energy ≈ exact_energy
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