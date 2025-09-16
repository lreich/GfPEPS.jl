using Revise
using Test
using GfPEPS
using JSON: parsefile

res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json")));

@testset "Energy BCS" begin
    @testset "d_wave" begin
    res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json")))
    @test res.optim_res ≈ res.exact_energy
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

# @time Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json")));

# bz = BrillouinZone2D(4,4,(:APBC,:PBC))
# GfPEPS.G_in_Fourier(bz,2)