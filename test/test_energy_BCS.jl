using Revise
using Test
using GfPEPS
using JSON: parsefile

@time res = Gaussian_fPEPS();


bz = GfPEPS.BrillouinZone2D(2,2,(:APBC,:PBC))
t  = 1
μ  = 1
Δ_x = 1
Δ_y = 1
pairing_type = "d_wave"
Nf = 2
Nv = 2

loss = GfPEPS.optimize_loss(t, μ, bz, Nf, Nv, pairing_type, Δ_x, Δ_y)
Γ, X = GfPEPS.rand_CM(Nf,Nv)

loss(X)



bz = GfPEPS.BrillouinZone2D(2,2,(:APBC,:PBC))

map(i -> begin
    return i
end, eachindex(eachcol(bz.kvals)))


GfPEPS.G_in_Fourier(bz,5)

@testset "Energy BCS" begin
    # @testset "d_wave" begin
    # res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json")))
    # @test res.optim_res ≈ res.exact_energy
    # end;

    # @testset "p_wave" begin
    #     res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_p_wave.json")))
    #     @test res.optim_res ≈ res.exact_energy
    # end;

    # @testset "s_wave" begin
    #     res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_s_wave.json")))
    #     @test res.optim_res ≈ res.exact_energy
    # end;
end;