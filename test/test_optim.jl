using Revise
using Test
using GfPEPS

@testset "optimize_loss" begin
    res = Gaussian_fPEPS()
    @test res.optim_res ≈ res.exact_energy
end;
