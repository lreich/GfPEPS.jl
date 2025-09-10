using Revise
using Test
using GfPEPS

# @testset "optimize_loss" begin
    res = Gaussian_fPEPS()
    @test res.optim_res ≈ res.exact_energy
# end

# bz = BrillouinZone2D(4,4,(:APBC,:PBC))

# G_in_Fourier(bz, 2)