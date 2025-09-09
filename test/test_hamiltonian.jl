using Test
using GfPEPS

Lx = 4
Ly = 4
bc = (:APBC,:PBC)
BZ = BrillouinZone2D(Lx,Ly,bc)

@testset "BCS Hamiltonian energy" begin

end