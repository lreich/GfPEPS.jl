using Test
using ITensors, ITensorMPS
using GfPEPS

Lx = 5
Ly = 5
bc = (:APBC,:PBC)
BZ = BrillouinZone2D(Lx,Ly,bc)

function BCS_H_MPO(Lx,Ly,t,μ,Δ_x,Δ_y; bc=(:APBC,:PBC))

    

    N = Lx*Ly
    os = opSum()
    for x in 1:Lx
        for y in 1:Ly

        end
    end
end

@testset "BCS Hamiltonian energy" begin



end