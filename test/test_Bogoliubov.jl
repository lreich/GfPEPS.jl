using Revise
using Test
using GfPEPS
using LinearAlgebra
using SkewLinearAlgebra
using JSON: parsefile

Nf = 2
Nv = 2
N = (Nf + 4*Nv)

Γ = GfPEPS.rand_CM(Nf, Nv)
H = GfPEPS.get_parent_hamiltonian(Γ)
M = GfPEPS.bogoliubov(H)

@testset "Bogoliubov transformation" begin
    U,V = GfPEPS.get_bogoliubov_blocks(M)
    V_conj = M[1:N, N+1:end]
    U_conj = M[N+1:end, N+1:end]

    @test U_conj ≈ conj(U)
    @test V_conj ≈ conj(V)
    @test U'U + V'V ≈ I
    @test transpose(U) * V ≈ - transpose(V) * U 
end;


U

function get_Ubar(U::AbstractMatrix)
    return conj(U)

end

C, D, Q, S, tildeU, tildeV, Recon, W = GfPEPS.fermionic_bloch_messiah(U,V)

tildeU

Recon ≈ W

@testset "Bloch-Messiah transformation" begin

end;
