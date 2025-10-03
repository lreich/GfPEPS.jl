using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using LinearAlgebra
using BlockDiagonals
using Random 
using MatrixFactorizations

Random.seed!(32178046)

Nf = 2
Nv = 2
N = (Nf + 4*Nv)

Γ_fiducial, X = GfPEPS.rand_CM(Nf,Nv)
H = GfPEPS.get_parent_hamiltonian(Γ_fiducial, Nf, Nv)
_, M = GfPEPS.bogoliubov(H)

Dmat, UVmat, Cmat = GfPEPS.bloch_messiah_decomposition(M)
