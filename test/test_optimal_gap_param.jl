using Revise
using TensorKit
using PEPSKit
using GfPEPS
using JLD2
using JSON: parsefile
using Test

t = 1
J = 0.4
δ_target = 0.16

L = 101
bz = BrillouinZone2D(L, L, (:APBC, :APBC))
Δ_opt = GfPEPS.find_optimal_Δ(t, J, δ_target, bz; pairing_type="d_wave")