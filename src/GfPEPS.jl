module GfPEPS

#= load external modules =#
using LinearAlgebra
using BlockDiagonals
using ITensors, ITensorMPS
using SparseArrays
using Optim
using Zygote
using JSON: parsefile
using Random
# using F_utilities

#= include local files =#
include("lib/helperFunctions.jl")
include("lib/brillouinZone.jl")
include("lib/hamiltonian.jl")
include("lib/GaussianMap.jl")
include("lib/constructor.jl")
include("lib/loss.jl")

include("exports.jl") # export functions

end
