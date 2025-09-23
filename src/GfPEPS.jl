module GfPEPS

#= load external modules =#
using MKL
using LinearAlgebra
using BlockDiagonals
# using ITensors, ITensorMPS
using Optim
using Zygote
# using Enzyme
using JSON: parsefile
using Random
using TensorOperations
using SkewLinearAlgebra

using SparseArrays: sparse, blockdiag, spdiagm
using TensorKit
using PEPSKit
import TensorKitTensors.HubbardOperators as hub
import TensorKitTensors.FermionOperators as FO

const V = FO.fermion_space()
const unit = TensorKit.id(V)

MKL.set_num_threads(Sys.CPU_THREADS) 

#= include local files =#
include("lib/utils.jl")
include("lib/brillouinZone.jl")
include("lib/hamiltonian.jl")
include("lib/GaussianMap.jl")
include("lib/constructor.jl")
include("lib/loss.jl")
include("lib/states.jl")
include("lib/translate.jl")
include("lib/bogoliubov.jl")
include("models/bcs.jl")

include("exports.jl") # export functions

end
