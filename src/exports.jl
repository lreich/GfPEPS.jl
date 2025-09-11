#= Brillouin zone functions =#
export BrillouinZone2D

#= Hamiltonian functions =#
export exact_energy_BCS_k

#= Gaussian map functions =#
export GaussianMap

#= loss functions =#
export optimize_loss

#= export constructors =#
export Gaussian_fPEPS

#= export global variables =#
const root = normpath(joinpath(@__DIR__, ".."))
const config_path = joinpath(root, "conf")
const test_config_path = joinpath(root, "conf", "test_conf")
export root
export config_path
export test_config_path