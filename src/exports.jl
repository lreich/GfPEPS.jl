#= Brillouin zone functions =#
export BrillouinZone2D

#= Hamiltonian functions =#
export get_2D_k_grid
export exact_energy_BCS_k
export exact_energy_BCS_k_average

#= Gaussian map functions =#
export GaussianMap

#= loss functions =#
export optimize_loss

#= export constructors =#
export Gaussian_fPEPS

#= export global variables =#
const root = normpath(joinpath(@__DIR__, ".."))
const config_path = joinpath(root, "conf")
export root
export config_path