using TensorKit
using PEPSKit
using GfPEPS
using JLD2
using JSON: parsefile

# conf = parsefile(joinpath(@__DIR__,"../../conf/conf_compute_EE.json"))
config = parsefile(joinpath(@__DIR__,"../conf/conf_kitaev.json"))
filename = config["file"]["name"]

Nf = config["params"]["N_physical_fermions_on_site"]
Nv = config["params"]["N_virtual_fermions_on_bond"]

# function gapless_bz(Nx::Int, Ny::Int; bc::NTuple{2,Symbol}=(:PBC, :PBC))
#     Nx_dirac = ((Nx + 2) ÷ 3) * 3
#     Ny_dirac = ((Ny + 2) ÷ 3) * 3
#     return GfPEPS.BrillouinZone2D(Nx_dirac, Ny_dirac, bc)
# end

# bz = gapless_bz(100, 100; bc = (:PBC, :PBC))

bz = GfPEPS.BrillouinZone2D(1200, 1200, (:PBC, :PBC))

# config["hamiltonian"]["Jx"] = 0
# config["hamiltonian"]["Jy"] = 0.25
# config["hamiltonian"]["Jz"] = 0.5

# Jx = 1.0
# Jy = 1.0
# Jz = 1.0

Jx = 1.0
Jy = 0.8
Jz = 1.0

function satisfy_triangle(Jx,Jy,Jz)
    res = true

    if abs(Jx) > abs(Jy) + abs(Jz)
        res = false
    end
    if abs(Jy) > abs(Jx) + abs(Jz)
        res = false
    end
    if abs(Jz) > abs(Jx) + abs(Jy)
        res = false
    end
    return res
end

@show satisfy_triangle(Jx,Jy,Jz)

kpairs = map(eachcol(bz.kvals)) do k return k end



params_Kitaev = GfPEPS.Kitaev(Jx, Jy, Jz)

Efunc(k) = GfPEPS.E(k, params_Kitaev)
@show minimum(Efunc.(kpairs));



# Evals2 = []
# for k in eachcol(bz.kvals)
#     push!(Evals2, GfPEPS.E(k, params_Kitaev))
# end

# display(sort(abs.(Evals2)))

# X_opt, optim_energy, exact_energy = GfPEPS.get_X_opt(;conf=config)
# Γ_opt = GfPEPS.Γ_fiducial(X_opt, Nv, Nf)

# etest = GfPEPS.energy_CM(Γ_opt, bz, Nf, params_Kitaev)

# @show etest
# @show optim_energy
# @show exact_energy

# # for k in eachcol(bz.kvals)
# #     ε_k = GfPEPS.energy_CM_k(Γ_opt, k, Nf, params_Kitaev)

# #     @show ε_k
# # end