using TensorKit
using PEPSKit
using GfPEPS
using JLD2
using JSON: parsefile
using Test

config = parsefile(joinpath(@__DIR__,"../conf/conf_kitaev.json"))
filename = config["file"]["name"]

Nf = config["params"]["N_physical_fermions_on_site"]
Nv = config["params"]["N_virtual_fermions_on_bond"]

bz = GfPEPS.BrillouinZone2D(24, 24, (:PBC, :PBC))

Jx = 0.25
Jy = 0.25
Jz = 0.5

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

config["hamiltonian"]["Jx"] = Jx
config["hamiltonian"]["Jy"] = Jy
config["hamiltonian"]["Jz"] = Jz

X_opt, optim_energy, exact_energy, info = GfPEPS.get_X_opt(;conf=config)

Γ_opt = GfPEPS.Γ_fiducial(X_opt, Nv, Nf)

Efunc2(k) = abs(GfPEPS.energy_CM_k(Γ_opt, k, Nf, params_Kitaev))
@show minimum(Efunc2.(kpairs));

@test minimum(Efunc.(kpairs)) ≈ minimum(Efunc2.(kpairs)) atol=1e-5

# etest = GfPEPS.energy_CM(Γ_opt, bz, Nf, params_Kitaev)

# @show etest
# @show optim_energy
# @show exact_energy

# # for k in eachcol(bz.kvals)
# #     ε_k = GfPEPS.energy_CM_k(Γ_opt, k, Nf, params_Kitaev)

# #     @show ε_k
# # end