using Revise
using Test
using GfPEPS
using JSON: parsefile
using LinearAlgebra

obj = Gaussian_fPEPS();

X_opt = obj.X_opt
Γ_opt_majoranas = -GfPEPS.Γ_fiducial(X_opt, obj.Nv, obj.Nf)

# bz = obj.bz

# ε_arr = Array{Float64,1}(undef, size(bz.kvals,2))
# for (i,k) in enumerate(eachcol(bz.kvals))
#     ε_arr[i] = -GfPEPS.E(k, obj.t, obj.μ, obj.Δ_options["pairing_type"], obj.Δ_options["Δ_x"], obj.Δ_options["Δ_y"])
# end
# ε_arr = sort(ε_arr)

# J = GfPEPS.build_J(0, size(bz.kvals,2))


# convert from majorana basis to complex fermion basis
Ω = [I(4*obj.Nv + obj.Nf) I(4*obj.Nv + obj.Nf);
     im*I(4*obj.Nv + obj.Nf) -im*I(4*obj.Nv + obj.Nf)]

Γ_opt_fermions = -0.5im .* Ω' * Γ_opt_majoranas * Ω 

E, V = eigen(Γ_opt_fermions)

E

V