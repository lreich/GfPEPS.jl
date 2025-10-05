using Revise
using Test
using GfPEPS
using LinearAlgebra
using SkewLinearAlgebra
using JSON: parsefile

Nf = 2
Nv = 2
N = (Nf + 4*Nv)

# all qq ordering dirac (f_1,f_1†, ..., f_Nf, f_Nf†, l_1, l_1†, r_1, r_1†, ..., l_Nv, l_Nv†, r_Nv, r_Nv†, d_1, d_1†, u_1, u_1†, ..., d_Nv, d_Nv†, u_Nv, u_Nv†)
Γ = Matrix{String}(undef, N, N)
# e.g. Γ[1,1] = "f_1f_1", Γ[1,2] = "f_1f_1†", Γ[2,1] = "f_1†f_1", etc.

twoN = 2 * N
modes = String[]

# physical f modes: f_1, f_1†, ..., f_Nf, f_Nf†
for k in 1:Nf
    push!(modes, "f_$k")
    push!(modes, "f_$(k)†")
end

# virtual left/right for p=1..Nv: l_p, l_p†, r_p, r_p† (repeated)
for p in 1:Nv
    push!(modes, "l_$p"); push!(modes, "l_$(p)†")
    push!(modes, "r_$p"); push!(modes, "r_$(p)†")
end

# virtual down/up for p=1..Nv: d_p, d_p†, u_p, u_p† (repeated)
for p in 1:Nv
    push!(modes, "d_$p"); push!(modes, "d_$(p)†")
    push!(modes, "u_$p"); push!(modes, "u_$(p)†")
end

Γ = Matrix{String}(undef, twoN, twoN)
for i in 1:twoN, j in 1:twoN
    Γ[i,j] = string(modes[i], modes[j])   # e.g. "f_1f_1†"
end

# bring to qp-ordering
perm = vcat(1:2:(2N), 2:2:(2N))
Γ_qp = Γ[perm, perm]
#= Now has the following ordering (qp)
    (f_1, ..., f_Nf, l_1, r_1, ..., l_Nv, r_Nv, d_1, u_1, ..., d_Nv, u_Nv, f_1†, ..., f_Nf†, l_1†, r_1†, ..., l_Nv†, r_Nv†, d_1†, u_1†, ..., d_Nv†, u_Nv†)
=#

# group virtual fermions as (l1,...,lNv,r1,...,rNv,d1,...,dNv,u1,...,uNv)
L = collect(1:2:2Nv)    # l1, l2, ...
R = collect(2:2:2Nv)    # r1, r2, ...
D = collect(2Nv+1:2:4Nv)  # d1, d2, ...
U = collect(2Nv+2:2:4Nv)  # u1, u2, ...
perm_virtual = vcat(L, R, D, U)

perm_total = vcat(
    1:Nf,                       # physical (already fine)
    Nf .+ perm_virtual,         # reorder virtuals
    (Nf+4Nv) .+ (1:Nf),         # f†
    (2Nf+4Nv) .+ perm_virtual    # reordered virtual†
)
Γ_group1 = Γ_qp[perm_total, perm_total]

# now reorder to (f,u,r,d,l)
L = collect(Nf+1:Nf+Nv)    # l1, l2, ...
R = collect(Nf+Nv+1:Nf+2Nv)   # r1, r2, ...
D = collect(Nf+2Nv+1:Nf+3Nv)  # d1, d2, ...
U = collect(Nf+3Nv+1:Nf+4Nv)  # u1, u2, ...
perm_virtual = vcat(U, R, D, L)

perm_reorder = vcat(1:Nf, 
    perm_virtual,
    (Nf+4Nv) .+ (1:Nf), # f†
    (Nf+4Nv) .+ perm_virtual # virtual†
)
Γ_group2 = Γ_group1[perm_reorder, perm_reorder]