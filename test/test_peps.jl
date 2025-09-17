using Revise
using Test
using GfPEPS
using JSON: parsefile
using SkewLinearAlgebra
using LinearAlgebra
using TensorKit
using PEPSKit

res = Gaussian_fPEPS(conf = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json")));

X_opt = res.X_opt
Γ_opt = GfPEPS.Γ_fiducial(X_opt, res.Nv, res.Nf)

B = Γ_opt[1:2*res.Nf, 2*res.Nf+1:end]
B2 = Γ_opt[2*res.Nf+1:end, 1:2*res.Nf]
B ≈ -transpose(B2)

res.Nf
N = (res.Nf + 4*res.Nv)

S = ComplexF64.(zeros(2N,2N))
for μ in 1:2N
    for v in 1:2N
        if iseven(μ)
            if v==μ/2
                S[μ,v] += 1
            end
            if v==μ/2+N
                S[μ,v] += 1
            end
        else
            if v==(μ+1)/2
                S[μ,v] += 1im
            end
            if v==(μ+1)/2+N
                S[μ,v] += -1im
            end
        end
    end
end
Sdag = 2*inv(S)

1/2 .* transpose(S) * conj(S) ≈ I

# 1/4 .* conj(S) * transpose(S) * transpose(S) * conj(S) ≈ I

# Γ_opt_dirac = real(1/4 .* conj(S) * transpose(S) * Γ_opt * transpose(S) * conj(S))

Γ_opt_dirac = 1/2 .* (transpose(S) * Γ_opt * conj(S)) 

# Test back transformation
R_conj = Γ_opt_dirac[1:N, 1:N]
Q_conj = Γ_opt_dirac[1:N, N+1:end]
Q = Γ_opt_dirac[N+1:end, 1:N]
R = Γ_opt_dirac[N+1:end, N+1:end]
@test R_conj ≈ conj(R)
@test Q_conj ≈ conj(Q)
@test R' ≈ -R
@test transpose(Q) ≈ -Q
@test Γ_opt_dirac'Γ_opt_dirac ≈ I

H = -im .* Γ_opt_dirac
H = Hermitian(H) 

E, W = eigen(H)
W'W ≈ I 
W*W' ≈ I
W'*H*W ≈ Diagonal(E)

U = W[1:N, 1:N]
V = W[N+1:end, 1:N]

V'V + U'U ≈ I
transpose(U) * V ≈ - transpose(V) * U 

inv(V)

Z = -V \ U
Z ≈ -transpose(Z)
Z .= 0.5 .* (Z .- transpose(Z))

N_norm = det(U)

Nv = res.Nv
Nf = res.Nf
# Output leg order: (f, l, u, r, d)
dims = (1 << Nf, 1 << Nv, 1 << Nv, 1 << Nv, 1 << Nv)
A = Array{ComplexF64}(undef, dims)

# Predefine ranges for each group of modes
r_f = 1:Nf
r_l = Nf .+ (1:Nv)
r_u = Nf + Nv .+ (1:Nv)
r_r = Nf + 2Nv .+ (1:Nv)
r_d = Nf + 3Nv .+ (1:Nv)

# Loop all configurations; amplitude is Pf(Z_S) for even |S|
@inbounds for  fmask in 0:(dims[1]-1), lmask in 0:(dims[2]-1), 
                umask in 0:(dims[3]-1), rmask in 0:(dims[4]-1), dmask in 0:(dims[5]-1)

    # Build full bit vector (little-endian within each group)
    bits = falses(N)

    for k in 1:Nf; bits[r_f[k]] = ((fmask >> (k-1)) & 0x1) == 1; end
    for k in 1:Nv; bits[r_u[k]] = ((umask >> (k-1)) & 0x1) == 1; end
    for k in 1:Nv; bits[r_l[k]] = ((lmask >> (k-1)) & 0x1) == 1; end
    for k in 1:Nv; bits[r_d[k]] = ((dmask >> (k-1)) & 0x1) == 1; end
    for k in 1:Nv; bits[r_r[k]] = ((rmask >> (k-1)) & 0x1) == 1; end

    occ = findall(identity, bits)
    if isodd(length(occ))
        amp = 0.0 + 0.0im
    elseif isempty(occ)
        # amp = N_norm
        amp = 1
    else
        ZS = @view Z[occ, occ]
        # amp = N_norm * pfaffian(Matrix(ZS))  # Pfaffian of antisymmetric submatrix
        amp = 1 * pfaffian(Matrix(ZS))  # Pfaffian of antisymmetric submatrix
    end

    # store with axes (f, l, u, r, d)
    A[ fmask+1, lmask+1, umask+1, rmask+1, dmask+1] = amp
end
A /= norm(A)

#= Convert to TensorKit =#
using TensorKit: FermionParity, Vect, ⊗, dual, TensorMap

# Helper: permutation that groups basis states by parity (even first, then odd)
# Returns (perm, n_even, n_odd) for n modes
function _parity_perm(n::Int)
    D = 1 << n
    if n == 0
        return collect(1:D), 1, 0
    end
    even = Vector{Int}(undef, D >> 1)
    odd  = Vector{Int}(undef, D >> 1)
    e = 1; o = 1
    @inbounds for m in 0:D-1
        if iseven(count_ones(m))
            even[e] = m + 1; e += 1
        else
            odd[o]  = m + 1; o += 1
        end
    end
    return vcat(even, odd), length(even), length(odd)
end

# Build parity-ordered index lists for each leg
perm_f, ne_f, no_f = _parity_perm(Nf)
perm_l, ne_l, no_l = _parity_perm(Nv)
perm_u, ne_u, no_u = _parity_perm(Nv)
perm_r, ne_r, no_r = _parity_perm(Nv)
perm_d, ne_d, no_d = _parity_perm(Nv)

# Reorder A along each axis so that within each leg: [even-block ; odd-block]
A_blk = @views A[perm_f, perm_l, perm_u, perm_r, perm_d]

# Construct FermionParity-graded spaces for each leg
fZ2 = FermionParity
f_space = (Nf == 0) ? Vect[fZ2](0 => 1) : Vect[fZ2](0 => ne_f, 1 => no_f)
# Use a single virtual bond space template and orient legs:
# left/up are dual, right/down are primal, so neighbors match
Vvirt = (Nv == 0) ? Vect[fZ2](0 => 1) : Vect[fZ2](0 => ne_l, 1 => no_l)  # ne_l == ne_u == ne_r == ne_d
l_space = dual(Vvirt)
u_space = dual(Vvirt)
r_space = Vvirt
d_space = Vvirt

# Make a fermionic TensorMap with convention: virtual legs in the domain, physical leg in the codomain
# Shape of A_blk is (f, l, u, r, d) so this matches (codomain ← domain)
A_tk = TensorMap(A_blk, f_space ← (l_space ⊗ u_space ⊗ r_space ⊗ d_space))
A_tk /= norm(A_tk)

peps = InfinitePEPS(A_tk; unitcell = (1, 1))
typeof(peps)

χs = [8, 16]
χmax = maximum(χs)
Espace = Vect[FermionParity](0 => χmax, 1 => χmax)
env = CTMRGEnv(randn, ComplexF64, peps, Espace)

peps₀ = InfinitePEPS(2, 2)
typeof(peps₀)
env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(8)), peps₀; tol=1e-10)

InfinitePEPS(2, 2)

env₀, = leading_boundary(CTMRGEnv(peps, ComplexSpace(16)), peps; tol=1e-10)

for χenv in χs
    trscheme = truncdim(χenv) & truncerr(1.0e-12)
    env, = leading_boundary(
        env, peps; tol = 1.0e-10, maxiter = 100, trscheme,
        alg = :sequential, projector_alg = :fullinfinite
    )
end

#= Test ipeps =#

function hamiltonian(
        T::Type{<:Number}, lattice::InfiniteSquare; t::Float64 = 1.0,
        Δx::Float64 = 0.5, Δy::Float64 = -0.5, mu::Float64 = 0.0
    )
    pspace = hub.hubbard_space(Trivial, Trivial)
    pspaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    num = hub.e_num(T, Trivial, Trivial)
    unit = TensorKit.id(T, pspace)
    hopping = (-t) * hub.e_hopping(T, Trivial, Trivial) -
        (mu / 4) * (num ⊗ unit + unit ⊗ num)
    pairing = sqrt(2) * hub.singlet_plus(T, Trivial, Trivial)
    pairing += pairing'
    return LocalOperator(
        pspaces,
        map(nearest_neighbours(lattice)) do bond
            return bond => hopping + pairing * (_is_xbond(bond) ? Δx : Δy)
        end...
    )
end
hamiltonian(lattice; t, Δx, Δy, mu) = hamiltonian(ComplexF64, lattice; t, Δx, Δy, mu)

t, Δx, Δy, mu = res.t, res.Δx, res.Δy, res.μ
ham = BCS.hamiltonian(ComplexF64, InfiniteSquare(1, 1); t, Δx, Δy, mu)
energy1 = expectation_value(peps, ham, env)

#=  =#


t1 = randn(ℂ^2 ⊗ ℂ^3, ℂ^2)

F = TensorKit.TensorKitSectors.FermionParity => Dict(0 => 1, 1 => 1)

GradedSpace(F)

f_space = GradedSpace(Nf; dual=true)

TensorKit.Sector.FermionParity

F = TensorKit.TensorKitSectors.FermionParity()

TensorKit.TensorKitSectors.FermionParity

phys_space = GradedSpace(FermionParity() => Dict(0 => 1, 1 => 1))

phys_space = GradedSpace(TensorKit.TensorKitSectors.FermionParity => Dict(0 => 1, 1 => 1))   # |0> even, |1> odd
virt_space = Index(fℤ₂, Dict(0 => 1, 1 => 1))   # 1 even, 1 odd state per bond

Vect[fℤ₂](0 => 1, 1 => 1)

TensorKit.FermionParity()

phys_d, left_d, up_d, right_d, down_d = size(A)
# create TensorKit indices (simple unlabeled indices with names)
p   = Index(phys_d, "p")
l   = Index(left_d, "l")
u   = Index(up_d, "u")
r   = Index(right_d, "r")
d_i = Index(down_d, "d")
