"""
    get_parent_hamiltonian(Γ_out::AbstractMatrix, Nf::Int, Nv::Int)

Given the output correlation matrix Γ_out, return the parent Hamiltonian in Dirac fermions.
"""
function get_parent_hamiltonian(Γ_out::AbstractMatrix)
    @assert eltype(Γ_out) <: Real && Γ_out ≈ -transpose(Γ_out)
    N = div(size(Γ_out, 1), 2)

    # convert from majorana basis to complex fermion basis
    Ω_single = [1 1;
                im -im]
    Ω = ⊕(Ω_single, N)

    # resulting fermion order is (f_1, f†_1, ..., f_N, f†_N)
    H = -0.5im .* Ω' * Γ_out * Ω

    # put annihilation in front of creation operators
    # (f_1, ..., f_N, f†_1, ..., f†_N)
    perm = vcat(1:2:(2N), 2:2:(2N))
    return Hermitian(H[perm, perm])
end

# This needs to be tested!
function corr_matrix_to_dirac(Γ_out::AbstractMatrix)
    @assert eltype(Γ_out) <: Real && Γ_out ≈ -transpose(Γ_out)
    N = div(size(Γ_out, 1), 2)

    # convert from majorana basis to complex fermion basis
    Ω_single = [1 1;
                im -im]
    Ω = ⊕(Ω_single, N)

    # resulting fermion order is (f_1, f†_1, ..., f_N, f†_N)
   return Ω' * Γ_out * Ω
end

function fiducial_hamiltonian(h_rho::AbstractMatrix, h_kappa::AbstractMatrix)
    @inline bits2Int(bk::AbstractVector{<:Integer}) = sum((bk[i] & 0x1) << (i-1) for i in eachindex(bk))

    N = size(h_rho, 1)

    H = zeros(ComplexF64, 2^N, 2^N)

    for i in 1:N
        for j in 1:N
            for k in 0:(2^N - 1)
                bk = digits(k, base = 2, pad = N) # bitstring that represents the occupation of each fermionic mode
                # parity (even/odd) depending on the occupation of modes between i and j
                parity = (sum(@view bk[(i+1):N]) + sum(@view bk[(j+1):N])) % 2
                ind = k + 1 # column index in H (1-based)
                if bk[j]==1 # hopping part
                    if bk[i] == 0 || i==j
                        bk[j] = 0
                        bk[i] = 1
                        target = bits2Int(bk) + 1
                        parity += Int(j > i)
                        H[target,ind] += 2 * h_rho[i,j] * (-1)^parity
                    elseif bk[i]==1 # pair annihilation part
                        bk[j] = 0
                        bk[i] = 0
                        target = bits2Int(bk) + 1
                        parity += Int(i > j)
                        H[target,ind] += h_kappa[i,j] * (-1)^parity
                    end
                elseif bk[j]==0 # pair creation part
                    if bk[i]==0
                        bk[j] = 1
                        bk[i] = 1
                        target = bits2Int(bk) + 1
                        parity += Int(i > j)
                        H[target,ind] -= 2 * conj(h_kappa[i,j]) * (-1)^parity
                    end
                end
            end
        end
    end

    return (H + H')/2
end

function parity_gate(Nv)
    n = 2^Nv
    S = diagm(ones(n))
    for i in 0:(n-1)
        if isodd(count_ones(UInt(i)))
            S[i+1,i+1] = -1.0
        end
    end
    return S
end

function fsign(n_list)
    result = 0
    for i in 2:length(n_list)
        result += n_list[i] * sum(@view n_list[1:(i-1)])
    end
    return (-1)^(result % 2)
end

function bondgate(Nv)
    p = zeros(2^Nv)
    for i in 0:(2^Nv - 1)
        n_list = reverse(digits(i, base=2, pad=Nv))
        p[i+1] = fsign(n_list)
    end
    return diagm(p)
end

function add_gates(tensor, Nv)
    Gb = bondgate(Nv)      # size: 2^Nv × 2^Nv
    Gp = parity_gate(Nv)   # size: 2^Nv × 2^Nv

    # 1) tensor = einsum("ulfdr,iu->ilfdr", tensor, Gb)   # apply on u (1st index)
    @tensor t1[i,l,f,d,r] := Gb[i,u] * tensor[u,l,f,d,r]

    # 2) tensor = einsum("ulfdr,il->uifdr", t1, Gb)       # apply on l (2nd index)
    @tensor t2[u,i,f,d,r] := t1[u,l,f,d,r] * Gb[i,l]

    # 3) tensor = einsum("ulfdr,iu->ilfdr", t2, Gp)       # apply on u (1st index)
    @tensor t3[i,l,f,d,r] := Gp[i,u] * t2[u,l,f,d,r]

    return t3
end


"""
Bogoliubov transformation of a fermionic bilinear Hamiltonian `H`. Returns 
- The (positive) energy spectrum `E`, in descending order;
- The transformation `W = [U V; V̄ Ū]` (such that `W * H * W' = diagm(vcat(E, -E))`);
"""
function bogoliubov(H::Hermitian)
    N = size(H, 1)
    E, W0 = eigen(H; sortby = (x -> -real(x)))
    n = div(N, 2)
    # construct the transformation W
    Wpos = W0[:, 1:n]
    U = Wpos[1:n, :]
    V = conj(Wpos[(n + 1):end, :])
    W = similar(W0)
    W[1:n, 1:n] = U
    W[1:n, (n + 1):(2n)] = V
    W[(n + 1):(2n), 1:n] = conj.(V)
    W[(n + 1):(2n), (n + 1):(2n)] = conj.(U)
    # check canonical constraint
    @assert W' * W ≈ I
    # check positiveness of energy
    @assert all(E[1:n] .> 0)
    return E[1:n], Matrix(W')
end

"""
Generate a random real orthogonal matrix
"""
function rand_orth(n::Int; special::Bool = false)
    M = randn(Float64, (n, n))
    F = qr(M)
    Q = Matrix(F.Q)
    R = F.R
    # absorb signs of diag(R) into Q
    λ = diag(R) ./ abs.(diag(R))
    Q .= Q .* λ'
    if special
        # ensure det(Q)=+1
        if det(Q) < 0
            Q[:, 1] .*= -1
        end
    end
    return Q
end

function fiducial_cormat(X::AbstractMatrix)
    @assert eltype(X) <: Real
    n2, m2 = size(X)
    @assert n2 == m2 && iseven(n2)
    U = @view X[1:2:end, :]
    V = @view X[2:2:end, :]
    return transpose(U) * V .- transpose(V) * U
end

"""
Generate a random correlation matrix `G` of 
a pure Gaussian state with even parity
"""
function generate_cormat(Np::Int, χ::Int)
    N = Np + 4χ
    while true
        X = rand_orth(2N)
        G = fiducial_cormat(X)
        H = parent_Hamiltonian_BdG(G)
        E, W = bogoliubov(H)
        if det(W) ≈ 1
            return X, G, H, E, W
        end
    end
    return
end

# function translate_to_PEPS(X::AbstractMatrix, Nf::Int, Nv::Int)
#     Γ_out = Γ_fiducial(X, Nv, Nf)

#     H = get_parent_hamiltonian(Γ_out, Nf, Nv)

#     @show size(H)

#     _, M = eigen(H)
#     # @assert det(M) ≈ -1.0 "Fiducial state has odd parity. Odd parity is currently not supported."

#     U = M[1:Nf+4Nv, 1:Nf+4Nv]
#     V = M[1:Nf+4Nv, Nf+4Nv+1:end]


#     @show size(U)
#     @show size(V)

#     -inv(U) * V


# end

"""
Construct the direct sum of [1 1; im -im] for `dup` times.
"""
function get_W2(dup::Int)
    w2 = sparse([1.0 1.0; 1.0im -1.0im])

    return blockdiag((w2 for _ in 1:dup)...)
end

function parent_Hamiltonian_BdG(G::AbstractMatrix)
    @assert eltype(G) <: Real && G ≈ -transpose(G)
    N = div(size(G, 1), 2)
    # change to complex fermion basis
    # c_{2j-1} = f_j + f†_j, c_{2i} = i(f_j - f†_j)
    # resulting fermion order is (f_1, f†_1, ..., f_N, f†_N)
    W = get_W2(N)
    H = W' * (-0.5im * G) * W
    # put annihilation in front of creation operators
    # (f_1, ..., f_N, f†_1, ..., f†_N)
    perm = vcat(1:2:(2N), 2:2:(2N))
    return Hermitian(H[perm, perm])
end

"""
Create the vacuum state for `n` spinless fermions
"""
function vacuum_state(T::Type{<:Number}, n::Int)
    vac = zeros(T, FO.fermion_space())
    vac.data[1] = 1.0
    return (n > 1) ? reduce(⊗, fill(vac, n)) : vac
end
vacuum_state(n::Int) = vacuum_state(ComplexF64, n)

"""
Construct the maximally entangled state (MES) on virtual bonds
for χ pairs of virtual fermions `(a1_i, a2_i)` (i = 1, ..., χ)
```
    |ω⟩ = ∏_{i=1}^χ 2⁻½ (1 + a1†_i a2†_i) |0⟩
```
"""
function virtual_state(T::Type{<:Number}, χ::Int)
    ff = FO.f_plus_f_plus(T)
    vac = vacuum_state(T, 2)
    # MES for one pair of (a1_i, a2_i) on the bond
    # the resulting fermion order is (a1_1, a2_1, ..., a1_χ, a2_χ)
    ω = (1 / sqrt(2)) * (unit ⊗ unit + ff) * vac
    if χ > 1
        # reorder fermions to (a1_1, ..., a1_χ, a2_1, ..., a2_χ)
        ω = reduce(⊗, fill(ω, χ))
        perm = Tuple(vcat(1:2:(2χ), 2:2:(2χ)))
        ω = TensorKit.permute(ω, (perm, ()))
    end
    return ω
end
virtual_state(χ::Int) = virtual_state(ComplexF64, χ)

"""
Construct the fully paired state `exp(a† A a† / 2)`, 
where A is an anti-symmetric matrix.
"""
function paired_state(T::Type{<:Number}, A::AbstractMatrix)
    N = size(A, 1)
    @assert A ≈ -transpose(A)
    ff = FO.f_plus_f_plus(T)
    ψ = vacuum_state(T, N)
    # apply exp(A_{ij} a†_i a†_j) (i < j)
    for i in 1:(N - 1)
        for j in (i + 1):N
            op = exp(A[i, j] * ff)
            idx_op = [-i, -j, i, j]
            idx_ψ = collect(-1:-1:(-N))
            idx_ψ[i], idx_ψ[j] = i, j
            ψ = ncon([op, ψ], [idx_op, idx_ψ])
        end
    end
    return ψ
end
paired_state(A) = paired_state(ComplexF64, A)

"""
Construct the local tensor of the fiducial state
`exp(a† A a† / 2)`, where A is an anti-symmetric matrix.

Input complex fermion order in `a` should be
(p_1, ..., p_{Np}, l_1, r_1, ..., l_χ, r_χ, d_1, u_1, ..., d_χ, u_χ)

The output complex fermion order will be
(p_1, ..., p_{Np}, l_1, ..., l_χ, r_1, ..., r_χ, d_1, ..., d_χ, u_1, ..., u_χ)
"""
function fiducial_state(T::Type{<:Number}, Np::Int, χ::Int, A::AbstractMatrix)
    ψ = paired_state(T, A)
    # reorder virtual fermions
    perm = vcat(1:2:(2χ), 2:2:(2χ))
    perm = Tuple(vcat(1:Np, perm .+ Np, perm .+ (Np + 2χ)))
    ψ = TensorKit.permute(ψ, (perm, ()))
    return ψ
end
function fiducial_state(Np::Int, χ::Int, A::AbstractMatrix)
    return fiducial_state(ComplexF64, Np, χ, A)
end

"""
Get PEPS tensor by contracting virtual axes of ⟨ω|F⟩,
where |ω⟩, |F⟩ are the virtual and the fiducial states.
```
            -2
            ↓
            ω
            ↑
            1  -1
            ↑ ↗
    -5  --←-F-→- 2 -→-ω-←- -3
            ↓
            -4
```
Input axis order
```
        5  1                2
        ↑ ↗                 ↑
    2-←-F-→-3   1-←-ω-→-2   ω
        ↓                   ↓
        4                   1
```
"""
function get_peps(ω::AbstractTensor{T, S, N1}, F::AbstractTensor{T, S, N2}) where {T, S, N1, N2}
    χ = div(N1, 2)
    Np = N2 - 4χ
    # merge physical and virtual axes
    fuser_p = isomorphism(Int, fuse(fill(V, Np)...), reduce(⊗, fill(V, Np)))
    fuser_v = isomorphism(Int, fuse(fill(V, χ)...), reduce(⊗, fill(V, χ)))
    ω = (fuser_v ⊗ fuser_v) * ω
    F = (fuser_p ⊗ reduce(⊗, fill(fuser_v, 4))) * F
    @tensor A[-1; -2 -3 -4 -5] := conj(ω[1 -2]) * conj(ω[2 -3]) * F[-1 -5 2 -4 1]
    return InfinitePEPS(A; unitcell = (1, 1))
end

"""
Extract the blocks A, B of bilinear fermion Hamiltonian H
`H = [A B; -B̄ -Ā]`, where `A = A'` and `Bᵀ = -B`; 
or U, V of the Bogoliubov transformation `W = [U V; V̄ Ū]`
"""
function bogoliubov_blocks(H::AbstractMatrix)
    N = div(size(H, 1), 2)
    return H[1:N, 1:N], H[1:N, (N + 1):end]
end

"""
Check if a 2-site bond is a nearest neighbor x-bond
"""
function _is_xbond(bond)
    return bond[2] - bond[1] == CartesianIndex(0, 1)
end

"""
Get the blocks of (Np + 4*χ)-dimensional real correlation matrix 
```
    G = [A B; -Bᵀ D]
```
"""
function cormat_blocks(G::AbstractMatrix, Np::Int = 2)
    @assert eltype(G) <: Real && G ≈ -transpose(G)
    return G[1:(2 * Np), 1:(2 * Np)],
        G[1:(2 * Np), (2 * Np + 1):end],
        G[(2 * Np + 1):end, (2 * Np + 1):end]
end

"""
Fourier components of the virtual state correlation matrix,
with χ species of complex virtual fermions along each direction.

The virtual Majorana fermions are ordered as
(l_1, r_1, ..., l_χ, r_χ, d_1, u_1, ..., d_χ, u_χ)
"""
function cormat_virtual(k::Vector{Float64}, χ::Int)
    expx1, expy1 = cispi(2 * k[1]), cispi(2 * k[2])
    expx2, expy2 = -1 / expx1, -1 / expy1
    xmat = [0 0 0 expx2; 0 0 expx2 0; 0 expx1 0 0; expx1 0 0 0]
    ymat = [0 0 0 expy2; 0 0 expy2 0; 0 expy1 0 0; expy1 0 0 0]
    return direct_sum([(n <= χ ? xmat : ymat) for n in 1:2χ]...)
end

"""
Momenta in the D-dimensional 1st Brillouin zone (under reciprocal lattice basis)
```
    kᵢ = (mᵢ + δ)/Nᵢ ∈ (−1/2, 1/2] 
```
with the shift δ=0 if `pbc==true` (periodic), and δ=1/2 if `pbc==false` (anti-periodic).
"""
struct BrillouinZone{D}
    # number of sites in each dimension
    Ns::NTuple{D, Int}
    # boundary condition in each dimension (true: PBC; false: anti-PBC)
    pbcs::NTuple{D, Bool}
    # all momenta in 1st Brillouin zone
    ks::Array{Vector{Float64}, D}
end

"""
Create the 1st Brillouin zone, where `Ns` and `pbcs` specify 
the number of sites and boundary condition along each dimension.
"""
function BrillouinZone(Ns::NTuple{D, Int}, pbcs::NTuple{D, Bool}) where {D}
    ranges = Vector{UnitRange{Int}}(undef, D)
    for (i, (N, pbc)) in enumerate(zip(Ns, pbcs))
        if pbc
            if iseven(N)
                mn = -(N ÷ 2) + 1
                mx = N ÷ 2
            else
                mn = -((N - 1) ÷ 2)
                mx = (N - 1) ÷ 2
            end
        else # anti-pbc
            if iseven(N)
                mn = -(N ÷ 2)
                mx = (N ÷ 2) - 1
            else
                # for odd N the range is same as periodic
                mn = -((N - 1) ÷ 2)
                mx = (N - 1) ÷ 2
            end
        end
        ranges[i] = mn:mx
    end
    # take the Cartesian product of all ranges
    shifts = collect(pbc ? 0.0 : 0.5 for pbc in pbcs)
    ks = map(Iterators.product(ranges...)) do ms
        collect(ms .+ shifts) ./ Ns
    end
    return BrillouinZone{D}(Ns, pbcs, ks)
end
function BrillouinZone(Ns::NTuple{D, Int}, pbcs::Bool) where {D}
    return BrillouinZone(Ns, ntuple(_ -> pbcs, D))
end

Base.size(bz::BrillouinZone) = bz.Ns