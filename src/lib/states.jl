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
for Nv pairs of virtual fermions `(a1_i, a2_i)` (i = 1, ..., Nv)
```
    |ω⟩ = ∏_{i=1}^Nv 2⁻½ (1 + a1†_i a2†_i) |0⟩
```
"""
function virtual_state(T::Type{<:Number}, Nv::Int)
    ff = FO.f_plus_f_plus(T)
    vac = vacuum_state(T, 2)
    # MES for one pair of (a1_i, a2_i) on the bond
    # the resulting fermion order is (a1_1, a2_1, ..., a1_χ, a2_χ)
    ω = (1 / sqrt(2)) * (unit ⊗ unit + ff) * vac
    if Nv > 1
        # reorder fermions to (a1_1, ..., a1_χ, a2_1, ..., a2_χ)
        ω = reduce(⊗, fill(ω, Nv))
        perm = Tuple(vcat(1:2:(2Nv), 2:2:(2Nv)))
        ω = TensorKit.permute(ω, (perm, ()))
    end
    return ω
end
virtual_state(Nv::Int) = virtual_state(ComplexF64, Nv)

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
(p_1, ..., p_{Nf}, l_1, r_1, ..., l_χ, r_χ, d_1, u_1, ..., d_χ, u_χ)

The output complex fermion order will be
(p_1, ..., p_{Nf}, l_1, ..., l_χ, r_1, ..., r_χ, d_1, ..., d_χ, u_1, ..., u_χ)
"""
function fiducial_state(T::Type{<:Number}, Nf::Int, Nv::Int, A::AbstractMatrix)
    ψ = paired_state(T, A)
    # reorder virtual fermions
    perm = vcat(1:2:(2χ), 2:2:(2χ))
    perm = Tuple(vcat(1:Nf, perm .+ Nf, perm .+ (Nf + 2χ)))
    ψ = permute(ψ, (perm, ()))
    return ψ
end
function fiducial_state(Nf::Int, Nv::Int, A::AbstractMatrix)
    return fiducial_state(ComplexF64, Nf, Nv, A)
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
    Nv = div(N1, 2)
    Nf = N2 - 4χ
    # merge physical and virtual axes
    fuser_p = isomorphism(Int, fuse(fill(V, Nf)...), reduce(⊗, fill(V, Nf)))
    fuser_v = isomorphism(Int, fuse(fill(V, Nv)...), reduce(⊗, fill(V, Nv)))
    ω = (fuser_v ⊗ fuser_v) * ω
    F = (fuser_p ⊗ reduce(⊗, fill(fuser_v, 4))) * F
    @tensor A[-1; -2 -3 -4 -5] := conj(ω[1 -2]) * conj(ω[2 -3]) * F[-1 -5 2 -4 1]
    return InfinitePEPS(A; unitcell = (1, 1))
end

"""
Translate the orthogonal matrix `X` produced by Gaussian-fPEPS
to Gaussian fPEPS in PEPSKit format.
"""
function translate(X::AbstractMatrix, Nf::Int, Nv::Int)
    N = Nf + 4Nv
    @assert size(X, 1) == 2N && X' * X ≈ I
    G = Γ_fiducial(X, Nv, Nf)

    @assert G ≈ -transpose(G) && G * G ≈ -I

    H = get_parent_hamiltonian(G)
    E, M = bogoliubov(H)
    detM = det(M)
    if !(det(M) ≈ 1.0)
        @assert det(M) ≈ -1
        error("det(M) = -1; fiducial state has odd parity.")
    end
    U = M[1:Nf+4Nv, 1:Nf+4Nv]
    V = M[1:Nf+4Nv, Nf+4Nv+1:end]

    ω = virtual_state(Nv)
    F = fiducial_state(Nf, Nv, -inv(U) * V)
    peps = get_peps(ω, F)
    return peps
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

"""
Generate a random correlation matrix `G` of 
a pure Gaussian state with even parity
"""
function generate_cormat(Nf::Int, Nv::Int)
    N = Nf + 4Nv
    while true
        X = rand_orth(2N)
        G = Γ_fiducial(X, Nv, Nf)
        H = get_parent_hamiltonian(G)
        E, W = bogoliubov(H)
        if det(W) ≈ 1
            return X, G, H, E, W
        end
    end
    return
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