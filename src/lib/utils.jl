"""
    ⊕(A::AbstractMatrix, n::Integer)

repeat A ⊕ A ⊕ ... (n times) via kron

"""
function ⊕(A::AbstractMatrix, n::Integer)
    @assert n >= 1

    Id = Matrix{eltype(A)}(I, n, n)
    return kron(Id, A)
end

"""
Generate a random real orthogonal matrix via QR decomposition.
"""
function rand_orth(n::Int)
    M = randn(Float64, (n, n))
    F = qr(M)
    Q = Matrix(F.Q)
    R = F.R
    # absorb signs of diag(R) into Q
    λ = diag(R) ./ abs.(diag(R))
    Q .= Q .* λ'
    return Q
end

"""
Generate a random covariance matrix `Γ` of 
a pure Gaussian state with even parity in Majorana representation.
"""
function rand_CM(Nf::Int, Nv::Int)
    N = Nf + 4Nv
    while true
        X = rand_orth(2N)
        Γ = Γ_fiducial(X, Nv, Nf)

        @show pfaffian(im .* Γ)
        if pfaffian(im .* Γ) ≈ 1 # for pure BCS state with even parity Pf(iΓ) = +1
            return Γ, X
        end
    end
end

# """
#     takagi(M)

# Implements Autonne-Takagi decomposition of a symmetric matrix

# ### Input
# - `M`  -- Array: Symmetric matrix
# ### Output

# Array: ``2N \\times 2N`` random symplectic matrix.
# """
# function takagi(M)
#     u, s, v = svd(M)
#     pref = u' * conj(v)
#     pref12 = normal_sqrtm(pref)
#     return s, u * pref12
# end

# function normal_sqrtm(A)
#     T, Z = schur(A)
#     return Z * sqrt(T) * transpose(conj(Z))
# end

# function bloch_messiah(S)
#     u, d, v = svd(S)
#     P = v * diagm(d) * v'
#     O = u * v'
#     n, m = size(P)
#     ell = div(n, 2)
#     A = P[1:ell, 1:ell]
#     B = P[ell+1:2*ell, 1:ell]
#     C = P[ell+1:2*ell, ell+1:2*ell]
#     M = A - C + im * (B + B')
#     Lam, W = takagi(M)
#     Lam = 0.5 * Lam
#     OO = [real(W) -imag(W); imag(W) real(W)]
#     sqrt1pLam2 = sqrt.(Lam .^ 2 .+ 1)
#     D = vcat(Lam + sqrt1pLam2, -Lam + sqrt1pLam2)
#     lO = O * OO
#     rO = OO'
#     return lO, D, rO
# end

# """
#     bloch_messiah(M; tol=1e-10)

# Bloch–Messiah decomposition of a fermionic Bogoliubov transformation

#     M = [ U  conj(V)
#           V  conj(U) ]

# Returns a named tuple with:
# - D, C :: unitary matrices
# - Ubar, Vbar :: canonical real blocks
#       Ubar = ⊕_p u_p σ⁰,  Vbar = ⊕_p i v_p σʸ  with u_p ≥ 0, v_p ≥ 0, u_p^2 + v_p^2 = 1
# - u, v :: vectors of (u_p, v_p) per pair (length N/2), and the used permutation `perm`.

# The factors satisfy:
# - U = D * Ubar * C
# - V = conj.(D) * Vbar * C
# - M ≈ [U conj(V); V conj(U)]
# """
# function bloch_messiah(M::AbstractMatrix{T}; tol::Real = 1e-10) where {T<:Complex}
#     N2 = size(M,1)
#     @assert N2 == size(M,2) "M must be square"
#     @assert iseven(N2) "M must be 2N × 2N"
#     N = N2 ÷ 2

#     # Extract U,V blocks
#     U, V = get_bogoliubov_blocks(M)

#     # SVD of U gives an initial D, C, and singular values u (nonnegative)
#     sv = svd(U)
#     D0 = sv.U                       # left unitary
#     uvals = sv.S                    # singular values of U
#     C0 = sv.Vt                      # right 'unitary' (adjoint), unitary nonetheless
#     Σ = Diagonal(uvals)

#     # Vbar in the (D0, C0) basis; note C† is adjoint(C), and Dᵀ is transpose(D)
#     Vbar0 = transpose(D0) * V * adjoint(C0)

#     # Determine a permutation that pairs modes; prefer equal u's; fallback to strongest coupling in Vbar0
#     unused = collect(1:N)
#     pairs = Vector{Tuple{Int,Int}}()
#     while !isempty(unused)
#         i = popfirst!(unused)
#         # try pair by u degeneracy
#         j_idx = findfirst(j -> abs(uvals[j] - uvals[i]) ≤ max(tol, 10*eps(real(T))), unused)
#         if j_idx === nothing
#             # fallback: pair by largest |Vbar0[i, j]|
#             if isempty(unused)
#                 # last one: should be a v≈0 (u≈1) leftover
#                 push!(pairs, (i, i))
#                 break
#             end
#             magnitudes = abs.(Vbar0[i, unused])
#             j_idx = argmax(magnitudes)
#         end
#         j = unused[j_idx]
#         deleteat!(unused, j_idx)
#         push!(pairs, (i, j))
#     end

#     # Build permutation vector: [i1,j1,i2,j2,...] (skip any i==j left-overs safely)
#     perm = Int[]
#     for (i,j) in pairs
#         if i == j
#             push!(perm, i)
#         else
#             push!(perm, i, j)
#         end
#     end
#     # Ensure perm is length N (if an odd leftover happened, it will have length N-1; add it)
#     if length(perm) < N
#         # add any missing index
#         for k in 1:N
#             if !(k in perm)
#                 push!(perm, k)
#             end
#         end
#     end
#     P = Matrix{T}(I, N, N)[:, perm]

#     # Reorder both sides consistently
#     D1 = D0 * P                 # left basis
#     C1 = transpose(P) * C0      # right basis
#     Σ1 = transpose(P) * Σ * P   # reordered diagonal
#     Vbar1 = transpose(P) * Vbar0 * P

#     # Phase alignment within each pair to make blocks exactly i*v*σ^y
#     # We construct S = diag(s_k) and apply D ← conj(S) * D1, C ← S * C1,
#     # which transforms Vbar: Vbar ← S† * Vbar1 * S† and leaves Ubar real.
#     s = ones(Complex{eltype(real(T))}, N)
#     # Iterate adjacent pairs (2p-1, 2p)
#     p = 1
#     while p ≤ N
#         if p == N
#             # unpaired leftover (only if N is odd): no change
#             break
#         end
#         i, j = p, p+1
#         z = Vbar1[i, j]
#         if abs(z) ≤ tol
#             # nothing to do, keep default phases
#             p += 2
#             continue
#         end
#         φ = angle(z)
#         # Choose s_i s_j = i e^{i φ} ⇒ pick s_i = e^{i φ/2}, s_j = -i e^{i φ/2}
#         s[i] = cis(φ/2)                 # e^{i φ/2}
#         s[j] = -1im * cis(φ/2)          # -i e^{i φ/2}
#         p += 2
#     end
#     S = Diagonal(s)

#     D = conj.(S) * D1
#     C = S * C1
#     Ubar = Σ1                        # stays real-diagonal (paired order)

#     # Update Vbar under the congruence by S† on both sides
#     Vbar = adjoint(S) * Vbar1 * adjoint(S)

#     # Build canonical block forms Ū = ⊕ u σ⁰ and V̄ = ⊕ i v σʸ
#     # We also extract (u_p, v_p)
#     Ubar_blk = zero(Ubar)
#     Vbar_blk = zero(Vbar)
#     up = Float64[]
#     vp = Float64[]

#     p = 1
#     while p ≤ N
#         if p == N
#             # leftover mode: u≈1, v≈0
#             u = real(Ubar[p,p])
#             push!(up, u)
#             push!(vp, 0.0)
#             Ubar_blk[p,p] = u
#             p += 1
#             continue
#         end
#         i, j = p, p+1
#         uij = (real(Ubar[i,i]) + real(Ubar[j,j]))/2
#         vij = abs(Vbar[i,j])
#         # Fill 2x2 blocks
#         Ubar_blk[i,i] = uij
#         Ubar_blk[j,j] = uij
#         Vbar_blk[i,j] = 1im * vij
#         Vbar_blk[j,i] = -1im * vij

#         push!(up, uij)
#         push!(vp, vij)
#         p += 2
#     end

#     # Optional: small cleanups
#     # Zero tiny entries
#     Ubar_blk = map(x -> abs(x) < tol ? zero(x) : x, Ubar_blk)
#     Vbar_blk = map(x -> abs(x) < tol ? zero(x) : x, Vbar_blk)

#     return (D = D, C = C, Ubar = Ubar_blk, Vbar = Vbar_blk, u = up, v = vp, perm = perm)
# end

# """
#     reconstruct_bogoliubov_from_bm(D, C, Ubar, Vbar)

# Given Bloch–Messiah factors, reconstruct U, V and the full 2N×2N matrix
# M = [U conj(V); V conj(U)].
# """
# function reconstruct_bogoliubov_from_bm(D::AbstractMatrix, C::AbstractMatrix,
#                                         Ubar::AbstractMatrix, Vbar::AbstractMatrix)
#     U = D * Ubar * C
#     V = conj.(D) * Vbar * C
#     M = [U                conj.(V);
#          V                conj.(U)]
#     return U, V, M
# end

# polar decomposition (returns unitary factor U_p such that X = U_p * H)
# using SVD: X = L * Σ * R^† -> U_p = L * R^†
function polar_unitary(X; atol=1e-12)
# handle near-zero matrix
F = svd(X)
U = F.U * F.Vt
return U
end

# Takagi (Autonne) factorization for a complex *skew-symmetric* or
# complex symmetric matrix M. We compute Q and S such that M = Q * S * Q^T,
# where S is nonnegative diagonal/block diag (Takagi singular values).
# This implementation follows the standard numerical recipe:
# - form Hermitian H = M * M^†
# - eigen-decompose H = W * Λ * W^† (Λ >= 0)
# - let Σ = sqrt(Λ)
# - build Q = M * W * inv(Σ) for nonzero singular values
# For zero singular values, extend Q by choosing orthonormal completion.
function takagi(M; tol = 1e-12)
    # M : complex matrix (N×N)
    N = size(M,1)
    @assert size(M,1) == size(M,2) "M must be square"

    # Hermitian positive semidefinite
    H = M * M'
    evals, W = eigen(H)
    # eigen returns real evals sorted ascending; make nonnegative clamp
    evals = real.(clamp.(evals, 0.0, Inf))

    # Sigma = sqrt(evals)
    Sigma = sqrt.(evals)


    # Build Q columns for nonzero singular values
    Q = zeros(ComplexF64, N, N)
    nz = findall(x -> x > tol, Sigma)
    zc = findall(x -> x <= tol, Sigma)


    # For nonzero singular values: q_j = (1/sigma_j) M * w_j
    for (j, idx) in enumerate(nz)
        sigma = Sigma[idx]
        w = W[:, idx]
        q = (M * w) / sigma
        Q[:, j] = q
    end

    # For zero singular values: we need to choose orthonormal completion
    # collect existing columns and complete using QR on a random basis orthogonal to span
    k = length(nz)
    if k < N
        # Build orthonormal basis from current Q[:,1:k]
        if k > 0
            Qk = Q[:, 1:k]
            # project random vectors and orthonormalize
            # use QR on nullspace-like construction
            Y = randn(ComplexF64, N, N - k)
            # orthogonalize Y w.r.t. Qk
            for j in 1:(N - k)
                v = Y[:, j]
                v -= Qk * (Qk' * v) # remove components
                Y[:, j] = v
            end
            # QR to orthonormalize remaining columns
            # Qrest, R = qr(Y) |> Tuple
            # # pick first N-k columns
            # for j in 1:(N - k)
            #     Q[:, k + j] = Qrest[:, j]
            # end
            F = qr(Y)
            Qrest = Matrix(F.Q)
            R = F.R
            # pick first N-k columns
            for j in 1:(N - k)
                Q[:, k + j] = Qrest[:, j]
            end
        else
            # no nonzero columns: choose any unitary
            Q = Matrix(qr(randn(ComplexF64, N, N))).Q
        end
    end

    # Now reorder Q so columns correspond to original eigenorder: we put nz first
    # Build permutation that places nz indices first in column order
    perm = vcat(nz, zc)
    Qfull = similar(Q)
    for j=1:N
        Qfull[:, j] = Q[:, j]
    end

    # Recompute Sigma matrix in the same column ordering
    Sigma_ord = Sigma[perm]

    # Finally build Takagi S diag matrix
    S = Diagonal(Sigma_ord)

    # We must ensure Q satisfies M = Q * S * Q^T up to numerical error.
    # If necessary, we can adjust phases of Q: enforce Q^T * M * Q = S
    # Compute B = Q' * M * Q (note Q' = conj(Q)') -> this should be symmetric
    B = Qfull' * M * Qfull
    # make diagonal phases real positive by absorbing phases into Q columns
    for j in 1:N
        phase = B[j,j]
        if abs(phase) > tol
            ph = phase / abs(phase)
            Qfull[:, j] *= conj(ph)^(1/2) # distribute half-phase to make diagonal real
        end
    end


    # recompute S to be the symmetric matrix Q^T M Q
    Smat = Qfull' * M * Qfull
    # force symmetry and take absolute(diagonal) as singulars
    Smat = (Smat + Smat')/2
    sval = abs.(diag(Smat))
    Sdiag = Diagonal(sval)

    return Qfull, Sdiag
end

# Build canonical Ũ, 𝚅̃ (tildeU, tildeV) from u and v arrays (length N, arranged in pairs)
# We'll produce N×N matrices where each pair (2×2) has structure as in the paper.
function build_tilde_uv(uvec, vvec)
    N = length(uvec)
    Ũ = zeros(ComplexF64, N, N)
    𝚅̃ = zeros(ComplexF64, N, N)

    # We assume modes are already paired (1,2), (3,4), ...; if N odd, last mode single
    i = 1
    while i <= N
        if i < N
            u = uvec[i]
            v = vvec[i]
            # Put 2x2 block in positions i,i+1
            Ũ[i,i+1] = u; Ũ[i+1,i] = u
            𝚅̃[i,i+1] = v; 𝚅̃[i+1,i] = -v
            i += 2
        else
            # leftover single mode -> put 1×1 block (u=1, v=0) by default
            Ũ[i,i] = 1.0
            𝚅̃[i,i] = 0.0
            i += 1
        end
    end
    return Ũ, 𝚅̃
end

# Main decomposition function
# Input: U,V (N×N) Bogoliubov matrices
# Output: C, D (as diagonal vector or matrix), tildeU, tildeV, and reconstruction checks
function fermionic_bloch_messiah(U::AbstractMatrix{<:Complex}, V::AbstractMatrix{<:Complex}; tol=1e-8)
    N = size(U,1)
    @assert size(U,2) == N && size(V)==(N,N) "U and V must be square N×N"

    # Form skew-symmetric M = U^T * V
    M = transpose(U) * V

    # Takagi: M = Q * S * Q^T, with S diagonal nonnegative
    Q, S = takagi(M)
    svals = diag(S)

    # From s = u*v and u^2 + v^2 = 1, solve for u and v (take u>=v>=0)
    uvec = zeros(Float64, N)
    vvec = zeros(Float64, N)
    for j in 1:N
        s = real(svals[j])
        # clamp s to [0, 0.5] realistically but numerics might exceed slightly
        if s < 0
            s = 0.0
        end
        # Solve u^2 + v^2 = 1, u*v = s  -> quadratic for u^2: (u^2)(1 - (s^2)/(u^2)) ???
        # Analytical solution: let t = u^2, then v^2 = 1 - t and u^2 v^2 = s^2 -> t (1-t) = s^2
        # -> t^2 - t + s^2 = 0 -> t = (1 ± sqrt(1 - 4 s^2)) / 2 . Choose + to make u >= v.
        disc = 1 - 4*s^2
        if disc < 0
            disc = 0.0
        end
        t = 0.5*(1 + sqrt(disc))
        u = sqrt(max(t, 0.0))
        v = s / max(u, eps())
        uvec[j] = u
        vvec[j] = v
    end

    # Build canonical tilde U and V in the pairing ordering of Q's columns
    tildeU, tildeV = build_tilde_uv(uvec, vvec)

    # We now want unitary C and diagonal D such that U ≈ C * D * tildeU
    # Let X = U * (tildeU)^(-1) if tildeU invertible (blockwise) else solve least squares
    # We'll compute X = U * pinv(D*tildeU) but unknown D — instead compute polar of U * pinv(tildeU)

    # Compute pseudo-inverse of tildeU
    pinv_tildeU = pinv(tildeU)

    X = U * pinv_tildeU
    C = polar_unitary(X)   # unitary factor

    # Absorb residual into D: D = C^† * U * pinv(tildeU)
    Dmat = C' * X

    # Force Dmat into block-diagonal / diagonal by zeroing off-diagonals within pairs
    # For simplicity, we take D to be the diagonal of Dmat (phases / amplitudes)
    Ddiag = Diagonal(diag(Dmat))

    # Final reconstruction
    W = [U  conj(V); V  conj(U)]
    Left = kron(C, [1])  # placeholder: we'll assemble full left = blockdiag(C, C*)
    # Assemble product: (C 0;0 C*) * (D 0;0 D*) * (tildeU  tildeV*; tildeV  tildeU*)
    bigC = [C zeros(ComplexF64,N,N); zeros(ComplexF64,N,N) conj(C)]
    bigD = [Ddiag zeros(ComplexF64,N,N); zeros(ComplexF64,N,N) conj(Ddiag)]
    bigT = [tildeU conj(tildeV); tildeV conj(tildeU)]

    # Recon = bigC * bigD * bigT
    Recon = bigD * bigT * bigC

    return (C=C, D=Ddiag, Q=Q, S=S, tildeU=tildeU, tildeV=tildeV, Recon=Recon, W=W)
end