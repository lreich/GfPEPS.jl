"""
get_bogoliubov_blocks(M::AbstractMatrix)
Extract the U,V blocks from the Bogoliubov transformation matrix `M = [U conj(V); V conj(U)]`.
"""
function get_bogoliubov_blocks(M::AbstractMatrix)
    N = div(size(M, 1), 2)
    U = M[1:N, 1:N]
    V = M[N+1:end, 1:N]
    return U, V
end

"""
Bogoliubov transformation of a fermionic bilinear Hamiltonian `H`. Returns 
- The (positive) energy spectrum `E`, in descending order;
- The transformation `M = [U conj(V); V conj(U)]` (such that `M' * H * M = diagm(vcat(E, -E))`);
"""
function bogoliubov(H::Hermitian)
    N = div(size(H, 1), 2)
    E, M0 = eigen(H; sortby = (x -> -real(x)))

    U = M0[1:N, 1:N]
    V = M0[N+1:end, 1:N]

    # bring to correct form
    M = similar(M0)
    M[1:N, 1:N] = U
    M[N+1:end, 1:N] = V
    M[1:N, N+1:end] = conj.(V)
    M[N+1:end, N+1:end] = conj.(U)

    # check canonical constraints
    @assert M' * M ≈ I
    @assert U'U + V'V ≈ I
    @assert transpose(U) * V ≈ - transpose(V) * U
    
    return E, M
end

"""
    skew_canonical_form(P::AbstractMatrix)

Returns S,X where: the transformation S, such that `transpose(S)*P*S` is in canonical form X (See: https://doi.org/10.1007/BF02906230)
"""
function skew_canonical_form(P::AbstractMatrix)
    # Check skew-symmetry
    @assert transpose(P) ≈ -P

    W = P'P
    @assert ishermitian(W)

    E, Φ = eigen(Hermitian(W); sortby = (x -> -real(x)))
    alphas = sqrt.(abs.(E))

    # build orthogonal eigenvectors
    Φ_prime = similar(Φ)
    for j in eachindex(alphas)
        Φ_prime[:, j] = (P'*conj.(Φ[:, j])) / alphas[j]
    end

    # build S
    S = similar(P)
    n_zeros = 0
    for j in eachindex(alphas)
        if alphas[j] ≈ 0.0
            S[:, end-n_zeros] = Φ[:, j]
            n_zeros += 1
        else
            if isodd(j)
                S[:, j] = Φ[:, j]
            else
                S[:, j] = Φ_prime[:, j-1]
            end
        end
    end

    X = S'*P*conj(S)

    # permutation to have positive elements in the upper-right of each 2x2 block
    perm_mat = canonical_skew_permutation(X)
    X = perm_mat' * X * perm_mat
    S = S * perm_mat

    # absorb phases into S to have X is real
    S, X = absorb_phases(S, X)

    X[abs.(X) .< 1e-12] .= 0.0
    return S,X
end

function absorb_phases(S::AbstractMatrix, X::AbstractMatrix)
    S2 = copy(S)
    X2 = copy(X)

    n = size(X2,1)
    i = 1
    while i <= n-1
        x = X2[i, i+1]
        y = X2[i+1, i]
        # If already real with nonnegative value, skip
        if !(abs(imag(x)) ≈ 0 && real(x) >= 0)
            φ = angle(x)
            d = exp(1im * φ/2)          # uniform phase for the pair
            @views S2[:, i  ] .*= d
            @views S2[:, i+1] .*= d
            # Block transforms by conj(d)^2
            # After this, value becomes real ≈ |x|
            X2[i, i+1] = abs(x)
            X2[i+1, i] = -abs(x)
        end
        i += 2
    end
    @assert isapprox(imag(X2), zeros(ComplexF64, n,n), atol=1e-12) "X2 should be real after phase absorption"

    return S2, real(X2)
end

# function skew_canonical_form(P::AbstractMatrix)
#     # Check skew-symmetry
#     @assert transpose(P) ≈ -P

#     W = P'P
#     @assert ishermitian(W)

#     n = size(W,1)
#     p = div(rank(W),2)
#     E, Φ = eigen(Hermitian(W); sortby = (x -> -real(x)))
#     alphas = sqrt.(abs.(E))

#     S = similar(P)

#     # For each twofold eigenspace:
#     #  - pick one eigenvector v,
#     #  - form partner w = P' * conj(v) / α and project it into the eigenspace,
#     #  - orthonormalize the two vectors inside the eigenspace (thin QR),
#     #  - enforce the sign convention so block = [0 +α; -α 0].
#     for μ in 1:p
#         Φb = Φ[:, 2μ-1:2μ]                    # basis of the 2D eigenspace
#         v = Φb[:, 1]                          # pick representative
#         w = (P' * conj(v)) / alphas[2μ-1]     # partner (α>0)
#         w_proj = Φb * (Φb' * w)               # project partner into same eigenspace

#         # orthonormalize inside the eigenspace
#         Q = qr(hcat(v, w_proj))
#         S[:, 2μ-1:2μ] = Matrix(Q.Q)[:, 1:2]

#         # ensure the off-diagonal (1,2) is positive: flip second column if needed
#         blk = real(transpose(S[:, 2μ-1:2μ]) * P * S[:, 2μ-1:2μ])
#         if blk[1,2] < 0
#             S[:, 2μ] .*= -1
#         end
#     end
#     # append any remaining zero-modes
#     if 2p < n
#         S[:, 2p+1:n] = Φ[:, 2p+1:n]
#     end
#     @assert S' * S ≈ I

#     X = real.(transpose(S) * P * S)
#     X[abs.(X) .< 1e-12] .= 0.0 # remove near zero entries for better readability

#     return S, X
# end


# Build a permutation matrix S_perm so that (S_perm' * P_bar1 * S_perm)
# has 2×2 skew blocks with Re(upper-right) ≥ 0 (i.e. "positive above, negative below")
function canonical_skew_permutation(P::AbstractMatrix)
    n = size(P,1)
    perm = collect(1:n)
    i = 1
    while i < n
        a = P[perm[i], perm[i+1]]
        b = P[perm[i+1], perm[i]]
        # Detect a 2×2 skew block (nonzero pair with b ≈ -a)
        if abs(a) > 0 && isapprox(b, -a; atol=1e-14, rtol=1e-12)
            # If real part of upper-right entry is < 0, swap the two indices
            if real(a) < 0
                perm[i], perm[i+1] = perm[i+1], perm[i]
            end
            i += 2
        else
            i += 1
        end
    end
    S = Matrix{eltype(P)}(I, n, n)
    return S[:, perm]
end

function bloch_messiah_decomposition(M::AbstractMatrix)
    N = div(size(M, 1), 2)

    U,V = GfPEPS.get_bogoliubov_blocks(M)

    Q = conj.(V) * transpose(V)
    @assert Q' ≈ Q
    P = conj.(V) * transpose(U)
    @assert transpose(P) ≈ -P
    @assert Q*P ≈ P*conj.(Q)

    _, B = eigen(Q; sortby = (x -> -real(x)))
    Q_bar = real(B'*Q*B)
    P_bar = B'*P*conj.(B)
    @assert P_bar ≈ - transpose(P_bar)

    # Bring P_bar to canonical form
    S, _ = skew_canonical_form(P_bar)
    P_canonical = S' * P_bar * conj.(S)

    A = permute_zero_cols_to_end(P_canonical)
    D = B * S * A

    @assert D'*P*conj(D) ≈ D'*conj(V)*transpose(U)*conj(D)

    F = MatrixFactorizations.rq(D' * U)
    R = Matrix(F.R)
    Q = Matrix(F.Q)

    # Fix phases so diagonal of R becomes positive real.
    d = diag(R)
    ph = similar(d)
    for i in eachindex(d)
        ph[i] = (abs(d[i]) > 0) ? d[i]/abs(d[i]) : one(d[i])   # unit-modulus (or 1 if zero)
    end
    Φ  = Diagonal(conj.(ph))            # multiply R on right by Φ to remove phases
    Rpos = R * Φ                        # now diagonal(Rpos) = abs.(d) ≥ 0 (real)
    Qnew = Φ' * Q                       # keep A invariant: (R Φ)(Φ' Q) = R Q
    Ubar = Rpos                         # Ū with positive diagonal
    C    = Qnew
    Vbar = transpose(D) * V * C'

    @assert D'*conj(V)*transpose(U)*conj(D) ≈ D'*conj(V)*transpose(C)*conj(C)*transpose(U)*conj(D)
    @assert D'*conj(V)*transpose(Q)*conj(Q)*transpose(U)*conj(D) ≈ D'*conj(V)*transpose(Q)*R

    @assert C'C ≈ I
    @assert Q'Q ≈ I

    @assert U ≈ D*Ubar*C
    @assert V ≈ conj.(D)*Vbar*C

    # remove numerical noise
    Ubar = real(Ubar)
    Ubar[abs.(Ubar) .< 1e-12] .= 0.0
    Vbar = real(Vbar)
    Vbar[abs.(Vbar) .< 1e-12] .= 0.0

    Dmat = [D zeros(N,N); zeros(N,N) conj.(D)]
    UV_mat = [Ubar Vbar; Vbar Ubar]
    Cmat = [C zeros(N,N); zeros(N,N) conj.(C)]

    @assert M ≈ Dmat * UV_mat * Cmat

    return Dmat, UV_mat, Cmat
end

function permute_zero_cols_to_end(P::AbstractMatrix)
    n = size(P,1)
    perm = collect(1:n)
    i = 1
    j = n
    while i < j
        if all(iszero, P[:, perm[i]])
            perm[i], perm[j] = perm[j], perm[i]
            j -= 1
        else
            i += 1
        end
    end
    A = Matrix{eltype(P)}(I, n, n)
    return A[:, perm]
end

function canonical_form2(P::AbstractMatrix)
    @assert P ≈ -transpose(P) "P must be skew-symmetric"

    # Step 1: Eigen decomposition of P†P
    F = P' * P
    λ, V = eigen(F)  # λ real and ≥ 0

    n = size(P, 1)
    cols = Vector{Vector{ComplexF64}}()
    s = Float64[]

    for j in 1:n
        λj = λ[j]
        vj = V[:, j]

        if λj > 1e-12  # nonzero eigenvalue
            # Step 2: Construct wj
            wj = (P' * conj(vj)) / sqrt(abs(λj))

            # Step 3: Add vj, wj as a pair
            push!(cols, vj)
            push!(cols, wj)

            push!(s, sqrt(abs(λj)))
        else
            # Step 4: Zero eigenvalue eigenvector
            push!(cols, vj)
        end
    end

    # Step 5: Build S from columns
    S = hcat(cols...)

    # Canonical form
    P̄ = S' * P * conj(S)

    return S, P̄, s
end