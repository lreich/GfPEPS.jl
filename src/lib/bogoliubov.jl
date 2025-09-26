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

    n = size(W,1)
    p = div(rank(W),2)
    E, Φ = eigen(W; sortby = (x -> -real(x)))
    alphas = sqrt.(E)

    # build S
    S = similar(Φ)

    # For each twofold eigenspace:
    #  - pick one eigenvector v,
    #  - form partner w = P' * conj(v) / α and project it into the eigenspace,
    #  - orthonormalize the two vectors inside the eigenspace (thin QR),
    #  - enforce the sign convention so block = [0 +α; -α 0].
    for μ in 1:p
        Φb = Φ[:, 2μ-1:2μ]                    # basis of the 2D eigenspace
        v = Φb[:, 1]                          # pick representative
        w = (P' * conj(v)) / alphas[2μ-1]     # partner (α>0)
        w_proj = Φb * (Φb' * w)               # project partner into same eigenspace

        # orthonormalize inside the eigenspace
        Q = qr(hcat(v, w_proj))
        S[:, 2μ-1:2μ] = Matrix(Q.Q)[:, 1:2]

        # ensure the off-diagonal (1,2) is positive: flip second column if needed
        blk = real(transpose(S[:, 2μ-1:2μ]) * P * S[:, 2μ-1:2μ])
        if blk[1,2] < 0
            S[:, 2μ] .*= -1
        end
    end
    # append any remaining zero-modes
    if 2p < n
        S[:, 2p+1:n] = Φ[:, 2p+1:n]
    end
    @assert S' * S ≈ I

    X = real.(transpose(S) * P * S)
    X[abs.(X) .< 1e-12] .= 0.0 # remove near zero entries for better readability

    return S, X
end


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
    P_bar1 = B'*P*conj.(B)

    S, P_bar = skew_canonical_form(P_bar1)
    D = B*conj.(S)

    # Enforce sign convention: (k,k+1) element of D' * P * conj(D) positive
    P_can = D' * P * conj(D)
    for k in 1:2:N-1
        if real(P_can[k, k+1]) < 0
            D[:, k+1] .*= -1                # flip second vector in the pair
            P_can[k, k+1] = -P_can[k, k+1]  # update cached block (optional)
            P_can[k+1, k] = -P_can[k+1, k]
        end
    end

    A = D' * U
    F = MatrixFactorizations.rq(A)
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

    Ubar = real(Rpos)                   # Ū with positive diagonal
    C    = Qnew
    Vbar = real(transpose(D) * V * C')

    @assert U ≈ D*Ubar*C
    @assert V ≈ conj.(D)*Vbar*C

    Dmat = [D zeros(N,N); zeros(N,N) conj.(D)]
    UV_mat = [Ubar Vbar; Vbar Ubar]
    Cmat = [C zeros(N,N); zeros(N,N) conj.(C)]

    @assert M ≈ Dmat * UV_mat * Cmat

    return Dmat, UV_mat, Cmat
end