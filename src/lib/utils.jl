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

        if pfaffian(Γ) ≈ 1 # for pure BCS state with even parity Pf(iΓ) = +1
            return Γ, X
        end
    end
end