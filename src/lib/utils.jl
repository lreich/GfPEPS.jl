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
    rand_orth(n::Int)

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
    rand_CM(Nf::Int, Nv::Int; parity::Int = 1)

Generate a random covariance matrix `Γ` of a pure Gaussian state with given parity in Majorana representation.

Returns `Γ`, `X` where `X` is the orthogonal matrix used to construct `Γ`.
"""
function rand_CM(Nf::Int, Nv::Int; parity::Int = 1)
    N = Nf + 4Nv
    info = parity == 1 ? "even" : "odd"

    while true
        X = rand_orth(2N)
        Γ = Γ_fiducial(X, Nv, Nf)

        if pfaffian(Γ) ≈ parity # for pure BCS state, parity = Pf(Γ) = +1 (even) / -1 (odd)
            @info "Created initial covariance matrix with $info parity"
            return Γ, X
        end
    end
end

#= PEPSKit helper =#

function initialize_ctmrg_env(peps::InfinitePEPS, χ0::Int, χ::Int; kwargs...)
    Espace = Vect[FermionParity](0 => χ0 ÷ 2, 1 => χ0 ÷ 2) 
    env = CTMRGEnv(rand, ComplexF64, peps, Espace) 

    χ_eff_array = begin
        arr = [χ0]
        while arr[end] < χ
            push!(arr, min(arr[end] * 2, χ))
        end

        arr
    end

    info = nothing
    for χ_eff in χ_eff_array 
        @info "Growing environment to χ_eff = $χ_eff"
        env, info = leading_boundary( 
            env, peps; tol=1e-5, maxiter=500, alg= :simultaneous, trunc = truncdim(χ_eff) 
        ) 
    end

    # do last step with fixed space truncation
    Espace = Vect[FermionParity](0 => χ÷2, 1 => χ÷2) 
    env, info = leading_boundary( 
        env, peps; kwargs..., trunc = truncspace(Espace) 
    )

    return env, info
end