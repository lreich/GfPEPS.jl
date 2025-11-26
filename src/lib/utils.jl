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

function init_ctmrg_env(peps)
    trivialspace = ProductSpace{GradedSpace{FermionParity, Tuple{Int64, Int64}}, 0}()

    corner_space = Vect[FermionParity](0 => 1, 1 => 1)

    C_ul = TensorMap([1.0], corner_space ← corner_space)
    C_dr = TensorMap([1.0], corner_space ← corner_space)
    C_dl = TensorMap([1.0], trivialspace ← corner_space ⊗ corner_space)
    C_ur = TensorMap([1.0], corner_space ⊗ corner_space ← trivialspace)

    χ_B = TensorKit.dim(domain(loc[i])[1])

    space_loc_l = codomain(loc[i])[1]
    space_loc_d = codomain(loc[i])[2]
    space_loc_r = domain(loc[i])[1]
    space_loc_u = domain(loc[i])[2]

    Tr_l = TensorMap(Matrix(1.0I,dim_loc_l,dim_loc_l), space_type^1 ← (space_loc_l)' ⊗ space_loc_l ⊗ space_type^1)
    Tr_d = TensorMap(Matrix(1.0I,dim_loc_d,dim_loc_d), space_type^1 ← space_type^1 ⊗ (space_loc_d)' ⊗ space_loc_d)
    Tr_r = TensorMap(Matrix(1.0I,dim_loc_r,dim_loc_r), (space_loc_r)' ⊗ space_loc_r ⊗ space_type^1 ← space_type^1)
    Tr_u = TensorMap(Matrix(1.0I,dim_loc_u,dim_loc_u), space_type^1 ⊗ (space_loc_u)' ⊗ space_loc_u ← space_type^1)
end