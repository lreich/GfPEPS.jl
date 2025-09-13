"""
    ⊕(A::AbstractMatrix, n::Integer)

repeat A ⊕ A ⊕ ... (n times) via kron

"""
function ⊕(A::AbstractMatrix, n::Integer)
    @assert n >= 1

    Id = Matrix{eltype(A)}(I, n, n)
    return kron(Id, A)
end

#= Correlation matrix function for the virtual bonds (G_in / Γ_in) =#
"""
    helper(k::Real)

Helper function to construct G_in_single_k(k::AbstractVector{<:Real}, Nv::Integer) where k is either kx or ky.

helper(k) = [  0   e^{i k} * σ_x
                    -e^{-i k} * σ_x   0  ]

"""
function helper(k::Real)
    σ_x = [0 1; 1 0]
    return [zeros(2,2) -cis(k)*σ_x;
            conj(cis(k))*σ_x zeros(2,2)]
end

"""
    G_in_single_k(k::AbstractVector{<:Real}, Nv::Integer)

Returns the Fourier transformed (F) covariance matrix of 1 virtual bond.

G_in_single_k(k, Nv) = [⊕_{i=1}^{Nv} G_in_single_k(kx)] ⊕ [⊕_{i=1}^{Nv} G_in_single_k(ky)]

"""
function G_in_single_k(k::AbstractVector{<:Real}, Nv::Integer)
    return Matrix(BlockDiagonal([⊕(helper(k[1]), Nv),⊕(helper(k[2]), Nv)]))
end

"""
    G_in_Fourier(bz::BrillouinZone2D, Nv::Int)

Returns the Fourier transformed (F) covariance matrix of all the virtual bonds: G_in = F Γ_in F†

"""
function G_in_Fourier(bz::BrillouinZone2D, Nv::Int)
    kvals = bz.kvals

    res = Array{ComplexF64,3}(undef, size(kvals,2), 8*Nv, 8*Nv)
    for (i, col) in enumerate(eachcol(kvals))
        res[i, :, :] = G_in_single_k(col, Nv)
    end
    return res
end

#= Correlation matrix function for the fiducial state (G_out / Γ_out) =#

"""
    build_J(Nv::Int)

Construct the symplectic matrix J for 4*Nv+2 modes.
"""
function build_J(Nv::Int,Nf::Int)
    return ⊕([0.0 1.0; -1.0 0.0], 4*Nv+Nf)
end
Zygote.@nograd build_J # constructing J is not something we need gradients through

"""
    Γ_fiducial(X::AbstractMatrix, Nv::Int)

Construct the correlation matrix for the fiducial state A from orthogonal matrix X.

Note:
- X must be an orthogonal matrix: X * X' = I 
- Γ_fiducial is either given in Fourier space or real space, depending on X
"""
function Γ_fiducial(X::AbstractMatrix, Nv::Int, Nf::Int)
    # return transpose(X) * ⊕([0.0 1.0; -1.0 0.0],4*Nv+2) * X
    return transpose(X) * build_J(Nv,Nf) * X
end

"""
    GaussianMap(Glocal, Gin)
Returns the Gaussian map: CM_out = B * inv(D + CM_in) * B' + A.

Keyword arguments:
- `CM_out::AbstractMatrix`: The covariance matrix of the fiducial state / the covariance matrix dual to the Gaussian map
- `CM_in::AbstractMatrix`: The covariance matrix of the virtual bonds
- `Nf::Int`: number of physical fermions
- `Nv::Int`: number of virtual fermions per bond

Note:
- CM_out must be a real antisymmetric matrix, i.e., CM_out² = -I
- The covariance matrices are currently only Fourier transformed TODO: also do for real space / no translation inv systems

"""
function GaussianMap(CM_out::AbstractMatrix, CM_in::AbstractArray, Nf::Int, Nv::Int)
    # get block matrices from CM_out (=Γ_fiducial)
    A = CM_out[1:2*Nf, 1:2*Nf]
    B = CM_out[1:2*Nf, 2*Nf+1:end]
    D = CM_out[2*Nf+1:end, 2*Nf+1:end]
    # Bt = transpose(B)

    # Gaussian map for each (kx,ky)
    mats = map(s -> B * ((D .+ s) \ transpose(B)) .+ A, eachslice(CM_in; dims=1))
    return cat(mats...; dims=3) |> x -> permutedims(x, (3,1,2))
end