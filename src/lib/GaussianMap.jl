using LinearAlgebra
using BlockDiagonals

#= Correlation matrix function for the virtual bonds (G_in / Γ_in) =#
"""
    helper(k::Real)

Helper function to construct G_in_single_k(k::AbstractVector{<:Real}, Nv::Integer) where k is either kx or ky.

helper(k) = [  0   e^{i k} * σ_x
                    -e^{-i k} * σ_x   0  ]

"""
function helper(k::Real)
    σ_x = [0 1; 1 0]
    return [zeros(2,2) cis(k)*σ_x;
            -conj(cis(k))*σ_x zeros(2,2)]
end

"""
    G_in_single_k(k::AbstractVector{<:Real}, Nv::Integer)

Returns the Fourier transformed (F) covariance matrix of 1 virtual bond.

G_in_single_k(k, Nv) = [⊕_{i=1}^{Nv} G_in_single_k(kx)] ⊕ [⊕_{i=1}^{Nv} G_in_single_k(ky)]

"""
function G_in_single_k(k::AbstractVector{<:Real}, Nv::Integer)
    # return vcat( hcat(⊕(helper(k[1]), Nv), spzeros(4*Nv,4*Nv)), hcat(spzeros(4*Nv,4*Nv), ⊕(helper(k[2]), Nv)))

    return Matrix(BlockDiagonal([⊕(helper(k[1]), Nv),⊕(helper(k[2]), Nv)]))
end

"""
    G_in_Fourier(Lx::Int, Ly::Int, Nv::Int)

Returns the Fourier transformed (F) covariance matrix of all the virtual bonds: G_in = F Γ_in F†

"""
function G_in_Fourier(Lx::Int, Ly::Int, Nv::Int)
    kvals = get_2D_k_grid(Lx,Ly)

    res = Array{ComplexF64}(undef, size(kvals,1), 8*Nv, 8*Nv)
    for (i, row) in enumerate(eachrow(kvals))
        res[i, :, :] = G_in_single_k(row, Nv)
    end
    return res
end

#= Correlation matrix function for the fiducial state (G_out / Γ_out) =#
"""
    Γ_fiducial(X::AbstractMatrix, Nv::Int)

Construct the correlation matrix for the fiducial state A from orthogonal matrix X.

Note:
- X must be an orthogonal matrix: X * X' = I
- Γ_fiducial is either given in Fourier space or real space, depending on X
"""
function Γ_fiducial(X::AbstractMatrix, Nv::Int)
    @assert X*transpose(X) ≈ I "Input must be an orthogonal matrix"

    return transpose(X) * ⊕([0.0 1.0; -1.0 0.0],4*Nv+2) * X
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

# function GaussianMap(CM_out::AbstractMatrix, CM_in::AbstractArray, Nf::Int, Nv::Int)
#     @assert CM_out^2 ≈ -I "CM_out must be a real antisymmetric matrix"

#     # get block matrices from CM_out (=Γ_fiducial)
#     A = CM_out[1:2*Nf, 1:2*Nf]
#     B = CM_out[1:2*Nf, 2*Nf+1:end]
#     D = CM_out[2*Nf+1:end, 2*Nf+1:end]
#     @assert size(A) == (2*Nf, 2*Nf)
#     @assert size(B) == (2*Nf, 8*Nv)
#     @assert size(D) == (8*Nv, 8*Nv)

#     # Gaussian map for each (kx,ky)
#     N = size(CM_in,1)      
#     return map(1:N) do i
#         return B * inv(D + (@view CM_in[i, :, :])) * transpose(B) + A
#     end
# end

function GaussianMap(CM_out::AbstractMatrix, CM_in::AbstractArray, Nf::Int, Nv::Int)
    @assert CM_out^2 ≈ -I "CM_out must be a real antisymmetric matrix"

    # get block matrices from CM_out (=Γ_fiducial)
    A = CM_out[1:2*Nf, 1:2*Nf]
    B = CM_out[1:2*Nf, 2*Nf+1:end]
    D = CM_out[2*Nf+1:end, 2*Nf+1:end]
    @assert size(A) == (2*Nf, 2*Nf)
    @assert size(B) == (2*Nf, 8*Nv)
    @assert size(D) == (8*Nv, 8*Nv)

    # Gaussian map for each (kx,ky)
    # N = size(CM_in,1)      
    # result = Array{eltype(CM_in)}(undef, N, 2*Nf, 2*Nf)
    # for i in 1:N
    #     result[i, :, :] = B * inv(D + CM_in[i, :, :]) * transpose(B) + A
    # end
    # return result

    # compute one output per k without mutating
    mats = map(s -> B * ((D .+ s) \ transpose(B)) + A, eachslice(CM_in; dims=1))
    # avoid splatting to prevent StackOverflow
    out3 = reduce((X,Y) -> cat(X, Y; dims=3), mats)   # (2Nf)×(2Nf)×N
    return permutedims(out3, (3, 1, 2))               # N×(2Nf)×(2Nf)
end

# function GaussianMap(CM_out::AbstractMatrix, CM_in::AbstractArray, Nf::Int, Nv::Int)
#     # @assert CM_out^2 ≈ -I "CM_out must be a real antisymmetric matrix"

#     # get block matrices from CM_out (=Γ_fiducial)
#     A = CM_out[1:2*Nf, 1:2*Nf]
#     B = CM_out[1:2*Nf, 2*Nf+1:end]
#     D = CM_out[2*Nf+1:end, 2*Nf+1:end]
#     # @assert size(A) == (2*Nf, 2*Nf)
#     # @assert size(B) == (2*Nf, 8*Nv)
#     # @assert size(D) == (8*Nv, 8*Nv)

#     N = size(CM_in,1)  
#     Bt = transpose(B)    

#     # Ensure temporaries have an element type that can hold both CM_out and CM_in
#     elty = promote_type(eltype(D), eltype(CM_in))
#     nA = size(A,1)           # = 2Nf
#     nD = size(D,1)           # = 8Nv

#     out = Array{elty}(undef, N, nA, nA)

#     # preallocated temporaries with promoted element type
#     work = similar(D, elty)                     # will hold D + Gin (mutated by lu!)
#     rhs  = similar(Bt, elty)                    # (nD x nA) RHS (transpose(B))
#     tmp  = similar(A, elty)                     # (nA x nA) temporary for B * sol
#     B_prom = convert(Matrix{elty}, B)           # convert B/A to promoted type once
#     A_prom = convert(Matrix{elty}, A)
#     Bt_prom = convert(Matrix{elty}, Bt)

#     @inbounds for i in 1:N
#         Gin = @view CM_in[i, :, :]        # view into input batch
#         copyto!(work, convert(Matrix{elty}, D))  # work = D (promoted)
#         work .+= Gin                      # work = D + Gin  (Gin may be complex)

#         copyto!(rhs, Bt_prom)             # rhs = transpose(B) (promoted)
#         lu = lu!(work)                    # factorize in-place; returns LU object
#         sol = lu \ rhs                    # solves (D+Gin) \ transpose(B)

#         mul!(tmp, B_prom, sol)            # tmp = B * sol  (reuses tmp storage)
#         tmp .+= A_prom                    # tmp = B*sol + A

#         @views out[i, :, :] .= tmp        # copy tmp into batch result
#     end

#     return out
# end