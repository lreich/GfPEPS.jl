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
            conj(cis(k))*σ_x zeros(2,2)] # Hackenbroich 2010 (lrud) (lrdu)
end

"""
    G_in_single_k(k::AbstractVector{<:Real}, Nv::Integer)

Returns the Fourier transformed covariance matrix for one k value.

The ordering of the majorana modes is (lrud) for each virtual fermion, i.e., (c_l1^1, c_l1^2, c_r1^1, c_r1^2, ..., c_lNv^1, c_lNv^2, c_rNv^1, c_rNv^2, c_u1^1, c_u1^2, c_d1^1, c_d1^2, ..., c_uNv^1, c_uNv^2, c_dNv^1, c_dNv^2) 

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


# function G_k_x(k::AbstractVector{<:Real}, unitcell::Tuple{Int,Int})
#     Nv_eff_x = Nv * unitcell[1]
#     Nv_eff_y = Nv * unitcell[2]
    
#     # 

#     # σ_x (e^{i k} - 1)
#     function helper(k::Real)
#         σ_x = [0 1; 1 0]

#         return [zeros(2,2) (cis(k)-1)*σ_x;
#                 (1 - cis(k))*σ_x zeros(2,2)]
#     end

#     return Matrix(BlockDiagonal([⊕(helper(k[1]), Nv),⊕(helper(k[2]), Nv)]))
# end

# function G_in_Fourier(bz::BrillouinZone2D, Nv::Int, unitcell::Tuple{Int,Int})
#     kvals = bz.kvals
#     N_uc = unitcell[1] * unitcell[2] # number of sites in the unit cell

#     function helper()
#         σ_x = [0 1; 1 0]

#         return [zeros(2,2) -σ_x;
#                 σ_x zeros(2,2)]
#     end

#     function G_in_single_k(k::AbstractVector{<:Real}, Nv::Integer)


#         # return Matrix(BlockDiagonal([⊕(helper(k[1]), Nv),⊕(helper(k[2]), Nv)]))
#     end

#     res = Array{ComplexF64,3}(undef, size(kvals,2), 8*Nv*N_uc, 8*Nv*N_uc)
#     for (i, col) in enumerate(eachcol(kvals))
#         res[i, :, :] = G_in_single_k(col, Nv)
#     end
#     return res
# end

# using LinearAlgebra
# using BlockDiagonals
# using GfPEPS

# # unitcell = (5,2)
# unitcell = (1,1)
# N_uc = unitcell[1] * unitcell[2]
# Nv = 2
# N = 8 * Nv * N_uc
# G = spdiagm(2 => -1 .* ones(N-2), 4 => -1 .* ones(N-4), 
#             -2 => 1 .* ones(N-2), -4 => 1 .* ones(N-4))



# unitcell = (5,2)
# N_uc = unitcell[1] * unitcell[2]
# σ_x = [0 1; 1 0]
# M = [zeros(2,2) -σ_x;
#     σ_x zeros(2,2)]
# Nv = 2
# BlockDiagonal([⊕(M, Nv),⊕(M, Nv)])


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
    Γ_fiducial(X::AbstractMatrix, Nv::Int, Nf::Int)

Construct the covariance matrix for the fiducial state A from orthogonal matrix X in the Majorana representation.
We choose Γ to be qq-ordered.

Γ_fiducial = [A B; -B' D]

Where A ∈ ℝ^(2Nf x 2Nf), B ∈ ℝ^(2Nf x 8Nv), D ∈ ℝ^(8Nv x 8Nv).
A and D are antisymmetric.

The modes of the A block are qq-ordered as: (c_1, c_2, ..., c_(2Nf))
The modes of the D block have the same ordering (lrud) as G_in_single_k, i.e., (c_l1^1, c_l1^2, c_r1^1, c_r1^2, ..., c_lNv^1, c_lNv^2, c_rNv^1, c_rNv^2, c_u1^1, c_u1^2, c_d1^1, c_d1^2, ..., c_uNv^1, c_uNv^2, c_dNv^1, c_dNv^2) 
The modes of the B block are ordered as above.

Note:
- X must be an orthogonal matrix: X * X' = I 
- Γ_fiducial is either given in Fourier space or real space, depending on X
"""
function Γ_fiducial(X::AbstractMatrix, Nv::Int, Nf::Int)
    Γ = transpose(X) * build_J(Nv,Nf) * X

    return (Γ - transpose(Γ)) / 2 # ensure exact antisymmetry
end

function get_Γ_blocks(Γ::AbstractMatrix, Nf::Int)
    A = Γ[1:2*Nf, 1:2*Nf]
    B = Γ[1:2*Nf, 2*Nf+1:end]
    D = Γ[2*Nf+1:end, 2*Nf+1:end]
    return A,B,D
end

"""
    GaussianMap(CM_out::AbstractMatrix, CM_in::AbstractArray, Nf::Int, Nv::Int)

Returns the Gaussian map: CM_out = B * inv(D + CM_in) * B' + A.
This contracts the virtual bonds and only the physical modes remain.

Keyword arguments:
- `CM_out::AbstractMatrix`: The covariance matrix of the fiducial state / the covariance matrix dual to the Gaussian map
- `CM_in::AbstractMatrix`: The covariance matrix of the virtual bonds
- `Nf::Int`: number of physical fermions
- `Nv::Int`: number of virtual fermions per bond

Note:
- CM_out must be a real antisymmetric matrix, i.e., CM_out² = -I
- The covariance matrices are currently only Fourier transformed TODO: also do for real space / no translation inv systems

Nv = Λ

"""
function GaussianMap(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractArray)
    Bt = transpose(B)

    # Gaussian map for each (kx,ky)
    # mats = map(s -> B * ((D .- s) \ transpose(B)) .+ A, eachslice(CM_in; dims=1)) # Kraus thesis
    mats = map(s -> B * ((D .+ s) \ Bt) .+ A, eachslice(CM_in; dims=1)) # Hong hao paper
    return permutedims(stack(mats, dims=3), (3,1,2))
end

function GaussianMap_single_k(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractMatrix)
    return B * ((D + CM_in) \ transpose(B)) .+ A
end
