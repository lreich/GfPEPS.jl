# Julia translation of Gaussian-fPEPS/src/gfpeps/GaussianLinearMap.py

using LinearAlgebra

"""
    GaussianLinearMap(Glocal, Gin)
Perform Gaussian linear map: Γ_out = A + B * inv(D + Γ_in) * B'.
If `Gin` is a batch (3D array), this broadcasts over the first dimension.
"""
function GaussianLinearMap(Glocal::AbstractMatrix, Gin)
    A, B, D = getABD(Glocal)
    if ndims(Gin) == 3
        n = size(Gin, 1)
        Telt = promote_type(eltype(Gin), eltype(A))
        out = Array{Telt}(undef, n, size(A, 1), size(A, 2))
        for i in 1:n
            out[i, :, :] = A + B * inv(D + view(Gin, i, :, :)) * transpose(B)
        end
        return out
    else
        return A + B * inv(D + Gin) * transpose(B)
    end
end
