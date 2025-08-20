# Julia translation of Gaussian-fPEPS/src/gfpeps/ABD.py
# Provides functions to build Gamma projectors and extract A,B,D blocks.

using LinearAlgebra

"""
    J(Nv::Integer)

Direct sum of [0 1; -1 0] repeated 4*Nv+2 times. Corresponds to 4 directions (2 per dir)
plus 4 physical fermion modes.
"""
function J(Nv::Integer)
    n = 4 * Nv + 2
    J2 = [0.0 1.0; -1.0 0.0]
    # block diagonal direct sum
    out = zeros(eltype(J2), 2 * n, 2 * n)
    for i in 0:(n - 1)
        out[2i + 1:2i + 2, 2i + 1:2i + 2] = J2
    end
    return out
end

"""
    GammaProjector(T::AbstractMatrix, Jmat::AbstractMatrix, Nv::Integer)
Despite a transpose, obtain Gamma projector: T' * J * T
"""
GammaProjector(T::AbstractMatrix, Jmat::AbstractMatrix, Nv::Integer) = transpose(T) * Jmat * T

"""
    getGammaProjector(T::AbstractMatrix, Nv::Integer)
Get Gamma local from orthogonal matrix T.
"""
getGammaProjector(T::AbstractMatrix, Nv::Integer) = GammaProjector(T, J(Nv), Nv)

"""
    getABD(GammaP::AbstractMatrix)
Slice Gamma projector into A,B,D blocks.
A = Γ[1:4,1:4], B = Γ[1:4,5:end], D = Γ[5:end,5:end]
"""
function getABD(GammaP::AbstractMatrix)
    A = GammaP[1:4, 1:4]
    B = GammaP[1:4, 5:end]
    D = GammaP[5:end, 5:end]
    return A, B, D
end

"""
    unitarize(R::AbstractMatrix)
Construct an orthogonal matrix from an arbitrary matrix R via exp(R - R').
"""
function unitarize(R::AbstractMatrix)
    return exp(R - transpose(R))
end
