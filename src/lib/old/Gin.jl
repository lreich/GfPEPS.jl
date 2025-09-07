# Julia translation of Gaussian-fPEPS/src/gfpeps/Gin.py

using LinearAlgebra

"""
    SingleGammaIn(k)
Create a single Gamma In (4x4 complex matrix) for scalar momentum k (Float64).
"""
function SingleGammaIn(k::Real)
    t = cis(k) # exp(im*k)
    ct = -conj(t) # -exp(-im*k)
    z = zero(typeof(t))
    return [z z z t;
            z z t z;
            z ct z z;
            ct z z z]
end

SingleGammaIn(pi)

"""
    NvGammaIn(k, Nv)
Direct sum of SingleGammaIn, repeated Nv times.
"""
function NvGammaIn(k::Real, Nv::Integer)
    temp = SingleGammaIn(k)
    T = eltype(temp)
    out = zeros(T, 4 * Nv, 4 * Nv)
    for i in 0:(Nv - 1)
        out[4i + 1:4i + 4, 4i + 1:4i + 4] = temp
    end
    return out
end

NvGammaIn(pi,2)

"""
    GammaIn(k::AbstractVector, Nv)
Block diagonal of NvGammaIn for each component of k (assumed length 2 for (kx, ky)).
"""
function GammaIn(k::AbstractVector{<:Real}, Nv::Integer)
    # concatenate blocks along diagonal
    blocks = [NvGammaIn(ki, Nv) for ki in k]
    n = size(blocks[1], 1)
    T = eltype(blocks[1])
    out = zeros(T, n * length(blocks), n * length(blocks))
    for (ib, blk) in enumerate(blocks)
        i0 = (ib - 1) * n
        out[i0 + 1:i0 + n, i0 + 1:i0 + n] = blk
    end
    return out
end

"""
    BatchK(Lx, Ly)
Return an array of size (Lx*Ly, 2) with APBC-PBC boundary conditions.
"""
function BatchK(Lx::Integer, Ly::Integer)
    X = [(i - 0.5) / Lx for i in 1:Lx]
    Y = [j / Ly for j in 0:Ly-1]
    G = Array{Float64}(undef, Lx * Ly, 2)
    idx = 1
    for y in Y, x in X
        G[idx, 1] = 2π * x
        G[idx, 2] = 2π * y
        idx += 1
    end
    return G
end

"""
    BatchGammaIn(Lx, Ly, Nv)
Vectorized construction of GammaIn over the Brillouin zone.
Returns a 3D array (Lx*Ly, 8Nv, 8Nv) of ComplexF64.
"""
function BatchGammaIn(Lx::Integer, Ly::Integer, Nv::Integer)
    K = BatchK(Lx, Ly)
    dim = 8 * Nv
    out = Array{ComplexF64}(undef, size(K, 1), dim, dim)
    for i in 1:size(K, 1)
        out[i, :, :] = GammaIn(view(K, i, :), Nv)
    end
    return out
end