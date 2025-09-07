"""
Tensor construction utilities translated from the Python `translate.py` script of Gaussian-fPEPS.

Provides a pipeline:
    Gamma -> local fiducial Hamiltonian -> ground-state vector -> reshape -> gate dressing

Resulting tensor has index order (u, l, f, d, r) matching the Python implementation comment (ulfdr).

Public API:
    translate(Gamma)::Vector        # ground-state flattened tensor (before reshape)
    reshape_tensor(vec, Nv)::Array  # reshape to rank-5 without gates
    add_gates(tensor, Nv)::Array    # apply bond & parity gates
    build_fiducial_tensor(Gamma, Nv)::Array  # full pipeline returning gated rank-5 tensor
"""
module Translate

using LinearAlgebra

export translate, reshape_tensor, add_gates, build_fiducial_tensor

# -- Core translation (Γ -> ground-state vector) -----------------------------

"""
    skew(M)
Return skew-symmetric part of matrix M.
"""
skew(M) = M - transpose(M)

"""
    cor_trans_matrix(cm)
Correlation transform S^T * cm * S with S = [[I I],[iI -iI]].
"""
function cor_trans_matrix(cm::AbstractMatrix)
    @assert size(cm,1)==size(cm,2)
    @assert iseven(size(cm,1))
    N = size(cm,1) ÷ 2
    Iₙ = Matrix{eltype(cm)}(I,N,N)
    S = [Iₙ Iₙ; im*Iₙ -im*Iₙ]
    return transpose(S) * cm * S
end

"""
    fiducial_hamiltonian(hρ, hκ)
Construct the many-body fiducial Hamiltonian matrix in occupation basis (size 2^N × 2^N)
from normal block hρ (Hermitian) and pairing block hκ (skew-symmetric), following Python reference.
"""
function fiducial_hamiltonian(hρ::AbstractMatrix, hκ::AbstractMatrix)
    N = size(hρ,1)
    @assert size(hρ,2)==N && size(hκ,1)==N && size(hκ,2)==N
    @assert norm(hρ - adjoint(hρ)) < 1e-8
    @assert norm(hκ + transpose(hκ)) < 1e-8
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)
    # iterate over basis states encoded by integers 0..2^N-1
    for k in 0:dim-1
        # occupation bit vector (LSB = mode 1); align with Python reversal logic
        for i in 1:N
            for j in 1:N
                b = digits(k, base=2, pad=N)  # least significant first
                # Python reversed then indexed; replicate logic carefully
                # We'll create array with index 1=Nth bit consistent ordering
                bk = reverse(b) # bk[1] corresponds to mode 1 like Python after reverse()
                parity = (sum(bk[i+1:end]) + sum(bk[j+1:end])) % 2
                if bk[j]==1
                    if bk[i]==0 || i==j
                        bk2 = copy(bk); bk2[j]=0; bk2[i]=1
                        target = foldl((acc,bit)->(acc<<1)+bit, bk2; init=0)
                        parity2 = parity + (j > i ? 1 : 0)
                        H[target+1,k+1] += 2*hρ[i,j]*((-1)^parity2)
                    elseif bk[i]==1
                        bk2 = copy(bk); bk2[i]=0; bk2[j]=0
                        target = foldl((acc,bit)->(acc<<1)+bit, bk2; init=0)
                        parity2 = parity + (i > j ? 1 : 0)
                        H[target+1,k+1] += hκ[i,j]*((-1)^parity2)
                    end
                elseif bk[j]==0 && bk[i]==0
                    bk2 = copy(bk); bk2[j]=1; bk2[i]=1
                    target = foldl((acc,bit)->(acc<<1)+bit, bk2; init=0)
                    parity2 = parity + (i > j ? 1 : 0)
                    H[target+1,k+1] -= conj(hκ[i,j])*((-1)^parity2)
                end
            end
        end
    end
    return (H + adjoint(H)) / 2
end

"""
    translate(Gamma)
Full pipeline: apply correlation transform to -Gamma, extract hρ,hκ blocks and diagonalize fiducial Hamiltonian; return lowest eigenvector.
"""
function translate(Gamma::AbstractMatrix)
    @assert size(Gamma,1)==size(Gamma,2) && iseven(size(Gamma,1))
    N = size(Gamma,1) ÷ 2
    trans_h = cor_trans_matrix(-Gamma)
    hρ = -im * transpose(trans_h[1:N, N+1:2N])
    hκ =  im * trans_h[1:N, 1:N]
    local_h = fiducial_hamiltonian(hρ, hκ)
    vals, vecs = eigen(Hermitian(local_h))
    return vecs[:, argmin(vals)]
end

# -- Gates -------------------------------------------------------------------

""" paritygate(n) -> Matrix{Float64}
Return diagonal parity gate of size n × n with entries (-1) for odd parity.
"""
function paritygate(n::Int)
    S = Matrix{Float64}(I, n, n)
    bits = ceil(Int, log(2, n))
    for i in 0:n-1
        # count set bits
        if count_ones(i) % 2 == 1
            S[i+1, i+1] = -1
        end
    end
    return S
end

"""
    fsign(n_vector)
Compute fermionic sign from list of occupations (0/1) via sum_{i>j} n_i n_j.
"""
function fsign(n::AbstractVector{<:Integer})
    result = 0
    for i in 2:length(n)
        result += n[i] * sum(n[1:i-1])
    end
    return (-1) ^ (result % 2)
end

""" bondgate(Nv)
Diagonal matrix implementing virtual fermion exchange sign structure.
Dimension is 2^Nv.
"""
function bondgate(Nv::Int)
    dim = 2 ^ Nv
    p = zeros(Float64, dim)
    for i in 0:dim-1
        # binary digits of length Nv
        bits = reverse(digits(i, base=2, pad=Nv))
        p[i+1] = fsign(bits)
    end
    return Diagonal(p)
end

"""
    add_bondgate(T, dim, Nv)
Apply bond gate on specified dimension (1-based) assuming that dimension has size 2^Nv.
"""
function add_bondgate(T::Array, dim::Int, Nv::Int)
    s = size(T)
    @assert s[dim] == 2 ^ Nv
    perm = collect(1:length(s))
    perm[1], perm[dim] = perm[dim], perm[1]
    Tp = permutedims(T, perm)
    shp = size(Tp)
    Tp = reshape(Tp, shp[1], :)
    Bg = Matrix(bondgate(Nv))
    Tp = Bg * Tp
    Tp = reshape(Tp, shp)
    invp = invperm(perm)
    return permutedims(Tp, invp)
end

""" add_paritygate(T, dim, Nv)
Apply parity gate (size 2^Nv) on given dimension.
"""
function add_paritygate(T::Array, dim::Int, Nv::Int)
    s = size(T)
    @assert s[dim] == 2 ^ Nv
    perm = collect(1:length(s))
    perm[1], perm[dim] = perm[dim], perm[1]
    Tp = permutedims(T, perm)
    shp = size(Tp)
    Tp = reshape(Tp, shp[1], :)
    Pg = paritygate(2 ^ Nv)  # matches Python paritygate which took 2^Nv
    Tp = Pg * Tp
    Tp = reshape(Tp, shp)
    invp = invperm(perm)
    return permutedims(Tp, invp)
end

""" add_gates(tensor, Nv)
Apply gates sequence: bond on dims 1 & 2, parity on dim 1.
"""
function add_gates(tensor::Array, Nv::Int)
    tensor = add_bondgate(tensor, 1, Nv)
    tensor = add_bondgate(tensor, 2, Nv)
    tensor = add_paritygate(tensor, 1, Nv)
    return tensor
end

# -- Reshape & pipeline ------------------------------------------------------

""" reshape_tensor(vec, Nv)
Reshape flat vector (length 4 * 2^(4Nv)? placeholder) into (2^Nv,2^Nv,4,2^Nv,2^Nv).
Caller must ensure length matches.
"""
function reshape_tensor(vec::AbstractVector, Nv::Int)
    d = 2 ^ Nv
    expected = d * d * 4 * d * d
    @assert length(vec) == expected
    T = reshape(vec, d, d, 4, d, d)
    # Python does transpose(4,3,2,1,0); that reverses outer order; mimic if needed.
    # Here keep original order (u,l,f,d,r) assuming alignment already correct.
    return T
end

""" build_fiducial_tensor(Gamma, Nv)
Full pipeline producing gated rank-5 tensor.
"""
function build_fiducial_tensor(Gamma::AbstractMatrix, Nv::Int)
    v = translate(Gamma)
    T = reshape_tensor(v, Nv)
    return add_gates(T, Nv)
end

end # module
