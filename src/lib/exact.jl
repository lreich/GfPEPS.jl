# Julia translation of Gaussian-fPEPS/src/gfpeps/exact.py

using LinearAlgebra

"""
    exact_energy_k(k, ht, D1X, D1Y, Mu)
Compute exact energy contribution for momentum k = (kx, ky).
"""
function exact_energy_k(k::AbstractVector{<:Real}, ht::Real, D1X::Real, D1Y::Real, Mu::Real)
    D = D1X * cos(k[1]) + D1Y * cos(k[2])
    c = cos(k[1]) + cos(k[2])
    t = [ -ht * c 0.0;
           0.0     -ht * c + Mu ]
    d = [ 0.0  D;
         -D    0.0]
    M = [ t  d;
         -d -t ]
    w = eigvals(Hermitian(M))
    N = length(w) ÷ 2
    return sum(w[1:N]) + tr(t)
end

"""
    eg(Lx, Ly, ht, D1X, D1Y, Mu)
Brillouin zone average of `exact_energy_k`.
"""
function eg(Lx::Integer, Ly::Integer, ht, D1X, D1Y, Mu)
    KSet = BatchK(Lx, Ly)
    acc = 0.0
    for i in 1:size(KSet, 1)
        acc += exact_energy_k(view(KSet, i, :), ht, D1X, D1Y, Mu)
    end
    return acc / size(KSet, 1)
end
