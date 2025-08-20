# Julia translation of Gaussian-fPEPS/src/gfpeps/deltatomu.py

using LinearAlgebra

"""
    nk(k, delta, mu)
Occupation number for BCS state.
"""
function nk(k::AbstractVector{<:Real}, delta::Real, mu::Real)
    kx, ky = k
    a = 1 / delta * ( -cos(kx) - cos(ky) + mu / 2 ) / abs(cos(kx) - cos(ky))
    return 1 / sqrt(1 + a^2) / (sqrt(1 + a^2) + a)
end

"""
    ntotal(delta, mu; L=101)
Average occupation over Brillouin zone grid of size LxL.
"""
function ntotal(delta::Real, mu::Real; L::Integer=101)
    KX = [(i - 0.5) / L for i in 1:L]
    KY = [j / L for j in 0:L-1]
    s = 0.0
    for ky in KY, kx in KX
        s += nk([2π * kx, 2π * ky], delta, mu)
    end
    return s / (L * L)
end

"""
    solve_mu(dxy, delta)
Solve mu from target filling 1 - delta using a scalar root find on ntotal.
"""
function solve_mu(dxy::Real, delta::Real; bracket::Tuple{Real,Real}=(0.0, 10.0))
    a, b = bracket
    fa = ntotal(dxy, a) - (1 - delta)
    fb = ntotal(dxy, b) - (1 - delta)
    for _ in 1:60
        c = (a + b) / 2
        fc = ntotal(dxy, c) - (1 - delta)
        if sign(fc) == sign(fa)
            a, fa = c, fc
        else
            b, fb = c, fc
        end
        if abs(b - a) < 1e-8
            break
        end
    end
    return (a + b) / 2
end
