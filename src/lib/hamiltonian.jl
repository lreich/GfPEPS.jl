using LinearAlgebra
"""
    get_kvals(::Val{:PBC}, L)

Returns the allowed momentum values for a 1D chain with periodic boundary conditions (PBC).

# Arguments
- `L`: System size (number of sites)

# Returns
- `Vector{Float64}`: Allowed momentum values 2π*m/L where:
  - If L is even: m ∈ {-(L-2)/2, ..., L/2}
  - If L is odd: m ∈ {-(L-1)/2, ..., (L-1)/2}
"""
function get_kvals(::Val{:PBC},L)
    if iseven(L)
        return [2π*m/L for m in (-(L-2)/2):L/2] 
    else
        return [2π*m/L for m in (-(L-1)/2):(L-1)/2] 
    end
end

"""
    get_kvals(::Val{:APBC}, L)

Returns the allowed momentum values for a 1D chain with anti-periodic boundary conditions (APBC).

# Arguments
- `L`: System size (number of sites)

# Returns
- `Vector{Float64}`: Allowed momentum values (2m-1)π/L where:
  - If L is even: m ∈ {1, ..., L/2}, returns both ±k values
  - If L is odd: m ∈ {1, ..., (L-1)/2}, returns ±k values plus π
"""
function get_kvals(::Val{:APBC},L)
    if iseven(L)
        kvals = [(2*m-1)*π/L for m in 1:L/2] 
		return vcat(-kvals,kvals)
    else
        kvals = [(2*m-1)*π/L for m in 1:(L-1)/2] 
		return vcat(-kvals,kvals,pi)
    end
end

"""
        get_2D_k_grid(Lx, Ly; x_periodic=Val(:APBC), x_offset=0.0, y_periodic=Val(:PBC),  y_offset=0.0)

Create the 2D momentum grid from 1D k-values (with optional offsets) and
return a meshgrid of the form: [[kx_1, ky_1]; 
                                [kx_2, ky_1]; 
                                    ... 
                                [kx_Lx, ky_1];
                                [kx_1, ky_2];
                                [kx_2, ky_2];
                                    ...
                                [kx_Lx, ky_2];
                                    ...
                                [kx_Lx, ky_Ly];

Returns
- Matrix of size (Lx*Ly)×2 where:
    - column 1 = vec(X) (kx repeated along rows)
    - column 2 = vec(Y) (ky repeated along columns)

Notes
- set the offsets, such that zero modes are avoided as those make the optimization of Γ harder.
"""
function get_2D_k_grid(Lx::Int, Ly::Int; 
    x_periodic::Union{Val{:APBC}, Val{:PBC}} = Val(:APBC),
    x_offset::Float64 = pi/2,
    y_periodic::Union{Val{:APBC}, Val{:PBC}} = Val(:PBC),
    y_offset::Float64 = pi/2)

    # TODO: test with correct kvals but first take from paper to compare
    # k_vals_x = sort(get_kvals(x_periodic, Lx) .+ x_offset)
    # k_vals_y = sort(get_kvals(y_periodic, Ly) .+ y_offset)

    # # create meshgrid
    # KX = repeat([kx for kx in k_vals_x], Ly)
    # KY = collect(Iterators.flatten(map(k_vals_y) do ky
    #     repeat([ky],Lx)
    # end))

    # return hcat(KX,KY)

    x = ((0:Lx-1) .- 0.5) ./ Lx
    y = (0:Ly-1) ./ Ly
    X = repeat(x', Ly, 1)      # (Ly, Lx), X[i,j] = x[j]
    Y = repeat(y, 1, Lx)       # (Ly, Lx), Y[i,j] = y[i]
    # row-major flatten to match NumPy/JAX
    Xf = vec(permutedims(X))
    Yf = vec(permutedims(Y))
    return 2π .* hcat(Xf, Yf)  # (Ly*Lx, 2)
end

ξ(k::AbstractVector,t::Real,μ::Real) = -2t * (cos(k[1]) + cos(k[2])) - μ
# Δ(k::AbstractVector,Δ_x::Real) = 2*Δ_x*(cos(k[1]) - cos(k[2]))
Δ(k::AbstractVector,Δ_x::Real,Δ_y::Real) = 2*(Δ_x*cos(k[1]) + Δ_y*cos(k[2]))
E(k::AbstractVector,t::Real,μ::Real,Δ_x::Real,Δ_y::Real) = sqrt(ξ(k,t,μ)^2 + Δ(k,Δ_x,Δ_y)^2)

# TODO: überarbeiten
function exact_energy_BCS_k(k::AbstractVector{<:Real}, t, Δ_x::Real, Δ_y::Real, μ::Real)
    T = [ ξ(k,t,μ)  0.0;
          0.0 -ξ(k,t,μ) ]
    D = [ 0.0  Δ(k,Δ_x,Δ_y);
         -Δ(k,Δ_x,Δ_y)  0.0 ]
    M = [ T  D;
         -D -T ]
    ε_k = eigvals(Hermitian(M))

    # return -E(k,t,μ,Δ_x,Δ_y) - μ

    return sum(ε_k[ε_k .< 0]) + tr(T)

end

"""
    exact_energy_BCS_BZ_average(Lx::Integer, Ly::Integer,t, Δ, μ)
Momentum average of `exact_energy_k`.
"""
function exact_energy_BCS_k_average(Lx::Integer, Ly::Integer,t, Δ_x, Δ_y, μ)
    kvals = get_2D_k_grid(Lx, Ly)
    return sum([exact_energy_BCS_k(k, t, Δ_x, Δ_y, μ) for k in eachrow(kvals)])/size(kvals,1)
end


#======================================================================================
Functions to solve μ from hole density
======================================================================================#

"""
    nk(k, delta, mu)
Occupation number for BCS state.
"""
function nk(k::AbstractVector{<:Real}, δ::Real, μ::Real)
    # kx, ky = k
    # a = 1 / δ * ( -cos(kx) - cos(ky) + μ / 2 ) / abs(cos(kx) - cos(ky))
    # return 1 / sqrt(1 + a^2) / (sqrt(1 + a^2) + a)
    ξ_k = -2t * (cos(k[1]) + cos(k[2])) - μ
    Δ_k = 2*Δ*(cos(k[1]) - cos(k[2]))
    E_k = sqrt(ξ_k^2 + Δ_k^2)


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