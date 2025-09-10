using LinearAlgebra
using Statistics
using Roots

#= Energy functions when diagonalised via Fourier transform =#
# in the paper it is - μ  but in the code it is + μ TODO: check sign convention
ξ(k::AbstractVector{<:Real},t::Real,μ::Real) = -2t * (cos(k[1]) + cos(k[2])) - μ

# different pairing types
function Δ(pairing_type::String, kwargs...)
    if pairing_type == "d_wave"
        return Δ(Val(:d_wave), kwargs...)
    elseif pairing_type == "s_wave"
        return Δ(Val(:s_wave), kwargs...)
    elseif pairing_type == "p_wave"
        return Δ(Val(:p_wave), kwargs...)
    else
        return Δ(Val(:default), kwargs...)
    end
end
Δ(::Val{:d_wave},k::AbstractVector{<:Real},Δ_x::Real,Δ_y::Real) = 2*(Δ_x*cos(k[1]) - Δ_y*cos(k[2]))
Δ(::Val{:s_wave},k::AbstractVector{<:Real},Δ_0::Real) = 2*Δ_0
Δ(::Val{:p_wave},k::AbstractVector{<:Real},Δ_x::Real,Δ_y::Real) = 2*(Δ_x*sin(k[1]) - Δ_y*sin(k[2])) + im*(Δ_x*sin(k[1]) + Δ_y*sin(k[2]))
Δ(::Val{:default},k::AbstractVector{<:Real},Δ_0::Real) = im*2*Δ_0(sin(k[1]) + sin(k[2]))
function E(k::AbstractVector{<:Real},t::Real,μ::Real,pairing_type::String,Δ_kwargs...) 
    Δ_kwargs = (k,Δ_kwargs...) # match input format of Δ
    return sqrt(ξ(k,t,μ)^2 + abs(Δ(pairing_type, Δ_kwargs...))^2)
end

"""
    exact_energy_BCS_k(bz::BrillouinZone2D, t::Real, μ::Real, Δ_kwargs...)

Returns the exact ground state energy per site of a BCS mean field Hamiltonian over the Brillouin zone `bz`.
"""
function exact_energy_BCS_k(bz::BrillouinZone2D, t::Real, μ::Real, pairing_type::String, Δ_kwargs...)
    return mean(map(eachrow(bz.kvals)) do k
        ξ(k,t,μ) - E(k, t, μ, pairing_type, Δ_kwargs...)
    end)
end

"""
    exact_energy_BCS(T,D)

Exact ground state energy of a BCS Hamiltonian in BdG form
    \\hat{H} = 1/2 ( c_1^\\dagger ... c_L^\\dagger, c_1 ... c_L )^T H ( c_1 ... c_L c_1^\\dagger ... c_L^\\dagger ) + 1/2 Tr(T)
given in block form H = [ T  D; -D -T ].
"""
function exact_energy_BCS(T::Matrix,D::Matrix; )
    H = [ T  D;
         -D -T' ]
    
    @assert ishermitian(H) "Hamiltonian is not Hermitian! Check if you have an error in T or D."

    ε = eigvals(H)
    return sum(ε[ε .< 0])/2 + tr(T)/2
end

#======================================================================================
Functions to solve μ from hole density
======================================================================================#

function exact_doping(bz::BrillouinZone2D, t::Real, μ::Real, Δ_x::Real, Δ_y::Real)
    return mean(map(eachcol(bz.kvals)) do k
        ξ(k,t,μ) / E(k, t, μ, Val(:d_wave), Δ_x, Δ_y)
    end)
end

function solve_for_mu(bz::BrillouinZone2D, δ::Real, t::Real, Δ_x::Real, Δ_y::Real; μ_range::NTuple{2, Float64} = (-5.0, 5.0))
    μ = find_zero(x -> δ - exact_doping(bz, t, x, Δ_x, Δ_y), μ_range)
    return μ
end

# solve_for_mu(BrillouinZone2D(4,4,(:APBC,:PBC)), 0.5, 1.0, 0.5, -0.5)