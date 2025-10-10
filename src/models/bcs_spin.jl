"""
BCS spin-1/2 Hamiltonian with singlet pairing terms on square lattice
```
    H = -t ∑_{i,v} (c†_{iα} c_{i+v,α} + h.c.) - μ ∑_i c†_{iα} c_{iα}
        + ∑_{i,v} (Δv ϵ_{αβ} c†_{iα} c†_{i+v,β} + h.c.)
```
where v sums over the basis vectors e_x, e_y. 

- s-wave state: Δy = Δx.
- d-wave state: Δy = -Δx.
- (d+id) state: Δy = i Δx.
"""
function BCS_spin_hamiltonian(
        T::Type{<:Number}, lattice::InfiniteSquare; pairing_type::String="d_wave", t::Float64 = 1.0,
        Δ_0::Float64 = 0.5, μ::Float64 = 0.0
    )
    Δx = Δ_0
    if pairing_type == "s_wave"
        Δy = Δx
    elseif pairing_type == "d_wave"
        Δy = -Δx
    end

    pspace = hub.hubbard_space(Trivial, Trivial)
    pspaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    num = hub.e_num(T, Trivial, Trivial)
    unit = TensorKit.id(T, pspace)
    hopping = (-t) * hub.e_hopping(T, Trivial, Trivial) -
        (μ / 4) * (num ⊗ unit + unit ⊗ num)
    pairing = sqrt(2) * hub.singlet_plus(T, Trivial, Trivial)
    pairing += pairing'
    return LocalOperator(
        pspaces,
        map(nearest_neighbours(lattice)) do bond
            return bond => hopping + pairing * (_is_xbond(bond) ? Δx : Δy)
        end...
    )
end
BCS_spin_hamiltonian(lattice; pairing_type="d_wave", Δ_0=1.0, μ=0.0) = BCS_spin_hamiltonian(ComplexF64, lattice; pairing_type=pairing_type, Δ_0=Δ_0, μ=μ)

"""
Check if a 2-site bond is a nearest neighbor x-bond
"""
function _is_xbond(bond)
    return bond[2] - bond[1] == CartesianIndex(0, 1)
end

#= Energy functions when diagonalised via Fourier transform =#
"""
    ξ(k::AbstractVector{<:Real},t::Real,μ::Real)

Returns:
```
    -2t * (cos(k_x) + cos(k_y)) - μ
```
"""
# ξ(k::AbstractVector{<:Real},t::Real,μ::Real) = -2t * (cos(k[1]) + cos(k[2])) - μ
ξ(k::AbstractVector{<:Real}, params::BCS) = -2 * params.t * (cos(k[1]) + cos(k[2])) - params.μ

"""
    Δ(pairing_type::String, kwargs...)

Returns pairing amplitude:
```
    ● d_wave: 2*Δ_0*(cos(k_x) - cos(k_y))
    ● s_wave: 2*Δ_0
    ● p+ip_wave: 2*Δ_0*(sin(k_x) - im*sin(k_y))
    ● kitaev_vortex_free: J*(e^{ik_x} + e^{ik_y})
```
"""
function Δ(k::AbstractVector{<:Real}, params::BCS)
    if params.pairing_type == "d_wave"
        return Δ(Val(:d_wave), k, params.Δ_0)
    elseif params.pairing_type == "s_wave"
        return Δ(Val(:s_wave), k, params.Δ_0)
    elseif params.pairing_type == "p_ip_wave"
        return Δ(Val(:p_ip_wave), k, params.Δ_0)
    else
        throw(ArgumentError("Unsupported pairing_type $(params.pairing_type) for BCS parameters"))
    end
end

Δ(::Val{:d_wave},k::AbstractVector{<:Real},Δ_0) = 2*Δ_0*(cos(k[1]) - cos(k[2]))
Δ(::Val{:s_wave},k::AbstractVector{<:Real},Δ_0::Real) = 2*Δ_0
Δ(::Val{:p_ip_wave},k::AbstractVector{<:Real},Δ_0::Real) = 2*Δ_0*(sin(k[1]) + im*sin(k[2]))
function E(k::AbstractVector{<:Real}, params::BCS)
    return sqrt(ξ(k, params)^2 + abs(Δ(k, params))^2)
end

"""
    exact_energy_BCS_k(bz::BrillouinZone2D, t::Real, μ::Real, Δ_kwargs...)

Returns the exact ground state energy per site of a BCS mean field Hamiltonian over the Brillouin zone `bz`.
"""
function exact_energy(params::BCS, bz::BrillouinZone2D)
    return mean(map(eachcol(bz.kvals)) do k
        ξ(k,params) - E(k, params)
    end)
end

# function exact_energy_BCS_k(bz::BrillouinZone2D, t::Real, μ::Real, pairing_type::String, Δ_0::Real) 
#     if pairing_type == "kitaev_vortex_free"
#         # Interpret Δ_0 as J (Jx=Jy=J), and μ = -2 Jz  => Jz = -μ/2
#         J = Δ_0
#         Jx = J; Jy = J; Jz = -μ/2

#         # Majorana band energy per unit cell: E_M(k) = 2 * |g(k)|
#         e_uc = -mean(map(eachcol(bz.kvals)) do k
#             2 * abs(g_kitaev(k; Jx=Jx, Jy=Jy, Jz=Jz))
#         end)

#         # return per site (honeycomb has 2 sites per unit cell)
#         return e_uc / 2
#     else
#         return mean(map(eachcol(bz.kvals)) do k
#             ξ(k,t,μ) - E(k, t, μ, pairing_type, Δ_0)
#         end)
#     end
# end

"""
    has_dirac_points(bz::BrillouinZone2D, t::Real, μ::Real, pairing_type::String, Δ_kwargs...)

Checks if there are Dirac points (zero-energy modes) in the energy spectrum over the Brillouin zone `bz`.
"""
function has_dirac_points(bz::BrillouinZone2D, params::BCS)
    dirac_point_found = false
    for k in eachcol(bz.kvals)
        if isapprox(E(k, params), 0.0; atol = 1e-6)
            @warn ("Dirac point found at k = $k. This may lead to convergence issues during optimization.")
            dirac_point_found = true
        end
    end
    return dirac_point_found
end

"""
The energy of a Gaussian fPEPS evaluated from 
the fiducial state correlation matrix `G`.
"""
function energy_CM(Γ_fiducial::AbstractMatrix, bz::BrillouinZone2D, Nf::Int, params::BCS)
    A = Γ_fiducial[1:2*Nf, 1:2*Nf]
    B = Γ_fiducial[1:2*Nf, 2*Nf+1:end]
    D = Γ_fiducial[2*Nf+1:end, 2*Nf+1:end]

    χ = div(size(Γ_fiducial, 1) - 2 * Nf, 8)
    return mean(
        map(eachcol(bz.kvals)) do k
            G_in = G_in_single_k(k, χ)
            Gf = A + B * inv(D + G_in) * transpose(B)
            # qq ordering of Majorana modes: (c_1, c_2, ..., c_(2(4Nv + Nf)))
            # return real(
            #     ξ(k,t,mu) * (2 - Gf[1, 2] - Gf[3, 4]) / 2 +
            #         Δ(pairing_type, k, Δ_0) * (Gf[1, 4] + Gf[2, 3] + 1.0im * (Gf[2, 4] - Gf[1, 3])) / 2 
            # )
            return real(
                ξ(k, params) * (2 - Gf[1, 2] - Gf[3, 4]) / 2 +
                    Δ(k, params) * (Gf[1, 4] + Gf[2, 3] + 1.0im * (Gf[2, 4] - Gf[1, 3])) / 2
            )
        end
    )
end

"""
Gutzwiller projector from Hubbard (spin-1/2) 
to the no-double-occupancy tJ space.

In Gutzwiller approximation, z = 2δ/(1+δ), 
where δ is doping before projection.
"""
function gutzwiller_projector(z::Float64)
    V_hub = hub.hubbard_space(Trivial, Trivial)
    V_tJ = tJ.tj_space(Trivial, Trivial)
    P = zeros(Float64, V_hub → V_tJ) # from hubbard (Vect[FermionParity](0=>2, 1=>2)) to tJ (Vect[FermionParity](0=>1, 1=>2)
    S = FermionParity
    P[(S(0), S(0))][1, 1] = sqrt(z) # |0> -> sqrt(z) |0>
    # P[(S(0), S(0))][1, 2] = 0     # |0> -> 0 |↑↓>
    P[(S(1), S(1))][1, 1] = 1.0     # |↑> -> |↑>
    P[(S(1), S(1))][2, 2] = 1.0     # |↓> -> |↓>
    return P
end

"""
Apply Gutzwiller projection to Hubbard (spin-1/2) PEPS
"""
function gutzwiller_project(z::Float64, peps::InfinitePEPS)
    P = gutzwiller_projector(z)
    return InfinitePEPS(collect(P * t for t in peps.A))
end

#======================================================================================
Functions to solve μ from hole density
======================================================================================#
function exact_doping(bz::BrillouinZone2D, t::Real, μ::Real, pairing_type::String, Δ_0::Real)
    bcs_params = BCS(t, μ, pairing_type, Δ_0)

    return mean(map(eachcol(bz.kvals)) do k
        ξ(k,bcs_params) / E(k, bcs_params)
    end)
end

function solve_for_mu(bz::BrillouinZone2D, δ::Real, t::Real, pairing_type::String, Δ_0::Real; μ_range::NTuple{2, Float64} = (-5.0, 5.0))
    μ = find_zero(x -> δ - exact_doping(bz, t, x, pairing_type, Δ_0), μ_range)
    return μ
end

"""
    doping_bcs(Γ::AbstractMatrix, bz::BrillouinZone, Nf::Int)

The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the fiducial state correlation matrix `Γ`.
"""
function doping_bcs(Γ::AbstractMatrix, bz::BrillouinZone2D, Nf::Int)
    A, B, D = get_Γ_blocks(Γ, Nf)
    Nv = div(size(Γ, 1) - 2 * Nf, 8)
    return mean(
        map(eachcol(bz.kvals)) do k
            G_in_k = G_in_single_k(k, Nv)
            Gf = GaussianMap_single_k(A, B, D, G_in_k)
            return real(Gf[1, 2] + Gf[3, 4]) / 2
        end
    )
end

"""
    doping_bcs(X::AbstractMatrix, bz::BrillouinZone2D, Nf::Int, Nv::Int)

The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the matrix `X` from which the fiducial state correlation matrix `Γ` is built.
"""
function doping_bcs(X::AbstractMatrix, bz::BrillouinZone2D, Nf::Int, Nv::Int)
    Γ = Γ_fiducial(X, Nv, Nf)
    return doping_bcs(Γ, bz, Nf)
end

"""
    doping_peps(peps::InfinitePEPS, env::CTMRGEnv)

The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the GfPEPS tensor.
"""
function doping_peps(peps::InfinitePEPS, env::CTMRGEnv)
    lattice = collect(space(t, 1) for t in peps.A)
    O = LocalOperator(lattice, ((1, 1),) => hub.e_num(Trivial, Trivial))
    return 1 - real(expectation_value(peps, O, env))
end

# Number operator in Gutzwiller projected space
function e_num_GW(V)
    t = zeros(T, V ← V)
    I = sectortype(t)
    t[(I(1), I(1))][1, 1] = 1
    t[(I(1), I(1))][2, 2] = 1
    return t
end

"""
    doping_pepsGW(peps::InfinitePEPS, env::CTMRGEnv)

The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the Gutzwiller projected GfPEPS tensor.
"""
function doping_pepsGW(peps::InfinitePEPS, env::CTMRGEnv)
    V = Vect[FermionParity](0 => 1, 1 => 2)

    lattice = collect(space(t, 1) for t in peps.A)
    O = LocalOperator(lattice, ((1, 1),) => e_num_GW(V))
    return 1 - real(expectation_value(peps, O, env))
end