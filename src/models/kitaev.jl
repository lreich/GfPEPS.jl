"""
Kitaev Hamiltonian mapped to Dirac fermions on a square lattice
```
    H = K ∑_{i,α} u_i,i+α (f†_{i} f_{i+α} + f†_{i} f†_{i+α}  + h.c.) + K ∑_i (2f†_{i} f_{i}-1)
```
where α=̂x, ̂y and u_i,i+α = ±1 are Z2 gauge fields.

"""
function Kitaev_hamiltonian(
        T::Type{<:Number}, lattice::InfiniteSquare; gauge_field::String="vortex_free", Jx::Real = 1.0,
        Jy::Real = 1.0, Jz::Real = 1.0
    )
    if gauge_field == "vortex_free"
        

    else
        @error("Only vortex_free gauge field is implemented.")
    end

    # Δx = Δ_0
    # if pairing_type == "s_wave"
    #     Δy = Δx
    # elseif pairing_type == "d_wave"
    #     Δy = -Δx
    # end

    # pspace = hub.hubbard_space(Trivial, Trivial)
    # pspaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    # num = hub.e_num(T, Trivial, Trivial)
    # unit = TensorKit.id(T, pspace)
    # hopping = (-t) * hub.e_hopping(T, Trivial, Trivial) -
    #     (μ / 4) * (num ⊗ unit + unit ⊗ num)
    # pairing = sqrt(2) * hub.singlet_plus(T, Trivial, Trivial)
    # pairing += pairing'
    # return LocalOperator(
    #     pspaces,
    #     map(nearest_neighbours(lattice)) do bond
    #         return bond => hopping + pairing * (_is_xbond(bond) ? Δx : Δy)
    #     end...
    # )
end
Kitaev_hamiltonian(lattice; gauge_field="vortex_free", Jx=1.0, Jy=1.0, Jz=1.0) = Kitaev_hamiltonian(ComplexF64, lattice; gauge_field=gauge_field, Jx=Jx, Jy=Jy, Jz=Jz)

"""
    ξ(k::AbstractVector{<:Real}, params::Kitaev)

(Vortex free configuration)

Returns:
```
    params.Jz - params.Jx* cos(k[1]) - params.Jy * cos(k[2])
```
"""
ξ(k::AbstractVector{<:Real}, params::Kitaev) = 2 * (params.Jz + params.Jx * cos(k[1]) + params.Jy * cos(k[2]))

Δ(k::AbstractVector{<:Real}, params::Kitaev) = 2 * (params.Jx * sin(k[1]) + params.Jy * sin(k[2]))
# Δ(k::AbstractVector{<:Real}, params::Kitaev) = 2*(params.Jx*cis(k[1]) + params.Jy*cis(k[2]))

function E(k::AbstractVector{<:Real}, params::Kitaev)
    return sqrt(ξ(k, params)^2 + abs(Δ(k, params))^2)
end

function exact_energy(params::Kitaev, bz::BrillouinZone2D)
    g_kitaev(k::AbstractVector{<:Real}; Jx::Real, Jy::Real, Jz::Real) = Jz + Jx * cis(k[1]) + Jy * cis(k[2])

    # Majorana band energy per unit cell: E_M(k) = 2 * |g(k)|
    E_uc = -mean(map(eachcol(bz.kvals)) do k
        2 * abs(g_kitaev(k; Jx=params.Jx, Jy=params.Jy, Jz=params.Jz))
    end)

    # return per site (honeycomb has 2 sites per unit cell)
    return E_uc / 2
end

"""
    has_dirac_points(bz::BrillouinZone2D, t::Real, μ::Real, pairing_type::String, Δ_kwargs...)

Checks if there are Dirac points (zero-energy modes) in the energy spectrum over the Brillouin zone `bz`.
"""
function has_dirac_points(bz::BrillouinZone2D, params::Kitaev)
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
function energy_CM(Γ_fiducial::AbstractMatrix, bz::BrillouinZone2D, Nf::Int, params::Kitaev)
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
            energy_k = real(
                ξ(k, params) * (2 - Gf[1, 2] - Gf[3, 4]) / 2 +
                    Δ(k, params) * (Gf[1, 4] + Gf[2, 3] + 1.0im * (Gf[2, 4] - Gf[1, 3])) / 2
            )
            return (energy_k - ξ(k, params)) / 2
        end
    )
end