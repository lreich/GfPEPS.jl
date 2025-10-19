"""
Kitaev Hamiltonian mapped to Dirac fermions on a square lattice
```
     H =  ∑_{r} [ J_x (c†_{r} + c_{r})*(c†_{r+r_1} - c_{r+r_1})
        + J_y (c†_{r} + c_{r})*(c†_{r+r_2} - c_{r+r_2})
        + J_z (2 c†_{r} c_{r} - 1) ]
```
where r sums over the lattice vectors and `r₁ = (0, 1)` and `r₂ = (1, 0)` correspond to the
nearest-neighbour bonds of the infinite square lattice.
"""
function Kitaev_Hamiltonian(
        T::Type{<:Number}, lattice::InfiniteSquare; gauge_field::String="vortex_free",
        Jx::Real = 1.0, Jy::Real = 1.0, Jz::Real = 1.0
    )
    gauge_field == "vortex_free" || throw(ArgumentError("Only vortex_free gauge field is implemented."))

    pspace = FO.fermion_space()
    spaces = fill(pspace, (lattice.Nrows, lattice.Ncols))

    coeff_Jx = convert(T, Jx)
    coeff_Jy = convert(T, Jy)
    coeff_Jz = convert(T, Jz)
    two = convert(T, 2)

    num = FO.f_num(T)
    id_site = TensorKit.id(T, pspace)
    onsite = coeff_Jz * (two * num - id_site)

    fpp = FO.f_plus_f_plus(T)
    fpm = FO.f_plus_f_min(T)
    fmp = FO.f_min_f_plus(T)
    fmm = FO.f_min_f_min(T)
    base_link = fpp - fpm + fmp - fmm
    op_x = coeff_Jx * base_link
    op_y = coeff_Jy * base_link

    bonds_x = Tuple{CartesianIndex, CartesianIndex}[]
    bonds_y = Tuple{CartesianIndex, CartesianIndex}[]
    for (a, b) in nearest_neighbours(lattice)
        δ = b - a
        if δ == CartesianIndex(0, 1)
            push!(bonds_x, (a, b))
        elseif δ == CartesianIndex(1, 0)
            push!(bonds_y, (b, a))
        else
            throw(ArgumentError("Unexpected bond displacement $δ for InfiniteSquare lattice."))
        end
    end

    return LocalOperator(
        spaces,
        ((site,) => onsite for site in vertices(lattice))...,
        (bond => op_x for bond in bonds_x)...,
        (bond => op_y for bond in bonds_y)...,
    )
end
Kitaev_Hamiltonian(lattice::InfiniteSquare; gauge_field::String="vortex_free", Jx::Real = 1.0, Jy::Real = 1.0, Jz::Real = 1.0) =
    Kitaev_Hamiltonian(ComplexF64, lattice; gauge_field=gauge_field, Jx=Jx, Jy=Jy, Jz=Jz)

Kitaev_hamiltonian(args...; kwargs...) = Kitaev_Hamiltonian(args...; kwargs...)



"""
    ξ(k::AbstractVector{<:Real}, params::Kitaev)

(Vortex free configuration)

Returns:
```
    params.Jz - params.Jx* cos(k[1]) - params.Jy * cos(k[2])
```
"""
ξ(k::AbstractVector{<:Real}, params::Kitaev) = 2 * (params.Jz - params.Jx * cos(k[1]) - params.Jy * cos(k[2]))

Δ(k::AbstractVector{<:Real}, params::Kitaev) = 2 * im* (params.Jx * sin(k[1]) + params.Jy * sin(k[2]))
# Δ(k::AbstractVector{<:Real}, params::Kitaev) = 2*(params.Jx*cis(k[1]) + params.Jy*cis(k[2]))

function A(k::AbstractVector{<:Real}, params::Kitaev)
    return [0  2*ξ(k, params); -2*ξ(k, params) 0]
end

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