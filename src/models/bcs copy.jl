# using Roots
# using TensorKit
# using PEPSKit
# using PEPSKit: nearest_neighbours
# using Statistics: mean
# import TensorKitTensors.HubbardOperators as hub
# import TensorKitTensors.TJOperators as tJ
# using ..GfPEPS
# using ..GfPEPS: _is_xbond
# using ..GfPEPS: cormat_blocks, cormat_virtual

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
function hamiltonian(
        T::Type{<:Number}, lattice::InfiniteSquare; t::Float64 = 1.0,
        Δx::Float64 = 0.5, Δy::Float64 = -0.5, mu::Float64 = 0.0
    )
    pspace = hub.hubbard_space(Trivial, Trivial)
    pspaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    num = hub.e_num(T, Trivial, Trivial)
    unit = TensorKit.id(T, pspace)
    hopping = (-t) * hub.e_hopping(T, Trivial, Trivial) -
        (mu / 4) * (num ⊗ unit + unit ⊗ num)
    pairing = sqrt(2) * hub.singlet_plus(T, Trivial, Trivial)
    pairing += pairing'
    return LocalOperator(
        pspaces,
        map(nearest_neighbours(lattice)) do bond
            return bond => hopping + pairing * (_is_xbond(bond) ? Δx : Δy)
        end...
    )
end
hamiltonian(lattice; t, Δx, Δy, mu) = hamiltonian(ComplexF64, lattice; t, Δx, Δy, mu)

function cal_xi(k::Vector{Float64}; t::Float64, mu::Float64)
    return -2t * (cos(2π * k[1]) + cos(2π * k[2])) - mu
end

function cal_Delta(k::Vector{Float64}; Δx::Float64, Δy::Float64)
    return 2Δx * cos(2π * k[1]) + 2Δy * cos(2π * k[2])
end

function cal_E(
        k::Vector{Float64}; t::Float64 = 1.0,
        Δx::Float64 = 0.5, Δy::Float64 = -0.5, mu::Float64 = 0.0
    )
    ξ = cal_xi(k; t, mu)
    Δ = cal_Delta(k; Δx, Δy)
    return sqrt(ξ^2 + Δ^2)
end

# ---- exact values of observables from Hamiltonian ----

"""
The exact ground state energy per site of a BCS mean field Hamiltonian
on a finite lattice specified by the BrillouinZone `bz`.
"""
function energy_exact(
        bz::BrillouinZone;
        t::Float64 = 1.0, Δx::Float64 = 0.5, Δy::Float64 = -0.5, mu::Float64 = 0.0
    )
    return mean(
        for k in eachcol(bz.kvals)
            cal_xi(k; t, mu) - cal_E(k; t, Δx, Δy, mu)
        end
    )
end

"""
Calculate doping for a given `mu`
"""
function doping_exact(
        bz::BrillouinZone; t::Float64 = 1.0,
        Δx::Float64 = 0.5, Δy::Float64 = -0.5, mu::Float64 = 0.0
    )
    return mean(
        for i in 1:size(bz.kvals, 2)
            k = bz.kvals[:, i]
            ξ = cal_xi(k; t, mu)
            Δ = cal_Delta(k; Δx, Δy)
            E = sqrt(ξ^2 + Δ^2)
            return ξ / E
        end
    )
end

"""
Solve for the value of `mu` at a given doping `δ`
"""
function solve_mu(
        bz::BrillouinZone, δ::Float64;
        mu_range::NTuple{2, Float64} = (-5.0, 5.0),
        t::Float64 = 1.0, Δx::Float64 = 0.5, Δy::Float64 = -0.5
    )
    mu = find_zero(x -> δ - doping_exact(bz; t, Δx, Δy, mu = x), mu_range)
    return mu
end

# ---- values of observables from fPEPS correlation matrix ----

"""
The energy of a Gaussian fPEPS evaluated from 
the fiducial state correlation matrix `G`.
"""
function energy_peps(
        G::AbstractMatrix, bz::BrillouinZone, Np::Int;
        t::Float64 = 1.0, Δx::Float64 = 0.5, Δy::Float64 = -0.5, mu::Float64 = 0.0
    )
    A, B, D = cormat_blocks(G, Np)
    χ = div(size(G, 1) - 2 * Np, 8)
    return mean(
        for i in 1:size(bz.kvals, 2)
            k = bz.kvals[:, i]
            Gω = cormat_virtual(k, χ)
            Gf = A + B * inv(D + Gω) * transpose(B)
            ξ = cal_xi(k; t, mu)
            Δ = cal_Delta(k; Δx, Δy)
            return real(
                ξ * (2 - Gf[1, 2] - Gf[3, 4]) / 2 -
                    Δ * (Gf[4, 1] + Gf[3, 2] + 1.0im * (Gf[4, 2] - Gf[3, 1])) / 2
            )
        end
    )
end

"""
The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the fiducial state correlation matrix `G`.
"""
function doping_peps(G::AbstractMatrix, bz::BrillouinZone, Np::Int)
    A, B, D = cormat_blocks(G, Np)
    χ = div(size(G, 1) - 2 * Np, 8)
    return mean(
        for i in 1:size(bz.kvals, 2)
            k = bz.kvals[:, i]
            Gω = cormat_virtual(k, χ)
            Gf = A + B * inv(D + Gω) * transpose(B)
            return real(Gf[1, 2] + Gf[3, 4]) / 2
        end
    )
end

"""
The average magnetization `<Sᵃ> = (1/2N) ∑_i σᵃ_{αβ} ⟨f†_{iα} f_{iβ}⟩`
evaluated from the fiducial state correlation matrix `G`.
"""
function mags_peps(G::AbstractMatrix, bz::BrillouinZone, Np::Int)
    A, B, D = cormat_blocks(G, Np)
    χ = div(size(G, 1) - 2 * Np, 8)
    N = bz.Lx * bz.Ly
    mags = zeros(Float64, 3)
    for i in 1:size(bz.kvals, 2)
        k = bz.kvals[:, i]
        Gω = cormat_virtual(k, χ)
        Gf = A + B * inv(D + Gω) * transpose(B)
        mags[1] += real(Gf[4, 1] + Gf[2, 3]) / (4N)
        mags[2] += real(Gf[3, 1] + Gf[4, 2]) / (4N)
        mags[3] += real(Gf[2, 1] + Gf[3, 4]) / (4N)
    end
    return mags
end

"""
The average hopping `T_{i,i+v} = (1/N) ∑_i ⟨f†_{iσ} f_{i+v,σ}⟩ + c.c.`
evaluated from the fiducial state correlation matrix `G`.

Note: the y-direction of `v` is opposite to the direction of increasing rows.
"""
function hopping_peps(G::AbstractMatrix, bz::BrillouinZone, Np::Int, v::Vector{Int})
    @assert length(v) == 2
    A, B, D = cormat_blocks(G, Np)
    χ = div(size(G, 1) - 2 * Np, 8)
    return mean(
        for i in 1:size(bz.kvals, 2)
            k = bz.kvals[:, i]
            Gω = cormat_virtual(k, χ)
            Gf = A + B * inv(D + Gω) * transpose(B)
            return real(2 - Gf[1, 2] - Gf[3, 4]) * cos(2π * k' * v)
        end
    )
end

"""
The singlet pairing `Δ_{i,i+v} = (1/√2N) ∑_i ϵ_{αβ} ⟨f_{iα} f_{i+v,β}⟩`

Note: the y-direction of `v` is opposite to the direction of increasing rows.
"""
function singlet_peps(G::AbstractMatrix, bz::BrillouinZone, Np::Int, v::Vector{Int})
    @assert length(v) == 2
    A, B, D = cormat_blocks(G, Np)
    χ = div(size(G, 1) - 2 * Np, 8)
    return mean(
        for i in 1:size(bz.kvals, 2)
            k = bz.kvals[:, i]
            Gω = cormat_virtual(k, χ)
            Gf = A + B * inv(D + Gω) * transpose(B)
            return -sqrt(2) / 4 * cos(2π * k' * v) *
                (Gf[1, 4] + Gf[2, 3] + 1.0im * (Gf[1, 3] - Gf[2, 4]))
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
    P = zeros(Float64, V_hub → V_tJ)
    S = FermionParity
    P[(S(0), S(0))][1, 1] = sqrt(z)
    P[(S(1), S(1))][1, 1] = 1.0
    P[(S(1), S(1))][2, 2] = 1.0
    return P
end

"""
Apply Gutzwiller projection to Hubbard (spin-1/2) PEPS
"""
function gutzwiller_project(z::Float64, peps::InfinitePEPS)
    P = gutzwiller_projector(z)
    return InfinitePEPS(collect(P * t for t in peps.A))
end
