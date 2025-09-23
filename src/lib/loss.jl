"""
    energy_loss(t::Real, Δ_x::Real, Δ_y::Real, μ::Real, bz::BrillouinZone2D)
Return a function energy(BatchGout) computing the mean energy density.

    Note: ony for Nf=2 BCS Hamiltonian
"""
function energy_loss(t::Real, μ::Real, bz::BrillouinZone2D, pairing_type::String, Δ_kwargs...)
    k_vals = bz.kvals

    ξk_batched = collect(ξ.(eachcol(k_vals),t,μ))
    Δk_batched = collect(Δ.(pairing_type, eachcol(k_vals),Δ_kwargs...))
    ξk_batched_summed = sum(ξk_batched)

    # divide by number of k-points
    invN = 1.0 / size(k_vals, 2) # actually faster when precomputed, because multiplication is faster than division

    function energy(CM_out::AbstractArray)
        # #= 
        #     qp-ordering of Majorana modes: (c_1, c_3, ..., c_(2(4Nv + Nf)-1), c_2, c_4, ..., c_(2(4Nv + Nf)))
        # =#
        # G13, G24, G14, G32, G34, G12 = CM_out[:, 1, 3], CM_out[:, 2, 4], CM_out[:, 1, 4], CM_out[:, 3, 2], CM_out[:, 3, 4], CM_out[:, 1, 2]
        # η = 0.25 .* (G14 .+ G32 .+ im .* (G34 .- G12))
        # @inbounds E = ξk_batched_summed - 0.5*(dot(ξk_batched, G13) + dot(ξk_batched, G24)) + 2*real(dot(Δk_batched, η))
        # return real(E * invN)

        #= 
            qq-ordering of Majorana modes: (c_1, c_2, ..., c_(2(4Nv + Nf)))
        =#
        G12, G34, G14, G23, G24, G13 = CM_out[:, 1, 2], CM_out[:, 3, 4], CM_out[:, 1, 4], CM_out[:, 2, 3], CM_out[:, 2, 4], CM_out[:, 1, 3]
        η = 0.25 .* (G14 .+ G23 .+ im .* (G24 .- G13))
        @inbounds E = ξk_batched_summed - 0.5*(dot(ξk_batched, G12) + dot(ξk_batched, G34)) + 2*real(dot(Δk_batched, η))
        return real(E * invN)
    end

    return energy
end

# function energy_loss2(t::Real, μ::Real, bz::BrillouinZone2D, pairing_type::String, Δ_kwargs...)
#     k_vals = bz.kvals

#     ξk_batched = collect(ξ.(eachcol(k_vals),t,μ))
#     Δk_batched = collect(Δ.(pairing_type, eachcol(k_vals),Δ_kwargs...))
#     ξk_batched_summed = sum(ξk_batched)

#     # divide by number of k-points
#     invN = 1.0 / size(k_vals, 2) # actually faster when precomputed, because multiplication is faster than division

#     function energy(CM_out::AbstractMatrix)
#         # #= 
#         #     qp-ordering of Majorana modes: (c_1, c_3, ..., c_(2(4Nv + Nf)-1), c_2, c_4, ..., c_(2(4Nv + Nf)))
#         # =#
#         # G13, G24, G14, G32, G34, G12 = CM_out[:, 1, 3], CM_out[:, 2, 4], CM_out[:, 1, 4], CM_out[:, 3, 2], CM_out[:, 3, 4], CM_out[:, 1, 2]
#         # η = 0.25 .* (G14 .+ G32 .+ im .* (G34 .- G12))
#         # @inbounds E = ξk_batched_summed - 0.5*(dot(ξk_batched, G13) + dot(ξk_batched, G24)) + 2*real(dot(Δk_batched, η))
#         # return real(E * invN)

#         return 

#         #= 
#             qq-ordering of Majorana modes: (c_1, c_2, ..., c_(2(4Nv + Nf)))
#         =#
#         G12, G34, G14, G23, G24, G13 = CM_out[:, 1, 2], CM_out[:, 3, 4], CM_out[:, 1, 4], CM_out[:, 2, 3], CM_out[:, 2, 4], CM_out[:, 1, 3]
#         η = 0.25 .* (G14 .+ G23 .+ im .* (G24 .- G13))
#         @inbounds E = ξk_batched_summed - 0.5*(dot(ξk_batched, G12) + dot(ξk_batched, G34)) + 2*real(dot(Δk_batched, η))
#         return real(E * invN)
#     end

#     return energy
# end

"""
    optimize_loss(t::Real, Δ_x::Real, Δ_y::Real, μ::Real, Lx::Int, Ly::Int, Nf::Int, Nv::Int)

Returns the energy from the CM_out as a function of the orthogonal matrix X, obtained from the minimization, using the Gaussian map.
"""
function optimize_loss(t::Real, μ::Real, bz::BrillouinZone2D, Nf::Int, Nv::Int, pairing_type::String, Δ_kwargs...)
    G_in = G_in_Fourier(bz, Nv)
    energy = energy_loss(t, μ, bz, pairing_type, Δ_kwargs...)
    # energy = energy_loss2(t, μ, bz, pairing_type, Δ_kwargs...)
    function loss(X)
        # CM_out = GaussianMap(Γ_fiducial(X, Nv), G_in, Nf, Nv)
        return real(energy(GaussianMap(Γ_fiducial(X, Nv, Nf), G_in, Nf, Nv)))

        # for (k,i) in enumerate(eachcol(bz.kvals))
        #     G_in_k = G_in_single_k(k, Nv)
        # end
    end
    return loss
end

"""
    optimize_loss_per_k(t::Real, μ::Real, bz::BrillouinZone2D, Nf::Int, Nv::Int, pairing_type::String, Δ_kwargs...)

Compute the loss by looping over k:
- For each k, build G_in_single_k(k, Nv)
- Contract with GaussianMap_single_k
- Accumulate the energy contribution for that k
- Return the mean over all k

Note: assumes Nf=2 (uses the first 4 Majorana modes for the physical sector, qq-ordering).
"""
function optimize_loss_per_k(t::Real, μ::Real, bz::BrillouinZone2D, Nf::Int, Nv::Int, pairing_type::String, Δ_kwargs...)
    k_vals = bz.kvals
    Nk = size(k_vals, 2)
    invN = 1.0 / Nk

    # Precompute ξ(k) and Δ(k) once
    ξk_vec = collect(ξ.(eachcol(k_vals), t, μ))
    Δk_vec = collect(Δ.(pairing_type, eachcol(k_vals), Δ_kwargs...))

    function loss(X)
        Γ = Γ_fiducial(X, Nv, Nf)
        # get block matrices from CM_out (=Γ_fiducial)
        A = Γ[1:2*Nf, 1:2*Nf]
        B = Γ[1:2*Nf, 2*Nf+1:end]
        D = Γ[2*Nf+1:end, 2*Nf+1:end]


        # Reuse a single buffer for G_in(k)
        Gin = Matrix{ComplexF64}(undef, 8Nv, 8Nv)

        acc = 0.0
        @inbounds for (i, k) in enumerate(eachcol(k_vals))
            G_in_single_k!(Gin, k, Nv)

            # Build G_in for this k and contract with the map
            CM_out = GaussianMap_single_k(A, B, D, Gin, Nf, Nv)

            η = 0.25 * (CM_out[1,4] + CM_out[2,3] + im * (CM_out[2,4] - CM_out[1,3]))

            ξk = ξk_vec[i]
            Δk = Δk_vec[i]

            # Per-k contribution (take real part for numerical safety)
            e_k = 0.5 * ξk * (2 - CM_out[1,2] - CM_out[3,4]) + 2 * real(Δk * η)
            acc += real(e_k)
        end

        return acc * invN
    end

    return loss
end

"""
    get_loss_function(energy_fct::Function, bz::BrillouinZone2D, Nf::Int, Nv::Int)

Returns the energy from the output state as a function of the orthogonal matrix X, obtained from the minimization, using the Gaussian map.

Keyword arguments:
- `energy_fct::Function`: function that computes the energy from the output CM
- `bz::BrillouinZone2D`: Brillouin zone
- `Nf::Int`: number of physical fermions
- `Nv::Int`: number of virtual fermions
"""
function get_loss_function(energy_fct::Function, bz::BrillouinZone2D, Nf::Int, Nv::Int)
    G_in = G_in_Fourier(bz, Nv)

    function loss(X)
        return real(energy_fct(GaussianMap(Γ_fiducial(X, Nv, Nf), G_in, Nf, Nv)))
    end
end
