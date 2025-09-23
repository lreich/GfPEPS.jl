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

"""
    optimize_loss(t::Real, Δ_x::Real, Δ_y::Real, μ::Real, Lx::Int, Ly::Int, Nf::Int, Nv::Int)

Returns the energy from the CM_out as a function of the orthogonal matrix X, obtained from the minimization, using the Gaussian map.
"""
function optimize_loss(t::Real, μ::Real, bz::BrillouinZone2D, Nf::Int, Nv::Int, pairing_type::String, Δ_kwargs...)
    G_in = G_in_Fourier(bz, Nv)
    energy = energy_loss(t, μ, bz, pairing_type, Δ_kwargs...)
    function loss(X)
        # CM_out = GaussianMap(Γ_fiducial(X, Nv), G_in, Nf, Nv)
        return real(energy(GaussianMap(Γ_fiducial(X, Nv, Nf), G_in, Nf, Nv)))
    end
    return loss
end