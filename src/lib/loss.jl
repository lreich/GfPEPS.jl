"""
    energy_loss(t::Real, Δ_x::Real, Δ_y::Real, μ::Real, bz::BrillouinZone2D)
Return a function energy(BatchGout) computing the mean energy density.
"""
function energy_loss(t::Real, μ::Real, bz::BrillouinZone2D, pairing_type::String, Δ_kwargs...)
    k_vals = bz.kvals

    ξk_batched = collect(ξ.(eachcol(k_vals),t,μ))
    Δk_batched = collect(Δ.(pairing_type, eachcol(k_vals),Δ_kwargs...))
    ξk_batched_summed = sum(ξk_batched)

    # divide by number of k-points
    invN = 1.0 / size(k_vals, 2) # actually faster when precomputed as multiplication is faster than division

    # TODO: energy convention verstehen
    # weird ordering here
    function energy(CM_out::AbstractArray)
        # as less allocations as possible for zygote
        # return @views real(sum( ξk .* (1 .- 1/2 .* ( CM_out[:,1,3] .+ CM_out[:,2,4])) .+
        #                 batch_delta .* (CM_out[:,1,4] .+ CM_out[:,3,2]) ) ./ N)

        # The following implementations are based on the exact formulas from the user-provided paper,
        # translated into the code's specific basis ordering.
        # Deduced Code Basis: (c_k,↑, c_k,↓, c†_{-k,↑}, c†_{-k,↓})
        # Paper's Basis (assumed): (c_k,↑, c†_{-k,↑}, c_k,↓, c†_{-k,↓})
        # Density calculation using the paper's formula translated to the code's basis:
        # Paper: n_up = 0.5 - 0.5 * G_paper[1,2]; n_down = 0.5 - 0.5 * G_paper[3,4]
        # Translated: n_up = 0.5 - 0.5 * G_code[1,3]; n_down = 0.5 - 0.5 * G_code[2,4]
        # Pairing correlator (kappa) using the paper's formula translated to the code's basis:
        # Paper: κ = 0.25 * [G_paper[1,4] + G_paper[2,3] + i*(G_paper[2,4] - G_paper[1,3])]
        # Translated: κ = 0.25 * [G_code[1,4] + G_code[3,2] + i*(G_code[3,4] - G_code[1,2])]
        
        
        # @views G13, G24, G14, G32 = CM_out[:, 1, 3], CM_out[:, 2, 4], CM_out[:, 1, 4], CM_out[:, 3, 2]
        # @inbounds E = ξk_batched_summed - 0.5*(dot(ξk_batched, G13) + dot(ξk_batched, G24)) + 0.5*(dot(Δk_batched, G14) + dot(Δk_batched, G32))
        G13, G24, G14, G32 = CM_out[:, 1, 3], CM_out[:, 2, 4], CM_out[:, 1, 4], CM_out[:, 3, 2]
        @inbounds E = ξk_batched_summed - 0.5*(dot(ξk_batched, G13) + dot(ξk_batched, G24)) + 0.5*(dot(Δk_batched, G14) + dot(Δk_batched, G32))
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