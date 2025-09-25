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
        η = 0.25 .* (CM_out[:, 1, 4] .+ CM_out[:, 2, 3] .+ im .* (CM_out[:, 2, 4] .- CM_out[:, 1, 3]))
        @inbounds E = ξk_batched_summed - 0.5*(dot(ξk_batched, CM_out[:, 1, 2]) + dot(ξk_batched, CM_out[:, 3, 4])) + 2*real(dot(Δk_batched, η))
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
        return real(energy(GaussianMap(get_Γ_blocks(Γ_fiducial(X, Nv, Nf), Nf)..., G_in)))
        # return real(energy(GaussianMap(Γ_fiducial(X, Nv, Nf), G_in, Nf, Nv)[1:2,:,:]))
    end
    return loss
end

function energy_bcs(t::Real, μ::Real, k::AbstractVector, pairing_type::String, Δ_kwargs...)
    ξk = ξ(k, t, μ)
    Δk = Δ(pairing_type, k, Δ_kwargs...)

    function energy(CM_out::AbstractMatrix)
        η = 0.25 * (CM_out[1,4] + CM_out[2,3] + im * (CM_out[2,4] - CM_out[1,3]))

        return real(0.5 * ξk * (2 - CM_out[1,2] - CM_out[3,4]) + 2 * real(Δk * η))
    end

    return energy
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
function get_loss_function_bcs(t::Real, μ::Real, bz::BrillouinZone2D, Nf::Int, Nv::Int, pairing_type::String, Δ_kwargs...)
    G_in = G_in_Fourier(bz, Nv)
    invN = 1.0 / size(bz.kvals, 2) # actually faster when precomputed, because multiplication is faster than division

    ξk_batched = collect(ξ.(eachcol(bz.kvals),t,μ))
    Δk_batched = collect(Δ.(pairing_type, eachcol(bz.kvals),Δ_kwargs...))

    function loss(X)
        Γ = Γ_fiducial(X, Nv, Nf)
        A, B, D = get_Γ_blocks(Γ, Nf)

        s = 0.0
        @inbounds for i in 1:size(bz.kvals, 2)
            CM = @views GaussianMap_single_k(A,B,D, G_in[i, :, :])
            η = 0.25 * (CM[1,4] + CM[2,3] + im*(CM[2,4] - CM[1,3]))
            s += real(0.5*ξk_batched[i]*(2 - CM[1,2] - CM[3,4]) + 2*real(Δk_batched[i]*η))
        end
        return s * invN

        # return real(mean(map(k_ind -> begin
        #     energy_fct = energy_bcs(t, μ, k_ind[2], pairing_type, Δ_kwargs...)
        #     energy_fct(GaussianMap_single_k(A, B, D, G_in[k_ind[1], :, :]))
        # end, enumerate(eachcol(bz.kvals)))))
    end
end

# function get_loss_function_bcs(t::Real, μ::Real, bz::BrillouinZone2D, Nf::Int, Nv::Int, pairing_type::String, Δ_kwargs...)
#     G_in = G_in_Fourier(bz, Nv)

#     # precompute k-dependent scalars
#     k_vals = bz.kvals
#     invN = 1.0 / size(k_vals, 2) # actually faster when precomputed, because multiplication is faster than division

#     ξk_batched = collect(ξ.(eachcol(k_vals),t,μ))
#     Δk_batched = collect(Δ.(pairing_type, eachcol(k_vals),Δ_kwargs...))
#     ξk_batched_summed = sum(ξk_batched)

#     function loss(X)
#         Γ = Γ_fiducial(X, Nv, Nf)
#         # compute all output CMs in one call (vectorized / batched)
#         CM_outs = GaussianMap(Γ, G_in, Nf, Nv)   # size (Nk, 2Nf, 2Nf)

#         # vectorized extraction across k (avoids deep reverse-mode recursion)
#         G12 = view(CM_outs, :, 1, 2)
#         G34 = view(CM_outs, :, 3, 4)
#         G14 = view(CM_outs, :, 1, 4)
#         G23 = view(CM_outs, :, 2, 3)
#         G24 = view(CM_outs, :, 2, 4)
#         G13 = view(CM_outs, :, 1, 3)

#         η = 0.25 .* (G14 .+ G23 .+ im .* (G24 .- G13))
#         E = ξk_batched_summed - 0.5*(dot(ξk_batched, G12) + dot(ξk_batched, G34)) + 2 * real(dot(Δk_batched, η))
#         return real(E * invN)
#     end

#     return loss
# end