"""
    energy_loss(t::Real, Δ_x::Real, Δ_y::Real, μ::Real, bz::BrillouinZone2D)
Return a function energy(BatchGout) computing the mean energy density.
"""
function energy_loss(t::Real, μ::Real, bz::BrillouinZone2D, pairing_type::String, Δ_kwargs...)
    k_vals = bz.kvals

    ξk_batched = ξ.(eachcol(k_vals),t,μ)
    Δk_batched = Δ.(pairing_type, eachcol(k_vals),Δ_kwargs...)

    N = size(k_vals, 2)  # number of k-points

    # TODO: energy convention verstehen
    # μ = - μ in the paper formula
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
        @views G13 = CM_out[:, 1, 3]
        @views G24 = CM_out[:, 2, 4]
        @views G14 = CM_out[:, 1, 4]
        @views G32 = CM_out[:, 3, 2]

        # @inbounds E = sum(ξk) - 0.5*(dot(ξk, G13) + dot(ξk, G24)) + 1/2 * (dot(Δk, G14) + dot(Δk, G32))
        @inbounds E = sum(ξk_batched) - 0.5*(dot(ξk_batched, G13) + dot(ξk_batched, G24)) + 1/2 * (dot(Δk_batched, G14) + dot(Δk_batched, G32))
        return real(E / N)

        # # expect CM_out :: N×4×4 (not Vector{Matrix})
        # @views up = CM_out[:, [1,3], [1,3]]
        # @views dn = CM_out[:, [2,4], [2,4]]
        # @views ud = CM_out[:, [1,3], [2,4]]

        # # Frobenius inner products per batch slice -> vectors length N
        # rhoup = 0.5 .+ 0.25 .* vec(sum(up .* J3, dims=(2,3)))
        # rhodn = 0.5 .+ 0.25 .* vec(sum(dn .* J3, dims=(2,3)))
        # rho   = rhoup .+ rhodn
        # kappa = 0.25 .* vec(sum(ud .* K3, dims=(2,3)))

        # E = -2 .* t .* rho .* batch_cosk .+ 4 .* batch_delta .* kappa .+ μ .* rho
        # return real(sum(E) / N)
    end
    # function energy(CM_out::AbstractArray)
        # up = @view CM_out[:, [1,3], [1,3]]
        # dn = @view CM_out[:, [2,4], [2,4]]
        # ud = @view CM_out[:, [1,3], [2,4]]

        # # Frobenius inner products per batch slice
        # rhoup = 0.5 .+ 0.25 .* map(s -> sum(s .* J), eachslice(up; dims=1))
        # rhodn = 0.5 .+ 0.25 .* map(s -> sum(s .* J), eachslice(dn; dims=1))

        # rho   = rhoup .+ rhodn
        # kappa = 0.25 .* map(s -> sum(s .* K), eachslice(ud; dims=1))

        # E = -2 .* t .* rho .* batch_cosk .+ 4 .* batch_delta .* kappa .+ μ .* rho
        # return real(sum(E) / N)
    # end
    # function energy(CM_out::AbstractArray)
        # CM_out :: N×4×4
        # E = 0
        # for i in 1:Lx*Ly
        #     E +=  -2t * batch_cosk[i] * 1/2 * (1 - CM_out[i][1,3])
        #     E +=  -2t * batch_cosk[i] * 1/2 * (1 - CM_out[i][2,4])
        #     E +=  batch_delta[i] * (CM_out[i][1,4] + CM_out[i][3,2])
        #     E +=  -μ * 1/2 * (CM_out[i][1,3] + CM_out[i][2,4]) + μ
        # end
        # return real(E / N)
    # end

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
        return real(energy(GaussianMap(Γ_fiducial(X, Nv), G_in, Nf, Nv)))
    end
    return loss
end