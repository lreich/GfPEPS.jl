"""
    energy_function(t::Real, Δ_x::Real, Δ_y::Real, μ::Real, Lx::Int, Ly::Int)
Return a function energy(BatchGout) computing the mean energy density.
"""

# TODO: different BCS cases implementieren
function energy_loss(t::Real, Δ_x::Real, Δ_y::Real, μ::Real, Lx::Int, Ly::Int)
    k_vals = get_2D_k_grid(Lx, Ly)

    ξk = -2t * (cos.(k_vals[:,1]) .+ cos.(k_vals[:,2])) .+ μ 
    Δk =  2 .* (Δ_x .* cos.(k_vals[:,1]) .+ Δ_y .* cos.(k_vals[:,2]))  

    cosk   = cos.(k_vals)                    # N×2
    batch_cosk  = vec(sum(cosk; dims=2))     # N
    batch_delta = Δ_x .* cosk[:, 1] .+ Δ_y .* cosk[:, 2]  # N

    J = [0.0 -1.0; 1.0 0.0]
    K = [0.0  1.0; 1.0 0.0]
    N = size(k_vals, 1)

    # make 2x2 kernels broadcast across batch dimension explicitly
    J3 = reshape(J, 1, size(J)... )   # 1×2×2
    K3 = reshape(K, 1, size(K)... )   # 1×2×2

    # TODO: energy convention verstehen
    # μ = - μ in the paper formula
    # weird ordering here
    function energy(CM_out::AbstractArray)
        # as less allocations as possible for zygote
        # return @views real(sum( ξk .* (1 .- 1/2 .* ( CM_out[:,1,3] .+ CM_out[:,2,4])) .+
        #                 batch_delta .* (CM_out[:,1,4] .+ CM_out[:,3,2]) ) ./ N)

        @views v13 = CM_out[:, 1, 3]
        @views v24 = CM_out[:, 2, 4]
        @views v14 = CM_out[:, 1, 4]
        @views v32 = CM_out[:, 3, 2]

        # algebraic rearrangement to avoid intermediate vectors:
        # sum( ξk .* (1 - 0.5*(v13+v24)) ) = sum(ξk) - 0.5*(dot(ξk,v13) + dot(ξk,v24))
        # sξ  = sum(ξk)
        # t1  = dot(ξk, v13)
        # t2  = dot(ξk, v24)
        # d1  = dot(batch_delta, v14)
        # d2  = dot(batch_delta, v32)

        @inbounds E = sum(ξk) - 0.5*(dot(ξk, v13) + dot(ξk, v24)) + (dot(batch_delta, v14) + dot(batch_delta, v32))
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
function optimize_loss(t::Real, Δ_x::Real, Δ_y::Real, μ::Real, Lx::Int, Ly::Int, Nf::Int, Nv::Int)
    G_in = G_in_Fourier(Lx, Ly, Nv)
    energy = energy_loss(t, Δ_x, Δ_y, μ, Lx, Ly)
    function loss(X)
        CM_out = GaussianMap(Γ_fiducial(X, Nv), G_in, Nf, Nv)
        return real(energy(CM_out))
    end
    return loss
end