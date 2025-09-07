# Julia translation of Gaussian-fPEPS/src/gfpeps/loss.py


"""
    energy_function(; hoping=1.0, DeltaX=0.0, DeltaY=0.0, Mu=0.0, Lx=100, Ly=100)
Return a function energy(BatchGout) computing the mean energy density.
"""
function energy_function(; hoping=1.0, DeltaX=0.0, DeltaY=0.0, Mu=0.0, Lx=100, Ly=100)
    batch_k = BatchK(Lx, Ly)
    batch_cosk = sum(cos.(batch_k), dims=2)[:, 1]
    batch_delta = map(1:size(batch_k, 1)) do i
        dot(cos.(view(batch_k, i, :)), [DeltaX, DeltaY])
    end
    batch_delta = collect(batch_delta)

    function energy(BatchGout)
        # rhoup
        eps = [0.0 -1.0; 1.0 0.0]
        rhoup = 0.5 .+ 0.25 .* [sum(view(BatchGout, i, 1:2:4, 1:2:4) .* eps) for i in 1:size(BatchGout, 1)]
        rhodn = 0.5 .+ 0.25 .* [sum(view(BatchGout, i, 2:2:4, 2:2:4) .* eps) for i in 1:size(BatchGout, 1)]
        rho = rhoup .+ rhodn
        # kappa
        s2 = [0.0 1.0; 1.0 0.0]
        kappa = 0.25 .* [sum(view(BatchGout, i, 1:2:4, 2:2:4) .* s2) for i in 1:size(BatchGout, 1)]
        return mean(-2 .* hoping .* rho .* batch_cosk .+ 4 .* batch_delta .* kappa .+ Mu .* rho)
    end
    return energy
end

"""
    optimize_runtime_loss(; Lx=100, Ly=100, Nv=2, hoping=1.0, DeltaX=0.0, DeltaY=0.0, Mu=0.0)
Return lossT(T) closure that builds Γ_local and maps to energy using GaussianLinearMap.
"""
function optimize_runtime_loss(; Lx=100, Ly=100, Nv=2, hoping=1.0, DeltaX=0.0, DeltaY=0.0, Mu=0.0)
    BatchGin = BatchGammaIn(Lx, Ly, Nv)
    energy = energy_function(; hoping=hoping, DeltaX=DeltaX, DeltaY=DeltaY, Mu=Mu, Lx=Lx, Ly=Ly)
    function lossT(T)
        Glocal = getGammaProjector(T, Nv)
        BatchGout = GaussianLinearMap(Glocal, BatchGin)
        return real(energy(BatchGout))
    end
    return lossT
end
