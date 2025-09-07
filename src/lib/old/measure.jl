# Julia translation of Gaussian-fPEPS/src/gfpeps/measure.py


"""
    measure(cfg, T)
Return (rhoup, rhodn, kappa) each as Lx x Ly arrays based on Γ_out.
`cfg` must provide fields: lattice.Lx, lattice.Ly, params.Nv.
"""
function measure(cfg, T)
    Lx = cfg.lattice.Lx
    Ly = cfg.lattice.Ly
    Nv = cfg.params.Nv
    BatchGin = BatchGammaIn(Lx, Ly, Nv)
    Glocal = getGammaProjector(T, Nv)
    BatchGout = GaussianLinearMap(Glocal, BatchGin)
    eps = [0.0 -1.0; 1.0 0.0]
    s2 = [0.0 1.0; 1.0 0.0]
    n = size(BatchGout, 1)
    rhoup = [0.5 + 0.25 * sum(view(BatchGout, i, 1:2:4, 1:2:4) .* eps) for i in 1:n]
    rhodn = [0.5 + 0.25 * sum(view(BatchGout, i, 2:2:4, 2:2:4) .* eps) for i in 1:n]
    kappa = [0.25 * sum(view(BatchGout, i, 1:2:4, 2:2:4) .* s2) for i in 1:n]
    rhoup = reshape(rhoup, Lx, Ly)
    rhodn = reshape(rhodn, Lx, Ly)
    kappa = reshape(kappa, Lx, Ly)
    return rhoup, rhodn, kappa
end
