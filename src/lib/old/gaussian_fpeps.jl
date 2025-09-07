# Julia translation of Gaussian-fPEPS/src/gfpeps/gaussian_fpeps.py
# Manifold optimization over Stiefel using a simple projected gradient with QR retraction.

using LinearAlgebra
using Random
using Manifolds
using ForwardDiff

# Relies on: loss.jl, loadwrite.jl, exact.jl, deltatomu.jl, measure.jl

"""
    gaussian_fpeps(cfg)

Optimize the orthogonal transformer T on the Stiefel manifold O(n) with a
projected gradient descent using QR-based retraction.

Required cfg fields:
- cfg.params.Nv::Int, cfg.params.seed::Int
- cfg.lattice.Lx::Int, cfg.lattice.Ly::Int
- cfg.file.LoadFile::Union{Nothing,String}, cfg.file.WriteFile::Union{Nothing,String}, cfg.file.SaveEachSteps::Bool
- cfg.hamiltonian.ht, .DeltaX, .DeltaY, .delta, .Mu, .solve_mu_from_delta::Bool
- cfg.optimizer.MaxIter::Int, cfg.optimizer.lr::Real (learning rate), cfg.optimizer.tol::Real (grad tol)
- cfg.backend (ignored here; CPU default)
"""
function gaussian_fpeps(cfg)
    # unpack cfg
    Random.seed!(cfg.params.seed)
    Nv = cfg.params.Nv
    Lx, Ly = cfg.lattice.Lx, cfg.lattice.Ly
    LoadKey, WriteKey = cfg.file.LoadFile, cfg.file.WriteFile

    cfgh = cfg.hamiltonian
    ht = cfgh.ht
    DeltaX, DeltaY = cfgh.DeltaX, cfgh.DeltaY
    delta, Mu = cfgh.delta, cfgh.Mu

    Tsize = 8 * Nv + 4
    T = initialT(LoadKey, Tsize)
    # Polar/orthonormalize via SVD (like U @ V^T); ensures T ∈ O(n)
    U, S, V = svd(T)
    T = U * transpose(V)

    if cfgh.solve_mu_from_delta
        @info "Overwrite Origin Mu"
        Mu = solve_mu(DeltaX, delta)
    end

    # build loss closure
    lossT = optimize_runtime_loss(; Nv=Nv, Lx=Lx, Ly=Ly, hoping=ht, DeltaX=DeltaX, DeltaY=DeltaY, Mu=Mu)

    Eg = eg(Lx, Ly, ht, DeltaX, DeltaY, Mu)
    @info "Eg = $(Eg)"

    # Stiefel manifold for orthogonal matrices
    M = Stiefel(Tsize, Tsize)
    proj_tangent(T, G) = G - T * Symmetric(transpose(T) * G)

    # AD-based gradient via ForwardDiff on a vectorized parameterization
    function grad_forwarddiff(f, X)
        n, m = size(X)
        fx(v) = f(reshape(v, n, m))
        g = ForwardDiff.gradient(fx, vec(X))
        return reshape(g, n, m)
    end

    # Optimization loop (ConjugateGradient substitute)
    maxiter = hasproperty(cfg, :optimizer) && hasproperty(cfg.optimizer, :MaxIter) ? cfg.optimizer.MaxIter : 400
    lr = hasproperty(cfg, :optimizer) && hasproperty(cfg.optimizer, :lr) ? cfg.optimizer.lr : 0.2
    tol = hasproperty(cfg, :optimizer) && hasproperty(cfg.optimizer, :tol) ? cfg.optimizer.tol : 1e-6

    log_cost = Float64[]
    log_gnorm = Float64[]

    for it in 1:maxiter
        fT = lossT(T)
        push!(log_cost, fT)
        # Euclidean gradient via AD
        G = grad_forwarddiff(lossT, T)
        GT = proj_tangent(T, G)
        gnorm = norm(GT)
        push!(log_gnorm, gnorm)

        @info "iter=$(it) cost=$(fT) gnorm=$(gnorm)"
        if gnorm < tol
            break
        end
        # Backtracking line search (Armijo) on manifold with retraction
        α = lr
        c = 1e-4
        ρ = 0.5
        inner = sum(GT .* (-GT))
        Tnew = retract(M, T, -α * GT)
        fnew = lossT(Tnew)
        while fnew > fT + c * α * inner
            α *= ρ
            # guard against too small step
            if α < 1e-8
                break
            end
            Tnew = retract(M, T, -α * GT)
            fnew = lossT(Tnew)
        end
        T = Tnew

        if cfg.file.SaveEachSteps && WriteKey !== nothing
            Xopt = T
            args = Dict("Mu"=>Mu,"DeltaX"=>DeltaX,"DeltaY"=>DeltaY,"delta"=>delta,
                        "ht"=>ht,"Lx"=>Lx,"Ly"=>Ly,"Nv"=>Nv,"seed"=>cfg.params.seed)
            savelog_trivial(string(WriteKey[1:end-3], "-iter", it, WriteKey[end-2:end]),
                            Xopt, lossT(Xopt), Eg, args, measure(cfg, Xopt))
        end
    end

    Xopt = T
    args = Dict("Mu"=>Mu,"DeltaX"=>DeltaX,"DeltaY"=>DeltaY,"delta"=>delta,
                "ht"=>ht,"Lx"=>Lx,"Ly"=>Ly,"Nv"=>Nv,"seed"=>cfg.params.seed)
    savelog_trivial(WriteKey, Xopt, lossT(Xopt), Eg, args, measure(cfg, Xopt))
    return Xopt
end
