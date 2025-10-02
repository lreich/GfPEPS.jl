function get_X_opt(Nf::Int, Nv::Int, t::Real, μ::Real, pairing_type::String, Δ_0::Real;
    δ::Float64 = 0.0,
    solve_μ_from_δ::Bool = false,
    λ::Float64 = 1e2, # lagrange multiplier for hole density
    Lx::Int = 5, 
    Ly::Int = 5,
    bc::Tuple{Symbol, Symbol}=(:APBC, :PBC),
    parity::Int = 1, # 1 for even, -1 for odd
    maxiter::Int=500,
    grad_tol::Float64=1.0e-8,
    f_reltol::Float64=1.0e-10,
    verbose::Bool=true,
    seed::Int=1234)

    Random.seed!(seed)

    # construct Brillouin zone
    bz = BrillouinZone2D(Lx,Ly,bc)
    
    Lx_init = isodd(Lx) ? 5 : 6
    Ly_init = isodd(Ly) ? 5 : 6
    bz_init = BrillouinZone2D(Lx_init,Ly_init,bc)

    # initial ortogonal matrix X to construct Γ_out with correct parity sector (even)
    _, X = rand_CM(Nf,Nv; parity=parity)

    if(solve_μ_from_δ)
        μ = solve_for_mu(bz,δ,t,pairing_type,Δ_0)
    end

    has_dirac_points(bz,t,μ,pairing_type,Δ_0) # warn if dirac points are present
    has_dirac_points(bz_init,t,μ,pairing_type,Δ_0) # warn if dirac points are present

    # build loss function
    loss_init_no_dens = optimize_loss(t, μ, bz_init, Nf, Nv, pairing_type, Δ_0)
    loss_no_dens = optimize_loss(t, μ, bz, Nf, Nv, pairing_type, Δ_0)

    # augmented loss that penalizes deviation from target hole density δ
    loss_init_dens(X) = begin
        return loss_init_no_dens(X) + λ * (doping_bcs(X, bz_init, Nf, Nv) - δ)^2
    end
    loss_dens(X) = begin
        return loss_no_dens(X) + λ * (doping_bcs(X, bz, Nf, Nv) - δ)^2
    end
    loss_init = solve_μ_from_δ ? loss_init_dens : loss_init_no_dens
    loss = solve_μ_from_δ ? loss_dens : loss_no_dens

    # build gradients
    g_init(x) = first(Zygote.gradient(loss_init, x))
    g_init!(G,x) = copyto!(G, g_init(x)) # better for optim
    g(x) = first(Zygote.gradient(loss, x))
    g!(G,x) = copyto!(G, g(x)) # better for optim

    # First, find a better initial guess for X by solving for smaller system sizes (see: 10.1103/PhysRevLett.129.206401) 
    @info "Finding better initial guess for X by solving smaller system sizes..."
    res_init = Optim.optimize(loss_init, g_init!, X, Optim.LBFGS(;m=20,manifold=Optim.Stiefel()), Optim.Options(
        iterations = maxiter,
        g_tol = grad_tol,
        show_trace = verbose,
        successive_f_tol = 10,
        f_reltol = f_reltol
    ))

    @info "Finding optimal X for full system size..."
    # optimize X for the full system size
    res = Optim.optimize(loss, g!, Optim.minimizer(res_init), Optim.LBFGS(;m=20,manifold=Optim.Stiefel()), Optim.Options(
        iterations = maxiter,
        g_tol = grad_tol,
        show_trace = verbose,
        successive_f_tol = 10,
        f_reltol = f_reltol
    ))

    @show Optim.minimum(res)
    exact_energy = exact_energy_BCS_k(bz,t,μ,pairing_type,Δ_0)
    println("Exact energy: $exact_energy")

    return Optim.minimizer(res), Optim.minimum(res), exact_energy
end

get_X_opt(;conf::Dict=parsefile(joinpath(GfPEPS.config_path, "conf_BCS_d_wave.json"))) = get_X_opt(
    conf["params"]["N_physical_fermions_on_site"],
    conf["params"]["N_virtual_fermions_on_bond"],
    conf["hamiltonian"]["t"],
    conf["hamiltonian"]["μ"],
    conf["hamiltonian"]["pairing_type"],
    conf["hamiltonian"]["Δ_0"];
    δ = conf["hamiltonian"]["hole_density"],
    solve_μ_from_δ = conf["hamiltonian"]["μ_from_hole_density"],
    λ = conf["hamiltonian"]["lagrange_multiplier_density"],
    Lx = conf["system_params"]["Lx"],
    Ly = conf["system_params"]["Ly"],
    bc = (Symbol(conf["system_params"]["x_bc"]), Symbol(conf["system_params"]["y_bc"])),
    parity = conf["system_params"]["parity"],
    maxiter = conf["params"]["maxiter"],
    grad_tol = conf["params"]["grad_tol"],
    f_reltol = conf["params"]["f_reltol"],
    verbose = conf["params"]["show_trace"],
    seed = conf["params"]["seed"]
)
