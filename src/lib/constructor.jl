mutable struct Gaussian_fPEPS
    Nf::Int # number of physical fermions
    Nv::Int # number of virtual fermions

    # lattice size 
    Lx::Int # horizontal extent 
    Ly::Int # vertical extent

    # quadratic Hamiltonian parameters
    t::Float64 # hopping amplitude
    μ::Float64 # chemical potential
    Δx::Float64 # pairing amplitude
    Δy::Float64 # pairing amplitude

    # optimizer
    maxiter::Int # maximum iterations
    tol::Float64 # gradient tolerance

    # GfPEPS tensors
    # tensors::Matrix{ITensor}

    # test
    optim_res::Float64
    exact_energy::Float64


    function Gaussian_fPEPS(;conf::Dict=parsefile(joinpath(GfPEPS.config_path, "conf_default_BCS.json")))
        Random.seed!(conf["params"]["seed"])
        Nf = conf["params"]["N_physical_fermions_on_site"]
        Nv = conf["params"]["N_virtual_fermions_on_bond"]

        Lx = conf["system_params"]["Lx"]
        Ly = conf["system_params"]["Ly"]
        x_bc = Symbol(conf["system_params"]["x_bc"])
        y_bc = Symbol(conf["system_params"]["y_bc"])
        pairing_type = conf["system_params"]["pairing_type"]

        bc = (x_bc, y_bc)
        # lattice_type = Symbol(conf["system_params"]["lattice_type"])

        # hamiltonian params
        t = conf["hamiltonian"]["t"]
        Δ_x = conf["hamiltonian"]["Δ_x"]
        Δ_y = conf["hamiltonian"]["Δ_y"]
        μ = conf["hamiltonian"]["μ"]
        δ = conf["hamiltonian"]["hole_density"]

        # construct Brillouin zone
        bz = BrillouinZone2D(Lx,Ly,bc)

        # initial ortogonal matrix X to construct Γ_out
        Xsize = 8*Nv+Nf*2
        X = rand(Xsize,Xsize)
        # Orthogonalize the initial guess (already done above via SVD; keep as T0)
        U,_,V = svd(X)
        X = U*V'
        
        if(conf["hamiltonian"]["μ_from_hole_density"])
            μ = solve_for_mu(bz,δ,t,Δ_x,Δ_y)
        end

        # build loss function
        loss = optimize_loss(t, μ, bz, Nf, Nv, pairing_type, Δ_x, Δ_y)

        # build gradients
        g(x) = first(Zygote.gradient(loss, x))
        g!(G,x) = copyto!(G, g(x)) # better for optim

        # First, find a better initial guess for X by solving for smaller system sizes (see: 10.1103/PhysRevLett.129.206401) 
        @info "Finding better initial guess for X by solving smaller system sizes..."
        Lx_init = 5
        Ly_init = 5
        bz_init = BrillouinZone2D(Lx_init,Ly_init,bc)
        has_dirac_points(bz_init,t,μ,pairing_type,Δ_x,Δ_y) # warn if dirac points are present
        loss = optimize_loss(t, μ, bz_init, Nf, Nv, pairing_type, Δ_x, Δ_y)
        res = Optim.optimize(loss, g!, X, Optim.ConjugateGradient(manifold=Optim.Stiefel()), Optim.Options(
            iterations = conf["params"]["maxiter"],
            g_tol = conf["params"]["grad_tol"],
            show_trace = true
        ))

        # @info "Finding optimal X for full system size..."
        # loss = optimize_loss(t, μ, bz, Nf, Nv, pairing_type, Δ_x, Δ_y)
        # has_dirac_points(bz,t,μ,pairing_type,Δ_x,Δ_y) # warn if dirac points are present
        # # optimize X for the full system size
        # res = Optim.optimize(loss, g!, Optim.minimizer(res_init), Optim.ConjugateGradient(manifold=Optim.Stiefel()), Optim.Options(
        #     iterations = conf["params"]["maxiter"],
        #     g_tol = conf["params"]["grad_tol"],
        #     show_trace = true
        # ))

        @show Optim.minimum(res)
        println("Exact energy:", exact_energy_BCS_k(bz,t,μ,pairing_type,Δ_x,Δ_y))

        return new(
            Nf,
            Nv,
            Lx,
            Ly,
            t,
            μ,
            Δ_x,
            Δ_y,
            conf["params"]["maxiter"],
            conf["params"]["grad_tol"],
            # zeros(ITensor, 0, 0),
            Optim.minimum(res),
            exact_energy_BCS_k(bz,t,μ,pairing_type,Δ_x,Δ_y)
        )
    end
end