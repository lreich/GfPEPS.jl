mutable struct Gaussian_fPEPS
    Nf::Int # number of physical fermions
    Nv::Int # number of virtual fermions

    # lattice size 
    Lx::Int # horizontal extent 
    Ly::Int # vertical extent
    bz::BrillouinZone2D # Brillouin zone

    # quadratic Hamiltonian parameters
    t::Float64 # hopping amplitude
    μ::Float64 # chemical potential
    Δ_options::Dict{String, Any} # pairing options

    # optimizer
    maxiter::Int # maximum iterations
    tol::Float64 # gradient tolerance

    # GfPEPS tensors
    # tensors::Matrix{ITensor}

    # test
    X_opt::Matrix{Float64}
    optim_res::Float64
    exact_energy::Float64


    function Gaussian_fPEPS(;conf::Dict=parsefile(joinpath(GfPEPS.config_path, "conf_default_BCS.json")))
        Random.seed!(conf["params"]["seed"])
        Nf = conf["params"]["N_physical_fermions_on_site"]
        Nv = conf["params"]["N_virtual_fermions_on_bond"]
        N = Nf + 4*Nv # total number of fermions per site

        Lx = conf["system_params"]["Lx"]
        Ly = conf["system_params"]["Ly"]
        x_bc = Symbol(conf["system_params"]["x_bc"])
        y_bc = Symbol(conf["system_params"]["y_bc"])

        bc = (x_bc, y_bc)
        # lattice_type = Symbol(conf["system_params"]["lattice_type"])

        # hamiltonian params
        t = conf["hamiltonian"]["t"]
        pairing_type = conf["hamiltonian"]["Δ_options"]["pairing_type"]
        Δ_options = conf["hamiltonian"]["Δ_options"]
        μ = conf["hamiltonian"]["μ"]
        δ = conf["hamiltonian"]["hole_density"]

        # build Δ_vec from dict
        Δ_vec = begin
            if pairing_type == "d_wave" || pairing_type == "p_wave" || pairing_type == "default"
                [Δ_options["Δ_x"], Δ_options["Δ_y"]]
            elseif pairing_type == "s_wave"
                [Δ_options["Δ_0"]]
            else
                error("Unknown pairing type: $pairing_type. Supported types are: d_wave, p_wave, s_wave.")
            end
        end

        # construct Brillouin zone
        bz = BrillouinZone2D(Lx,Ly,bc)

        # initial ortogonal matrix X to construct Γ_out
        Xsize = 8*Nv+Nf*2
        X = rand(Xsize,Xsize)
        # Orthogonalize the initial guess (already done above via SVD; keep as T0)
        U,_,V = svd(X)
        X = U*V'
        
        # ensure correct parity sector
        _, X = rand_CM(Nf,Nv)
        @info "Created initial covariance matrix with even parity sector"

        if(conf["hamiltonian"]["μ_from_hole_density"])
            μ = solve_for_mu(bz,δ,t,Δ_x,Δ_y)
        end

        # build loss function
        loss = optimize_loss(t, μ, bz, Nf, Nv, pairing_type, Δ_vec...)

        # build gradients
        g(x) = first(Zygote.gradient(loss, x))
        g!(G,x) = copyto!(G, g(x)) # better for optim

        # First, find a better initial guess for X by solving for smaller system sizes (see: 10.1103/PhysRevLett.129.206401) 
        @info "Finding better initial guess for X by solving smaller system sizes..."
        Lx_init = 5
        Ly_init = 5
        bz_init = BrillouinZone2D(Lx_init,Ly_init,bc)
        has_dirac_points(bz_init,t,μ,pairing_type,Δ_vec...) # warn if dirac points are present
        loss = optimize_loss(t, μ, bz_init, Nf, Nv, pairing_type,Δ_vec...)
        res_init = Optim.optimize(loss, g!, X, Optim.ConjugateGradient(manifold=Optim.Stiefel()), Optim.Options(
        # res_init = Optim.optimize(loss, g!, X, Optim.LBFGS(;m=20,manifold=Optim.Stiefel()), Optim.Options(
            iterations = conf["params"]["maxiter"],
            g_tol = conf["params"]["grad_tol"],
            show_trace = conf["params"]["show_trace"]
        ))

        @info "Finding optimal X for full system size..."
        loss = optimize_loss(t, μ, bz, Nf, Nv, pairing_type, Δ_vec...)
        has_dirac_points(bz,t,μ,pairing_type,Δ_vec...) # warn if dirac points are present
        # optimize X for the full system size
        res = Optim.optimize(loss, g!, Optim.minimizer(res_init), Optim.ConjugateGradient(manifold=Optim.Stiefel()), Optim.Options(
            iterations = conf["params"]["maxiter"],
            g_tol = conf["params"]["grad_tol"],
            show_trace = conf["params"]["show_trace"]
        ))

        @show Optim.minimum(res)
        println("Exact energy:", exact_energy_BCS_k(bz,t,μ,pairing_type,Δ_vec...))

        # X_opt = Optim.minimizer(res)

        # U,V = bogoliubov_blocks_from_X(X_opt)
        # Z,norm = pairing_from_X(X_opt)
        # A_fiducial = fiducial_tensor_from_X(X_opt, Nf, Nv)

        # display(A_fiducial)

        return new(
            Nf,
            Nv,
            Lx,
            Ly,
            bz,
            t,
            μ,
            Δ_options,
            conf["params"]["maxiter"],
            conf["params"]["grad_tol"],
            # zeros(ITensor, 0, 0),
            Optim.minimizer(res),
            Optim.minimum(res),
            exact_energy_BCS_k(bz,t,μ,pairing_type,Δ_vec...)
        )
    end
end