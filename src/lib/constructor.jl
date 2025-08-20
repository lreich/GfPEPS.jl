mutable struct Gaussian_fPEPS
    N_majoranas::Int # number of virtual majoranas

    # lattice size 
    Lx::Int # horizontal extent 
    Ly::Int # vertical extent

    # quadratic Hamiltonian parameters
    t::Float64 # hopping amplitude
    μ::Float64 # chemical potential
    Δ::Float64 # pairing amplitude

    # optimizer
    maxiter::Int # maximum iterations
    lr::Float64 # learning rate
    tol::Float64 # gradient tolerance

    # GfPEPS tensors
    tensors::Matrix{ITensor}

    function Gaussian_fPEPS(;N_majoranas::Int=4)
        # return new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, zeros(ITensor, 0, 0))
    end
end