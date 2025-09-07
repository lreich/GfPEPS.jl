mutable struct Gaussian_fPEPS
    Nf::Int # number of physical fermions
    Nv::Int # number of virtual fermions

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

    function Gaussian_fPEPS(;Nv::Int=4)
        # return new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, zeros(ITensor, 0, 0))
    end
end