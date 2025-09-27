using TensorKit
using PEPSKit
import PEPSKit: σˣ, σʸ, σᶻ

"""
    kitaev_hc_hamiltonian(T::Type{<:Number}, lattice::InfiniteSquare; Jx::Float64 = 1.0, Jy::Float64 = 1.0, Jz::Float64 = 1.0)

Construct the Kitaev honeycomb Hamiltonian

```
    H = -Jx ∑_{x-links} σ^x_j σ^x_k - Jy ∑_{y-links} σ^y_j σ^y_k - Jz ∑_{z-links} σ^z_j σ^z_k
```

on a brick-wall embedding of the honeycomb onto an `InfiniteSquare` lattice (see: doi:10.21468/SciPostPhysLectNotes.86)

```
A = even sites
B = odd sites

        |           |
--A--x--B--y--A--x--B--y--A--
  |           |           |
  z           z           z 
  |           |           |
--B--y--A--x--B--y--A--x--B--
        |           |
```
"""
function kitaev_hc_hamiltonian(
        T::Type{<:Number}; lattice::InfiniteSquare = InfiniteSquare(2, 2), Jx::Float64 = 1.0,
        Jy::Float64 = 1.0, Jz::Float64 = 1.0
    )

    # single-spin Pauli operators
    σx = σˣ(T, Trivial)
    σy = σʸ(T, Trivial)
    σz = σᶻ(T, Trivial)
    pspace = domain(σx)[1]
    I1 = id(pspace)

    # We treat ONE lattice site = ONE honeycomb *cell* containing TWO spins (A,B).
    # To present this composite cell as a single site to LocalOperator we FUSE the two spins.
    unfused_space = pspace ⊗ pspace          # A ⊗ B
    fused_space   = fuse(unfused_space)      # single site space
    to_fused      = isomorphism(fused_space ← unfused_space)
    from_fused    = inv(to_fused)

    # Helpers: embed operator acting on A or B of a cell
    opA(op) = to_fused * (op ⊗ I1) * from_fused
    opB(op) = to_fused * (I1 ⊗ op) * from_fused

    # Intra–cell x bond (A,B): acts on ONE fused site
    Hx = -Jx * to_fused * (σx ⊗ σx) * from_fused          # fused_space ← fused_space

    # Inter-cell y bond: B_left -- A_right  => (I⊗σy) ⊗ (σy⊗I)
    Hy = -Jy * (opB(σy) ⊗ opA(σy))

    # Inter-cell z bond: A_top -- B_bottom  => (σz⊗I) ⊗ (I⊗σz)
    Hz = -Jz * (opA(σz) ⊗ opB(σz))

    # Lattice site spaces (each site now = fused cell)
    spaces = fill(fused_space, (lattice.Nrows, lattice.Ncols))
    nns = nearest_neighbours(lattice)

    # Displacement (dr, dc) between cell indices
    disp(a, b) = (b.I[1] - a.I[1], b.I[2] - a.I[2])

    # Select oriented bonds only once: east (0, +1), south (+1, 0)
    y_bonds = Tuple{CartesianIndex,CartesianIndex}[]
    z_bonds = Tuple{CartesianIndex,CartesianIndex}[]
    # for (a,b) in nns
    #     dr, dc = disp(a,b)
    #     if dr == 0 && dc == 1      # east
    #         push!(y_bonds, (a,b))
    #     elseif dr == 1 && dc == 0  # south
    #         push!(z_bonds, (a,b))
    #     end
    # end
    nR, nC = lattice.Nrows, lattice.Ncols
    for r in 1:nR
        for c in 1:nC
            a = CartesianIndex(r,c)

            # east neighbour with wrap
            c2 = (c == nC) ? 1 : c+1
            b_east = CartesianIndex(r, c2)
            push!(y_bonds, (a, b_east))

            # south neighbour with wrap
            r2 = (r == nR) ? 1 : r+1
            b_south = CartesianIndex(r2, c)
            push!(z_bonds, (a, b_south))
        end
    end
    # now counts should all match
    @assert length(y_bonds) == nR*nC
    @assert length(z_bonds) == nR*nC

   # Build term iterators with correct site index counts
    x_terms = ((site,) => Hx for site in vertices(lattice))
    y_terms = ((a,b)  => Hy for (a,b) in y_bonds)                # two distinct cells
    z_terms = ((a,b)  => Hz for (a,b) in z_bonds)

    return LocalOperator(
        spaces,
        x_terms...,
        y_terms...,
        z_terms...,
    )
end
kitaev_hc_hamiltonian(; lattice=InfiniteSquare(2, 2), Jx=1.0, Jy=1.0, Jz=1.0) = kitaev_hc_hamiltonian(ComplexF64; lattice=lattice, Jx=Jx, Jy=Jy, Jz=Jz)

H = kitaev_hc_hamiltonian();

Dbond = 2
χenv = 16;

boundary_alg = (; tol = 1.0e-10, trscheme = (; alg = :fixedspace));
optimizer_alg = (; alg = :lbfgs, tol = 1.0e-4, maxiter = 100, lbfgs_memory = 16);
reuse_env = true
verbosity = 3;

peps₀ = InfinitePEPS(randn, ComplexF64, 4, Dbond; unitcell = (2,2))
env_random = CTMRGEnv(randn, ComplexF64, peps₀, ℂ^χenv);
env₀, info_ctmrg = leading_boundary(env_random, peps₀; boundary_alg...);
@show info_ctmrg.truncation_error;

# find groundstate
peps, env, E, info_opt = fixedpoint(
    H, peps₀, env₀; boundary_alg, optimizer_alg, reuse_env, verbosity
);

@show info_opt.fg_evaluations info_opt.gradnorms[1:10:end];
@show E;