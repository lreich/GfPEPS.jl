function tj_model_NNN(
    particle_symmetry::Type{<:Sector},
    spin_symmetry::Type{<:Sector},
    lattice::InfiniteSquare;
    t1=1.0,
    t2=0.5,
    J1=1.0,
    J2=0.5,
    slave_fermion::Bool=false,
)
    hopping =
        TJOperators.e_plusmin(particle_symmetry, spin_symmetry; slave_fermion) +
        TJOperators.e_minplus(particle_symmetry, spin_symmetry; slave_fermion)
    num = TJOperators.e_number(particle_symmetry, spin_symmetry; slave_fermion)
    heis =
        TJOperators.S_exchange(particle_symmetry, spin_symmetry; slave_fermion) -
        (1 / 4) * (num ⊗ num)
        
    spaces = fill(domain(term_AA)[1], (lattice.Nrows, lattice.Ncols))

    h_NN = (-t1) * hopping + J1 * heis
    h_NNN = (-t2) * hopping + J2 * heis

    return LocalOperator(
        spaces,
        (neighbor => h_NN for neighbor in PEPSKit.nearest_neighbours(lattice))...,
        (neighbor => h_NNN for neighbor in PEPSKit.next_nearest_neighbours(lattice))...,
    )
end

function tj_model_with_doping(
        T::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector},
        lattice::InfiniteSquare;
        t = 2.5, J = 1.0, mu = 0.0, slave_fermion::Bool = false, n::Integer =0,
    )
    hopping =
        TJOperators.e_plusmin(particle_symmetry, spin_symmetry; slave_fermion) +
        TJOperators.e_minplus(particle_symmetry, spin_symmetry; slave_fermion)
    num = TJOperators.e_number(particle_symmetry, spin_symmetry; slave_fermion)
    heis =
        TJOperators.S_exchange(particle_symmetry, spin_symmetry; slave_fermion) -
        (1 / 4) * (num ⊗ num)
    pspace = space(num, 1)
    unit = TensorKit.id(pspace)
    h = (-t) * hopping + J * heis - (mu / 4) * (num ⊗ unit + unit ⊗ num)
    if T <: Real
        h = real(h)
    end
    spaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    H = PEPSKit.nearest_neighbour_hamiltonian(spaces, h)

    if particle_symmetry === Trivial
        iszero(n) || throw(ArgumentError("imposing particle number requires `U₁` symmetry"))
    elseif particle_symmetry === U1Irrep
        full_charge_sector = fℤ₂(0) ⊠ U1Irrep(n)

        H = MPSKit.add_physical_charge(H, fill(full_charge_sector, size(spaces)...))
    else
        throw(ArgumentError("symmetry not implemented"))
    end

    return H
end

function tj_model_with_doping_old(
    T::Type{<:Number},
    particle_symmetry::Type{<:Sector},
    spin_symmetry::Type{<:Sector},
    lattice::InfiniteSquare;
    t=2.5,
    J=1.0,
    λ=1.0,
    δ_target=0.0,
    slave_fermion::Bool=false,
)
    hopping =
        TJOperators.e_plusmin(particle_symmetry, spin_symmetry; slave_fermion) +
        TJOperators.e_minplus(particle_symmetry, spin_symmetry; slave_fermion)
    num = TJOperators.e_number(particle_symmetry, spin_symmetry; slave_fermion)
    heis =
        TJOperators.S_exchange(particle_symmetry, spin_symmetry; slave_fermion) -
        (1 / 4) * (num ⊗ num)
    pspace = space(num, 1)
    unit = TensorKit.id(pspace)

     # Average doping (hole density) on the bond: 1 - (ni + nj) / 2
    doping = unit ⊗ unit - (num ⊗ unit + unit ⊗ num) / 2

    # Factor of 2 because there are two sites on the bond.
    doping_target = δ_target * (unit ⊗ unit)

    h = (-t) * hopping + J * heis + λ * (doping - doping_target)^2
    if T <: Real
        h = real(h)
    end
    return PEPSKit.nearest_neighbour_hamiltonian(fill(pspace, size(lattice)), h)
end

"""
    find_optimal_Δ(t::Real, J::Real, δ_target::Real; pairing_type::String="d_wave")

Returns the optimal gap parameter Δ for the t-J model at given hopping amplitude `t`, exchange interaction `J`, and target doping `δ_target`, assuming a specified pairing symmetry.:

# Keyword arguments
- `t::Real`: Hopping amplitude
- `J::Real`: Exchange interaction
- `δ_target::Real`: Target doping level
- `bz::BrillouinZone2D`: Brillouin zone

# Optional arguments
- `pairing_type::String="d_wave"`: Choose from:
    - d_wave
    - s_wave
"""
function find_optimal_Δ(t::Real, J::Real, δ_target::Real, bz::BrillouinZone2D; pairing_type::String="d_wave")
    # Gutzwiller approximation factors
    gt = 2*δ_target/(1+δ_target)
    gs = 4/(1+δ_target)^2

    t = gt * t

    function loss(x)
        Δ_eff = gs*J*x[1]

        μ = solve_for_mu(bz, δ_target, t, pairing_type, Δ_eff; μ_range=(-10.0, 10.0))

        params = BCS(t, μ, pairing_type, Δ_eff)

        return exact_energy(params,bz) + 2*gs*J*x[1]^2
    end

    Δ_min = 1e-10
    Δ_max = 10.0
    # res = Optim.optimize(loss, [Δ_min, Δ_max], Optim.NelderMead(), Optim.Options(
            # iterations = 100,
            # g_tol = 1e-6,
            # show_trace = true,
            # f_reltol = 1e-6))
    res = Optim.optimize(loss, Δ_min, Δ_max, Brent())

    return Optim.minimizer(res)[1]
end