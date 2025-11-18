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