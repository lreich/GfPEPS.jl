"""
Get PEPS tensor by contracting virtual axes of ⟨ω|F⟩,
where |ω⟩, |F⟩ are the virtual and the fiducial states.
```
            -2
            ↓
            ω
            ↑
            1  -1
            ↑ ↗
    -5  --←-F-→- 2 -→-ω-←- -3
            ↓
            -4
```
Input axis order
```
        5  1                2
        ↑ ↗                 ↑
    2-←-F-→-3   1-←-ω-→-2   ω
        ↓                   ↓
        4                   1
```
"""
function get_peps(ω::AbstractTensor{T, S, N1}, F::AbstractTensor{T, S, N2}) where {T, S, N1, N2}
    V = fermion_space()
    Nv = div(N1, 2)
    Nf = N2 - 4Nv

    #= Notes:
        - fuse(V, V, ...) gives us the representation that is the result of fusing all the single fermion spaces together.
            This represents the whole tensor product, but in a fused basis.
        - reduce(⊗, fill(V, N) gives V ⊗ V ⊗ V ⊗ ... (N times) (This is the explicit tensor product space before fusion)
        - isomorphism() constructs an isometric map between two (isomorphic) spaces.
            returns the canonical change-of-basis map (isomorphism) between:
                reduce(⊗, fill(V, Nf)) — the explicit tensor product basis, and
                fuse(fill(V, Nf)...) — the fused representation space.
    =#
    fuser_physical = isomorphism(Int, fuse(fill(V, Nf)...), reduce(⊗, fill(V, Nf)))
    fuser_virtual = isomorphism(Int, fuse(fill(V, Nv)...), reduce(⊗, fill(V, Nv)))

    # The maximally entangled bond state ω is in the full tensor product basis of the two virtual fermions (Λ flavors).
    # We now transform ω to to the explicit tensor product basis of |l> ⊗ |r> ( or |d> ⊗ |u> ).
    ω = (fuser_virtual ⊗ fuser_virtual) * ω

    # The fiducial state F, obtained from the paired state is in the completed fused basis of all physical and auxiliary fermions.
    # We now transform F to the explicit tensor product basis of |f> ⊗ |l> ⊗ |r> ⊗ |d> ⊗ |u>.
    F = (fuser_physical ⊗ reduce(⊗, fill(fuser_virtual, 4))) * F 

    # contract virtual legs by computing: A=⟨ω|F⟩
    # @tensor A[-1; -2 -3 -4 -5] := conj(ω[1 -2]) * conj(ω[2 -3]) * F[-1 -5 2 -4 1]
     @tensor A[-1; -2 -3 -4 -5] := conj(ω[1 -2]) * conj(ω[2 -3]) * F[-1 1 2 -4 -5]
    
    return InfinitePEPS(A; unitcell = (1, 1))
end

"""
Translate the orthogonal matrix `X` defining the fiducial state's covariance matrix,
to Gaussian fPEPS in PEPSKit format.
"""
function translate(X::AbstractMatrix, Nf::Int, Nv::Int)
    @assert X*X' ≈ I "X must be orthogonal"
    Γ = Γ_fiducial(X, Nv, Nf)

    @assert Γ ≈ -transpose(Γ) "Covariance matrix must be antisymmetric"
    @assert Γ * Γ ≈ -I "Covariance matrix must satisfy Γ² = -I for a pure Gaussian state"
    @assert pfaffian(Γ) ≈ 1  "Pure BCS states must have even parity Pf(iΓ) = +1"

    H = get_parent_hamiltonian(Γ, Nf, Nv)
    _, M = bogoliubov(H)

    U,V = get_bogoliubov_blocks(M)
    Z = V * inv(U) # pairing matrix 
    # Z = conj(V * inv(U)) # pairing matrix 
    @assert Z ≈ -transpose(Z) "Pairing matrix must be antisymmetric"
    Z = (Z - transpose(Z)) / 2  # ensure exact antisymmetry

    ω = virtual_bond_state(Nv)
    F = fiducial_state(Nf, Nv, Z)
    peps = get_peps(ω, F)

    return PEPSKit.peps_normalize(peps)
end

"""
    function qp_to_qq_ordering_transformation(N::Int)

Construct the transformation matrix F from qp-ordering to qq-ordering for N complex fermions.

F: (c1,c3,...,c(2N-1), c2,c4,...,c2N)  → (c1,c2,...,c2N)

"""
function qp_to_qq_ordering_transformation(N::Int)
    F = zeros(Int, 2N, 2N)

    for i in 1:2N
        ind = isodd(i) ? ((i + 1) ÷ 2) : (N + (i ÷ 2))
        F[i, ind] = 1
    end

    return F
end
Zygote.@nograd qp_to_qq_ordering_transformation

function get_Dirac_to_Majorana_qq_transformation(N::Int)
    Ω = ComplexF64.(zeros(2N,2N))
    for i in 1:2N
        for j in 1:2N
            if isodd(i)
                if j==(i+1)÷2
                    Ω[i,j] += 1
                end
                if j==(i+1)÷2+N
                    Ω[i,j] += 1
                end
            else
                if j==i÷2
                    Ω[i,j] += 1im
                end
                if j==i÷2+N
                    Ω[i,j] += -1im
                end
            end
        end
    end
    # Ωdag = 2*inv(Ω)

    return Ω
end

function get_Dirac_to_Majorana_qp_transformation(N::Int)
    Ω = [I(N) I(N);
         im*I(N) -im*I(N)]

    return Ω
end


"""
    get_parent_hamiltonian(Γ::AbstractMatrix)

Return the parent Hamiltonian in Dirac representation (qp-ordering) of the fiducial state correlation matrix `Γ` in Majorana representation (qq-ordering).
"""
function get_parent_hamiltonian(Γ::AbstractMatrix, Nf::Int, Nv::Int)
    N = div(size(Γ, 1), 2)

    # Transform to Dirac fermions (qq-ordering)
    Ω0 = [1  1; im  -im]
    Ω = kron(I(N), Ω0)
    Γ_fiducial_dirac = 1/4 .* Ω' * Γ * Ω
    #= Now has the following ordering (qq)
        (f_1,f_1†, ..., f_Nf, f_Nf†, l_1, l_1†, r_1, r_1†, ..., l_Nv, l_Nv†, r_Nv, r_Nv†, d_1, d_1†, u_1, u_1†, ..., d_Nv, d_Nv†, u_Nv, u_Nv†)
    =#

    # bring to qp-ordering
    perm = vcat(1:2:(2N), 2:2:(2N))
    Γ_fiducial_dirac = Γ_fiducial_dirac[perm, perm]
    #= Now has the following ordering (qp)
        (f_1, ..., f_Nf, l_1, r_1, ..., l_Nv, r_Nv, d_1, u_1, ..., d_Nv, u_Nv, f_1†, ..., f_Nf†, l_1†, r_1†, ..., l_Nv†, r_Nv†, d_1†, u_1†, ..., d_Nv†, u_Nv†)
    =#

    # group virtual fermions as (l1,...,lNv,r1,...,rNv,d1,...,dNv,u1,...,uNv)
    L = collect(1:2:2Nv)    # l1, l2, ...
    R = collect(2:2:2Nv)    # r1, r2, ...
    D = collect(2Nv+1:2:4Nv)  # d1, d2, ...
    U = collect(2Nv+2:2:4Nv)  # u1, u2, ...
    perm_virtual = vcat(L, R, D, U)
    
    perm_total = vcat(
        1:Nf,                       # physical (already fine)
        Nf .+ perm_virtual,         # reorder virtuals
        (Nf+4Nv) .+ (1:Nf),         # f†
        (2Nf+4Nv) .+ perm_virtual    # reordered virtual†
    )
    Γ_fiducial_dirac = Γ_fiducial_dirac[perm_total, perm_total]

    # now reorder to (f,u,r,d,l)
    L = collect(Nf+1:Nf+Nv)    # l1, l2, ...
    R = collect(Nf+Nv+1:Nf+2Nv)   # r1, r2, ...
    D = collect(Nf+2Nv+1:Nf+3Nv)  # d1, d2, ...
    U = collect(Nf+3Nv+1:Nf+4Nv)  # u1, u2, ...
    perm_virtual = vcat(U, R, D, L)

    perm_reorder = vcat(1:Nf, 
        perm_virtual,
        (Nf+4Nv) .+ (1:Nf), # f†
        (Nf+4Nv) .+ perm_virtual # virtual†
    )
    Γ_fiducial_dirac = Γ_fiducial_dirac[perm_reorder, perm_reorder]

    @assert Γ_fiducial_dirac' ≈ -Γ_fiducial_dirac "Fiducial state CM in Dirac representation must be anti-hermitian"
    @assert Γ_fiducial_dirac*Γ_fiducial_dirac' ≈ I / 4 "Fiducial state CM in Dirac representation must be pure"

    return Hermitian(-2im .* Γ_fiducial_dirac)
end

""" 
    get_empty_peps_tensor(Nf::Int, Nv::Int)

Create an empty fPEPS tensor with the correct dimensions and spaces for given number of physical (Nf) and virtual (Nv) fermions.
"""
function get_empty_peps_tensor(Nf::Int, Nv::Int)
    physical_spaces = Vect[fℤ₂](0 => 2^(Nf - 1), 1 => 2^(Nf - 1))
    V_bonds = Vect[fℤ₂](0 => 2^(Nv - 1), 1 => 2^(Nv - 1))
    virtual_spaces = V_bonds ⊗ V_bonds ⊗ V_bonds' ⊗ V_bonds'

    T = zeros(ComplexF64, dim(physical_spaces), dim(virtual_spaces))
    T = reshape(T, (2^Nf, 2^Nv, 2^Nv, 2^Nv, 2^Nv))

    return T, physical_spaces, virtual_spaces
end

function translate_new(X::AbstractMatrix, Nf::Int, Nv::Int)
    Γ_fiduc = Γ_fiducial(X, Nv, Nf)

    H = get_parent_hamiltonian(Γ_fiduc, Nf, Nv)
    _, M = bogoliubov(H)

    # Bloch Messiah decomposition
    Dmat,UVmat,Cmat = bloch_messiah_decomposition(M)
    Dmat_prime,UVmat_prime,Cmat_prime = truncated_bloch_messiah(Dmat, UVmat, Cmat)

    D, Ubar, Vbar, C = get_mats_from_bloch_messiah(Dmat_prime, UVmat_prime, Cmat_prime)

    M_A = size(Vbar, 1)
    parity = mod(M_A,2)
    v_prod = prod([abs(Vbar[i-1, i]) for i in 2:2:M_A])

    # compute full matrices for overlap
    R_mat_full = D*Vbar
    Q_mat = Ubar*Vbar

    @assert Q_mat ≈ - transpose(Q_mat)
    Q_mat = (Q_mat - transpose(Q_mat)) / 2 # enforce exact skew-symmetry

    states_f = 0:(2^Nf - 1)
    states_v = 0:(2^Nv - 1)

    # Cartesian product; store as tuples
    states = [(f,u,r,d,l) for f in states_f for u in states_v for r in states_v
                                   for d in states_v for l in states_v]

    ind_f_dict = translate_occ_to_TM_dict(Nf)
    ind_v_dict = translate_occ_to_TM_dict(Nv)

    T, physical_spaces, virtual_spaces = get_empty_peps_tensor(Nf, Nv)

    # get tensor elements with overlap formula from 10.1103/PhysRevB.107.125128
    Threads.@threads for state in states
    # for state in states
        f_occ, u_occ, r_occ, d_occ, l_occ = state

        # convert occ to bitstrings
        f = (digits(f_occ, base=2, pad=Nf))
        u = (digits(u_occ, base=2, pad=Nv))
        l = (digits(l_occ, base=2, pad=Nv))
        d = (digits(d_occ, base=2, pad=Nv))
        r = (digits(r_occ, base=2, pad=Nv))

        # Boolean occupation vector to select rows from R_mat_full (true if occupied)
        occ_bool = vcat(f, u, r, d, l) .== 1
        M_prime = sum(occ_bool)

        parity_f = mod(sum(f), 2)
        parity_v = mod(sum(l) + sum(u) + sum(r) + sum(d), 2)

        if mod(M_prime,2) != parity || parity_f != parity_v # skip if parity doesn't match
            continue
        end

        if M_prime!=0  
            # build R_mat
            R_mat = R_mat_full[occ_bool,:]
            fsign = isodd((M_prime * (M_prime - 1)) ÷ 2) ? -1 : 1 # fermionic sign from reordering
            pf = pfaffian([zeros(M_prime,M_prime) R_mat; -transpose(R_mat) Q_mat])

            T[ind_f_dict[f], ind_v_dict[u], ind_v_dict[r], ind_v_dict[d], ind_v_dict[l]] = fsign * pf / v_prod
        else # all unoccupied
            T[1,1,1,1,1] = pfaffian(Q_mat) / v_prod
        end
    end

    peps = InfinitePEPS(TensorMap(T, physical_spaces ← virtual_spaces))
    return peps
end

function translate_occ_to_TM_dict(N)
    nstates = 2^N
    even = []
    odd  = []
    for x in 0: nstates-1
        d = (digits(x, base=2, pad=N))
        if isodd(sum(d))
            push!(odd, d)
        else
            push!(even, d)
        end
    end
    mapping = Dict{Vector{Int}, Int}()
    for (i,x) in enumerate((even..., odd...))
        mapping[x] = i
    end
    return mapping
end