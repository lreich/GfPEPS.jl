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
    # We now transform ω to to the explicit tensor product basis of |l> ⊗ |r> ( or |u> ⊗ |d> ).
    ω = (fuser_virtual ⊗ fuser_virtual) * ω

    # The fiducial state F, obtained from the paired state is in the completed fused basis of all physical and auxiliary fermions.
    # We now transform F to the explicit tensor product basis of |f> ⊗ |l> ⊗ |r> ⊗ |u> ⊗ |d>.
    F = (fuser_physical ⊗ reduce(⊗, fill(fuser_virtual, 4))) * F 

    # contract virtual legs by computing: A=⟨ω|F⟩
    @tensor A[-1; -2 -3 -4 -5] := conj(ω[1 -2]) * conj(ω[2 -3]) * F[-1 -5 2 -4 1]
    return InfinitePEPS(A; unitcell = (1, 1))
end

"""
Translate the orthogonal matrix `X` defining the fiducial state's covariance matrix,
to Gaussian fPEPS in PEPSKit format.
"""
function translate(X::AbstractMatrix, Nf::Int, Nv::Int)
    N = Nf + 4Nv
    Γ = Γ_fiducial(X, Nv, Nf)

    @assert Γ ≈ -transpose(Γ) "Covariance matrix must be antisymmetric"
    @assert Γ * Γ ≈ -I "Covariance matrix must satisfy Γ² = -I for a pure Gaussian state"
    @assert pfaffian(im .* Γ) ≈ 1  "Pure BCS states must have even parity Pf(iΓ) = +1"

    H = get_parent_hamiltonian(Γ)
    _, M = bogoliubov(H)

    U,V = get_bogoliubov_blocks(M)
    # Z = -inv(U) * V # pairing matrix 
    Z = -U \ conj(V)

    display(Z)

    ω = virtual_bond_state(Nv)
    F = fiducial_state(Nf, Nv, Z)
    peps = get_peps(ω, F)
    return peps
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

get_Dirac_to_Majorana_qq_transformation(2)

function get_Dirac_to_Majorana_qp_transformation(N::Int)
    Ω = [I(N) I(N);
         im*I(N) -im*I(N)]

    return Ω
end


"""
    get_parent_hamiltonian(Γ_out::AbstractMatrix, Nf::Int, Nv::Int)

Given the output correlation matrix Γ_out in Majorana representation, return the parent Hamiltonian in Dirac fermions.
"""
function get_parent_hamiltonian(Γ_out::AbstractMatrix, Nf::Int, Nv::Int)
    @assert eltype(Γ_out) <: Real && Γ_out ≈ -transpose(Γ_out)
    N = div(size(Γ_out, 1), 2)

    # convert from majorana basis to complex fermion basis
    F = qp_to_qq_ordering_transformation(Nf)
    P_full = BlockDiagonal([F, Matrix(I(8*Nv))])
    Γ_out_qq = P_full * Γ_out * P_full'
    @assert Γ_out_qq ≈ -transpose(Γ_out_qq)

    S = get_Dirac_to_Majorana_qq_transformation(N)
    Γ_out_dirac = inv(S) * Γ_out_qq * transpose(inv(S))
    @assert Γ_out_dirac ≈ -transpose(Γ_out_dirac)


    # Ω_qp = get_Dirac_to_Majorana_qp_transformation(Nf)
    # Ω_qq = get_Dirac_to_Majorana_qq_transformation(4Nv)
    # @assert Ω_qq*Ω_qq' ≈ 2I
    # @assert Ω_qp*Ω_qp' ≈ 2I

    # Ω = BlockDiagonal([Ω_qp, Ω_qq])

    # Γ_out_dirac = 1/2 .* (Ω * Γ_out * Ω')
    H = Hermitian(-im .* Γ_out_dirac)

    # # put annihilation in front of creation operators
    # # (f_1, ..., f_N, f†_1, ..., f†_N)
    # perm = vcat(1:2:(2N), 2:2:(2N))
    # return Hermitian(H[perm, perm])

    return H

    # Ω_single = [1 1;
    #             im -im]
    # Ω = ⊕(Ω_single, N)

    # resulting fermion order is (f_1, f†_1, ..., f_N, f†_N)
    # H = -0.5im .* Ω' * Γ_out * Ω



    # put annihilation in front of creation operators
    # (f_1, ..., f_N, f†_1, ..., f†_N)
    # perm = vcat(1:2:(2N), 2:2:(2N))
    # return Hermitian(H[perm, perm])
end