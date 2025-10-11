fermion_space() = Vect[fℤ₂](0 => 1, 1 => 1) # single fermion / Z2 graded space

"""
    two_site_operator(T::Type{<:Number} = ComplexF64)

Create the vacuum state for `n` spinless fermions
"""
function two_site_operator(T::Type{<:Number} = ComplexF64)
    V = fermion_space()
    return zeros(T, V ⊗ V ← V ⊗ V)
end

"""
    f_dag_f_dag(T::Type{<:Number} = ComplexF64)

Creates the two-body operator f1† f2†
"""
function f_dag_f_dag(T::Type{<:Number} = ComplexF64)
    op = two_site_operator(T)
    I = sectortype(op)
    # only non-zero matrix element is ⟨1,1| f1† f2† |0,0⟩ = 1
    op[(I(1), I(1), dual(I(0)), dual(I(0)))] .= 1
    return op
end

"""
    vacuum_state(T::Type{<:Number}, n::Int)

Create the vacuum state for `n` spinless fermions
"""
function vacuum_state(T::Type{<:Number}, n::Int)
    vac = zeros(T, fermion_space())
    vac.data[1] = 1.0
    return (n > 1) ? reduce(⊗, fill(vac, n)) : vac
end
vacuum_state(n::Int) = vacuum_state(ComplexF64, n)

"""
    virtual_bond_state(T::Type{<:Number}, Nv::Int)

Construct the maximally entangled state (MES) on virtual bonds
for Nv flavours of virtual fermions `(a1_i, a2_i)` (i = 1, ..., Nv)
```
    |ω⟩ = ∏_{α=1}^Nv 1/sqrt(2) (1 + a1†_α a2†_α) |0⟩

    For horizontal bond: a1†_α=r†_iα, a2†_α=l†_(i+̂x)α
    For vertical bond:   a1†_α=d†_iα, a2†_u=d†_(i+̂y)α
```
"""
function virtual_bond_state(T::Type{<:Number}, Nv::Int)
    ff = f_dag_f_dag(T)
    vac = vacuum_state(T, 2)
    # MES for one pair of (a1_i, a2_i) on the bond
    # the resulting fermion order is (a1_1, a2_1, ..., a1_Λ, a2_Λ)
    ω = (1 / sqrt(2)) * (unit ⊗ unit + ff) * vac
    if Nv > 1
        # reorder fermions to (a1_1, ..., a1_Λ, a2_1, ..., a2_Λ)
        ω = reduce(⊗, fill(ω, Nv))
        perm = Tuple(vcat(1:2:(2*Nv), 2:2:(2*Nv)))
        ω = permute(ω, (perm, ()))
    end
    return ω
end
virtual_bond_state(Nv) = virtual_bond_state(ComplexF64, Nv)

"""
Construct the fully paired state `exp(a† Z a† / 2)`, 
where Z is an anti-symmetric matrix.
"""
function paired_state(T::Type{<:Number}, Z::AbstractMatrix)
    N = size(Z, 1)
    @assert Z ≈ -transpose(Z)
    ff = f_dag_f_dag(T)
    ψ = vacuum_state(T, N)
    # apply exp(Z_{ij} a†_i a†_j) (i < j)
    for i in 1:(N - 1)
        for j in (i + 1):N
            op = exp(Z[i, j] * ff)
            idx_op = [-i, -j, i, j]
            idx_ψ = collect(-1:-1:(-N))
            idx_ψ[i], idx_ψ[j] = i, j
            ψ = ncon([op, ψ], [idx_op, idx_ψ])
        end
    end
    return ψ
end
paired_state(Z) = paired_state(ComplexF64, Z)

"""
Construct the local tensor of the fiducial state
`exp(a† Z a† / 2)`, where Z is an anti-symmetric matrix.

Input complex fermion order of Z is
(f_1, ..., f_{Nf}, l_1, r_1, ..., l_χ, r_χ, d_1, u_1, ..., d_χ, u_χ)

The output complex fermion order will be
(f_1, ..., f_{Nf}, l_1, ..., l_χ, r_1, ..., r_χ, d_1, ..., d_χ, u_1, ..., u_χ)
"""
function fiducial_state(T::Type{<:Number}, Nf::Int, Nv::Int, Z::AbstractMatrix)
    ψ = paired_state(T, Z)
    # reorder virtual fermions (TensorKit automaticially handles fermionic signs)
    # perm = vcat(1:2:(2Nv), 2:2:(2Nv))
    # perm = Tuple(vcat(1:Nf, perm .+ Nf, perm .+ (Nf + 2Nv)))
    # ψ = TensorKit.permute(ψ, (perm, ()))
    return ψ
end
fiducial_state(Nf::Int, Nv::Int, Z::AbstractMatrix) = fiducial_state(ComplexF64, Nf, Nv, Z)