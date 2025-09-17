function get_Dirac_to_Majorana_transformation(N::Int)
    Ω = ComplexF64.(zeros(2N,2N))
    for μ in 1:2N
        for v in 1:2N
            if iseven(μ)
                if v==μ/2
                    Ω[μ,v] += 1
                end
                if v==μ/2+N
                    Ω[μ,v] += 1
                end
            else
                if v==(μ+1)/2
                    Ω[μ,v] += 1im
                end
                if v==(μ+1)/2+N
                    Ω[μ,v] += -1im
                end
            end
        end
    end
    Ωdag = 2*inv(Ω)

    return Ω,Ωdag
end

Ω, Ωdag = get_Dirac_to_Majorana_transformation(4)

"""
    get_parent_hamiltonian(Γ_out::AbstractMatrix, Nf::Int, Nv::Int)

Given the output correlation matrix Γ_out, return the parent Hamiltonian in Dirac fermions.
"""
function get_parent_hamiltonian(Γ_out::AbstractMatrix)
    @assert eltype(Γ_out) <: Real && Γ_out ≈ -transpose(Γ_out)
    N = div(size(Γ_out, 1), 2)

    # convert from majorana basis to complex fermion basis
    Ω, Ωdag = get_Dirac_to_Majorana_transformation(N)
    @assert Ω*Ωdag ≈ 2I

    Γ_out_dirac = 1/2 .* (transpose(Ω) * Γ_out * conj(Ω))

    return Hermitian(-im .* Γ_out_dirac)

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