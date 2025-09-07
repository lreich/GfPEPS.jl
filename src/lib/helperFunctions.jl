"""
    ⊕(A::AbstractMatrix, n::Integer)

repeat A ⊕ A ⊕ ... (n times) via kron

"""
function ⊕(A::AbstractMatrix, n::Integer; sparse::Bool=true)
    @assert n >= 1

    if sparse
        return kron(spdiagm(0 => ones(eltype(A), n)), A)
    else
        Id = Matrix{eltype(A)}(I, n, n)
        return kron(Id, A)
    end
end