"""
    get_kvals(::Val{:PBC}, L)

Returns the allowed momentum values for a 1D chain with periodic boundary conditions (PBC).

# Arguments
- `L`: System size (number of sites)

# Returns
- `Vector{Float64}`: Allowed momentum values 2π*m/L where:
  - If L is even: m ∈ {-(L-2)/2, ..., L/2}
  - If L is odd: m ∈ {-(L-1)/2, ..., (L-1)/2}
"""
function get_kvals(::Val{:PBC},L)
    if iseven(L)
        return [2π*m/L for m in (-(L-2)/2):L/2] 
    else
        return [2π*m/L for m in (-(L-1)/2):(L-1)/2] 
    end
end

"""
    get_kvals(::Val{:APBC}, L)

Returns the allowed momentum values for a 1D chain with anti-periodic boundary conditions (APBC).

# Arguments
- `L`: System size (number of sites)

# Returns
- `Vector{Float64}`: Allowed momentum values (2m-1)π/L where:
  - If L is even: m ∈ {1, ..., L/2}, returns both ±k values
  - If L is odd: m ∈ {1, ..., (L-1)/2}, returns ±k values plus π
"""
function get_kvals(::Val{:APBC},L)
    if iseven(L)
        kvals = [(2*m-1)*π/L for m in 1:L/2] 
		return vcat(-kvals,kvals)
    else
        kvals = [(2*m-1)*π/L for m in 1:(L-1)/2] 
		return vcat(-kvals,kvals,pi)
    end
end

"""
        get_2D_k_grid(Lx, Ly; x_periodic=Val(:APBC), x_offset=0.0, y_periodic=Val(:PBC),  y_offset=0.0)

Create the 2D momentum grid from 1D k-values (with optional offsets) and
return a meshgrid of the form: [[kx_1, ky_1]; 
                                [kx_2, ky_1]; 
                                    ... 
                                [kx_Lx, ky_1];
                                [kx_1, ky_2];
                                [kx_2, ky_2];
                                    ...
                                [kx_Lx, ky_2];
                                    ...
                                [kx_Lx, ky_Ly];

Returns
- Matrix of size (Lx*Ly)×2 where:
    - column 1 = vec(X) (kx repeated along rows)
    - column 2 = vec(Y) (ky repeated along columns)

Notes
- set the offsets, such that zero modes are avoided as those make the optimization of Γ harder.
"""
function get_2D_k_grid(Lx::Int, Ly::Int; 
    x_periodic::Union{Val{:APBC}, Val{:PBC}} = Val(:APBC),
    x_offset::Float64 = 0.0,
    y_periodic::Union{Val{:APBC}, Val{:PBC}} = Val(:PBC),
    y_offset::Float64 = 0.0)

    k_vals_x = get_kvals(x_periodic, Lx) .+ x_offset
    k_vals_y = get_kvals(y_periodic, Ly) .+ y_offset

    # create meshgrid
    KX = repeat([kx for kx in k_vals_x], Ly)
    KY = collect(Iterators.flatten(map(k_vals_y) do ky
        repeat([ky],Lx)
    end))

    return hcat(KX,KY)
end