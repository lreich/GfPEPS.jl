"""
# Julia translation of Gaussian-fPEPS/src/gfpeps/loadwrite.py (using JLD2)
"""

using JLD2
using Random

_normkey(k::AbstractString) = startswith(k, "/") ? k[2:end] : k

"""
    initialT(loadfile::Union{Nothing,String}, Tsize::Integer)
Load T from JLD2 at path "transformer/T" if present; else return random T.
"""
function initialT(loadfile::Union{Nothing,String}, Tsize::Integer)
    if loadfile !== nothing && isfile(loadfile)
        @info "Try to initialize T from $(loadfile)"
        jldopen(loadfile, "r") do f
            key = "transformer/T"
            if haskey(f, key) || haskey(f, _normkey(key))
                T = haskey(f, key) ? read(f, key) : read(f, _normkey(key))
                return reshape(T, Tsize, Tsize)
            else
                @debug "Load Failed! No transformer/T in $(loadfile) switch to random initialize!"
                return rand(Tsize, Tsize)
            end
        end
    end
    return rand(Tsize, Tsize)
end

"""
    savelog_trivial(writefile, x, fun, Eg, args::Dict, cor)
Save log to JLD2 with energy and correlation observables.
"""
function savelog_trivial(writefile::Union{Nothing,String}, x, fun, Eg, args::Dict, cor)
    isnothing(writefile) && return
    rhoup, rhodn, kappa = cor
    @info "Save T to $(writefile)"
    jldopen(writefile, "w") do f
        write(f, "transformer/T", x)
        write(f, "energy/EABD", fun)
        write(f, "energy/Eg", Eg)
        write(f, "model/Mu", args["Mu"])
        write(f, "model/DeltaX", args["DeltaX"])
        write(f, "model/DeltrY", args["DeltaY"]) # preserve key spelling
        write(f, "model/Hoping", args["ht"])     # preserve key spelling
        write(f, "model/Nv", args["Nv"])
        write(f, "model/seed", args["seed"])
        write(f, "model/Lx", args["Lx"])
        write(f, "model/Ly", args["Ly"])
        write(f, "model/delta", args["delta"])
        write(f, "cor/rhoup", rhoup)
        write(f, "cor/rhodn", rhodn)
        write(f, "cor/kappa", kappa)
    end
    return nothing
end
