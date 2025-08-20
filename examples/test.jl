using JLD2

cd(@__DIR__)
jldopen("../X_opt.jld2") do file
    display(file)

    Xopt = read(file, "transformer/T")

    display(Xopt)

    # Read the optimized tensor T
    # Topt = read(file, "Topt")
end