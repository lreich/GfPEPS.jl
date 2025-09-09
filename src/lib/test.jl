using Revise
using GfPEPS
using LinearAlgebra, ITensors, ITensorMPS
using JSON: parsefile
using ForwardDiff
using Manopt, Manifolds, LinearAlgebra, ManifoldDiff
using Zygote
using Random
using FiniteDifferences, ManifoldDiff, ADTypes
using Printf
import GfPEPS: energy_loss, G_in_Fourier, GaussianMap, Γ_fiducial, ξ, Δ

# load config
configs = parsefile(joinpath(GfPEPS.config_path, "conf_test.json"))

Random.seed!(configs["params"]["seed"])
Nf = configs["params"]["N_physical_fermions_on_site"]
Nv = configs["params"]["N_virtual_fermions_on_bond"]

Lx = configs["system_params"]["Lx"]
Ly = configs["system_params"]["Ly"]
lattice_type = configs["system_params"]["lattice_type"]

# hamiltonian params
t = configs["hamiltonian"]["t"]
Δ_x = configs["hamiltonian"]["Δ_x"]
Δ_y = configs["hamiltonian"]["Δ_y"]
μ = configs["hamiltonian"]["μ"]
δ = configs["hamiltonian"]["hole_density"]

# initial Γ
Tsize = 8*Nv+Nf*2
T = rand(Tsize,Tsize)

# T = Float64[
#  9.29616093e-01 3.16375555e-01 1.83918812e-01 2.04560279e-01 5.67725029e-01 5.95544703e-01 9.64514520e-01 6.53177097e-01 7.48906638e-01 6.53569871e-01 7.47714809e-01 9.61306736e-01 8.38829794e-03 1.06444377e-01 2.98703714e-01 6.56411183e-01 8.09812553e-01 8.72175914e-01 9.64647597e-01 7.23685347e-01;
#  6.42475328e-01 7.17453621e-01 4.67599007e-01 3.25584678e-01 4.39644606e-01 7.29689083e-01 9.94014586e-01 6.76873712e-01 7.90822518e-01 1.70914258e-01 2.68492758e-02 8.00370244e-01 9.03722538e-01 2.46762104e-02 4.91747318e-01 5.26255167e-01 5.96366010e-01 5.19575451e-02 8.95089528e-01 7.28266180e-01;
#  8.18350011e-01 5.00222753e-01 8.10189409e-01 9.59685257e-02 2.18950044e-01 2.58719062e-01 4.68105754e-01 4.59373203e-01 7.09509780e-01 1.78053006e-01 5.31449884e-01 1.67742229e-01 7.68813918e-01 9.28170549e-01 6.09493658e-01 1.50183495e-01 4.89626704e-01 3.77344954e-01 8.48601412e-01 9.11097229e-01;
#  3.83848721e-01 3.15495903e-01 5.68394153e-01 1.87818035e-01 1.25841544e-01 6.87595805e-01 7.99606718e-01 5.73536565e-01 9.73229982e-01 6.34054377e-01 8.88421725e-01 4.95414759e-01 3.51616530e-01 7.14230369e-01 5.03929116e-01 2.25637607e-01 2.44974440e-01 7.92800700e-01 4.95172415e-01 9.15093673e-01;
#  9.45371834e-01 5.33232230e-01 2.52492595e-01 7.20862058e-01 3.67438764e-01 4.98648443e-01 2.26575047e-01 3.53565647e-01 6.50851787e-01 3.12932895e-01 7.68735447e-01 7.81837103e-01 8.52409483e-01 9.49905740e-01 1.07322912e-01 9.10725356e-01 3.36055162e-01 8.26380427e-01 8.98100635e-01 4.27153043e-02;
#  1.95794999e-01 2.94501322e-01 6.26999881e-01 8.62231051e-02 1.42945020e-01 5.15826519e-01 6.89341330e-01 8.56625811e-01 6.47361683e-01 5.81618676e-01 7.11115955e-01 2.52416857e-01 9.00159683e-01 4.42293693e-01 2.05208247e-02 9.59661014e-01 6.52225422e-01 5.13206250e-01 6.82356383e-01 4.89540391e-01;
#  9.26490171e-01 5.15879772e-01 7.21598817e-02 5.67508298e-01 6.15243184e-01 9.41546294e-01 4.15363355e-01 2.64439975e-01 9.73931654e-02 4.85844222e-01 4.64662863e-01 2.97593170e-02 6.94277462e-01 7.16947112e-01 7.29811423e-01 4.14351017e-01 1.50988448e-02 9.08975157e-01 7.89378718e-01 1.65199169e-01;
#  3.12785961e-01 6.10945306e-01 3.64490287e-01 1.56038589e-01 1.77303813e-01 8.67889671e-01 2.90094668e-01 5.85179621e-01 4.53994876e-01 4.11178132e-01 8.82634445e-01 6.92708015e-01 2.79273355e-01 6.44402312e-02 1.98623614e-01 9.31682745e-01 8.54413568e-01 9.54734735e-01 5.22533482e-02 5.79471681e-01;
#  4.80496267e-01 2.17089790e-02 3.73620464e-01 4.14091801e-01 6.03907234e-01 6.71748727e-01 8.38865700e-01 7.79526208e-01 4.00701044e-01 7.94529231e-01 8.93124310e-01 2.62489691e-01 9.89197007e-01 8.53307099e-01 7.31477156e-01 3.55565625e-01 8.83289491e-01 8.67959091e-01 9.55766446e-01 1.07256520e-04;
#  1.65410419e-02 3.14070057e-01 9.95316175e-01 1.48844223e-01 1.67377119e-01 7.58572037e-01 6.95296662e-02 7.05473439e-01 4.69165282e-01 1.01882161e-02 7.74823863e-01 7.94201008e-01 1.49569452e-01 2.37036318e-02 7.62063771e-01 2.23670180e-01 2.62174398e-01 4.56869503e-01 2.49926919e-01 5.68283562e-01;
#  8.46942997e-01 3.78099537e-01 4.32465116e-01 8.32619176e-01 3.71131936e-01 4.05532719e-02 5.54671486e-01 4.51246243e-01 7.25300977e-01 3.78450522e-01 8.40662496e-01 4.69314694e-01 5.62643429e-01 6.61199339e-01 4.62241868e-01 6.23636945e-01 2.21880631e-01 7.32863117e-01 3.81682089e-01 1.94834542e-01;
#  2.71162775e-01 2.49225052e-01 1.52139062e-01 7.71373704e-01 2.55411732e-01 1.27543405e-01 6.65167153e-01 4.12804949e-01 6.67767766e-01 6.59813517e-01 3.05377672e-01 2.01224241e-01 2.22027126e-01 1.19970863e-01 3.71454410e-02 3.41315835e-02 2.30496548e-01 2.28353706e-01 6.24910622e-01 8.92561185e-01;
#  7.79728016e-01 7.21451269e-01 3.10442159e-01 3.63083416e-01 1.96081784e-01 9.35491798e-01 5.61734000e-01 8.17291537e-01 3.48913981e-01 7.99714263e-01 1.04104463e-01 7.12120165e-01 9.17448496e-01 8.03585369e-01 3.27113146e-01 2.55107179e-01 4.95002174e-01 4.13610954e-01 4.24937095e-01 7.43783048e-02;
#  6.03390651e-01 7.47199734e-01 7.97576227e-01 3.81500523e-01 7.97158153e-01 4.71797370e-01 7.21399171e-01 1.89212689e-01 4.34595709e-01 8.23468109e-01 8.26272628e-01 1.89525483e-01 2.01021702e-02 7.03491386e-01 3.47270119e-01 4.19503816e-01 7.68088903e-01 9.62878066e-01 8.92565307e-01 1.34942267e-01;
#  7.97804715e-01 6.82090608e-01 5.30052887e-01 8.65981655e-01 7.53248095e-01 9.32025868e-02 3.29439432e-01 4.12362696e-01 9.64607766e-04 5.97563563e-01 6.82561542e-01 3.66280762e-01 3.96498857e-01 4.75359146e-01 5.81079901e-01 1.37137128e-01 9.99414173e-01 5.07327209e-01 4.93066705e-01 1.86874755e-01;
#  2.96088015e-01 7.98515059e-01 2.94778293e-01 6.97121593e-01 2.72959229e-01 9.12528451e-01 2.98539234e-01 8.89859772e-01 9.46722294e-01 1.68912823e-01 3.59291406e-01 6.82177303e-01 9.21391970e-01 1.44924344e-01 9.98743850e-02 4.11602777e-01 9.75125371e-01 4.15065409e-02 1.86914295e-01 6.08767988e-01;
#  8.72187151e-02 3.16288154e-01 6.10135777e-01 2.01194713e-01 6.91897716e-01 2.44626971e-01 6.80371696e-01 4.85937729e-01 2.60494812e-01 2.82740319e-01 9.05451620e-01 2.75842687e-01 8.41148043e-01 1.96424375e-01 3.30583429e-01 9.46629885e-01 5.06071850e-01 4.03415858e-01 2.70615384e-02 6.21773689e-01;
#  3.47348507e-01 2.76799228e-01 6.40547739e-02 3.34390519e-01 6.77099425e-02 3.82424826e-03 4.09678034e-02 3.48954866e-01 9.19543108e-02 5.20315726e-01 2.15749813e-01 9.91666545e-01 2.67187161e-01 6.80168053e-01 5.26246119e-01 4.39923070e-01 6.61555246e-01 1.34785175e-01 7.87849005e-01 1.37977701e-01;
#  6.32923298e-01 2.13665358e-01 2.33753147e-02 6.26933973e-01 6.42163727e-01 6.42103337e-01 8.64620900e-01 3.37201403e-01 4.10958387e-01 6.10318684e-01 9.40843191e-01 1.52077270e-01 1.84058006e-01 1.54438288e-01 7.12824326e-01 9.27983161e-01 4.22687180e-01 5.45177264e-01 3.22388965e-01 1.60704683e-01;
#  7.03152025e-01 7.30248406e-01 9.60404538e-01 3.70490571e-01 7.16698392e-01 9.86061218e-01 1.10767559e-01 7.58228549e-01 1.57554121e-01 8.16435255e-01 5.54106546e-01 5.81710221e-01 9.36145583e-01 4.34347755e-01 4.89127186e-01 6.66130467e-01 6.30154001e-01 1.30204751e-01 3.53843971e-01 6.35123385e-01
# ]

# Orthogonalize the initial guess (already done above via SVD; keep as T0)
U,S,V = svd(T)
T = U*V'

if(configs["hamiltonian"]["μ_from_hole_density"])
    μ = solve_mu(Δ_x,δ)
end

E_fct = energy_loss(t, Δ_x, Δ_y, μ, Lx, Ly)
G_in = G_in_Fourier(Lx, Ly, Nv);
CM_out = GaussianMap(Γ_fiducial(T, Nv), G_in, Nf, Nv)
@show E_fct(CM_out)
@time E_fct(CM_out)

# build loss closure
lossT = optimize_loss(t, Δ_x, Δ_y, μ, Lx, Ly, Nf, Nv)

# Manifold
M = Stiefel(Tsize, Tsize)

# cost function is the same, as T is an orthogonal matrix which is in the Stiefel manifold
cost_fct(M,x) = lossT(x)
grad_fct(M,x) = project(M, x, first(Zygote.gradient(lossT, x)))
# grad_fct(M,x) = first(Zygote.gradient(lossT, x))

println("Initial Loss =",cost_fct(M,T))
# display(grad_fct(M,T))
# println(size(grad_fct(M,T)))

# gradient (reverse-mode)
# egrad_zygote(x) = first(Zygote.gradient(lossT, x))
# egrad_zygote(M,x) = egrad_zygote(x)
# gradient (forward-mode)
# egrad_fd(x) = ForwardDiff.gradient(lossT, x)
# X1 = egrad_zygote(T)
# X2 = egrad_fd(T)
# norm(M, T, X1 - X2)

# function grad_f2_AD(M, p)
#     b = Manifolds.RiemannianProjectionBackend(AutoFiniteDifferences(central_fdm(5, 1)))
#     return Manifolds.gradient(M, lossT, p, b)
# end
# X3 = grad_f2_AD(M, T)
# norm(M, T, X1 - X3)


# res = conjugate_gradient_descent(M, cost_fct, grad_fct, T; 
#     coefficient = Manopt.HestenesStiefelCoefficient(M),
#     # stepsize=ArmijoLinesearch(
#     #     M; 
#     #     initial_stepsize=5.0,
#     #     stop_when_stepsize_less=0.1),
#     debug=[:Iteration,
#     (:Cost, " Cost: %1.11f | "), 
#     (:GradientNorm, " Gradient Norm: %1.3e "), "\n",
#     :Stop, 1],
#     stopping_criterion = StopWhenGradientNormLess(1e-7) | StopAfterIteration(150)
# )

using Optim
using ForwardDiff

# very slow...
# g_test(T) = ForwardDiff.gradient(lossT, T)
g(x) = first(Zygote.gradient(lossT, x))
g!(G,x) = copyto!(G, g(x))

# @time lossT(T);
# @time g(T);

bz = BrillouinZone2D(Lx, Ly, (:APBC, :PBC) )

res = Optim.optimize(lossT, g!, T, Optim.ConjugateGradient(manifold=Optim.Stiefel()), Optim.Options(
    iterations = 500,
    g_tol = 1e-7,
    show_trace = true
))

@show Optim.minimum(res)
println("Exact energy:", exact_energy_BCS_k(bz,t,μ,Val(:d_wave),Δ_x,Δ_y))
T_opt = Optim.minimizer(res)

sdsd

#= Now translate X to fiducial state A =#

function permuteG(G::AbstractMatrix, Nv::Integer)
    function permutation_order(Nv)
        if Nv == 1
            order = [8, 6, 0, 1, 10, 4, 9, 7, 2, 3, 11, 5]
        elseif Nv == 2
            order = [12, 16, 6, 10, 0, 1, 14, 18, 4, 8, 13, 17, 7, 11, 2, 3, 15, 19, 5, 9]
        elseif Nv == 3
            order = [16, 20, 24, 6, 10, 14, 0, 1, 18, 22, 26, 4, 8, 12, 17, 21, 25, 7, 11, 15, 2, 3, 19, 23, 27, 5, 9, 13]
        elseif Nv == 4
            order = [20, 24, 28, 32, 6, 10, 14, 18, 0, 1, 22, 26, 30, 34, 4, 8, 12, 16, 21, 25, 29, 33, 7, 11, 15, 19, 2, 3, 23, 27, 31, 35, 5, 9, 13, 17]
        elseif Nv == 5
            order = [24, 28, 32, 36, 40, 6, 10, 14, 18, 22, 0, 1, 26, 30, 34, 38, 42, 4, 8, 12, 16, 20, 25, 29, 33, 37, 41, 7, 11, 15, 19, 23, 2, 3, 27, 31, 35, 39, 43, 5, 9, 13, 17, 21]
        else
            throw(ArgumentError("Nv must be 1,2,3,4,5"))
        end
        return order .+ 1  # convert zero-based Python indices -> 1-based Julia indices
    end

    perm = permutation_order(Nv)
    n = length(perm)
    M = zeros(eltype(G), n, n)
    for i in 1:n
        M[perm[i], i] = 1.0
    end
    P = M
    return transpose(P) * G * P
end

function getG(T::AbstractMatrix, Nv::Integer)
    # local skew (x - xᵀ)
    skew(x) = x .- transpose(x)

    n = 8*Nv + 4
    # build J as Float64 matrix with J[i,j] = 1 iff (j-i==1) && (j % 2 == 0) (1-based indices)
    J = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        J[i,j] = ( (j - i == 1) && (j % 2 == 0) ) ? 1.0 : 0.0
    end
    J = skew(J)

    optim_G = transpose(T) * J * T
    return permuteG(optim_G, Nv)
end

function cor_trans_matrix(cm::AbstractMatrix)
    N = size(cm, 1) ÷ 2
    one = Matrix{eltype(cm)}(I, N, N)
    S = [one  one;
         1im*one  -1im*one]
    return transpose(S) * cm * S
end

function fiducial_hamiltonian(hρ::AbstractMatrix, hκ::AbstractMatrix)
    N = size(hρ, 1)

    @assert size(hρ,2) == N && size(hκ,1) == N && size(hκ,2) == N
    @assert norm(hρ - adjoint(hρ)) < 1e-10
    @assert norm(hκ + transpose(hκ)) < 1e-10

    dim = 1 << N
    H = zeros(ComplexF64, dim, dim)

    for i0 in 0:(N-1)
        for j0 in 0:(N-1)
            for k in 0:(dim-1)
                # bk: LSB-first bit vector of length N, matching Python's reversed bitarray
                bk = [(k >> p) & 1 for p in 0:(N-1)]

                # parity = (count of 1s to the right of i) + (count to the right of j) mod 2
                count_i = (i0+2 <= N) ? sum(bk[(i0+2):N]) : 0
                count_j = (j0+2 <= N) ? sum(bk[(j0+2):N]) : 0
                parity = (count_i + count_j) % 2

                if bk[j0+1] == 1
                    if bk[i0+1] == 0 || i0 == j0
                        bk2 = copy(bk)
                        bk2[j0+1] = 0
                        bk2[i0+1] = 1
                        target = sum(bk2 .* (1 .<< (0:(N-1))))
                        parity += (j0 > i0) ? 1 : 0
                        H[target+1, k+1] += 2 * hρ[i0+1, j0+1] * (-1)^parity
                    elseif bk[i0+1] == 1
                        bk2 = copy(bk)
                        bk2[i0+1] = 0
                        bk2[j0+1] = 0
                        target = sum(bk2 .* (1 .<< (0:(N-1))))
                        parity += (i0 > j0) ? 1 : 0
                        H[target+1, k+1] += hκ[i0+1, j0+1] * (-1)^parity
                    end
                else # bk[j0+1] == 0
                    if bk[i0+1] == 0
                        bk2 = copy(bk)
                        bk2[j0+1] = 1
                        bk2[i0+1] = 1
                        target = sum(bk2 .* (1 .<< (0:(N-1))))
                        parity += (i0 > j0) ? 1 : 0
                        H[target+1, k+1] -= conj(hκ[i0+1, j0+1]) * (-1)^parity
                    end
                end
            end
        end
    end

    return (H + adjoint(H)) / 2
end

function translate(Gamma::AbstractMatrix)
    @assert size(Gamma,1) % 2 == 0
    N = size(Gamma,1) ÷ 2

    trans_h = cor_trans_matrix(-Gamma)

    hρ = -1im * transpose(trans_h[1:N, N+1:2N])
    hκ =  1im * trans_h[1:N, 1:N]

    local_h = fiducial_hamiltonian(hρ, hκ)
    ev = eigen(Hermitian(local_h))
    return ev.vectors[:, 1]   # lowest eigenvector (eigenvalues are in ascending order)
end

function paritygate(Nv::Integer)
    n = 1 << Nv
    p = [ isodd(popcount(i)) ? -1 : 1 for i in 0:(n-1) ]
    return Diagonal(p)   # can be used as a matrix-mult operator
end

function fsign(n_list::AbstractVector{<:Integer})
    result = 0
    L = length(n_list)
    for i in 2:L            # Python loop: for i in range(1,len(n_list))
        result += n_list[i] * sum(n_list[1:(i-1)])
    end
    return (-1)^(result % 2)
end

function bondgate(Nv::Integer)
    n = 1 << Nv
    p = similar(collect(0:(n-1)), Float64)
    for i in 0:(n-1)
        # MSB-first bit list (to match Python int2ba(..., Nv).to01())
        bits = [ ((i >> (Nv-1-j)) & 1) for j in 0:(Nv-1) ]
        p[i+1] = fsign(bits)
    end
    return Diagonal(p)
end

function add_gates(tensor::Array, Nv::Integer)
    # helper: multiply matrix G along axis `ax` of `tensor`
    function apply_gate_along_axis(tensor, G, ax::Int)
        dims = size(tensor)
        nd = length(dims)
        perm = (ax, setdiff(1:nd, ax)...)
        tperm = permutedims(tensor, perm)
        mat = reshape(tperm, dims[ax], :)
        mat = G * mat
        newshape = (size(G,1), dims[setdiff(1:nd, ax)]...)
        tperm = reshape(mat, newshape)
        # invert permutation
        invperm = similar(perm)
        for i in 1:length(perm)
            invperm[perm[i]] = i
        end
        return permutedims(tperm, invperm)
    end

    tensor = apply_gate_along_axis(tensor, Matrix(bondgate(Nv)), 1) # "ulfdr,iu->ilfdr" -> axis 1
    tensor = apply_gate_along_axis(tensor, Matrix(bondgate(Nv)), 2) # "ulfdr,il->uifdr" -> axis 2
    tensor = apply_gate_along_axis(tensor, Matrix(paritygate(Nv)), 1) # parity on u

    return tensor
end

X = arr = Float64[
 0.32182476 -0.25160221 -0.20778703 -0.22893257  0.29964938  0.09461802  0.32069745  0.26584104  0.14479773 -0.18062258  0.01197357  0.32001403 -0.42662338 -0.12585553 -0.02032329 -0.22732544  0.07470626  0.12077167 -0.07415359  0.1831205;
 -0.06704315  0.02736454  0.17697064 -0.27316797 -0.16483162 -0.02998903  0.39696806 -0.11075602  0.17138633 -0.03410764 -0.26063392 -0.02348476  0.33811788 -0.51492781  0.24619506 -0.06912271  0.12353865 -0.02175725  0.33797201  0.13059465;
 0.40408303 -0.13113194  0.39384301 -0.19203238 -0.13996641 -0.12504492  0.01103004  0.07817102  0.03410992 -0.18194277  0.09220374 -0.22565693  0.00622671  0.55989123  0.32350117 -0.0144991   0.16758994  0.00226533  0.19623787  0.08604134;
 -0.22974298  0.18533797 -0.20960165 -0.38431499 -0.13521574  0.1490139   0.18007881 -0.19041675  0.2427294   0.19365957  0.36734339  0.20731648  0.23416845  0.37709532 -0.05051022 -0.32523337 -0.06113229 -0.09158816 -0.10431307  0.09668693;
 0.41619984  0.08386979  0.03659977  0.253311    0.01604985  0.10656737  0.04761811 -0.14132078  0.10629512 -0.29312246  0.14955027  0.36546254  0.35343352  0.02224001 -0.44532431  0.1039751  -0.0321571  -0.03526368  0.32771079 -0.15175742;
 0.03128016  0.00642671 -0.03260298 -0.24493745 -0.30266599 -0.10588869  0.00799272  0.49682558 -0.00608683  0.25982837  0.43016476 -0.15168056 -0.09894156 -0.19933893 -0.31082521  0.32413129 -0.0019616  -0.05294008  0.23575591  0.05496858;
 0.07248414  0.24495539 -0.14345483  0.01938706  0.23595429  0.29074604 -0.08431563  0.33036113 -0.31633585  0.06203892  0.02020298 -0.0434664   0.28523938  0.04103562  0.26278825 -0.10391476 -0.37229949  0.3789929   0.26422015  0.19016204;
 0.23681977 -0.10155288 -0.04399597 -0.13895121 -0.11969376  0.35145368 -0.17009239 -0.30095576 -0.07878623  0.16645361  0.06735895 -0.01163718  0.1310392  -0.11272688 -0.02581888  0.29768616  0.40878377  0.43960153 -0.28681633  0.23375618;
 -0.23156331 -0.62041661  0.06898293  0.07544731  0.2435977   0.16907928  0.07790648  0.05311348 -0.18991192  0.18974533  0.16573174 -0.1312921   0.21318331  0.033734   -0.12203599 -0.26801952  0.26321912  0.04647323  0.21545715 -0.2936664;
 -0.25739156 -0.14433485  0.53314181  0.19745989 -0.22098269  0.43823971 -0.0726974   0.0651099   0.19882236 -0.04481537  0.20061855  0.31743248 -0.22177223 -0.0855794   0.17829209  0.04770965 -0.2411247   0.06692204 -0.00860384  0.0588231;
 0.26961823 -0.21905729  0.03459341  0.08989195  0.02850331 -0.41991624  0.11616108 -0.04177462  0.21418597  0.38916605  0.14488029  0.15326419  0.19295346 -0.08759092  0.25008582  0.07722353 -0.34685172  0.26489886 -0.23750187 -0.27488495;
 -0.07144765 -0.22080729 -0.11218845  0.36073619 -0.02199652 -0.05523274  0.21997405 -0.13966376  0.25189704  0.26946628 -0.21518307 -0.15882873 -0.08651958  0.24595788 -0.21587228  0.10434522 -0.15127522  0.10166867  0.24477309  0.55298707;
 0.11116874  0.13645606  0.07501773  0.16405371 -0.39819213  0.17683221  0.36531294  0.35408079 -0.19453442  0.25520767 -0.41825545  0.15144849  0.05147953  0.1995642  -0.12761302 -0.11011875  0.14712189  0.03357348 -0.22267484 -0.20912773;
 -0.10645341  0.42985203  0.28167984 -0.08664914  0.37380171 -0.05963819  0.18051651 -0.18507639  0.0116581   0.30093727  0.01343914  0.09162175 -0.34032387  0.10448381 -0.06733026  0.13577199  0.20457676  0.28787084  0.2818925  -0.23185301;
 0.09858148  0.21701615  0.09347522  0.45712409 -0.06548413 -0.22138798  0.15838944 -0.06768088 -0.2796683   0.0592957   0.44609743  0.0048945  -0.04851462 -0.19995653  0.12047591 -0.35872951  0.26628825 -0.06710712 -0.08249685  0.30660003;
 -0.0812825   0.20086841 -0.07176991  0.26278749  0.14637507  0.04858429 -0.15579751  0.38414887  0.66103941 -0.06214931  0.02518966 -0.19025627  0.18180714 -0.01041951  0.07812519 -0.06661442  0.3599183   0.13411699 -0.09761756 -0.06912856;
 -0.24616228 -0.02051588  0.17490895 -0.00434747  0.36604817 -0.08403051  0.36384598  0.18834433 -0.14671086 -0.17705529  0.09197046  0.11527715  0.30169655  0.13479097  0.01476432  0.48864263  0.05568537 -0.15196573 -0.32223393  0.21594702;
 -0.16909968 -0.10297849 -0.38098595  0.11574486 -0.15423213 -0.15275813 -0.15788059  0.06486962 -0.0744108   0.05500171 -0.02995319  0.52359189 -0.02949278  0.13531709  0.40359149  0.24031745  0.31134752 -0.0573264   0.31974144 -0.0105602;
 0.20567542  0.04753458 -0.28061788  0.17352263  0.04176036  0.43715804  0.34192108 -0.15114663  0.06161     0.08382932  0.17333003 -0.30202046 -0.14426792 -0.02294337  0.3264113   0.2600063  -0.05648696 -0.34921911  0.03941839 -0.24434925;
 0.26674169  0.0199187   0.2131009  -0.07542839  0.30051761  0.11815181 -0.3252556   0.07454134  0.05494982  0.49403435 -0.16421909  0.17782009  0.09812812 -0.06511428  0.01452149 -0.05913561  0.0528474  -0.5418933  -0.00349947  0.20499933
]


Gamma = getG(X, Nv)
tensor_0 = translate(Gamma)

tensor_1 = reshape(tensor_0, (1<<Nv, 1<<Nv, 4, 1<<Nv, 1<<Nv))
tensor_1 = permutedims(tensor_1, (5,4,3,2,1))  # matches Python .transpose(4,3,2,1,0)


sdsd

tensor_final = add_gates(tensor_1, Nv)