using PyPlot
using LinearAlgebra

include("utils.jl")

pygui(true)



function V(x, Vr, V0)
    N = size(x)[1]
    V = zeros(N)
    start = Int(round(N/3))
    num = Int(start*2-start)
    V[start:start+num-1] = V0*ones(num)
    V[start+num:end] = Vr*ones(N-(start+num)+1)
    return V
end

function plot_eigenvals_1(N, V0)
    Vrs = LinRange(-100, 100, 11)
    for Vr in Vrs
        H = make_H0(N, x->V(x, Vr, V0))
        l, v = eigs(H, nev = 2, which=:SM)
        diff = l[1] - l[2]
        print(diff)
        plot(Vr, diff, ".")
    end
end


V0 = 100
N = Int(1e5)

n = 6
Vrs = LinRange(-100, 100, n)
H = make_H0(N, x->V(x, 0, V0))
l1, g0 = get_eigs(H, 1)
vs = Array{Float64}(undef, (n, 2))
hs = Array{Float64}(undef, (n, 2, 2))
for i in 1:n
    H = make_H0(N, x->V(x, Vrs[i], V0))
    l, v = get_eigs(H, 2)
    tau = inner(v[:, 2], H*g0)
    diff = (l[2] - l1[1])/2
    hs[i, :, :] = [[-diff, tau] [tau, diff]]
    m = ceil(Int, N/2)
    v1 = ((v[:, 1] + v[:, 2])/sqrt(2))[1:m]
    v2 = ((v[:, 1] + v[:, 2])/sqrt(2))[m:end]
    vs[i, :] = [sqrt(inner(v1, v1)), sqrt(inner(v2, v2))]
end

for i in 1:n
    print(eigvecs(hs[i, :, :]), "\n")
    print(vs[i, :], "\n")
    print("\n")
end