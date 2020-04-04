using PyPlot
using LinearAlgebra

include("utils.jl")

function V(x,V0=10^3)
    N = size(x)[1]
    V = zeros(N)
    start = Int(round(N/3))
    num = Int(start*2-start)
    V[start:start+num-1] = V0*ones(num)
    return V
end

function plot_eigenvecs(N, v, nev)
    x = LinRange(1/N, 1-1/N, N-1)
    fig, ax = subplots()
    ax2 = ax.twinx()
    ax2.plot(x, V(x))
    for i in 1:nev
        ax.plot(x, v[:, i])
    end
    show()
end

function plot_eigenvals(N, l, nev)
    ns = 1:nev
    fig, ax = subplots()
    ax.plot(ns, l, ".")
    show()
end

function time_evolve(alpha, v, l, T)
    n = size(alpha)[1]
    evolve = alpha .* exp.(-1im*l[1:n]*T)
    v_new = zeros(size(v)[1])
    for i in 1:n v_new += evolve[i] * v[:, i] end
    return v_new
end

function f(l, V0=10^3)
    k = sqrt(l)
    kappa = sqrt(V0 - l)
    core_p = (kappa*sin(k/3) + k*cos(k/3))^2
    core_m = (kappa*sin(k/3) - k*cos(k/3))^2
    return exp(kappa/3)*core_p - exp(-kappa/3)*core_m
end


# returns start and end of consecutive indexes
function get_chunks(indx)
    chunks = [indx[1]]
    old = indx[1]
    for i in 2:size(indx)[1]
        new = indx[i]
        if new-old > 1
            append!(chunks, [old, new])
        end
        old = new
    end
    append!(chunks, [old])
    return convert(Int, chunks)
end


function find_eigenvals(l_max, N, V0=1e3)
    x = LinRange(0, l_max, N)
    vals = f.(x)
    indx = findall(vals .< 0)
    chunks = get_chunks(indx)
    n = size(chunks)[1]
    l = Array{Float64}(undef, n)
    for i in 1:n/2
        l[2*i] = (x[chunks[2*i]] - x[chunks[2*i+1]-1]) / 2
        l[2*i+1] = (x[chunks[2*i+1]] - x[chunks[2*i+1]-1]) / 2
    end
    return l
end

# N = 10000
# nev = 5
# H = make_H0(N, V)
# l, v = eigs(H, nev=nev, which=:SM, tol=1-20)

l = find_eigenvals(400, 100000)
print(l)


# l = eigvals(Matrix(H))
# x = LinRange(1/N, 1-1/N, N-1)
# alpha = [0, 1/sqrt(2), 1/sqrt(2)]

# fig, ax = subplots()
# v0 = time_evolve(alpha, v, l, 0)
# v_new = time_evolve(alpha, v, l, pi/(l[2] - l[1]))

# ax.plot(x, real(v0))
# ax.plot(x, imag(v0), "--")
# ax.plot(x, real(v_new))
# ax.plot(x, imag(v_new),"--")
# show()


# ls = LinRange(73.92, 73.95, 10000)

# print(analytic_eigenvals.(ls))
# plot(ls, zeros(size(ls)))
# plot(ls, analytic_eigenvals.(ls))
# plot(l[1]*ones(10), LinRange(-1, 1, 10))
# plot(l[2]*ones(10), LinRange(-1, 1, 10))
