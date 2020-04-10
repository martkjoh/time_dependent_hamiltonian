using PyPlot

pygui(true)

include("utils.jl")

function V(x, V0)
    N = size(x)[1]
    V = zeros(N)    
    start = ceil(Int, N/3)
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


function time_evolve(v, l, T, alpha = [1])
    n = size(alpha)[1]
    evolve = alpha .* exp.(-1im*l[1:n]*T)
    v_new = zeros(size(v)[1])
    for i in 1:n v_new += evolve[i] * v[:, i] end
    return v_new
end


function plot_roots(V0)
    fig, ax = subplots()
    l = roots(f, 1., V0)
    print(l, "\n")
    print(f.(l, V0), "\n")
    print((l[1:end-1] + l[2:end])/2, "\n")
    print(f.((l[1:end-1] + l[2:end])/2, V0), "\n")
    ls = LinRange(0, V0, 100000)
    plot(ls, f.(ls, V0))
    ax.set_title("$(size(l)[1]) roots")
    for a in l plot(a, 0, "x") end
    return fig, ax
end


function plot_num_roots()
    n = 200
    V0 = LinRange(0.1, 4, n)
    num_roots = Array{Int}(undef, n)
    for i in 1:n
        l = roots(f, 0.1, 10^V0[i])
        num_roots[i] = size(l)[1]
    end
    plot((10).^V0, num_roots, ".")
    F(x) = sqrt(x)
    plot((10).^V0, num_roots[end]/F((10)^V0[end])*F.((10).^V0), "--", label = "\$ C\\sqrt{V_0}\$")
    legend()
end


function plot_time_evolve1(V0, N)
    H = make_H0(N, x -> V(x, V0))
    l, v = eigs(H, nev=2, which=:SM)
    x = LinRange(1/N, 1-1/N, N-1)
    alpha = [1, 1]/sqrt(2)
    v0 = time_evolve(v, l, 0, alpha)
    v_new = time_evolve(v, l, pi/(l[1] - l[2]), alpha)
    fig, ax = subplots()
    ax.plot(x, V(x, V0))
    ax2 = ax.twinx()
    ax2.plot(x, abs.(v0).^2)
    ax2.plot(x, abs.(v_new).^2, "--")
    show()
end


function plot_wave_func(V0, N, nev)
    H = make_H0(N, x -> V(x, V0))
    l, v = get_eigs(H, nev)
    x = LinRange(1/N, 1-1/N, N-1)
    fig, ax = subplots()
    ax.plot(x, V(x, V0))
    ax2 = ax.twinx()
    for i in 1:nev
        print(l[i], "\n")
        ax2.plot(x, v[:, i], label = "\$\\psi_$i\$")
        ax2.legend()
    end
    savefig("figs/box_w_barrier/wave_func.png")
    close(fig)
end


function plot_timesteps(method)
    N = 1000
    dt = 0.000001
    T = 10*dt
    V0 = 1e2
    nev = 1

    H = make_H0(N, x->V(x, V0))
    l, v = get_eigs(H, nev)
    v0 = v
    x = LinRange(1/N, 1-1/N, N-1)
    for i in 1:10
        v = method(v, H, T, dt)
        fig, ax = subplots()
        plot(x, abs.(v).^2)
        plot(x, abs.(v0).^2, "--")
        sleep(1)
        close(fig)
    end
end

plot_wave_func(1e3, Int(1e4), 4)
print(roots(f, 0.1, 1e3))