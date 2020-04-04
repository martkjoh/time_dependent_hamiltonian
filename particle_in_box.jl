using PyPlot
using Printf

include("utils.jl")


function plot_vector(N, nev)

    V(x) = zeros(size(x))
    H = make_H0(N, V)
    l, v = eigs(H, nev=nev, which=:SM)

    x = LinRange(1/(N-1), 1-1/(N-1), N-1)
    fig, ax = subplots(nev, sharex=true)
    ax[end].set_xlabel("\$x\$")
    
    for i in 1:nev
        if v[2, i]-v[1, i]<0 v[:, i]=-v[:, i] end
        
        ax[i].plot(x, v[:, i], label="\$\\psi_$i\$")
        
        psi = sin.(pi*i*x)
        c = sqrt(psi' * psi)
        ax[i].plot(x, psi/c, "--k", label="\$c\\sin($i\\pi x)\$")
        
        ax[i].set_ylabel("\$\\psi(x)\$")
        ax[i].legend(loc=1)
    end
    
    ax[1].set_title("\$N=$N\$")
    fig.tight_layout()
    
    fig.savefig("figs/particle_in_box/vector_N=$N.png")
    close(fig)
end


function plot_value(Ns, nev)
    V(x) = zeros(size(x))
    k = size(Ns)[1]
    fig, ax = subplots(k, 2, sharex=true, figsize = (10, 6))
    
    for i in 1:k
        H = make_H0(Ns[i], V)
        l, v = eigs(H, nev=nev, which=:SM)
        
        n = 1:nev
        ax[i, 1].plot(n, l, label="\$ E_n \$")
        ax[i, 1].plot(n, (pi*n).^2, "--k", label="\$(n \\pi)^2\$")
        ax[i, 2].plot(n, l-(pi*n).^2, label = "\$ (n\\pi)^2 - E_n \$")

        ax[i, 1].set_title("\$N=$(Ns[i])\$")
        ax[i, 1].legend()
        ax[i, 2].legend()
    end
    
    ax[1, 1].set_ylabel("\$E / [2mL/\\hbar^2] \$")
    ax[end, 1].set_xlabel("\$n\$")
    ax[end, 2].set_xlabel("\$n\$")
    ax[1, 1].set_title("eigenvalues")
    ax[1, 2].set_title("\$\\Delta E_n \$")
    fig.tight_layout()
    
    fig.savefig("figs/particle_in_box/values.png")
    close(fig)
end

function plot_error(N, ns)
    V(x) = zeros(size(x))
    fig, ax = subplots(1, 3, figsize=(12, 3))

    for i in 1:size(ns)[1]
        n = ns[i]
        Ns = n+2:N
        k = size(Ns)[1]
        v_num = Array{Float64}(undef, k)
        v_ana = (pi*n)^2

        for j in 1:k
            H = make_H0(Ns[j], V)
            l, v = eigs(H, nev=n, which=:SM)
            v_num[j] = l[n]
        end

        Ns = n+2:N
        k = size(Ns)[1]
        label = "\$$(@sprintf("%.2f", (v_ana-v_num[1]) * Ns[1]^2)) / N^2\$"
        err = map(x -> (v_ana-v_num[1]) * Ns[1]^2/(x)^2, Ns)
        ax[i].plot(Ns, err, "--k", label = label)
        
        label = "\$ ($n\\pi)^2 - E_$n\$"
        ax[i].plot(Ns, v_ana*ones(k) - v_num[1:k], "x", label = label)
        
        ax[i].set_title("Error for eigenvalue n = $n")
        ax[i].set_xlabel("N")
        ax[i].legend()
    end

    ax[1].set_ylabel("\$ \\Delta E  / [2mL/\\hbar^2] \$")
    fig.suptitle("\$ N = $N \$", y=1)

    fig.tight_layout()
    fig.savefig("figs/particle_in_box/error.png")
    close(fig)
end


function particle_in_box()
    nev= 3
    N = 10
    plot_vector(N, nev)
    N = 100
    plot_vector(N, nev)
    Ns = [20, 100, 5000]
    plot_value(Ns, 10)
    ns = [1, 5, 10]
    N = 50
    plot_error(N, ns)

    N = 100
    nev = 5
    V(x) = 0*x
    H = make_H0(N, V)
    l, v = eigs(H, nev = nev)
    check_ortho(v, 5)
end
