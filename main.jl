using SparseArrays
using Arpack
using PyPlot

function make_H0(N, V)

    I = Array{Int}(undef, 3*(N-1) - 2)
    J = Array{Int}(undef, 3*(N-1) - 2)
    H = Array{Float64}(undef, 3*(N-1) - 2)
    x = LinRange(0, 1, N-1)

    I[1:3:end] = J[1:3:end] = 1:N-1     # Diagonal
    I[2:3:end] = J[3:3:end] = 1:N-2     # To the right (ii+1)
    I[3:3:end] = J[2:3:end] = 2:N-1     # To the left  (ii-1)
    H[1:3:end] = V(x) + 2*N^2*ones(N-1)
    H[2:3:end] = H[3:3:end] = -N^2*ones(N-2)

    return sparse(I, J, H)
end

function plot_vector_particle_in_box(N, nev)

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

function plot_value_particle_in_box(Ns, nev)
    V(x) = zeros(size(x))
    k = size(Ns)[1]
    fig, ax = subplots(k, 2)
    for i in 1:k
        H = make_H0(Ns[i], V)
        l, v = eigs(H, nev=nev, which=:SM)
        n = 1:nev
        ax[i, 1].plot(n, l, label="\$N=$(Ns[i])\$")
        ax[i, 1].plot(n, (pi*n).^2, "--k", label="\$(\\pi n)^2\$")
        ax[i, 2].plot(n, l-(pi*n).^2)
        ax[i, 1].legend()
    end
    ax[1, 1].set_title("eigenvalues")
    ax[1, 2].set_title("\$\\Delta E_n\$")
    fig.tight_layout()
    fig.savefig("figs/particle_in_box/values.png")
    close(fig)
end

function particle_in_box()
    nev= 3
    N = 10
    plot_vector_particle_in_box(N, nev)
    N = 100
    plot_vector_particle_in_box(N, nev)
    Ns = [50, 100, 5000]
    plot_value_particle_in_box(Ns, 10)
end

particle_in_box()