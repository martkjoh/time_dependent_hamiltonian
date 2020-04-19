import numpy as np
from numpy import pi, sin, sqrt

from plotting import *
from utils import get_eig

FIG_PATH = "figs/particle_in_box/"

def V(x):
    return np.zeros_like(x)

def plot_eigvecs(N, V, nev):
    l, v = get_eig(N, V, nev)
    x = np.linspace(1/N, 1-1/N, N-1)
    fig, ax = plt.subplots(nev, sharex=True, figsize=(5, 6))
    ax[0].set_title("$N={}$".format(N))
    for i in range(nev):
        v_exact = sin(pi*(i+1)*x)
        c = 1/sqrt(v_exact@v_exact)
        ax[i].plot(x, v[:, i], "-", color=color(i, nev), label="$\psi_{}(x)$".format(i))
        ax[i].plot(x, c*v_exact, "k:", label="$c\sin(2\pi{}x)$".format(i+1))
        ax[i].legend(loc=1)
        ax[i].set_ylabel("$\psi/[1]$")
    ax[-1].set_xlabel("$x / [L]$")

    plt.tight_layout()
    plt.savefig(FIG_PATH + "vector_N={}.pdf".format(N))

def plot_eigvals(Ns, V, nev):
    n = np.arange(1, nev+1)
    m = len(Ns)
    fig, ax = plt.subplots(m, 2, figsize=(12, 6), sharex=True)
    for i in range(m):
        N = Ns[i]
        l, v = get_eig(N, V, nev)
        ax[i, 0].plot(n, l, label="$E_n$")
        ax[i, 0].plot(n, (pi*n)**2, "k--")
        ax[i, 0].set_ylabel("$E / [2mL/\hbar^2]$")
        ax[i, 0].set_title("$N={}$".format(N))
        ax[i, 0].legend()
        ax[i, 1].plot(n, abs((l - (pi*n)**2)/pi**2), label="$|E_n-(\pi n)^2|/\pi^2$")
        ax[i, 1].set_ylabel("rel. error")
        ax[i, 1].legend()
    ax[-1, 0].set_xlabel("$n$")
    ax[-1, 1].set_xlabel("$n$")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "values.pdf")

def plot_error(ns, N_max, V):
    Ns = np.arange(10, N_max+1)
    m = len(ns)
    fig, ax = plt.subplots(1, m, figsize=(12, 3), sharex=True)
    for i in range(m):
        n = ns[i]
        ls = []
        for N in Ns:
            l, v = get_eig(N, V, n)
            ls.append(l[n-1])
        ls = np.array(ls)
        ax[i].plot(Ns, abs(ls-(pi*n)**2)/pi**2, label="$|E_{}-(\pi {})^2|/\pi^2$".format(n, n))
        C = (Ns[0]**2*abs(ls[0]-(pi*n)**2)/pi**2)
        ax[i].plot(Ns, C/Ns**2, "k:", label="${:.3f}/N^2$".format(C))
        ax[i].set_ylabel("rel. error")
        ax[i].set_xlabel("$N$")
        ax[i].legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "error.pdf")


plot_eigvecs(10, V, 3)
plot_eigvecs(100, V, 3)
plot_eigvals([20, 1_000], V, 10)
plot_error([1, 5, 8], 50, V)