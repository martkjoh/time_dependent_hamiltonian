import numpy as np
from numpy import ceil, sqrt, pi

from plotting import *
from utils import get_eig, roots, f, get_x, time_evolve, pade_step, euler_step, inner

FIG_PATH = "figs/box_w_barrier/"
N = 10_000
V0 = 1e3

def V(x, V0 = V0):
    n = len(x)
    m = int(ceil(n/3))
    V = np.zeros(n)
    V[m:2*m] = V0*np.ones(m)
    return V

def plot_eigvecs(N, nev):
    l, v = get_eig(N, V, nev)
    x = np.linspace(1/N, 1-1/N, N-1)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(x, V(x), "--k")
    ax.set_ylabel("$E / [2mL/\hbar^2]$")
    ax.set_xlabel("$x / [L]$")
    ax.set_title("$N={}$".format(N))
    ax2 = ax.twinx()
    ax2.set_ylabel("$\Psi / [1]$")

    for i in range(nev):
        ax2.plot(x, v[:, i].real, 
        color=cm.viridis(i/nev),
        label="$\\psi_{}$".format(i))

    ax2.legend()
    plt.savefig(FIG_PATH + "eigenvecs.pdf")

def plot_eigvals(N, nev):
    l, v = get_eig(N, V, nev)
    n = np.arange(nev) + 1

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(n, l, "o")
    ax.set_xlabel("$n$")
    ax.set_ylabel("$E / [2mL/\hbar^2]$")
    plt.savefig(FIG_PATH + "eigenvals.pdf")

def plot_roots():
    l = roots(f, 0.1, V0)
    ls = np.linspace(0, V0, 1000)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ls, f(ls, V0),  label="$f(x)$")
    ax.set_title("${}$ roots".format((len(l))))
    ax.plot(ls, np.zeros_like(ls), "--k", lw=1)
    ax.set_xlabel("$x / [2mL/\hbar^2]$")
    ax.set_ylabel("$f(x)$")
    for a in l:
        leg = ax.plot(a, 0, "xk", ms=12)
    plt.legend((leg), ("roots", ))
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "roots.pdf")

def plot_error(Ns):
    l = roots(f, 0.1, V0)
    nev = len(l)
    n = np.arange(1, nev+1)
    m = len(Ns)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel("$n$")
    ax.set_ylabel("$|E_i-x_i|/x_0$")
    for i in range(m):
        N = Ns[i]
        l2, v = get_eig(N, V, nev)
        ax.plot(n, abs((l-l2)/l[0]), "x", 
            label="$N={}$".format(Ns[i]),
            ms=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "roots_error.pdf")

def plot_superpos(N):
    l, v = get_eig(N, lambda x:V(x, V0), 2)
    alpha = np.array([1, 1]) / sqrt(2)
    x = get_x(N)

    fig, ax = plt.subplots()
    ax.plot(x, V(x, V0), "k--")
    ax.set_ylabel("$E / [2mL/\hbar^2]$")
    ax.set_xlabel("$x / [L]$")
    ax.set_title("$N={}$".format(N))
    ax2 = ax.twinx()
    ax2.set_ylabel("$\Psi / [1]$")
    ax2.plot(x, time_evolve(v, l, 0, alpha), label="$\Psi(x, 0) \in \\bfR$")
    T = pi / (l[0]-l[1])
    ax2.plot(x, time_evolve(v, l, T, alpha).real, label="$\Re(\\Psi(x, T))$")
    ax2.plot(x, time_evolve(v, l, T, alpha).imag, label="$\Im(\\Psi(x, T))$")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_PATH + "super_pos.pdf")

def plot_time_evolve(N):
    l, v = get_eig(N, lambda x:V(x, V0), 2)
    alpha = np.array([1, 1]) / sqrt(2)
    x = get_x(N)

    fig, ax = plt.subplots()
    ax.plot(x, V(x, V0), "k--")
    ax.set_ylabel("$E / [2mL/\hbar^2]$")
    ax.set_xlabel("$x / [L]$")
    ax.set_title("$N={}$".format(N))
    ax2 = ax.twinx()
    ax2.set_ylabel("$|\Psi|^2 / [1]$")
    n = 5
    T = pi / (l[0]-l[1]) / n
    for i in range(n+1):
        v_new = time_evolve(v, l, T*i, alpha)
        label = "$|\\Psi(x, {}T/{})|^2$".format(i, n)
        ax2.plot(x, abs(v_new)**2, label=label, color=color(i, n))
        ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_PATH + "time_evolve.pdf")

def plot_time_evolve_step_error(Ns, step, dts):
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    for N in Ns:
        n = 50
        for dt in dts:
            l, v = get_eig(N, V, 1)
            v = v[:, 0]
            A = step(N, V, dt)
            a = np.empty(n, dtype=np.complex128)
            a[0] = inner(v, v)
            for i in range(1, n):
                v = A@v
                a[i] = inner(v, v)

            print(2**(np.log(a[-1]- a[-2])/60-1))

            ax.plot(np.arange(n), a, label="CFL={}".format(dt*N**2))

    ax.legend()
    plt.show()

def plot_time_evolve_step(N, step, f, T, dt):
    x = get_x(N)
    v0 = f(x)
    v = np.copy(v0)
    print("Making step")
    A = step(N, V, dt)
    n = int(T/dt)
    print("walking {} steps".format(n))
    fig, ax = plt.subplots()
    ax.plot(x, v0)
    for _ in range(3):
        for _ in range(int(n/3)):
            v = A@v
        ax.plot(x, v)
    plt.show()
    

nev = 8
# plot_eigvecs(N, nev)

# plot_eigvals(N, 12)
# plot_roots()
# plot_error([100, 1000, 10_000])

# plot_superpos(N)
# plot_time_evolve(N)
# plot_time_evolve_step_error([1_000, 5_000], euler_step, [1e-6, 5e-6])
# plot_time_evolve_step(1_000, pade_step, lambda x:-1/2*x**2+1/2*x, 1, 0.001)

