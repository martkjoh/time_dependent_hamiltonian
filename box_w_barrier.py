import numpy as np
from numpy import ceil, sqrt, pi

from plotting import *
from utils import get_eig, roots, f, get_x, time_evolve, time_evolve_step, pade_step


def V(x, V0):
    n = len(x)
    m = int(ceil(n/3))
    V = np.zeros(n)
    V[m:2*m] = V0*np.ones(m)
    return V

def plot_eigvecs(N, V, nev):
    l, v = get_eig(N, V, nev)
    fig, ax = plt.subplots()
    x = np.linspace(1/N, 1-1/N, N-1)
    ax.plot(x, V(x))
    ax2 = ax.twinx()
    for i in range(nev):
        ax2.plot(x, v[:, i].real, color=cm.viridis(i/nev))

    plt.show()

def plot_roots(V0):
    fig, ax = plt.subplots()
    l = roots(f, 0.1, V0)
    ls = np.linspace(0, V0, 1000)
    ax.plot(ls, f(ls, V0))
    ax.set_title("${}$ roots".format((len(l))))
    ax.plot(ls, np.zeros_like(ls), "--")
    for a in l:
        ax.plot(a, 0, "x")
    plt.show()

def plot_superpos(N, V0):
    l, v = get_eig(N, lambda x:V(x, V0), 2)
    alpha = np.array([1, 1]) / sqrt(2)
    x = get_x(N)

    fig, ax = plt.subplots()
    ax.plot(x, V(x, V0))
    ax2 = ax.twinx()
    ax2.plot(x, time_evolve(v, l, 0, alpha))
    T = pi / (l[0]-l[1])
    ax2.plot(x, time_evolve(v, l, T, alpha), "--")
    plt.show()

def plot_time_evolve_step(N, V0, step, dt):
    l, v = get_eig(N, lambda x : V(x, V0), 1)
    x = get_x(N)
    T = 1
    print("I'm walking here")
    A = step(N, lambda x:V(x, V0), dt)
    v_new = time_evolve_step(A, v, int(T/dt))
    fig, ax = plt.subplots()
    ax.plot(x, V(x, V0), "k--")
    ax2 = ax.twinx()
    ax2.plot(x, v)
    ax2.plot(x, v_new.real, "--")
    ax2.plot(x, v_new.imag, "--")
    plt.show()

    

V0 = 1e3
N = int(1e4)
nev = 4

# plot_eigvecs(N, lambda x: V(x, V0), nev)
# plot_roots(V0)
# plot_superpos(v)
plot_time_evolve_step(N, V0, pade_step, 0.001)
