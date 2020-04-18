import numpy as np
from numpy import ceil, sin
from numpy.linalg import eig

from plotting import *
from utils import get_eig, get_H_eff, inner, get_x, get_prob

FIG_PATH = "figs/periodic_detuning/"
V0 = 1e2
N = 10_000
eps = 1

def V(x, Vr, V0 = V0):
    n = len(x)
    m = int(ceil(n/3))
    V = np.zeros(n)
    V[m:2*m] = V0*np.ones(m)
    V[2*m:] = Vr*np.ones(n-2*m)
    return V

def plot_vecs(N, Vr):
    l, v = get_eig(N, lambda x: V(x, Vr), 2)
    x = get_x(N)

    fig, ax = plt.subplots()
    ax.plot(x, v[:, 0])
    ax.plot(x, v[:, 1])
    ax2 = ax.twinx()
    ax2.plot(x, V(x, Vr), "k--")

    y1 = np.max(abs(v))
    y2 = np.max(abs(V(x, Vr)))
    ax.set_ylim(-y1*1.1, y1*1.1)
    ax2.set_ylim(-y2*1.1, y2*1.1)
    plt.plot()
    plt.show()

def plot_diff_vals(N, Vrs):
    n = len(Vrs)
    diff1 = []
    diff2 = []
    l0, v0 = get_eig(N, lambda x: V(x, 0, V0), 2)
    for i in range(n):
        Vr = Vrs[i]

        l1, _ = get_eig(N, lambda x: V(x, Vr), 2)

        H_eff = get_H_eff(N, l0, v0, V, Vr)
        l2, _ = eig(H_eff)

        diff1.append(l1[1] - l1[0])
        diff2.append(l2[1] - l2[0])

    fig, ax = plt.subplots()
    ax.plot(Vrs, diff1, label="full system")
    ax.plot(Vrs, diff2, label="effective system")
    ax.legend()
    ax.set_ylabel("$\Delta E/ [2mL/\hbar^2]$")
    ax.set_xlabel("$V_r/ [2mL/\hbar^2]$")
    plt.show()

def plot_H_eff_vecs(N, Vrs):
    n = len(Vrs)
    fig, ax = plt.subplots(2)
    for i in range(n):
        Vr = Vrs[i]
        H_eff = get_H_eff(N, V, Vr)
        l, v = eig(H_eff)
        indx = np.argsort(l)
        v = v[:, indx]
        l, v2 = get_eig(N, lambda x:V(x, Vr), 2)
        x = get_x(N)

        ax[0].plot([0.25, 0.75], abs(v[:, 0])**2, "kx")
        ax2 = ax[0].twinx()
        ax2.plot(x, abs(v2[:, 0])**2, color=color(i, n))
        ax[1].plot([0.25, 0.75], abs(v[:, 1])**2, "kx")
        ax3 = ax[1].twinx()
        ax3.plot(x, abs(v2[:, 1])**2, color=color(i, n))

    plt.show()

def plot_prob():
    N = 2000
    T = 500
    tau = 0.02

    l0, v0 = get_eig(N, lambda x: V(x, 0, 0), 2)
    p = get_prob(v0[0], N, T, tau)

    t = np.linspace(0, T, N)
    plt.plot(t, p)
    plt.plot(t, sin(t*tau/2)**2)
    plt.show()


Vrs = np.linspace(-10, 10, 51)

# plot_vecs(N, 0)
# plot_diff_vals(N, Vrs)
# plot_H_eff_vecs(N, [-0.05])
plot_prob()
