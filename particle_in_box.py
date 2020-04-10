import numpy as np
from numpy import pi

from plotting import *
from utils import get_eig


def V(x):
    return np.zeros_like(x)

def plot_eigvecs(N, V, nev):
    l, v = get_eig(N, V, nev)
    fig, ax = plt.subplots()
    x = np.linspace(1/N, 1-1/N, N-1)
    ax.plot(x, V(x))
    ax2 = ax.twinx()
    for i in range(nev):
        ax2.plot(x, v[:, i].real, color=cm.viridis(i/nev))

    plt.show()

def plot_eigvals(N, V, nev):
    l, v = get_eig(N, V, nev)
    n = np.arange(1, nev+1)
    fig, ax = plt.subplots()
    ax.plot(n, (pi*n)**2, "--")
    ax.plot(n, l, "x")
    plt.show()



N = int(1e5)
nev = 10
V0 = 1e3
