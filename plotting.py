from matplotlib import pyplot as plt
from matplotlib import cm


font = {'family' : 'serif', 
        'weight' : 'normal', 
        'size'   : 11}
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc("lines", lw=2)
plt.rc('font', **font)

def color(i, N):
        return cm.coolwarm((i + N/2)/(2*N))