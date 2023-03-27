import numpy as np
import time
import matplotlib.ticker
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import make_interp_spline
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import cmath
import os



from sympy.functions.special.delta_functions import Heaviside
start_time = time.time()

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
font = {'family' : 'normal',
    'weight' : 'bold',
        'size'   : 25}
rc('font', **font)


import csv
package_dir = os.path.dirname(os.path.abspath(__file__))
##############importing the .txt data#########################################
Parameters    = np.loadtxt(fname = 'arc_chi_100.txt')
Lifetime = np.loadtxt(fname = 'PDpchannel_test.txt')
#######################plot##########################################

def mkPlot(args, fout="PDtestpion.pdf"):
    
    import matplotlib.pyplot as plt
    scale = 1.25
    fig, ax = plt.subplots(1,1,figsize=(14,14))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    
    plt.rcParams['font.size'] = 20
    x1 = Parameters[:,2] #free parameter on the x axis
    y1 = Lifetime #lifetime on the y axis
    z = np.log10(Parameters[:,6]) #still lifetime
   # presentbound = (33.84-z.min())/(z.max()-z.min())
    #futurebound =  (34.5-z.min())/(z.max()-z.min())

    #colors = [(0,(0,0,0)),(presentbound,(0,0.7,0)),(futurebound,(0.7,0,0)), (1,(0.7,0,0))] #B->R->G
    #n_bin = 100
    #cmap_name = 'my_list'
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N = n_bin)
    #colormap2 = cmap
   
    #colormap2 = plt.cm.bwr
    #colormap2 = plt.cm.PuBu
    #or any other colormap#
    #normalize1 = matplotlib.colors.Normalize(vmin=z1.min(), vmax=z1.max())
    #normalize = matplotlib.colors.Normalize(vmin=z.min(), vmax=z.max())

    ##boundary norm

    cmapp = matplotlib.colors.ListedColormap(['r','g','b'])
    boundary_norm = matplotlib.colors.BoundaryNorm([z.min(), 33.84,34.5,z.max()],cmapp.N)

    plot1 = ax.scatter(x1, y1, c='black', s=10, cmap=cmapp, norm=boundary_norm, marker="o")
    #ax.set_ylim([-10,10])
    ax.set_yscale('log')
    #ax.set_xlim([-0.03,0.03])
    #ax[0,0].set(xlabel=r"$a_1$ (°)", ylabel=r"$a_2$ (°)")
    
    cax1 = fig.add_axes([0.175, .04, 0.7, 0.03])
    cbar1= fig.colorbar(plot1,cax1, orientation = 'horizontal')
    #cax2 = fig.add_axes([0.590, .48, 0.3, 0.02])
   # cbar2= fig.colorbar(plot2,cax2, orientation = 'horizontal')

    plt.savefig(fout)
    #    plt.show()
    exit()
if __name__ == "__main__":
    import sys
    fnames = sys.argv[1:]
    mkPlot(fnames)
