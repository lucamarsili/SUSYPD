import numpy as np
import time
import matplotlib.ticker
import matplotlib as mpl 
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

Lifetime = np.loadtxt(fname = 'PDknuchannel_test.txt')
#######################plot##########################################


def load_parameter2():
        #points = np.loadtxt("s/GUTFIT.txt"%args[0])
        #oldscan = np.loadtxt("ParaTable_10.txt") ##good for testing
        #N = len(points[0,:]);
        Parameters = np.loadtxt("GOODSCAN2404/GUTFITpara.txt")
        likelihood = np.loadtxt("GOODSCAN2404/GUTFITlikelihood.txt")
        F_N_P = 190000
        #print(x[-1])
        return Parameters[F_N_P:], likelihood[F_N_P:]


def mkPlot(args, fout="PDknu.pdf"):
    
    import matplotlib.pyplot as plt
    scale = 1.25
    fig, (ax1,ax2) = plt.subplots(1,2,gridspec_kw={'width_ratios':[30,1]})
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    
    Parameters, likelihood =load_parameter2()
    #The idea is to make a colored band, the color is the likelihood of each point, then we plot a line for each points in function of the susy scale 
    #then we just colour in grey the constrained area
    plt.rcParams['font.size'] = 20
   
    #colormap2 = plt.cm.bwr
    
    
    x1 = np.logspace(4,9,15)#free parameter on the x axis

        
    
    z = likelihood

    print(z)
    cmap = plt.cm.cool
    color_gradients = cmap((z-z.min())/(z.max()-z.min()))
    
    for i,_ in enumerate(z): 
        y1 = Lifetime[i,:]
        ax1.plot(x1,y1,c = color_gradients[i],lw = 0.5) 
    
    
    normalize = matplotlib.colors.Normalize(vmin=z.min(), vmax=z.max())
    
    cb = mpl.colorbar.ColorbarBase(ax2 ,cmap = cmap, norm = normalize, orientation= 'vertical')
    #plot1 = ax.scatter(x1, y1, c=0.1, s=10, cmap=colormap2, norm=normalize1, marker="o")
    ax1.axhline(y=6.5e33, color='black',lw = 1)
    ax1.axhline(y=3e34, linestyle = 'dashed',color='black',lw = 1)
    
    # presentbound = (33.84-z.min())/(z.max()-z.min())
    #futurebound =  (34.5PDknuchannel_test.txt-z.min())/(z.max()-z.min())

    #colors = [(0,(0,0,0)),(presentbound,(0,0.7,0)),(futurebound,(0.7,0,0)), (1,(0.7,0,0))] #B->R->G
    #n_bin = 100
    #cmap_name = 'my_list'
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N = n_bin)
    #colormap2 = cmap
   
    #colormap2 = plt.cm.PuBu
    #or any other colormap#
    
    ##boundary norm

    #cmapp = matplotlib.colors.ListedColormap(['r','g','b'])
    #boundary_norm = matplotlib.colors.BoundaryNorm([z.min(), 33.84,34.5,z.max()],cmapp.N)

    #ax.set_ylim([-10,10])
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim([10e+32,10e+35])
    ax1.set_xlim([10e+04,10e+06])
    #ax[0,0].set(xlabel=r"$a_1$ (°)", ylabel=r"$a_2$ (°)")
    
    #cax1 = fig.add_axes([0.175, .04, 0.7, 0.03])
    #cbar1= fig.colorbar(plot1,cax1, orientation = 'horizontal')
    #cax2 = fig.add_axes([0.590, .48, 0.3, 0.02])
   # cbar2= fig.colorbar(plot2,cax2, orientation = 'horizontal')

    plt.savefig(fout)
    #    plt.show()
    exit()
if __name__ == "__main__":
    import sys
    fnames = sys.argv[1:]
    mkPlot(fnames)
