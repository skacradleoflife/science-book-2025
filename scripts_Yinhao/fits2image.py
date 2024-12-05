bmaj = 0.025
bmin = 0.022
modelname = 'J_v5_2.4cm_150pc_4096grids'
noise_dev_std =	0.65e-6

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.visualization import (MinMaxInterval, SqrtStretch,ImageNormalize)
from astropy.visualization import quantity_support
import numpy as np
from beam import *

quantity_support()

import os
basedir = os.path.abspath(os.path.dirname(__file__))
from matplotlib import rcParams
rcParams.update({'font.size': 25})

# Open FITS File
fits_file = basedir+'/inputs/'+modelname+'.fits'
hdul = fits.open(fits_file)
data = hdul[0].data
cdelt = np.abs(-5.9839233370598E-08*3600.)
nx = 4096
ny = 4096
image_data = data


############ BEAM and NOISY ############

beam =  (np.pi/(4.*np.log(2.)))*bmaj*bmin/(cdelt**2.)
noise_dev_std_Jy_per_pixel = noise_dev_std / np.sqrt(0.5*beam)  # 1D
noise_array = np.random.normal(0.0,noise_dev_std_Jy_per_pixel,size=nx*ny)
noise_array = noise_array.reshape(nx,ny)
image_data += noise_array

#beam =  (np.pi/(4.*np.log(2.)))*bmaj*bmin/(cdelt**2.)
stdev_x = (bmaj/(2.*np.sqrt(2.*np.log(2.)))) / cdelt
stdev_y = (bmin/(2.*np.sqrt(2.*np.log(2.)))) / cdelt
smooth = Gauss_filter(image_data*1.e6, stdev_x, stdev_y, 0, Plot=False)*1.e3

# figure size
fig = plt.figure(figsize=(8.,8.))
plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)

norm = ImageNormalize(smooth, interval=MinMaxInterval()) #, stretch=SqrtStretch())

wcs_original = WCS(hdul[0].header)

wcs = wcs_original.sub([1, 2])

wcs.wcs.cunit[0] = u.arcsec
wcs.wcs.cunit[1] = u.arcsec

ax = plt.subplot()

dpix = 0.0
a0 = cdelt*(nx//2.-dpix)   # >0
a1 = -cdelt*(nx//2.+dpix)  # <0
d0 = -cdelt*(nx//2.-dpix)  # <0
d1 = cdelt*(nx//2.+dpix)   # >0
# da positive definite
da = np.maximum(abs(a0),abs(a1))
mina = da
maxa = -da
xlambda = mina - 0.166*da
ax.set_ylim(-0.1,0.1)
ax.set_xlim(0.1,-0.1)      # x (=R.A.) increases leftward
dmin = -da
dmax = da

im = ax.imshow(smooth, origin='lower', norm=norm, cmap='magma',extent=[a0,a1,d0,d1])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="2.5%", pad=0.12)
cb =  plt.colorbar(im, cax=cax, orientation='horizontal')
cax.xaxis.tick_top()
cax.xaxis.set_tick_params(labelsize=20, direction='out')
# title on top
cax.xaxis.set_label_position('top')
cax.set_xlabel(r'$\mu$Jy/beam$\times10^{3}$')
cax.xaxis.labelpad = 8

from matplotlib.ticker import FuncFormatter
def custom_formatter(x, pos):
    if x == 0:
        return "0"  
    else:
        return f"{x:.3f}"  
cax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
cax.xaxis.set_major_locator(plt.MaxNLocator(6))

ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.set_xlabel('RA offset [arcsec]')
ax.set_ylabel('Dec offset [arcsec]')

from matplotlib.patches import Ellipse
e = Ellipse(xy=[0.08,-0.08], width=bmaj, height=bmin, angle=0)
e.set_clip_box(ax.bbox)
e.set_facecolor('white')
e.set_alpha(0.8)
ax.add_artist(e)

ax.text(0.09,0.08,'12.5 GHz', color = 'white',weight='bold',horizontalalignment='left')
ax.text(-0.09,0.08,'100 h', color = 'white',weight='bold',horizontalalignment='right')
ax.plot(0,0.05/1.5,'+',color='white',markersize=15)
ax.text(-0.09,-0.09,r'$\mathbf{\alpha}$ = $\mathbf{10^{-5}}$', color = 'white',weight='bold',horizontalalignment='right')

plt.savefig(basedir+'/outputs/ska1_'+modelname+'_100h.png', dpi=300)

