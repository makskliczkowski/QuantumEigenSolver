import os
from .__helper__ import *

from scipy.special import psi
from scipy.special import polygamma
from scipy.special import erf, erfinv
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import struct

FLOAT_SIZE = 8

kPSep = os.sep

colors_ls = (list(mcolors.TABLEAU_COLORS)[:120])
colors_ls_cyc = itertools.cycle(list(mcolors.TABLEAU_COLORS)[:120])
markers_ls = ['o','s','v', '+', 'o', '*']
markers = itertools.cycle(markers_ls)

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['figure.dpi'] = 400
#mpl.rcParams['figure.figsize'] = 400
mpl.rcParams['lines.markersize']='4'
from matplotlib.ticker import ScalarFormatter, NullFormatter


# -------------------------------------------- PAGE --------------------------------------------
def page_result(d_a, d_b):
    if d_a <= d_b:
        return psi(d_a * d_b + 1) - psi(d_b + 1) - (d_a - 1)/(2*d_b)
    else:
        return psi(d_a * d_b + 1) - psi(d_a + 1) - (d_b - 1)/(2*d_a)

def page_result_var(d_a, d_b):
    return (((d_a + d_b)/(d_a*d_b + 1.0)) * polygamma(1, d_b + 1)) - polygamma(1, d_a*d_b + 1) - ((d_a-1)*(d_a + 2.0 * d_b - 1.0))/(4.0 * d_b * d_b * (d_a * d_b  + 1.0))

def page_thermodynamic(f, L):
    return f * L * np.log(2.) - np.power(2., -np.abs(1.-2.*f)*L - 1.0)

def page_thermodynamic_var(f, L):
    return (1/2 - 1/4 * (1.0 if f == 1/2 else 0.0)) * np.power(2., -(1+np.abs(1-2*f))*L)

def their_result(fraction_h, V):
    frac = fraction_h if fraction_h <= 1 else (V+2)*fraction_h/np.power(2.0, V)
    val = (V-1.0)*np.log(2)/2.0
    val += 2.0 * (np.exp(-np.power(erfinv(frac), 2.0)) - 1.0) / (frac * np.pi)
    return  val + (-2.0 + 2.0 * frac + np.exp(-np.power(erfinv(frac), 2.0)))*erfinv(frac)/(2.0*frac*np.sqrt(np.pi)) 
# -------------------------------------------- OTHER --------------------------------------------
# standard maximum value
page_difference = lambda x, l: np.array(page_thermodynamic(0.5, l) - x).flatten()
# page results value not in thermodynamic limit
digamma_difference = lambda x, l: np.array(page_result(np.power(2, l//2), np.power(2, l//2)) - x).flatten()
# page results value not in thermodynamic limit
digamma_difference_v = lambda x, l: np.array(x-page_result_var(np.power(2, l//2), np.power(2, l//2))).flatten()

their_limit_l = 200
# -------------------------------------------- PLOT ALL TOGETHER --------------------------------------------

