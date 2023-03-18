import os
from scipy.special import psi
from scipy.special import polygamma
from scipy.special import erf, erfinv
from scipy.optimize import curve_fit
from scipy.linalg import svd
current_dir = os.getcwd() 
import tenpy
import numpy as np
import scipy
import random
from datetime import datetime
from common.__common__ import *
kPSep = os.sep

def page_result(d_a, d_b):
    if d_a <= d_b:
        return psi(d_a * d_b + 1) - psi(d_b + 1) - (d_a - 1)/(2*d_b)
    else:
        return psi(d_a * d_b + 1) - psi(d_a + 1) - (d_b - 1)/(2*d_a)

def page_result_var(d_a, d_b):
    return (((d_a + d_b)/(d_a*d_b + 1.0)) * polygamma(1, d_b + 1)) - polygamma(1, d_a*d_b + 1) - ((d_a-1)*(d_a + 2.0 * d_b - 1.0))/(4.0 * d_b * d_b * (d_a * d_b  + 1.0))

def printDictionary(dict):
    string = ""
    for key in dict:
        value = dict[key]
        string += f'{key},{value},'
    return string[:-1]

def random_gaussian_state(N : int):
    a = np.random.normal(0., 1., N)
    return a / np.sqrt(np.sum(np.square(a)))

def gaussianity(arr : np.ndarray):
    return np.mean(np.square(arr))/np.square(np.mean(arr))

rescale_vec = lambda x : (x - np.mean(x))/np.sqrt(np.var(x))
random.seed(datetime.now().microsecond)

def entro_schmidt(state : np.ndarray, L : int, La : int):    
	dimA = 2 ** La
	dimB = 2 ** (L-La)
	N = dimA * dimB

	# reshape array to matrix
	rho = state.reshape(dimA, dimB);

	# get schmidt coefficients from singular-value-decomposition
	U, schmidt_coeff, _ = svd(rho)

	# calculate entropy
	entropy = 0;
	for i in range(len(schmidt_coeff)):
		value = schmidt_coeff[i] * schmidt_coeff[i]
		entropy += ((-value * np.log(value)) if (abs(value) > 0) else 0)
	return entropy

def av_random_gauss_states(L, realizations = 200, frac = 100):
    gauss = []
    entropies = []
    entropies_var = []
    N = 2**L
    random.seed(datetime.now().microsecond)
    for r in np.arange(realizations):
        vec = np.array([random_gaussian_state(N) for i in range(frac)])
        v = vec.flatten()
        v = rescale_vec(v)
        v = np.abs(v)   
        
        g = gaussianity(v)
        gauss.append(g)
        entro = [entro_schmidt(vec[i], L, L//2) for i in np.arange(frac)]
        e = np.mean(entro)
        entropies.append(e)
        ev = np.var(entro)
        entropies_var.append(ev)
    
    return {
        "g_mean" : np.mean(gauss) - np.pi/2,
        "g_var" : np.var(gauss),
        "e_mean" : np.mean(entropies),
        "e_var" : np.mean(entropies_var)
        } , {
        "g" : gauss,
        "e" : entropies,
        "e_v" : entropies_var
    }
    
def av_random_mat(L, realizations = 200, frac = 100):    
    gauss = []
    entropies = []
    entropies_var = []
    N = 2**L
    for r in np.arange(realizations):
        H = tenpy.linalg.random_matrix.GOE((N, N))
        _, vec = scipy.linalg.eigh(H)
        print("finished diagonalizing", r)
        v = vec.flatten()
        v = rescale_vec(v)
        v = np.abs(v)   
        
        g = gaussianity(v)
        gauss.append(g)
        vec = vec.T
        entro = [entro_schmidt(vec[i], L, L//2) for i in np.arange(frac)]
        e = np.mean(entro)
        entropies.append(e)
        ev = np.var(entro)
        entropies_var.append(ev)
        
    return {
        "g_mean" : np.mean(gauss) - np.pi/2,
        "g_var" : np.var(gauss),
        "e_mean" : np.mean(entropies),
        "e_var" : np.mean(entropies_var)
    } , {
        "g" : gauss,
        "e" : entropies,
        "e_v" : entropies_var
    }
    

def __main__():
    now = datetime.now()
    current_time = now.microsecond
    
    frac = 100
    realizations = 50
    Ls = np.array([15])
    Ns = np.array([2**L for L in Ls])
    pages = np.array([page_result(2**(L//2), 2**(L//2)) for L in Ls])
    pages_var = np.array([page_result_var(2**(L//2), 2**(L//2)) for L in Ls])
    
    g = []
    gv = []
    e = []
    ev = []
    gs = []
    gsv = []
    es = []
    esv = []
    color_GOE = 'black'
    color_state = 'blue'
    
    entropies = np.zeros((len(Ls), realizations))
    entropies_mat = np.zeros((len(Ls), realizations))

    entropies_var = np.zeros_like(entropies)
    entropies_var_mat = np.zeros_like(entropies)
    
    gausses = np.zeros_like(entropies)
    gausses_mat = np.zeros_like(entropies)

    for i, L in enumerate(Ls):
        print("Doing", L)
        dic_m_v, dic_v = av_random_gauss_states(L, realizations, frac)
        entropies[i] = dic_v['e']
        gausses[i] = dic_v['g']
        entropies_var[i] = dic_v['e_v']
        
        g.append(dic_m_v['g_mean'])
        e.append(dic_m_v['e_mean'])
        gv.append(dic_m_v['g_var'])
        ev.append(dic_m_v['e_var'])
        print("Fermionic = ", printDictionary(dic_m_v))
        # -----------------------------------------------------------
        # dic_m_h, dic_h = av_random_mat(L, realizations, frac)
        # entropies_mat[i] = dic_h['e']
        # gausses_mat[i] = dic_h['g']
        # entropies_var_mat[i] = dic_h['e_v']
        
        # gs.append(dic_m_h['g_mean'])
        # es.append(dic_m_h['e_mean'])
        # gsv.append(dic_m_h['g_var'])
        # esv.append(dic_m_h['e_var'])
        # print("GOE = ", printDictionary(dic_m_h))
        
    # fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
    # plt.suptitle(f"Random matrix prediction size scaling. $\Gamma = {frac}, N_r={realizations}$")
    # ############### GAUSS
    # ax[0][0].plot(Ns, np.abs(g), marker = '.', markersize = 10, label = 'random state', color = color_state)
    # ax[0][0].plot(Ns, np.abs(gs), marker = '*', markersize = 10, label = f'random matrix', color = color_GOE)
    # ax[0][0].legend()
    # ax[0][0].set_xscale('log')
    # ax[0][0].set_yscale('log')
    # ax[0][0].set_xlim(1e1, 1e5)
    # ax[0][0].set_ylabel(r"$\bar{\Gamma} _\Psi - \pi / 2$")
    # ax[0][0].set_xticklabels([])

    # # ############### ENTRO
    # ax[0][1].plot(1/Ls, pages - np.array(e), marker = '.', markersize = 10, color = color_state)
    # ax[0][1].plot(1/Ls, pages - np.array(es), marker = '*', markersize = 10, color = color_GOE)
    # ax[0][1].set_ylabel(r"$\bar{S} _A$")
    # ax[0][1].set_ylim(0, None)
    # ax[0][1].set_xlim(0, 1/6)
    # ax[0][1].set_xticklabels([])
    # # ############### GAUSS VAR
    # ax[1][0].plot(Ns, np.array(gv), marker = '.', markersize = 10, color = color_state)
    # ax[1][0].plot(Ns, np.array(gsv), marker = '*', markersize = 10, color = color_GOE)
    # ax[1][0].set_xscale('log')
    # ax[1][0].set_yscale('log')
    # ax[1][0].set_ylabel(r"$E[(\Gamma _\Psi - \bar{\Gamma} _\Psi)^2]$")
    # ax[1][0].set_xlabel(r"$\mathcal{D}$")
    # ax[1][0].set_xlim(1e1, 1e5)

    # # ############### ENTRO VAR
    # ax[1][1].plot(1/Ls, np.abs(pages_var - np.array(ev)), marker = '.', markersize = 10, color = color_state)
    # ax[1][1].plot(1/Ls, np.abs(pages_var - np.array(esv)), marker = '*', markersize = 10, color = color_GOE)
    # ax[1][1].set_xlim(0, 1/6)
    # ax[1][1].set_ylim(0, None)
    # ax[1][1].set_ylabel(r"$E[(S_A - \bar{S} _A)^2]$")
    # ax[1][1].set_xlabel(r"$1/L$")
    # plt.tight_layout(pad=1.5, w_pad=0.1, h_pad=0.1)
    
    folder = current_dir + kPSep + "random_matrices" + kPSep
    createFolder([folder])
    # plt.savefig(folder + "figure.pdf")
    
    entropies = pd.DataFrame(entropies, index=Ls)
    entropies.to_csv(folder + f'entropies_{current_time}.dat')
    entropies_var = pd.DataFrame(entropies_var, index=Ls)
    entropies_var.to_csv(folder + f'entropies_var_{current_time}.dat')
    gausses = pd.DataFrame(gausses, index=Ls)
    gausses.to_csv(folder + f'gauss_{current_time}.dat')

    entropies = pd.DataFrame(entropies_mat, index=Ls)
    # entropies.to_csv(folder + f'm_entropies_{current_time}.dat')
    entropies_var = pd.DataFrame(entropies_var_mat, index=Ls)
    entropies_var.to_csv(folder + f'm_entropies_var_{current_time}.dat')
    gausses = pd.DataFrame(gausses_mat, index=Ls)
    # gausses.to_csv(folder + f'm_gauss_{current_time}.dat')
    # np.save(folder + "gauss_state", np.array(g))
    # np.save(folder + "gauss_GOE", np.array(gs))
    # np.save(folder + 'entropy_state', np.array(e))
    # np.save(folder + 'entropy_state', np.array(es))
    # np.save(folder + )

    
__main__()