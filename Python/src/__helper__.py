from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import os
import itertools
import random as random
import shutil
from scipy.special import psi
from scipy.special import polygamma
from scipy.special import erf, erfinv
from scipy.optimize import curve_fit

goe = 0.5307
constant_page_correction = lambda n: (n * np.log(2.0) / 2.0) - 0.5 + ((0.5 - np.log(2.0)) / 2.0)
page_val = lambda n : (n * np.log(2.0) / 2.0) - 0.5

#################################### STATISTICS ####################################

'''
Gaussian
'''
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

'''
Folded normals sum giving the error function distribution
'''
def sum_of_folded_normals(bins, sig1, sig2):
    sqrt_var = np.sqrt(sig1*sig1 + sig2*sig2)
    val = np.sqrt(2.0/np.pi)/sqrt_var
    val *= np.exp(-np.square(bins)/(2.0*sqrt_var*sqrt_var))
    val *= (erf(sig1 * bins/np.sqrt(2.0) / sig2 / sqrt_var) + erf(sig2 * bins/np.sqrt(2.0) / sig1 / sqrt_var))
    return val

'''
Chi2 distribution
'''
def sum_of_squares_normals(bins):
    return np.exp(-bins/2.0)/2.0

'''
Chi distribution
'''
def sum_of_squares_normals_sqrt(bins):
    return bins * np.exp(-np.square(bins)/2.0)

'''
Moving average
'''
def moving_average(df, window = 10):
    df_tmp = df.copy()
    for i, row in df_tmp.iterrows():
        df_tmp.loc[i] = savgol_filter(row, window, 3)
    return df_tmp 
    return df.rolling(window, axis=1).mean().dropna(axis = 1), en[:-window+1]


#################################### FINDERS ####################################

''' Find maximum index in a Dataframe'''
def find_maximum(df : pd.DataFrame):
    return df.idxmax(axis=1)

''' Find the nearest index to the value given in the DataFrame '''
def find_nearest_idx(df : pd.DataFrame, col : str, val : float):
    return (df[col]-val).abs().idxmin()

''' Find the nearest value to the value given in a numpy array'''
def find_nearest_np(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

''' Find the index of a value closest to the given value in a numpy array'''
def find_nearest_idx_np(array, value):
    return (np.abs(array - value)).argmin()


# function cause i am foking stupid
def crop_array(arr, av):
    i = 0
    moved = abs(arr - av)
    for i in range(len(arr)-1):
        if moved[i] < moved[i+1]:
            break
        i+=1
    return np.take(arr, np.arange(0, i+1))
 
 
#################################### FITS ####################################
    
def fit_one_over_v(x, a, c):
    return a/x + c

def fit_power(x, a, b):
    return a * np.power(1/x, b)

def fit_one_over_v2(x, a, b, c):
    return a/x + b/x/x + c

def two_to_minus_x(x, a, c):
    return a * np.power(2.0, -x) + c

####################################################### VAL NUM #######################################################

'''
Returns the bounds around a given index
'''
def get_values_num(fraction, cols_to_take, L, idx):
    bad = False
    N = len(cols_to_take)
    values_num = round(fraction * N) if fraction < 1.0 else int(fraction)
    
    # if it is the same as index dataframe fraction
    if fraction == 1.0:
        return bad, idx, 0, N-1
    
    if values_num > len(cols_to_take):
        return True, idx, 0, N-1
        
    # bounds - take all around those
    lower = idx - round(values_num / 2.)
    if lower < 0:
        return True, idx, 0, N-1
        
    upper = idx + round(values_num / 2.)
    if upper >= len(cols_to_take):
        return True, idx, 0, N-1
    
    return False, idx, lower, upper

####################################################### REDUCED DENSITY MATRIX #######################################################

'''
Calculate the reduced density matrix out of a state
'''
def reduced_density_matrix(state : np.ndarray, A_size : int, L : int):
        dimA = int(( (2 **      A_size ) ));
        dimB = int(( (2 ** (L - A_size)) ));
        N = dimA * dimB;
        rho = np.zeros((dimA, dimA))
        for n in range(0, N, 1):					
            counter = 0;
            for m in range(n % dimB, N, dimB):
                idx = n // dimB;
                rho[idx, counter] += np.conj(state[n]) * state[m]
                counter+=1
        return rho

####################################################### CALCULATORS #######################################################

'''
Calculate the gap ratio around the mean energy in a sector
'''
def gap_ratio(en, fraction = 0.3, use_mean_lvl_spacing = True, return_mean = True):
    mean = np.mean(en)
    mean_idx = find_nearest_idx_np(en, mean)
    #print(mean, en[mean_idx])
    
    bad, _, lower, upper = get_values_num(fraction, np.arange(1, len(en)), 14, mean_idx)
    if bad: 
        return -1
    energies = en[lower:upper]
    #print(lower, upper, mean_idx, len(en), len(energies))
    # delta energies
    d_en = energies[1:]-energies[:-1]
    # if we use mean level spacing divide by it
    if use_mean_lvl_spacing:
        d_en /= np.mean(d_en)
    
    # calculate the gapratio
    gap_ratios = np.minimum(d_en[:-1], d_en[1:]) / np.maximum(d_en[:-1], d_en[1:])
            
    return np.mean(gap_ratios) if return_mean else gap_ratios.flatten()

'''
Calculate the average entropy in a given DataFrame
- df : DataFrame with entropies
- row : row number (-1 for half division of a system)
'''
def mean_entropy(df : pd.DataFrame, row : int):
    return np.mean(df.iloc[row])

'''
Calculate the gaussianity <|Oab|^2>/<|Oab|>^2 -> for normal == pi/2
'''
def gaussianity(arr : np.ndarray):
    return np.mean(np.square(arr))/np.square(np.mean(arr))

'''
Calculate the modulus fidelity - should be 2/pi for gauss
- states : np.array of eigenstates
'''
def modulus_fidelity(states : np.ndarray):
    Ms = []
    for i in range(0, states.shape[-1] - 1):
        Ms.append(np.dot(states[:, i], states[:, i+1]))
    return np.mean(Ms)

'''
Calculate the information entropy for given states
'''
def info_entropy(states : np.ndarray, model_info : str):
    try:
        entropies = []
        for state in states.T:
            square = np.square(np.abs(state))
            entropies.append(-np.sum(square * np.log(square)))
        return np.mean(entropies)
    except:
        print(f'\nHave some problem in {model_info}\n')
        return -1.0

####################################################### DATAFRAME PARSE #######################################################

'''
Parses the dataframe according to some given dictionary of parameters
'''
def parse_dataframe(df : pd.DataFrame, params : dict):
    tmp = df.copy()
    for key in params.keys():
        tmp = tmp[tmp[key].isin(params[key])]
    return tmp

'''

'''
def print_dict(dic:dict):
    r = ''
    for key in dic.keys():
        r += f'{key}={dic[key]},'
    return r

