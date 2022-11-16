from scipy.signal import savgol_filter
import numpy as np
import pandas as pd

import itertools
import random as random
import shutil




#################################### STATISTICS ####################################

'''
Gaussian
'''
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

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

def find_maximum(df):
    return df.idxmax(axis=1)

def find_nearest_idx(df, col, val):
    return (df[col]-val).abs().idxmin()

def find_nearest_np(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx_np(array, value):
    return (np.abs(array - value)).argmin()


# function cause i am foking stupid
def crop_array(arr : np.array, av):
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

#################################### CALCULATORS ####################################


def gap_ratio(en, fraction = 0.3):
    mean = np.mean(en)
    mean_idx = find_nearest_idx_np(en, mean)
    #print(mean, en[mean_idx])
    
    bad, _, lower, upper = get_values_num(fraction, np.arange(1, len(en)), 14, mean_idx)
    if bad: 
        return -1
    energies = en[lower:upper]
    #print(lower, upper, mean_idx, len(en), len(energies))
    d_en = energies[1:]-energies[:-1]

    gap_ratios = np.minimum(d_en[:-1], d_en[1:]) / np.maximum(d_en[:-1], d_en[1:])
    #for i in np.arange(len(d_en) - 1):
    #    gap_ratio.append(min(d_en[i], d_en[i+1])/max(d_en[i], d_en[i+1]))
        
    return np.mean(gap_ratios)
