import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf

import time
import os

from tqdm import tqdm
from IPython.display import display, Latex

'''
Create folders from the list of directories.
'''
def createFolder(directories : list, silent = False):
    for folder in directories:
        try:
            if not os.path.isdir(folder):
                os.makedirs(folder)
                #if not silent:
                    #printV(f"Created a directory : {folder}", silent)
        except OSError:
            print("Creation of the directory %s failed" % folder, silent)      
            

# ---------------------------------------           binary operations           ---------------------------------------

''' binary search '''
def binary_search(arr, l_point, r_point, element):
    if r_point >= len(arr):
        #except(f"Out of range in {arr}")
        print(f"cannot be bigger than {len(arr)}")
        return -1
    if r_point >= l_point:
        middle = l_point + (r_point-l_point)//2
        if arr[middle] == element : 
            return middle
        elif arr[middle] < element : 
            return binary_search(arr, middle + 1, r_point, element)
        else: 
            return binary_search(arr, l_point, middle - 1, element)
    return -1


''' converts given number to vector of given representatio '''
''' n -> number to be transformed '''
''' b -> base of the representation '''
''' size -> length of the vector '''
def numberToBase(n, b, size = 4):
    digits = np.zeros(size, dtype = int)
    i = size - 1
    while n:
        digits[i] = int(n%b)
        n //= b
        i-=1
    return digits

''' converts given vector to the number by using given base representation '''
def baseToNumber(vec, base):
    num = 0
    power = 1
    for i in np.arange(len(vec)-1, -1, -1):
        num += vec[i] * power
        power*=base
    return num

''' save the binary powers '''
binaryPowers = {i**2 for i in range(32)}

''' Rotates the binary representation of the input decimal number by one left shift '''
''' n -> number to rotate '''
''' L -> vector length '''
def rotateLeft(n : np.int64, L : int):
    maxP = binaryPowers[L-1]
    return (n - maxP * 2 + 1) if n >= maxP else n * 2

''' Checks the bit at given k position in binary representation of n '''
''' n -> number '''
''' k -> position sought '''
def checkBit(n : np.int64, k : int):
    return n & (np.int64(1) << k)

''' Flips all the bits in the binary representation of n giving correct number '''
def flipAll(n : np.int64, L):
    return binaryPowers[L] - n - 1

''' Flips the given k'th bit in the binary representation of n. k is counted from right to left '''
def flipSingle(n, k):
    return np.int64(n) - np.int64(binaryPowers[k]) if checkBit(n, k) else np.int64(n) +  np.int64(binaryPowers[k])

''' state printer in the human readable form and also sorted according to the coefficients'''
def printState(n,v,b=2, threshold=1e-3, sort = True):
    whole = '$'
    if sort:
        v = np.sort(v)[::-1]
    for i, val in enumerate(v):
        if val > threshold:
            vec = numberToBase(i,b,n)
            vec_str = f' {val:.2f}|'
            for a in vec:
                vec_str += (f'\\downarrow ') if a == 0 else (f'\\uparrow ')
            vec_str += f'\\rangle + '
            whole += vec_str
    return whole[:-2] + '$'

''' state printer in the human readable form and also sorted according to the coefficients from a map '''
def printStateMap(n, mapka, b = 2, threshold=1e-3, sort = True):
    whole = '$'
    for i, v in mapka.items():
        if v > threshold:
            vec = numberToBase(i,b,n)
            vec_str = f' {v:.2f}|'
            for a in vec:
                vec_str += (f'\\downarrow ') if a == 0 else (f'\\uparrow ')
            vec_str += f'\\rangle + '
            whole += vec_str
    return whole[:-2] + '$'