'''
file    : Algebra/hamil_energy_helper.py
author  : Maksymilian Kliczkowski
date    : 2025-02-01

This module contains the function to unpack operator terms for Hamiltonian models.
description:
    The function `unpack_operator_terms` unpacks a list of operator terms into separate lists of functions,
    indices, and multiplicative factors, ensuring that there are at least 'ns' operator entries.
    It checks if the provided list of operator terms has fewer than 'ns' entries.
    If so, it appends default operator terms (a tuple with a lambda function returning (x, 0.0),
    and empty lists for indices and multipliers) until its length is equal to 'ns'.
'''

from typing import Callable, Tuple, Optional, Any, List
from numba.typed import List as TypedList

# ListType = TypedList
ListType = list

################################################################################

# Define a default operator function.
def default_operator(x, i):
    '''Default operator function'''
    return (x, 0.0)

def default_operator_njit(x, i):
    '''Default operator function for numba'''
    return (x, 0.0)

def unpack_operator_terms(
    ns              : int,
    operator_terms  : List[Tuple[List[Callable], List[List[int]], List[Any]]]):
    """
    Unpacks a list of operator terms into separate lists of functions, indices, and multiplicative factors,
    ensuring that there are at least 'ns' operator entries.
    
    This function checks if the provided list of operator terms has fewer than 'ns' entries.
    If so, it appends default operator terms (a tuple with a lambda function returning (x, 0.0),
    and empty lists for indices and multipliers) until its length is equal to 'ns'. The function then
    unpacks each operator term into three separate numba typed lists:
        - a list of callable functions,
        - a list of index lists,
        - a list of multiplicative factors.
    
    ---
    Parameters:
        ns (int):
            The expected number of operator entries.
        operator_terms (List[Tuple[List[Callable], List[List[int]], List[Any]]]):
            A list where each element is a tuple representing an operator term. Each tuple contains:
                - A list of callable functions.
                - A list of lists of indices corresponding to the operator term.
                - A list of factors to multiply the operator term.
                
    ---
    Returns:
        Tuple[numba.typed.List, numba.typed.List, numba.typed.List]:
            A tuple containing three numba.typed.Lists:
                - operator_funcs: A list of lists, with each sublist containing callable operator functions.
                - operator_indices: A list of lists, with each sublist containing index lists.
                - operator_mult: A list of lists, with each sublist containing multiplicative factors.
    """

    
    if len(operator_terms) < ns:
        for ii in range(ns - len(operator_terms)):
            operator_terms.append((TypedList(), [[ii]], [0.0]))
    
    operator_funcs      = ListType()
    operator_indices    = ListType()
    operator_mult       = ListType()
    
    # handle the unpacking now
    for i in range(ns):
        funcs = ListType()
        sites = ListType()
        mults = ListType()
        for term in operator_terms[i]:
            funcs.append(term[0])
            sites.append(term[1])
            mults.append(term[2])
        if len(funcs) != 0:
            operator_funcs.append(funcs)
            operator_indices.append(sites)
            operator_mult.append(mults)
        else:
            operator_funcs.append(ListType())
            operator_indices.append(ListType())
            operator_mult.append(0.0)
            
    return operator_funcs, operator_indices, operator_mult

################################################################################