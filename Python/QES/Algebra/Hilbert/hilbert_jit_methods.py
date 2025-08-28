"""
file        : QES/Algebra/hilbert_jit_methods.py
description : JIT methods for Hilbert space operations.
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl

version     : 1.0.1
changes     : 
"""

import numpy as np
import numba

#! private
_INT_BINARY_REPR       = 64
_INT_HUGE_REPR         = np.iinfo(np.int64).max
_SYM_NORM_THRESHOLD    = 1e-12
from general_python.common.binary import bin_search
from general_python.algebra.utils import get_backend, JAX_AVAILABLE, ACTIVE_INT_TYPE, Array, maybe_jit


####################################################################################################
#! NUMBA METHODS
####################################################################################################

if True:
    
    # a) mapping
    
    @numba.njit
    def get_mapping(mapping, state):
        """
        Get the mapping of the state.
        
        Args:
            mapping (list):
                The mapping of the states.
            state (int):
                The state to get the mapping for.
        
        Returns:
            int:
                The mapping of the state.
        """
        return mapping[state] if len(mapping) > state else state

    # b) find the representative of a given state
    
    @numba.jit(nopython=False)
    def find_repr_int(state,
                    _sym_group,
                    _reprmap    : np.ndarray = None):
        """
        Find the representative of a given state. This corresponds 
        to finding the lowest state in the symmetry sector.
        as the representative is the state with the lowest integer
        value in the symmetry sector.
        
        Args:
            state (int):
                The state to find the representative for.
            _sym_group (list):
                The symmetry group of the system.
            _reprmap (np.ndarray):
                The mapping of the states to their representatives.
                If None, the representative is calculated.
        Returns:
            int:
                The representative of the state.
        """
                
        # If mapping exists, return saved representative.
        if _reprmap is not None and len(_reprmap) > 0:
            idx     = _reprmap[state, 0]
            sym_eig = _reprmap[state, 1]
            return idx, sym_eig
        
        # If no mapping exists, find the representative.
        if _sym_group is None or len(_sym_group) == 0:
            return state, 1.0

        _sec = _INT_HUGE_REPR
        _val = 1.0
        
        # loop over all states in symmetry sectors
        for g in _sym_group:
            _st, _retval = g(state)
            if _st < _sec:
                _sec = _st
                _val = _retval
        return _sec, _val

    # c) find the representative of a given state in a given symmetry sector

    @numba.jit(nopython=False)
    def find_representative_int(
                            _state                  : int,
                            _mapping                : np.ndarray,
                            _normalization          : np.ndarray,
                            _normalization_beta     : float,
                            _sym_group,
                            _reprmap                : np.ndarray = None
            ):
        """
        Find the representative of a given state.
        """
        if _mapping is None or len(_mapping) == 0:
            return (_state, 1.0)
        
        # if the map exists, use it!
        if _reprmap is not None and len(_reprmap) > 0:
            idx, sym_eig    = _reprmap[_state, 0], _reprmap[_state, 1]
            sym_eigc        = sym_eig.conjugate() if hasattr(sym_eig, "conjugate") else sym_eig
            return (idx, _normalization[idx] / _normalization_beta * sym_eigc)
        
        mapping_size = len(_mapping)
        
        # find the representative already in the mapping (can be that the matrix element already 
        # represents the representative state)
        idx = bin_search.binary_search_numpy(_mapping, 0, mapping_size - 1, _state)
        
        if idx != bin_search._BAD_BINARY_SEARCH_STATE:
            return (idx, _normalization[idx] / _normalization_beta)
        
        # otherwise, we need to find the representative by acting on the state with the symmetry operators
        # and finding the one that gives the smallest value - standard procedure
        idx, sym_eig = find_repr_int(_state, _sym_group, _reprmap)
        if idx != bin_search._BAD_BINARY_SEARCH_STATE:
            sym_eigc = sym_eig.conjugate() if hasattr(sym_eig, "conjugate") else sym_eig
            return (idx, _normalization[idx] / _normalization_beta * sym_eigc)
        
        # didn't find the representative - this may be different sector
        return (_state, 0.0)

    # d) get the matrix element of a given state

    @numba.jit(nopython=False)
    def get_matrix_element(
            k               : int,
            new_k           : int,
            kmap            = None,
            h_conj          = False,
            _mapping        : np.ndarray = None,
            _norm           : np.ndarray = None,
            _sym_group                   = None,
            _reprmap        : np.ndarray = None
        ):
        '''
        Get the matrix element of a given state using information provided from the symmetry group and 
        a given Hilbert space.
        Args:
            k (int):
                The state to get the matrix element for.
            new_k (int):
                The new state to get the matrix element for.
            kmap (int):
                The mapping of the states.
            h_conj (bool):
                A flag to indicate if the Hamiltonian is conjugated.
            _mapping (list):
                The mapping of the states.
            _norm (list):
                The normalization of the states.
            _sym_group:
                The symmetry group.
            _reprmap:
                The mapping of the representatives.
        '''
        
        # check the mapping, if it is None, we need to get the mapping
        if kmap is None:
            kmap = get_mapping(_mapping, k)
        
        # try to process the elements
        if kmap == new_k:
            # the element k is already the same as new_k and obviously we 
            # and we add this at k (not kmap as it only checks the representative)
            return (new_k, k), 1
        
        # otherwise we need to check the representative of the new k
        # get the norm of the k'th element of the Hilbert space - how to return to the representative
        # of the new k
        norm = _norm[k] if _norm is not None else 1.0
        # find the representative of the new k
        idx, symeig = find_representative_int(new_k, _mapping, _norm, norm, _sym_group, _reprmap)
        return ((idx, k), symeig) if not h_conj else ((k, idx), symeig)

    ###############################

    @numba.jit(nopython=False)
    def jitted_find_repr_int(state, _sym_group, _reprmap = None):
        return find_repr_int(state, _sym_group, _reprmap)

    @numba.jit(nopython=False)
    def jitted_find_representative_int(_state, _mapping, _normalization, _normalization_beta, _sym_group, _reprmap = None):
        return find_representative_int(_state, _mapping, _normalization, _normalization_beta, _sym_group, _reprmap)
        
    @numba.njit
    def jitted_get_mapping(mapping, state):
        return get_mapping(mapping, state)

    @numba.jit(nopython=False)
    def jitted_get_matrix_element(k, new_k, kmap = None, h_conj = False, _mapping = None, _norm = None, _sym_group = None, _reprmap = None):
        return get_matrix_element(k, new_k, kmap, h_conj, _mapping, _norm, _sym_group, _reprmap)

####################################################################################################