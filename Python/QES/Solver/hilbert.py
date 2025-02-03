
"""
High-level Hilbert space class for quantum many-body systems.
@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""

from abc import ABC, abstractmethod

# import Lattice class
from ..general_python.lattices import Lattice

''' Other imports '''
import numpy as np


class HilbertSpace(ABC):
    """
    A class to represent a Hilbert space either in Many-Body Quantum Mechanics or Quantum Information Theory and non-interacting systems.
    
    
    """
    
    def __init__(self,
                sym_gen     : dict | None   = None,
                global_syms : list | None   = None,
                gen_mapping : bool          = False,
                state_type  : str           = "integer",
                **kwargs):
        """
        Initialize the Hilbert space.
        
        Args:
            sym_gen (dict)      : A dictionary of symmetry generators {operator : value}.
            global_syms (list)  : A list of global symmetries {operator : value}.
            gen_mapping (bool)  : A flag to generate the mapping of the representatives to the original states.
            Ns (int)            : The number of sites in the system.
            Nhl (int)           : The local Hilbert space dimension - 2 for spin-1/2, 4 for spin-1, etc (default is 2).
        """
        
        # handle the system phyisical size dimension
        if "Ns" in kwargs and "lattice" not in kwargs:
            self._Ns        = kwargs.get('Ns', 1)       # number of sites in the system
            self._lattice   = None                      # lattice object
        elif "lattice" in kwargs:
            self._lattice   = kwargs.get('lattice')     # lattice object
            self._Ns        = self.lattice.get_Ns()     # number of sites in the system
        else:
            raise ValueError("Either 'Ns' or 'lattice' must be provided.")
        
        # handle local Hilbert space properties
        self._Nhl   = kwargs.get('Nhl', 2)              # local Hilbert space dimension
        self._Nhint = kwargs.get('Nhint', 0)            # number of modes (fermions, bosons, etc. on each site)
        
        # initialize the Hilbert space etc.
        if state_type.lower() == "integer" or state_type.lower() == "int":
            self._state_type = int
        else:
            self._state_type = np.ndarray
        
        self._NhFull        = self._Nhl ** (self._Nhint * self._Ns) # full Hilbert space dimension
        self._normalization = []                                    # normalization of the states
        self._sym_group     = []                                    # symmetry group
        self._mapping       = []                                    # mapping of the states
        self._reprmap       = []                                    # mapping of the representatives (vector of tuples (state, representative value))
        
        # handle symmetries
        self._global_syms   = global_syms               # global symmetries
        self._init_mapping(sym_gen, gen_mapping)        # initialize the mapping
    
    ####################################################################################################
    
    def _init_mapping(self, gen : list, gen_mapping : bool = False):
        """
        Initialize the mapping of the states.
        
        Args:
            gen (list)         : A list of symmetry generators.
            gen_mapping (bool) : A flag to generate the mapping of the representatives to the original states.
        """
        self._gen_sym_group(gen)    # generate the symmetry group
        
        if len(gen) > 0:
            print(f"Symmetry group generator: {self.get_sym_info()}") # print the symmetry group
        
        # generate the mapping of the states
        
        
            
            
    # --------------------------------------------------------------------------------------------------
    
    def _gen_sym_group(self, gen : list):
        """
        Generate the symmetry group.
        
        Args:
            gen (list) : A list of symmetry generators.
        """
		
        # no symmetries! - there are no global and local symmetries to be used, therefore return
        if (gen is None or (gen is not None and len(gen) == 0)) and not self.check_global_symmetry():
            return
        
        # copy the generators to modify them if needed
        sym_gen = gen.copy()
        
        contains_translation        = False
        contains_cpx_translation    = False
        
        # GLOBALS
        contains_U1                 = False
        
        # go through the local symmetries
        for (generator, sec) in sym_gen:
            
            # proceed if this is a translation
            if generator == "T":
                contains_translation = True
                if sec == "C":
                    contains_cpx_translation = True
                continue
    
    ####################################################################################################
    
    # Getters and checkers for the Hilbert space
    
    ####################################################################################################
    
    def check_global_symmetry(self):
        """
        Check if there are any global symmetries.
        """
        return self._global_syms is not None and (self._global_syms is not None and len(self._global_syms) > 0)
    
    #---------------------------------------------------------------------------------------------------
    
    def get_sym_info(self):
        """
        Creates the information string about the Hilbert space and symmetries.
        
        Returns:
            str: A string containing the information about all the symmetries.
        """
        tmp = ""
        if self._sym_group:
            # start with local symmetries
            for gen, val in self._sym_group:
                tmp += f"{gen}={val},"
        if self.check_global_symmetry():
            # start with global symmetries
            for sym in self._global_syms:
                tmp += f"{sym[0]}={sym[1]:.2f},"
        
        # remove last ","
        if tmp:
            tmp = tmp[:-1]
        
        return tmp
    
    
    ####################################################################################################
    
    # Representation of the Hilbert space
    
    ####################################################################################################
    
    def __str__(self):
        """
        Return a string representation of the Hilbert space.
        
        Returns:
            str: A string representation of the Hilbert space.
        """
        if self._NhFull == self._Nhl ** (self._Nhint * self._Ns):
            return (
            f"Produced the full Hilbert space - no symmetries are used. Spin modes = {self._Nhl}\n"
            f"Number of lattice sites (Ns) = {self._Ns}\n"
            f"Hilbert space size (Nh) = {self._NhFull}\n"
            )
        elif self._NhFull <= 0:
            return "No states in the Hilbert space"
        else:
            sym_info = (
                f"Reduced Hilbert space produced using symmetries.\n"
                f"Spin modes = {self._Nhl}\n"
                f"Number of lattice sites (Ns) = {self._Ns}\n"
                f"Number of fermionic modes (Nhint) = {self._Nhint}\n"
                f"Full Hilbert space size (NhFull) = {self._NhFull}\n"
                f"Reduced Hilbert space size (Nh) = {len(self._mapping)}\n"
                f"Number of symmetry sectors = {len(self._sym_group)}\n"
            )
            if self.check_global_symmetry():
                sym_info += "Global symmetries used:\n"
                for sym in self._global_syms:
                    sym_info += f" - {sym[0]} with value {sym[1]}\n"
            else:
                sym_info += "No global symmetries applied.\n"

            if self._sym_group:
                sym_info += "Local symmetry group operators applied.\n"
            else:
                sym_info += "No local symmetry group operators applied.\n"

            return sym_info
    
    ####################################################################################################
    
    @abstractmethod
    def __len__(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        pass
    
    ####################################################################################################
    
    @abstractmethod
    def __getitem__(self, i : int):
        """
        Return the i-th basis state of the Hilbert space.
        
        Args:
            i (int): The index of the basis state to return.
        
        Returns:
            np.ndarray: The i-th basis state of the Hilbert space.
        """
        pass
    
    ####################################################################################################
    
    @abstractmethod
    def __contains__(self, state : np.ndarray):
        """
        Check if a state is in the Hilbert space.
        
        Args:
            state (np.ndarray): The state to check.
        
        Returns:
            bool: True if the state is in the Hilbert space, False otherwise.
        """
        pass
    