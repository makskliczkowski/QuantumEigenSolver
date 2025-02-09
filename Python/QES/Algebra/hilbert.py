
"""
High-level Hilbert space class for quantum many-body systems.
@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""

from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple, List     # type hints for the functions and methods
import numpy as np

# import Lattice class
from .. import Lattice
from . import Operator, SymmetryGenerators

class HilbertSpace(ABC):
    """
    A class to represent a Hilbert space either in Many-Body Quantum Mechanics or Quantum Information Theory and non-interacting systems.
    
    
    """
    
    ####################################################################################################
    
    def __init__(self,
                sym_gen     : Union[dict, None]             = None,
                global_syms : Union[List[Operator], None]   = None,
                gen_mapping : bool                          = False,
                state_type  : str                           = "integer",
                single_part : bool                          = False,
                **kwargs):
        """
        Initialize the Hilbert space. 
        
        Args:
            sym_gen (dict)      : A dictionary of symmetry generators {operator : value}.
            global_syms (list)  : A list of global symmetries {operator : value}.
            gen_mapping (bool)  : A flag to generate the mapping of the representatives to the original states - this 
                                means that a map between a current index (state) and a representative index is created.
            ns (int)            : The number of sites in the system.
            nhl (int)           : The local Hilbert space dimension - 2 for spin-1/2, 4 for spin-1, etc (default is 2).
            single_particle     : A flag to indicate if the system is a single-particle system (default is False).
        """
        
        # check if the arguments match the requirements
        if not isinstance(sym_gen, dict) and sym_gen is not None:
            raise ValueError("The symmetry generators must be provided as a dictionary.")
        if not isinstance(global_syms, list) and global_syms is not None:
            raise ValueError("The global symmetries must be provided as a list.")
        if not isinstance(gen_mapping, bool):
            raise ValueError("The flag for generating the mapping must be a boolean.")            
        
        # handle the system phyisical size dimension - distinguish between the number of sites and the lattice object
        # if the lattice object is provided, the number of sites is calculated from the lattice object
        if "ns" in kwargs and "lattice" not in kwargs:
            self._ns        = kwargs.get('ns', 1)       # number of sites in the system
            self._lattice   = None                      # lattice object
        elif "lattice" in kwargs:
            self._lattice   = kwargs.get('lattice')     # lattice object provided
            self._ns        = self._lattice.get_Ns()    # number of sites in the system
        else:
            raise ValueError("Either 'ns' or 'lattice' must be provided.")
        
        # handle local Hilbert space properties
        self._nhl   = kwargs.get('nhl', 2)              # local Hilbert space dimension
        self._nhint = kwargs.get('nhint', 1)            # number of modes (fermions, bosons, etc. on each site)
        
        # initialize the Hilbert space etc.
        if state_type.lower() == "integer" or state_type.lower() == "int":
            self._state_type = int
        else:
            self._state_type = np.ndarray
        
        # initialize the Hilbert space properties like the full Hilbert space dimension, normalization, symmetry group, etc.
        self._single_part   = single_part                           # single particle system flag
        if self._single_part:
            self._nhfull    = self._ns * self._nhl                  # single particle system Hilbert space dimension (each site has its own Hilbert space)
        else:
            self._nhfull        = self._nhl ** (self._nhint * self._ns) # full Hilbert space dimension
            
        # may be eddited later by modifying the symmetry group
        self._nh            = self._nhfull                          # Hilbert space dimension
        
        # initialize the properties of the Hilbert space
        self._normalization = []                                    # normalization of the states
        self._sym_group     = []                                    # symmetry group
        self._mapping       = []                                    # mapping of the states
        self._reprmap       = []                                    # mapping of the representatives (vector of tuples (state, representative value))
        
        # handle symmetries - save the global symmetries and initialize the mapping 
        self._global_syms   = global_syms               # global symmetries
        self._init_mapping(sym_gen, gen_mapping)        # initialize the mapping
    
    ####################################################################################################
    
    # Generate the symmetry group and all properties of the generation
    
    ####################################################################################################
    
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
        sym_gen                     = gen.copy()
        
        contains_translation        = False
        contains_cpx_translation    = False
        
        # globals
        contains_u1                 = False
        
        # go through the local symmetries by going through the generators
        if sym_gen is not None:
            # go through the generators and check the symmetries
            for gen in sym_gen:
                # get the generator and the value
                generator, sec = gen
                
                # proceed if this is a translation
                if isinstance(generator, (SymmetryGenerators.Translation_x, SymmetryGenerators.Translation_y, SymmetryGenerators.Translation_z)):
                    contains_translation == True if (self._lattice is not None and self._lattice.get_bc() == PBC) else False
                    if sec == "C":
                        contains_cpx_translation = True
                    continue
    
    # --------------------------------------------------------------------------------------------------
    
    def _init_mapping(self, gen : list, gen_mapping : bool = False):
        """
        Initialize the mapping of the states. This function is used to generate the mapping of the states to the representatives.
        It uses a list of symmetry generators to generate the mapping.
        
        Args:
            gen (list)         : A list of symmetry generators.
            gen_mapping (bool) : A flag to generate the mapping of the representatives to the original states.
        """
        self._gen_sym_group(gen)                                    # generate the symmetry group
        
        if len(gen) > 0:
            print(f"Symmetry group generator: {self.get_sym_info()}") # print the symmetry group
        
        # generate the mapping of the states
        
        
            
            

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
    
    # --------------------------------------------------------------------------------------------------
    
    def get_lattice(self):
        """
        Return the lattice object.
        
        Returns:
            Lattice: The lattice object.
        """
        return self._lattice
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def sites(self):
        """
        Return the number of sites in the system.
        
        Returns:
            int: The number of sites in the system.
        """
        return self._ns
    
    @property
    def Ns(self):
        """
        Return the number of sites in the system.
        
        Returns:
            int: The number of sites in the system.
        """
        return self._ns
    
    def get_Ns(self):
        """
        Return the number of sites in the system.
        
        Returns:
            int: The number of sites in the system.
        """
        return self._ns
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def local(self):
        """
        Return the local Hilbert space dimension.
        
        Returns:
            int: The local Hilbert space dimension.
        """
        return self._nhl
    
    @property
    def Nhl(self):
        """
        Return the local Hilbert space dimension.
        
        Returns:
            int: The local Hilbert space dimension.
        """
        return self._nhl
    
    def get_Nhl(self):
        """
        Return the local Hilbert space dimension.
        
        Returns:
            int: The local Hilbert space dimension.
        """
        return self._nhl
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def modes(self):
        """
        Return the number of modes (fermions, bosons, etc. on each site).
        
        Returns:
            int: The number of modes.
        """
        return self._nhint
    
    @property
    def Nhint(self):
        """
        Return the number of modes (fermions, bosons, etc. on each site).
        
        Returns:
            int: The number of modes.
        """
        return self._nhint
    
    def get_Nhint(self):
        """
        Return the number of modes (fermions, bosons, etc. on each site).
        
        Returns:
            int: The number of modes.
        """
        return self._nhint
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def full(self):
        """
        Return the full Hilbert space dimension.
        
        Returns:
            int: The full Hilbert space dimension.
        """
        return self._nhfull
    
    @property
    def Nhfull(self):
        """
        Return the full Hilbert space dimension.
        
        Returns:
            int: The full Hilbert space dimension.
        """
        return self._nhfull

    def get_Nh_full(self):
        """
        Return the full Hilbert space dimension.
        
        Returns:
            int: The full Hilbert space dimension.
        """
        return self._nhfull
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def dimension(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
    @property
    def Nh(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
    def get_Nh(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
    ####################################################################################################
    
    # Representation of the Hilbert space
    
    ####################################################################################################
    
    def __str__(self):
        """
        Return a string representation of the Hilbert space.
        
        Returns:
            str: A string representation of the Hilbert space.
        """
        if self._nhfull == self._nhl ** (self._nhint * self._ns):
            return (
                f"Produced the full Hilbert space - no symmetries are used. Spin modes = {self._nhl}\n"
                f"Number of lattice sites (Ns) = {self._ns}\n"
                f"Hilbert space size (Nh) = {self._nhfull}\n"
            )
        elif self._nhfull <= 0:
            return "No states in the Hilbert space"
        else:
            sym_info = (
                f"Reduced Hilbert space produced using symmetries.\n"
                f"Spin modes = {self._nhl}\n"
                f"Number of lattice sites (Ns) = {self._ns}\n"
                f"Number of fermionic modes (Nhint) = {self._nhint}\n"
                f"Full Hilbert space size (NhFull) = {self._nhfull}\n"
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
    
    def __repr__(self):
        """
        Return a string representation of the Hilbert space.
        
        Returns:
            str: A string representation of the Hilbert space.
        """
        symmetries = self.get_sym_info()
        return f"{'Many body' if not self._single_part else 'Single particle'} Hilbert space with {self._nh}" +    \
            f" states and {self._ns} sites : {self._nhl} modes on each site." +                                    \
            f" Symmetries: {symmetries}" if symmetries else ""
    
    ####################################################################################################
    
    @abstractmethod
    def __len__(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
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
    
    ####################################################################################################