
"""
High-level Hilbert space class for quantum many-body systems.
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Date    : 2025-02-01
Version : 1.0.0
Changes : 
    - 2025.02.01 : 1.0.0 - Initial version of the Hilbert space class. - MK
"""

import math
import time
import numpy as np
from numba import njit, jit
from abc import ABC
from itertools import combinations
from typing import Union, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# already imported from general_python
from Python.QES.Algebra.Hilbert.hilbert_jit_states import get_backend, JAX_AVAILABLE, ACTIVE_INT_TYPE
####################################################################################################
from general_python.lattices.lattice import Lattice, LatticeBC, LatticeDirection
from general_python.common.flog import get_global_logger, Logger
if JAX_AVAILABLE:
    from general_python.algebra.utils import pad_array_jax
from general_python.common.binary import binary_search, __BAD_BINARY_SEARCH_STATE, binary_search_numpy
####################################################################################################
from Algebra.Operator.operator import Operator, SymmetryGenerators, GlobalSymmetries, operator_identity, OperatorFunction
from Algebra.globals import GlobalSymmetry
from Algebra.symmetries import choose, translation

####################################################################################################
#! WRAPPER FOR JIT AND NUMBA
####################################################################################################

from Python.QES.Algebra.Hilbert.hilbert_jit_methods import get_mapping, find_repr_int, find_representative_int, get_matrix_element, jitted_find_repr_int, jitted_get_mapping, jitted_get_matrix_element

if JAX_AVAILABLE:
    import jax.numpy as jnp
    
    @jax.jit
    def get_mapping_jax(mapping, state):
        """
        Get the mapping of the state.
        
        Args:
            mapping (list): The mapping of the states.
            state (int): The state to get the mapping for.
        
        Returns:
            int: The mapping of the state.
        """
        return mapping[state] if len(mapping) > state else state

#####################################################################################################
#! Hilbert space class
#####################################################################################################

class HilbertSpace(ABC):
    """
    A class to represent a Hilbert space either in Many-Body Quantum Mechanics or Quantum Information Theory and non-interacting systems.
    """
    
    #################################################################################################
    
    _ERRORS = {
        "sym_gen"       : "The symmetry generators must be provided as a dictionary.",
        "global_syms"   : "The global symmetries must be provided as a list.",
        "gen_mapping"   : "The flag for generating the mapping must be a boolean.",
        "ns"            : "Either 'ns' or 'lattice' must be provided.",
        "lattice"       : "Either 'ns' or 'lattice' must be provided.",
        "nhl"           : "The local Hilbert space dimension must be an integer.",
        "nhint"         : "The number of modes must be an integer.",
        "single_part"   : "The flag for the single particle system must be a boolean.",
        "state_type"    : "The state type must be a string.",
        "backend"       : "The backend must be a string or a module.",
    }
    
    def __init__(self,
                # core definition - elements to define the modes
                ns          : Union[int, None]                  = None,
                lattice     : Union[Lattice, None]              = None,
                nh          : Union[int, None]                  = None,
                # mode specificaton
                is_manybody : bool                              = True,
                part_conserv: Optional[bool]                    = True,
                # local space properties - for many body
                nhl         : Optional[int]                     = 2,
                nhint       : Optional[int]                     = 1,
                sym_gen     : Union[dict, None]                 = None,
                global_syms : Union[List[GlobalSymmetry], None] = None,
                gen_mapping : bool                              = False,
                # general parameters
                state_type  : str                               = "integer",
                backend     : str                               = 'default',
                dtype                                           = np.float64,
                logger      : Optional[Logger]                  = None,
                **kwargs):
        """
        Initializes a HilbertSpace object with specified system and local space properties, symmetries, and backend configuration.
        
        Parameters:
            ns (Union[int, None], optional):
                Number of sites in the system. If not provided, inferred from `lattice` or `nh`.
            lattice (Union[Lattice, None], optional):
                Lattice object defining the system structure. If provided, `ns` is set from `lattice.ns`.
            nh (Union[int, None], optional):
                Full Hilbert space dimension. Used if neither `ns` nor `lattice` is provided.
            is_manybody (bool, optional):
                Flag indicating if the system is many-body. Default is True.
            nhl (Optional[int], optional):
                Local Hilbert space dimension (per site). Default is 2.
            nhint (Optional[int], optional):
                Number of internal modes per site (e.g., fermions, bosons). Default is 1.
            sym_gen (Union[dict, None], optional):
                Dictionary specifying symmetry generators. Default is None.
            global_syms (Union[List[GlobalSymmetry], None], optional): 
                List of global symmetry objects. Default is None.
            gen_mapping (bool, optional):
                Whether to generate state mapping based on symmetries. Default is False.
            state_type (str, optional):
                Type of state representation (e.g., "integer"). Default is "integer".
            backend (str, optional):
                Backend to use for vectors and matrices. Default is 'default'.
            dtype (optional):
                Data type for Hilbert space arrays. Default is np.float64.
            logger (Optional[Logger], optional):
                Logger instance for logging. Default is None.
            **kwargs:
                Additional keyword arguments, such as 'threadnum' (number of threads to use).
        Raises:
            ValueError: If provided arguments do not match expected types or required parameters are missing.
        Notes:
            - If both `ns` and `lattice` are None, and `nh` is provided, `ns` is inferred from `nh` and `nhl`.
            - Initializes symmetry groups, state mappings, and backend configuration.
            - Sets up logging and threading options.
        """
        
        self._logger        = logger if logger is not None else get_global_logger()
        # initialize the backend for the vectors and matrices
        self._backend, self._backend_str, self._state_type = self.reset_backend(backend, state_type)
        self._dtype         = dtype if dtype is not None else self._backend.float64
        self._is_many_body  = is_manybody
        self._is_quadratic  = not is_manybody
        
        # check if the arguments match the requirements
        if not isinstance(sym_gen, dict) and sym_gen is not None:
            raise ValueError(HilbertSpace._ERRORS["sym_gen"])
        if not isinstance(global_syms, list) and global_syms is not None:
            raise ValueError(HilbertSpace._ERRORS["global_syms"])
        if not isinstance(gen_mapping, bool):
            raise ValueError(HilbertSpace._ERRORS["gen_mapping"])
        
        # handle the system physical size dimension - distinguish between the number of sites and the lattice object
        # if the lattice object is provided, the number of sites is calculated from the lattice object
        if ns is not None:
            self._ns        = ns
            self._lattice   = lattice
            if self._lattice is not None and self._lattice.ns != ns:
                self._log(f"Warning: The number of sites in the lattice ({self._lattice.ns}) is different than the number of sites provided ({ns}).",
                            lvl = 1, color = 'yellow', log = 'info')
        elif lattice is not None:
            self._ns        = lattice.ns
            self._lattice   = lattice
        elif nh is not None and self._is_many_body:
            try:
                self._ns        = int(math.log(nh, nhl))
                if nhl**(self._ns * nhl) < nh:
                    raise ValueError(HilbertSpace._ERRORS["ns"])
                self._log(f"Inferred Ns={self._ns} from Nh={nh} and Nhl={nhl}", log='info', lvl=2)
            except ValueError as e:
                raise ValueError(f"Cannot infer Ns from Nh={nh}, Nhl={nhl}. Provide Ns or Lattice. Error: {e}")
        elif nh is not None and self._is_quadratic:
            self._ns        = nh                        # Assume Nh directly represents Ns for simple quadratic
            self._lattice   = None
            self._log(f"Assuming Ns={self._ns} from provided Nh={nh} in quadratic mode.", log='info', lvl=2)
        else:
            raise ValueError(HilbertSpace._ERRORS["ns"])
        
        # set locals
        self._nhl           = nhl
        self._nhint         = nhint
        # Nhfull: Potential full many-body space size
        try:
            self._nhfull    = self._nhl**(self._nhint * self._ns) if self._ns > 0 else 0
        except OverflowError:
            self._nhfull    = float('inf')
            self._log(f"Warning: Full Hilbert space size NhFull exceeds standard limits (Ns={self._ns}, Nhl={self._nhl}).", log='warning', lvl=0)

        # Nh: Effective dimension of the *current* representation
        # Initial estimate:
        if self._is_quadratic:
            # For quadratic, the "dimension" is often Ns (or 2Ns if pairing)
            # We'll set it initially to Ns, can be adjusted later if needed (e.g., for Bogoliubov)
            self._nh = self._ns
            self._log(f"Initialized HilbertSpace in quadratic mode: Ns={self._ns}, effective Nh={self._nh}.",
                        log='debug', lvl=1, color='green')
        else: # Many-body
            # If symmetries will be applied, Nh will be reduced later.
            # Start with the full size, potentially reduced by global syms only initially.
            self._nh = self._nhfull
            self._log(f"Initialized HilbertSpace in many-body mode: Ns={self._ns}, Nhl={self._nhl}, initial Nh={self._nh} (potentially reducible).",
                        color = 'green', log='debug', lvl=1)
    
        #! Initialize the symmetries
        self._normalization         = []                            # normalization of the states - how to return to the representative
        self._sym_group             = []
        self._sym_group_sec         = []
        self._global_syms           = global_syms if global_syms is not None else []
        self._particle_conserving   = part_conserv
    
        # initialize the properties of the Hilbert space
        self._mapping       = None
        self._reprmap       = None                                  # mapping of the representatives (vector of tuples (state, representative value))
        self._fullmap       = None                                  # mapping of the full Hilbert space
        
        self._getmapping_fun= None                                  # function to get the mapping of the states
        # setup the logger instance for the Hilbert space
        self._threadnum     = kwargs.get('threadnum', 1)            # number of threads to use
        
        if self._is_many_body and (sym_gen or self._global_syms):
            if gen_mapping:
                self._log("Explicitly requested immediate mapping generation.", log='debug', lvl=2)
            self._init_mapping(sym_gen, gen_mapping=gen_mapping)    # gen_mapping True here enables reprmap
        elif self._is_quadratic:
            self._log("Quadratic mode: Skipping symmetry mapping generation.", log='debug', lvl=2)
            # Ensure mapping attributes are None
            self._mapping       = None
            self._normalization = None
            self._reprmap        = None
            # Setup sym group if generators provided, but don't build map
            if sym_gen:
                self._gen_sym_group(sym_gen) #!TODO: How to define this?
        
        if self._sym_group == []:
            self._sym_group = [operator_identity(self._backend)]
    
    # --------------------------------------------------------------------------------------------------
    #! Resets
    # --------------------------------------------------------------------------------------------------
    
    @staticmethod
    def reset_backend(backend : str, state_type : str):
        """
        Reset the backend for the Hilbert space.
        
        Args:
            backend (str): The backend to use for the Hilbert space.
        """
        if isinstance(backend, str):
            _backend_str   = backend
            _backend       = get_backend(backend)
        else:
            _backend_str   = 'np' if backend == np else 'jax'
            _backend       = backend
        
        statetype = HilbertSpace.reset_statetype(state_type, _backend)
        return _backend, _backend_str, statetype
        
    @staticmethod
    def reset_statetype(state_type : str, backend):
        """
        Reset the state type for the Hilbert space.
        
        Args:
            state_type (str): The state type to use for the Hilbert space.
        """
        if state_type.lower() == "integer" or state_type.lower() == "int":
            return int
        return backend.array
        
    def reset_local_symmetries(self):
        """
        Reset the local symmetries of the Hilbert space.
        """
        self._log("Reseting the local symmetries. Can be now recreated.", lvl = 2, log = 'debug')
        self._sym_group     = []
        self._sym_group_sec = []
    
    # --------------------------------------------------------------------------------------------------
    
    def log(self, msg : str, log : Union[int, str] = Logger.LEVELS_R['info'], lvl : int = 0, color : str = "white", append_msg = False):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (Union[int, str]) : The flag to log the message (default is 'info').
            lvl (int) : The level of the message.
        """
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        self._log(msg, log=log, lvl=lvl, color=color, append_msg=append_msg)
    
    def _log(self, msg : str, log : Union[int, str] = Logger.LEVELS_R['info'], lvl : int = 0, color : str = "white", append_msg = True):
        """
        Log the message.
        
        Args:
            msg (str) : The message to log.
            log (int) : The flag to log the message (default is 'info').
            lvl (int) : The level of the message.
        """
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        if append_msg:
            msg = f"[HilbertSpace] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log = log, lvl = lvl)
    
    ####################################################################################################
    #! Generate the symmetry group and all properties of the generation
    ####################################################################################################
    
    #! Translation related
        
    def _gen_sym_group_check_t(self, sym_gen : list) -> (list, Tuple[bool, bool], Tuple[Operator, LatticeDirection]):
        '''
        Helper function to check the translation symmetry. This function is used to check the translation symmetry.
        It gets the translation generator and checks if it satisfies the symmetry conditions. If the translation
        is not possible with the boundary conditions provided, the function returns.
        
        Args:
            sym_gen (list) : A list of symmetry generators.
        Returns:
            list, tuple : A list of symmetry generators and a tuple of flags (has_translation, has_cpx_translation).
        '''
        if self._lattice is None:
            has_translation = any(g[0].has_translation() for g in sym_gen if hasattr(g,'__len__') and len(g)>0)
            if has_translation:
                self._log("Warning: Translation symmetry requested but no Lattice provided. Ignoring translation.", log='warning', lvl=1)
            # Remove translation generators if found
            sym_gen_out = [g for g in sym_gen if not (hasattr(g,'__len__') and len(g)>0 and g[0].has_translation())]
            return sym_gen_out, (False, False), (None, LatticeDirection.X)
        
        has_cpx_translation = False
        has_translation     = False
        t                   = None  
        direction           = LatticeDirection.X
        for idx, (gen, sec) in enumerate(sym_gen):
            
            # proceed if this is a translation
            if gen.has_translation():
                
                # Check if the lattice has periodic boundary conditions
                has_translation = (self._lattice.get_BC() == LatticeBC.PBC)
                
                if has_translation:
                    self._sym_group_sec.append((gen, sec))
                    
                # Remove the translation generator from the list
                sym_gen.pop(idx)
                
                # create the operator
                
                direction   = LatticeDirection.X
                direction   = LatticeDirection.Y if gen == SymmetryGenerators.TranslationY else direction
                direction   = LatticeDirection.Z if gen == SymmetryGenerators.TranslationZ else direction 
                kx          = sec if direction == LatticeDirection.X else 0.0
                ky          = sec if direction == LatticeDirection.Y else 0.0
                kz          = sec if direction == LatticeDirection.Z else 0.0
                t           = translation(self._lattice, kx = kx, ky = ky, kz = kz, direction = direction, backend = self._backend)
                
                # Check for complex sector: nonzero momentum (unless at half filling with even Ns)
                if sec != 0 and not (sec == self.Ns // 2 and self.Ns % 2 == 0):
                    has_cpx_translation = True
                break
            
        if has_translation:
            self._log("Translation symmetry is present.", lvl = 1)
            if has_cpx_translation:
                self._log("Translation in complex sector...", lvl = 2, color = 'blue')
        return sym_gen, (has_translation, has_cpx_translation), (t, direction)

    def _gen_sym_apply_t(self, sym_gen_op : list, t : Optional[Operator] = None, direction : LatticeDirection = LatticeDirection.X):
        """
        Apply the translation symmetry.
        """
        if t is not None:
            self._log("Adding translation to symmetry group combinations.", lvl = 2, color = 'yellow')
            
            # check the direction
            size        = self._lattice.lx if direction == LatticeDirection.X else 1
            size        = self._lattice.ly if direction == LatticeDirection.Y else size
            size        = self._lattice.lz if direction == LatticeDirection.Z else size
            sym_get_out = sym_gen_op.copy()
            sym_gen_in  = sym_gen_op.copy()
            t_in        = t
            for _ in range(1, size):
                # this is the number of translations to apply
                for op_local in sym_gen_in:
                    sym_get_out.append(t % op_local)
                # move the translation to the next cell
                t_in = t_in % t
            return sym_get_out
        return sym_gen_op

    #! Global symmetries related
    
    def _gen_sym_group_check_u1(self) -> (bool, float):
        """
        Check if a U(1) global symmetry is present.
        Returns (has_U1, U1_value).
        """
        has_u1, u1_val = self.check_u1()
        if has_u1:
            self._log("U(1) global symmetry is present.", lvl = 2, color = 'blue')
        return has_u1, u1_val

    #! Removers for the symmetry generators

    def _gen_sym_remove_reflection(self, sym_gen : list, has_cpx_translation : bool):
        """
        Helper function to remove reflections from the symmetry generators if the complex translation is present.
        
        Args:
            sym_gen (list)         : A list of symmetry generators.
            has_cpx_translation (bool) : A flag for complex translation - momentum is different than 0 or pi.
        """
        if has_cpx_translation and sym_gen is not None and hasattr(sym_gen, "__iter__"):
            sym_gen = [gen for gen in sym_gen if not isinstance(gen[0], SymmetryGenerators.Reflection)]
            self._log("Removed reflection symmetry from the symmetry generators.", lvl = 2, color = 'blue')
        return sym_gen
    
    def _gen_sym_remove_parity(self, sym_gen : list, has_u1 : bool, has_u1_sec : float):
        """
        If U(1) is present but the system is not at half-filling (or has odd size),
        remove parity generators in the X and/or Y directions.
        """
        if has_u1:
            
            self._log("U(1) symmetry detected. Checking parity generators...", log = 1, lvl = 1, color = 'yellow')
            
            new_sym_gen = []
            for (gen, sec) in sym_gen:
                if gen in (SymmetryGenerators.ParityX, SymmetryGenerators.ParityY) and \
                   ((int(has_u1_sec) != self._ns // 2) or (self._ns % 2 != 0)):
                    self._log(f"Removing parity {gen} due to U(1) constraint.", log = 1, lvl = 2, color = 'blue')
                else:
                    new_sym_gen.append((gen, sec))
            sym_gen = new_sym_gen
        return sym_gen
    
    #! Printer
    def _gen_sym_print(self, t: Optional[Operator] = None) -> None:
        """
        Print the symmetry group.

        Parameters
        ----------
        has_t : bool
            Flag for the translation symmetry.
        t : Operator
            The translation operator.        
        """
        
        self._log("Using local symmetries:", lvl = 1, color = 'green')
        for (g, sec) in self._sym_group_sec:
            self._log(f"{g}: {sec}", lvl = 2, color = 'blue')
        if t is not None:
            self._log(f"{t}: {t.eigval}", lvl = 2, color = 'blue')
        self._log("Using global symmetries:", lvl = 1, color = 'green')
        for g in self._global_syms:
            self._log(f"{g}: {g.get_val()}", lvl = 2, color = 'blue')
    
    #! Final symmetry group generation
    
    def _gen_sym_group(self, gen : list):
        """
        Generate the symmetry group of the Hilbert space. 
        This is done by going through the symmetry generators and checking the symmetries.
        
        Args:
            gen (list) : A list of symmetry generators.
        """
		
        if (not gen or len(gen) == 0) and not self.check_global_symmetry():
            self._log("No local or global symmetries provided; symmetry group is empty.", lvl = 1, log = 'debug', color = 'green')
            return
        
        # copy the generators to modify them if needed
        sym_gen                     = gen.copy() if gen is not None and hasattr(gen, "__iter__") else []
        
        # Reset symmetry groups.
        self.reset_local_symmetries()
                
        #! globals - check the global symmetries
        
        # Check global U(1) symmetry.
        has_u1, u1_val              = self._gen_sym_group_check_u1()
        
        # process translation symmetries
        sym_gen, (_, has_cpx_t), (t, direction) = self._gen_sym_group_check_t(sym_gen)
        
        # remove reflections from the symmetry generators if the complex translation is present
        sym_gen                     = self._gen_sym_remove_reflection(sym_gen, has_cpx_t)

        # check the existence of the parity when U(1) is present
        sym_gen                     = self._gen_sym_remove_parity(sym_gen, has_u1, u1_val)
        
        # save all again for convenience
        for gen, sec in sym_gen:
            self._sym_group_sec.append(sec)
        
        # ------------------------------
        # Generate all combinations of the local generators.
        # For each subset of indices from sym_gen, combine their operators.
        _size_gen = len(sym_gen)
        for r in range(_size_gen + 1):
            # r = 0, 1, ..., _size_gen - goes through subsets of sym_gen
            
            for indices in combinations(range(_size_gen), r):
                # combine the operators from the subset of sym_gen
                
                # Start with a neutral (identity) operator.
                operator_in = operator_identity()
                
                # Combine the operators from the subset of sym_gen.
                for idx in indices:
                    op_i = choose(sym_gen[idx], ns = self._ns, lat = self._lattice, backend = self._backend)
                    operator_in = operator_in % op_i # use composition of operators 
                self._sym_group.append(operator_in)
        
        # apply the translation symmetry
        self._sym_group = self._gen_sym_apply_t(self._sym_group, t, direction)

        self._gen_sym_print(t)
    
    # --------------------------------------------------------------------------------------------------
    
    def _init_mapping(self, gen : list, gen_mapping : bool = False):
        """
        Initialize the mapping of the states. This function is used to generate the mapping of the states to the representatives.
        It uses a list of symmetry generators to generate the mapping.
        
        Args:
            gen (list)         : A list of symmetry generators.
            gen_mapping (bool) : A flag to generate the mapping of the representatives to the original states.
        """
        if not self._is_many_body:
            self._log("Skipping mapping initialization in quadratic mode.", log='debug', lvl=2)
            return
        if not gen and not self._global_syms:
            self._log("No symmetries provided, mapping is trivial (identity).", log='debug', lvl=2)
            self._nh            = self._nhfull
            self._mapping       = None
            self._normalization = None
            return
        
        t0 = time.time()
        
        self._gen_sym_group(gen)    # generate the symmetry group
        
        if gen is not None and len(gen) > 0:
            self._log("Generating the mapping of the states...", lvl = 1, color = 'green')

        # generate the mapping of the states
        if self._state_type == int:
            self._generate_mapping_int(gen_mapping)
        else:
            self._generate_mapping_base(gen_mapping)
        
        if gen is not None and len(gen) > 0 or len(self._mapping) > 0:
            t1 = time.time()
            self._log(f"Generated the mapping of the states in {t1 - t0:.2f} seconds.", lvl = 2, color = 'green')
            self._mapping           = self._backend.array(self._mapping, dtype = self._backend.int64)
        else:
            self._log("No mapping generated.", lvl = 1, color = 'green', log = 'debug')

    # --------------------------------------------------------------------------------------------------

    ####################################################################################################
    #! Getters and checkers for the Hilbert space
    ####################################################################################################
    
    # GLOBAL SYMMETRIES
    
    def check_global_symmetry(self):
        """
        Check if there are any global symmetries.
        """
        return len(self._global_syms) > 0 if self._global_syms is not None else False
    
    def check_u1(self):
        """
        Check if there is a U(1) symmetry.
        """
        if self._global_syms is not None:
            for sym in self._global_syms:
                if sym.get_name() == GlobalSymmetries.U1:
                    return True, sym.get_val()
        return False, None
    
    #---------------------------------------------------------------------------------------------------
    
    def get_sym_info(self):
        """
        Creates the information string about the Hilbert space and symmetries.
        
        Returns:
            str: A string containing the information about all the symmetries.
        """
        tmp = ""
        if self._sym_group_sec:
            # start with local symmetries
            for gen, val in self._sym_group_sec:
                tmp += f"{gen}={val},"
        if self.check_global_symmetry():
            # start with global symmetries
            for g in self._global_syms:
                tmp += f"{g.get_name()}={g.get_val():.2f},"
        
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
    def dtype(self):
        """
        Return the data type of the Hilbert space.
        
        Returns:
            type: The data type of the Hilbert space.
        """
        return self._dtype
    
    @property
    def modifies(self):
        """
        Return the flag for modifying the Hilbert space.
        
        Returns:
            bool: The flag for modifying the Hilbert space.
        """
        if self._is_quadratic:
            return False
        
        if self._mapping is None and (self._sym_group or self._global_syms):
            # Symmetries exist but mapping not generated, assume it *will* modify
            # This might be slightly inaccurate if symmetries don't reduce the space
            return True
        elif self._mapping is not None:
            return self._nh != self._nhfull
        else: # No symmetries, no modification
            return False
        
        return self._nh != self._nhfull
    
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
    
    @property
    def ns(self):
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
    def mapping(self):
        """
        Return the mapping of the states.
        
        Returns:
            list: The mapping of the states.
        """
        return self._mapping
    
    @property
    def sym_group(self):
        """
        Return the symmetry group.
        
        Returns:
            list: The symmetry group.
        """
        return self._sym_group
    
    @property
    def reprmap(self):
        """
        Return the mapping of the representatives.
        
        Returns:
            list: The mapping of the representatives.
        """
        return self._reprmap
    
    @property
    def normalization(self):
        """
        Return the normalization of the states.
        
        Returns:
            list: The normalization of the states.
        """
        return self._normalization
    
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
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def quadratic(self):
        return self._is_quadratic
    
    @property
    def many_body(self):
        return self._is_many_body
    
    @property
    def particle_conserving(self):
        return self._particle_conserving
    
    # --------------------------------------------------------------------------------------------------
    
    @property
    def logger(self):
        """
        Return the logger instance.
        
        Returns:
            Logger: The logger instance.
        """
        return self._logger

    # --------------------------------------------------------------------------------------------------
    
    def norm(self, state):
        """
        Return the normalization of a given state.
        
        Args:
            state (int): The state to get the normalization for.
        
        Returns:
            float: The normalization of the state.
        """
        return self._normalization[state] if state < len(self._normalization) else 1.0

    # --------------------------------------------------------------------------------------------------
    
    ####################################################################################################
    #! Representation of the Hilbert space
    ####################################################################################################
    
    def __str__(self):
        """ Return a string representation of the Hilbert space. """
        mode = "Many-Body" if self._is_many_body else "Quadratic"
        info = f"Mode: {mode}\n"
        info += f"Ns (sites/modes): {self._ns}\n"
        if self._is_many_body:
            info    += f"Nhl (local dim): {self._nhl}\n"
            info    += f"NhFull (potential MB size): {self._nhfull}\n"

        if self._mapping is not None:
            info    += f"Reduced Hilbert space size (Nh): {self._nh}\n"
            info    += f"Number of symmetry operators considered: {len(self._sym_group)}\n"
        elif self._is_many_body:
            info    += f"Effective Hilbert space size (Nh): {self._nh} (Full space, no reduction applied)\n"
        else: # Quadratic
            info    += f"Effective Hilbert space size (Nh): {self._nh} (Quadratic basis size)\n"

        gs_info     = [f"{g.get_name_str()}={g.get_val()}" for g in self._global_syms]
        if gs_info:
            info += f"Global Symmetries: {', '.join(gs_info)}\n"
        ls_info     = [f"{g}={sec}" for g, sec in self._sym_group_sec]
        if ls_info:
            info += f"Local Symmetries: {', '.join(ls_info)}\n"
        return info

    def __repr__(self):
        sym_info = self.get_sym_info()
        base = "Single particle" if self._single_part else "Many body"
        return f"{base} Hilbert space with {self._nh} states and {self._ns} sites; {self._nhl} modes per site. Symmetries: {sym_info}" if sym_info else ""
    
    ####################################################################################################
    #! Find the representative of a state
    ####################################################################################################
    
    def _find_sym_norm_base(self, state):
        """
        Finds the normalization for a given state (baseIdx) by summing the eigenvalues
        over all symmetry operators that return the same state.
        
        Returns sqrt(sum of eigenvalues).
        """
        pass
    
    def _find_sym_norm_int(self, state):
        """
        Find the symmetry normalization of a given state.
        
        Args:
            state (int): The state to find the symmetry normalization for.
        
        Returns:
            float: The symmetry normalization of the state.
        """
        norm = 0.0
        for g in self._sym_group:
            new_idx, val = g(state)
            if new_idx == state:
                norm += val
        return math.sqrt(norm)
    
    def find_sym_norm(self, state) -> Union[float, complex]:
        """
        Finds the normalization for a given state (baseIdx) by summing the eigenvalues
        over all symmetry operators that return the same state.
        
        Returns sqrt(sum of eigenvalues).
        """
        if isinstance(state, int):
            return self._find_sym_norm_int(state)
        return self._find_sym_norm_base(state)
    
    # --------------------------------------------------------------------------------------------------
    
    def _find_repr_base(self, state):
        pass
    
    def _find_repr_int(self, state):
        """
        Find the representative of a given state.
        
        Args:
            state (int): The state to find the representative for.
        
        Returns:
            int: The representative of the state.
        """
                
        # If mapping exists, return saved representative.
        return find_repr_int(state, self._mapping, self._sym_group, self._reprmap)
    
    def find_repr(self, state):
        """
        Find representatives of another state using various combinations of symmetry generators.

        This method computes the smallest representative possible by applying different combinations of symmetry 
        generators. It also determines the symmetry eigenvalue associated with returning to the original state.

        Args:
            state (int or state type): The state representation. If an integer is provided, an integer-based approach 
                                    is used; otherwise, a base state representation is assumed.

        Returns:
            tuple: A pair containing the representative index and the corresponding symmetry eigenvalue.
        """
        if isinstance(state, int):
            return self._find_repr_int(state)
        return self._find_repr_base(state)
    
    # --------------------------------------------------------------------------------------------------
    
    def find_representative_base(self, state, normalization_beta):
        """
        Find the representative of a given state.
        
        Args:
            state (np.ndarray): The state to find the representative for.
            normalization_beta (float): The normalization in sector beta.
        
        Returns:
            np.ndarray: The representative of the state.
        """
        pass
    
    def find_representative_int(self, state, normalization_beta):
        """
        Find the representative of a given state.
        """
        return find_representative_int(state, self._mapping,
                self._normalization, normalization_beta, self._sym_group, self._reprmap)
    
    def find_representative(self, state, normalization_beta):
        """
        Finds the representative for a given base index in the "sector alfa".

        This procedure is handy when acting with a matrix on a representative |¯n>,
        which is transformed to a state |m> by the matrix. The representative |¯m> for
        this new state is then found, and the normalization in sector beta is applied.

        It means that we have a base index in sector alfa and we look for the representative
        after acting on it. Then, the normalization in sector beta is taken into account.

        Args:
            state: The state for which to find the representative.

        Returns:
            A tuple containing the representative binary number and the corresponding eigenvalue
            derived from the symmetry operations.
        """
        if isinstance(state, int):
            return self.find_representative_int(state, normalization_beta)
        return self.find_representative_base(state, normalization_beta)
    
    # --------------------------------------------------------------------------------------------------
    
    def _mapping_kernel_int(self, start: int, stop: int, t: int):
        """
        For a given range of states in the full Hilbert space, find those states
        that are representatives (i.e. the smallest state under symmetry operations)
        and record their normalization.
        Parameters:
            start (int)             : The starting index of the range.
            stop (int)              : The stopping index of the range.
            t (int)                 : The thread number.
        """
        map_threaded = []
        norm_threaded = []
        
        for j in range(start, stop):
            global_checker = True
            if self._global_syms:
                for g in self._global_syms:
                    global_checker = global_checker and g(j)
                    
            # if the global symmetries are not satisfied, skip the state
            if not global_checker:
                continue
            
            rep, _ = self._find_repr_int(j)
            if rep == j:
                n = self._find_sym_norm_int(j)
                if abs(n) > __SYM_NORM_THRESHOLD:
                    map_threaded.append(j)
                    norm_threaded.append(n)
        return map_threaded, norm_threaded
    
    def _mapping_kernel_int_repr(self):
        """
        For all states in the full Hilbert space, determine the representative.
        For each state j, if global symmetries are not conserved, record a bad mapping.
        Otherwise, if j is already in the mapping, record trivial normalization.
        Otherwise, find the representative and record its normalization.
        
        This function is created whenever one wants to create a full map for the Hilbert space 
        and store it in the mapping.
        """
        
        # initialize the mapping and normalization if necessary
        self._reprmap = []
        
        for j in range(self._nhfull):
            global_checker = True
            
            if self._global_syms:
                for g in self._global_syms:
                    global_checker = global_checker and g(j)
                    
            # if the global symmetries are not satisfied, skip the state
            if not global_checker:
                self._reprmap.append((__BAD_BINARY_SEARCH_STATE, 0.0))
                continue
            
            mapping_size    = len(self._mapping)
            idx             = binary_search(self._mapping, 0, mapping_size - 1, j)
            if idx != __BAD_BINARY_SEARCH_STATE and idx < mapping_size:
                self._reprmap.append((idx, 1.0))
                continue
            
            # find the representative
            rep, sym_eig    = self.find_repr(j)
            idx             = binary_search(self._mapping, 0, mapping_size - 1, rep)
            if idx != __BAD_BINARY_SEARCH_STATE and idx < mapping_size:
                sym_eigc = sym_eig.conjugate() if hasattr(sym_eig, "conjugate") else sym_eig
                self._reprmap.append((idx, np.conj(sym_eigc)))
            else:
                self._reprmap.append((__BAD_BINARY_SEARCH_STATE, 0.0))
    
    # --------------------------------------------------------------------------------------------------
    
    def _mapping_kernel_base(self):
        """
        """
        pass
    
    def _mapping_kernel_base_repr(self):
        """
        """
        pass
    
    # --------------------------------------------------------------------------------------------------
    
    def _generate_mapping_int(self, gen_mapping : bool = False):
        """
        Generate the mapping of the states for the Hilbert space.
        
        Args:
            gen_mapping (bool): A flag to generate the mapping of the representatives to the original states.
        """
        
        # no symmetries - no mapping
        if  (self._sym_group is None or len(self._sym_group) == 0) and (self._global_syms is None or len(self._global_syms) == 0):
            self._nh = self._nhfull
            return
        
        fuller      = self._nhfull
        # For demonstration, use self.threadNum if set; otherwise default to 1.
        if self._threadnum > 1:
            self._mapping       = []
            self._normalization = []
            futures             = []
            
            # create the threads
            with ThreadPoolExecutor(max_workers=self._threadnum) as executor:
                
                for t in range(self._threadnum):
                    start   = int(fuller * t / self._threadnum)
                    stop    = fuller if (t+1) == self._threadnum else int(fuller * (t+1) / self._threadnum)
                    futures.append(executor.submit(self._mapping_kernel_int, start, stop, t))
                for future in as_completed(futures):
                    m, n    = future.result()
                    self._mapping.extend(m)
                    self._normalization.extend(n)
        else:
            self._mapping, self._normalization = self._mapping_kernel_int(0, fuller, 0)
        self._nh = len(self._mapping)
        
        if gen_mapping:
            self._mapping_kernel_int_repr()

    def _generate_mapping_base(self, gen_mapping : bool = False):
        """
        Generate the mapping of the states for the Hilbert space.
        
        Args:
            gen_mapping (bool): A flag to generate the mapping of the representatives to the original states.
        """
        pass
    
    # --------------------------------------------------------------------------------------------------
    
    def get_matrix_element(self, k, new_k, kmap = None, h_conj = False):
        """
        Compute the matrix element between two states in the Hilbert space.
        This method determines the matrix element corresponding to the transition between a given state |k⟩ and a new state defined by new_k.
        It accounts for the possibility that the new state may not be in its representative form, in which case it finds the representative state
        and applies the corresponding normalization factor or symmetry eigenvalue. The ordering of the returned tuple elements may be
        reversed based on the flag h_conj; if h_conj is False, the result is ((representative, k), factor), otherwise ((k, representative), factor).

        Imagine a situation where an operator acts on a state |k> and gives a new state <new_k|.
        We use this Hilbert space to find the matrix element between these two states. It may happen
        that the new state is not the representative of the original state, so we need to find the
        representative and the normalization factor.

        Args:
            k: An index or identifier for the original state in the Hilbert space.
            new_k: An index or identifier representing the new state after the operator action.
            h_conj (bool, optional): A flag to determine the order of the tuple. If False (default), the tuple is (representative, k), 
                                    otherwise it is (k, representative).
        Returns:
            tuple: A tuple consisting of:
                - A tuple of two elements representing the indices (or identifiers) of the representative state and the original state,
                    ordered based on the value of h_conj.
                - The normalization factor or symmetry eigenvalue associated with the new state. 
        Note:
            This function uses the find_representative function from this module to find the representative.        
        """
        return get_matrix_element(k, new_k, kmap, h_conj, self._mapping, 
                        self._normalization, self._sym_group, self._reprmap)
    
    ####################################################################################################
    #! Full Hilbert space generation
    ####################################################################################################
    
    def generate_full_map_int(self):
        """
        Generate the full mapping of the Hilbert space.
        """
        self._fullmap = []
        if self._global_syms:
            for j in range(self._nhfull):
                global_checker = True
                for g in self._global_syms:
                    global_checker = global_checker and g(j)
                # if the global symmetries are satisfied, add the state to the full map
                if global_checker:
                    self._fullmap.append(j)

    def get_full_map_int(self):
        """
        Generate the full mapping of the Hilbert space.
        """
        if self._fullmap is not None and len(self._fullmap) > 0:
            return self._fullmap
        self.generate_full_map_int()
        return self._fullmap
    
    @maybe_jit
    def _cast_to_full_jax(self, state, backend : str = "jax"):
        """
        Cast the state to the full Hilbert space.
        
        Args:
            state: The state to cast to the full Hilbert space.
        
        Returns:
            int: The state cast to the full Hilbert space.
        """
        import jax.numpy as jnp
        # Create a full state vector of zeros.
        f_s = jnp.zeros((self._nhfull,), dtype=self._dtype)
        # Convert self._fullmap to a JAX array if it isn't one already.
        fullmap = jnp.array(self._fullmap)
        # Use the vectorized .at[].set() operation to update f_s at indices in fullmap.
        f_s = f_s.at[fullmap].set(state)
        return f_s
    
    def cast_to_full(self, state):
        """
        Cast the state to the full Hilbert space.
        
        Args:
            state: The state to cast to the full Hilbert space.
        
        Returns:
            int: The state cast to the full Hilbert space.
        """
        if not self.check_global_symmetry():
            return state
        if self._fullmap is None or len(self._fullmap) == 0:
            self.generate_full_map_int()
        if isinstance(state, np.ndarray):
            final_state = self._backend.zeros(self._nhfull, dtype=self._dtype)
            for i, idx in enumerate(self._fullmap):
                final_state[idx] = state[i]
            return final_state
        return self._cast_to_full_jax(state)
    
    ####################################################################################################
    #! Operators for the Hilbert space
    ####################################################################################################
    
    def __len__(self):
        """
        Return the dimension of the Hilbert space.
        
        Returns:
            int: The dimension of the Hilbert space.
        """
        return self._nh
    
    def get_mapping(self, i):
        """
        Return the mapping of the states.
        
        Returns:
            list: The mapping of the states.
        Note:
            This function uses the get_mapping function from this module to get the mapping.
        """
        return get_mapping(self.mapping, i)
    
    def __getitem__(self, i):
        """
        Return the i-th basis state of the Hilbert space.
        
        Args:
            i: The index of the basis state to return or a state to find the representative for.
        
        Returns:
            np.ndarray: The i-th basis state of the Hilbert space.
        """
        if isinstance(i, (int, np.integer)):
            return self._mapping[i] if (self._mapping is not None and len(self._mapping) > 0) else i
        return self._mapping[i]
    
    def __call__(self, i):
        """
        Return the representative of the i-th basis state of the Hilbert space.
        """
        return self.find_repr(i)
        
    def __contains__(self, state):
        """
        Check if a state is in the Hilbert space.
        
        Args:
            state: The state to check.
        
        Returns:
            bool: True if the state is in the Hilbert space, False otherwise.
        """
        if isinstance(state, int):
            return state in self._mapping
        #! TODO: implement the state finding
        return NotImplementedError("Only integer indexing is supported.")
    
    ################################################################################################
    
####################################################################################################

def set_operator_elem(operator, hilbert : HilbertSpace, k : int, val, new_k : int, conj = False):
    """
    Set the matrix element of the operator.
    
    Args:
        operator (Operator)     : The operator to set the matrix element for.
        hilbert (HilbertSpace)  : The Hilbert space object.
        i                       : The index of the matrix element.
        val                     : The value of the matrix element.
        j                       : The index of the matrix element.
    Returns:
        Operator: The operator with the matrix element set.
    """
    (row, col), sym_eig = hilbert.get_matrix_element(k, new_k, h_conj = conj)
    
    # check if operator is numpy array
    if isinstance(operator, np.ndarray):
        operator[row, col]  += val * sym_eig
    else:
        operator = operator.at[row, col].add(val * sym_eig)
    return operator # for convenience

def get_operator_elem(hilbert : HilbertSpace, k : int, new_k : int, conj = False):
    """
    Get the matrix element of the operator.
    
    Args:
        hilbert (HilbertSpace)  : The Hilbert space object.
        k                       : The index of the matrix element.
        new_k                   : The new index of the matrix element.
        conj                    : Whether to take the complex conjugate (default is False).
    Returns:
        float: The matrix element of the operator.
    """
    (row, col), sym_eig = hilbert.get_matrix_element(k, new_k, h_conj = conj)
    return (row, col), sym_eig

####################################################################################################

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from jax import jit, lax
    from functools import partial
    
    # @partial(jit, static_argnames=('funct', 'max_padding'))
    def process_matrix_elem_jax(funct: Callable, k, k_map, params, max_padding: int):
        """
        Process the matrix element. It is assumed that the function is returning the [rows], [cols], [vals].
        
        Args:
            funct (Callable): The function to apply.
            k (int)         : The index of the matrix element - by default it is assumed to be the row index.
            k_map (int)     : The mapped index of the matrix element - if the mapping is used.
            params          : The parameters for the matrix element - for example the lattice indices.
            max_padding (int): Maximum number of padding to apply.
        """
        # assume that the function is returning the [rows], [cols], [vals]
        all_results     = jax.vmap(lambda p: funct(k, k_map, p))(params)
        _, cols, vals   = all_results
        
        # Debug print to check shape and type
        # jax.debug.print("Shape of cols: {}", cols.shape)
        # jax.debug.print("Type of cols: {}", cols.dtype) 
        # jax.debug.print("Shape of vals: {}", vals.shape)
        # jax.debug.print("Type of vals: {}", vals.dtype) 
        
        # Flatten the results
        cols            = cols.reshape(-1)
        vals            = vals.reshape(-1)
        
        # Sort the columns and values
        sort_idx        = jnp.argsort(cols)
        cols_sorted     = cols[sort_idx]
        vals_sorted     = vals[sort_idx]
        
        # Debug print to check shape and type
        # jax.debug.print("Shape of cols_sorted: {}", cols_sorted.shape)
        # jax.debug.print("Type of cols_sorted: {}", cols_sorted.dtype)  

        # Find unique column indices and sum values.
        unique_cols, inv, counts    = jnp.unique(
            cols_sorted, return_inverse=True, return_counts=True,size=cols_sorted.shape[0]
        )
        summed_vals                 = jax.ops.segment_sum(vals_sorted, inv, num_segments=unique_cols.shape[0])

        # Padding as before.
        pad_width                   = max_padding - unique_cols.shape[0]
        # jax.debug.print("Padding width: {}", pad_width)
        # jax.debug.print("max_padding: {}", max_padding)
        
        # Calculate pad_width using JAX, keep it as JAX array
        unique_cols_padded = pad_array_jax(unique_cols, max_padding, -1)
        summed_vals_padded = pad_array_jax(summed_vals, max_padding, 0.0)

        return unique_cols_padded, summed_vals_padded, unique_cols.shape[0]

    @partial(jit, static_argnames=('funct', 'hilbert', 'max_padding', 'batch_start', 'batch_end'))
    def process_matrix_batch_jax(funct: Callable, batch_start, batch_end, hilbert : HilbertSpace, params, max_padding: int):
        '''
        Process a batch of matrix elements using JAX.
        
        Args:
            funct (Callable)    : The function to process each matrix element.
            batch_start (int)   : The starting index of the batch.
            batch_end (int)     : The ending index of the batch.
            hilbert (HilbertSpace): The Hilbert space object.
            params (Any)        : Additional parameters for processing.
            max_padding (int)   : Maximum number of padding to apply.
        '''
        ks      = jnp.arange(batch_start, batch_end, dtype=ACTIVE_INT_TYPE)
        k_maps  = jnp.array([hilbert.get_mapping(k) for k in ks], dtype=ACTIVE_INT_TYPE)
        # Vectorize process_matrix_elem_jax over the rows in the batch.
        cols_, vals_, counts_ = jax.vmap(
            lambda r, k_map: process_matrix_elem_jax(funct, r, k_map, params, max_padding)
        )(ks, k_maps)
        return cols_, vals_, counts_

def process_matrix_elem_np(funct : Callable, k, k_map, params):
    """
    Process the matrix element for a given set of parameters.
    This can be for example a set of real lattice indices 
    
    Args:
        k (int)     : The index of the matrix element - by default it is assumed to be the row index.
        k_map (int) : The mapped index of the matrix element - if the mapping is used.
        params      : The parameters for the matrix element - for example the lattice indices.
    """
    # assume that the function is returning the [rows], [cols], [vals]
    rows, cols, vals    = np.vectorize(lambda p: funct(k, k_map, p))(params)
    count               = len(rows)
    return rows, cols, vals, count

def process_matrix_batch_np(batch_start, batch_end, hilbert : HilbertSpace, funct : Callable, params):
    """
    Process the matrix batch. This function processes the matrix batch using the given function.
    This is assumed to be a start of index and end of index for the batch.
    
    Runs the process_matrix_elem_np function for each element in the batch.
    
    Args:
        batch_start (int)   : The starting index of the batch.
        batch_end (int)     : The ending index of the batch.
        funct (Callable)    : The function to process the matrix batch.
    """
    # create
    k_maps  = [hilbert.get_mapping(k) for k in range(batch_start, batch_end)]
    ks      = np.arange(batch_start, batch_end, dtype=np.int64)
    # Vectorize process_row over the rows in the batch.
    unique_cols_batch, summed_vals_batch, counts_batch  = np.vectorize(lambda k, k_map: process_matrix_elem_np(funct, k, k_map, params))(ks, k_maps)
    total_counts                                        = np.sum(counts_batch)
    return unique_cols_batch, summed_vals_batch, counts_batch, total_counts

#####################################################################################################