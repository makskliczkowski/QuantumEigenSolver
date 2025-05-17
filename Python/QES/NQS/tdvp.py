'''
file    : QES/NQS/tdvp.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
'''

import numpy as np
import numba

from contextlib import contextmanager
from typing import Callable, Optional, Union, NamedTuple
from general_python.algebra.utils import get_backend, JAX_AVAILABLE, Array
from general_python.common.timer import timeit

import general_python.algebra.solvers.stochastic_rcnfg as sr
import general_python.algebra.solvers as solvers
import general_python.algebra.preconditioners as precond

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jax = None
    jnp = None

#################################################################

class TDVPStepInfo(NamedTuple):
    mean_energy     : Array
    std_energy      : Array
    failed          : Optional[bool] = None
    sr_converged    : Optional[bool] = None
    sr_executed     : Optional[bool] = None
    sr_iterations   : Optional[int]  = None
    times           : Optional[dict] = None


#################################################################

class TDVP:
    r'''
    This class implements the Time-Dependent Variational Principle (TDVP) for quantum systems.
    
    In principle, it is used to calculate the time evolution (or imaginary time evolution) of a quantum state.
    The TDVP is a variational method that uses the time-dependent Schrödinger equation to evolve a quantum state in time.
    
    It is used to be able to calculate:
    
    a) The force vector (can be treated already as a gradient in the first order)
        :math:`F_k=\\langle \\mathcal O_{\\theta_k}^* E_{loc}^{\\theta}\\rangle_c`
        
        i) :math:`\\mathcal O_{\\theta_k}^*` is logarithmic derivative of the wavefunction
        ii) :math:`E_{loc}^{\\theta}` is the local energy of the wavefunction
        iii) :math:`\\langle \\cdots \\rangle_c` is the connected correlation function
        iv*) The force vector can be appended with penalty terms:
            :math:`F_k=\\langle \\mathcal O_{\\theta_k}^* E_{loc}^{\\theta}\\rangle_c + \\sum_j \\beta_j
            \\langle (\\frac{\\psi _{W_j}}{\\psi _{W}}) - \\langle \\frac{\\psi _{W_j}}{\\psi _{W}} \\rangle ) O_k^\dag \\rangle_c`
            \\langle \\frac{\\psi _{W}}{\\psi _{W_j}} \\rangle`
    b) The quantum Fisher matrix 
        :math:`S_{k,k'} = \\langle (\\mathcal O_{\\theta_k})^* \\mathcal O_{\\theta_{k'}}\\rangle_c`
    
    and for real parameters :math:`\\theta\\in\\mathbb R`, the TDVP equation reads
        :math:`q\\big[S_{k,k'}\\big]\\dot\\theta_{k'} = -q\\big[xF_k\\big]`
        
    Here, either :math:`q=\\text{Re}` or :math:`q=\\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.
    
    For ground state search a regularization controlled by a parameter :math:`\\rho` can be included
    by increasing the diagonal entries and solving

        :math:`q\\big[(1+\\rho\\delta_{k,k'})S_{k,k'}\\big]\\theta_{k'} = -q\\big[F_k\\big]`

    The `TDVP` class solves the TDVP equation by computing a pseudo-inverse of :math:`S` via
    eigendecomposition yielding

        :math:`S = V\\Sigma V^\\dagger`

    with a diagonal matrix :math:`\\Sigma_{kk}=\\sigma_k`
    Assuming that :math:`\\sigma_1` is the smallest eigenvalue, the pseudo-inverse is constructed 
    from the regularized inverted eigenvalues

        :math:`\\tilde\\sigma_k^{-1}=\\frac{1}{\\Big(1+\\big(\\frac{\\epsilon_{SVD}}{\\sigma_j/\\sigma_1}\\big)^6\\Big)\\Big(1+\\big(\\frac{\\epsilon_{SNR}}{\\text{SNR}(\\rho_k)}\\big)^6\\Big)}`

    with :math:`\\text{SNR}(\\rho_k)` the signal-to-noise ratio of :math:`\\rho_k=V_{k,k'}^{\\dagger}F_{k'}` (see `[arXiv:1912.08828] <https://arxiv.org/pdf/1912.08828.pdf>`_ for details).
    
    '''
    
    def __init__(
            self,
            use_sr          : bool                              = True,
            use_minsr       : bool                              = False,
            rhs_prefact     : Union[float, complex]             = 1.0,
            # Stochastic reconfiguration parameters
            sr_lin_solver   : Optional[solvers.Solver]          = None,
            sr_precond      : Optional[precond.Preconditioner]  = None,
            sr_snr_tol      : float                             = 1e-3,
            sr_pinv_tol     : float                             = 1e-14,
            sr_pinv_cutoff  : float                             = 1e-8,
            sr_diag_shift   : float                             = 0.0,
            sr_lin_solver_t : Optional[solvers.SolverForm]      = solvers.SolverForm.GRAM,
            sr_lin_x0       : Optional[Array]                   = None,
            sr_maxiter      : int                               = 100,
            # Backend
            backend         : str                               = 'default',
        ):
        
        self.backend         = get_backend(backend)
        self.is_jax          = not (backend == 'numpy' or backend == 'np' or backend == np)
        self.is_np           = not self.is_jax
        self.backend_str     = 'jax' if self.is_jax else 'numpy'
            
        self.use_sr          = use_sr
        self.use_minsr       = use_minsr
        self.rhs_prefact     = rhs_prefact
        self.form_matrix     = False    # flag to indicate if the full matrix is formed
        
        #! handle the stochastic reconfiguration parameters
        self.sr_snr_tol      = sr_snr_tol
        self.sr_pinv_tol     = sr_pinv_tol
        self.sr_pinv_cutoff  = sr_pinv_cutoff
        self.sr_diag_shift   = sr_diag_shift
        self.sr_maxiter      = sr_maxiter
        
        #! handle the solver
        self.sr_solve_lin    = None
        self.sr_solve_lin_t  = sr_lin_solver_t
        self.sr_solve_lin_fn = None
        self.set_solver(sr_lin_solver)
        
        #! handle the preconditioner
        self.sr_precond      = None
        self.sr_precond_fn   = None
        self.set_preconditioner(sr_precond)

        self.meta            = None
                
        # Helper storage
        self._e_local_mean   = None
        self._e_local_std    = None
        self._solution       = None     # solution of the TDVP equation
        self._f0             = None     # force vector obtained from the covariance of loss and derivative
        self._s0             = None     # Fisher matrix obtained from the covariance of derivatives
        self._n_samples      = None     # number of samples
        self._full_size      = None     # full size of the covariance matrix
        self._x0             = sr_lin_x0
        
        self.timings         = {
            'prepare'   : 0.0,
            'gradient'  : 0.0,
            'covariance': 0.0,
            'x0'        : 0.0,
            'solve'     : 0.0,
        }
        
        #! functions
        self._init_functions()

    ###################
    #! TIMING
    ###################

    @contextmanager
    def _time(self, phase: str, fn, *args, **kwargs):
        """
        Context manager to time a function call and store elapsed time.

        Yields:
            result of fn(*args, **kwargs)
        """
        result, elapsed = timeit(fn, *args, **kwargs)
        self.timings[phase] = elapsed
        yield result

    ###################
    #! SETTERS
    ###################
    
    def _init_functions(self):
        """
        Initializes and assigns function handles for gradient, loss, derivatives, covariance, and preparation routines
        based on the current backend (JAX or NumPy) and configuration flags.
        - Selects appropriate functions from the `sr` module depending on whether JAX or NumPy is used (`self.is_jax`).
        - Chooses between standard and minimal SR covariance functions based on `self.use_minsr`.
        - Assigns both standard and modified preparation functions.
        - If using JAX, applies `jax.jit` to the selected functions for just-in-time compilation and stores them with `_j` suffix.
        - If not using JAX, stores the original function references with `_j` suffix for interface consistency.
        This method should be called during initialization to ensure all computational routines are set up according to the
        current backend and configuration.
        """
        
        self._gradient_fn    = sr.gradient_jax if self.is_jax else sr.gradient_np
        self._loss_c_fn      = sr.loss_centered_jax if self.is_jax else sr.loss_centered
        self._deriv_c_fn     = sr.derivatives_centered_jax if self.is_jax else sr.derivatives_centered
        
        if self.use_minsr:
            self._covariance_fn  = sr.covariance_jax_minsr if self.is_jax else sr.covariance_np_minsr
        else:
            self._covariance_fn  = sr.covariance_jax if self.is_jax else sr.covariance_np
            
        # modified and standard preparation functions
        self._prepare_fn     = sr.solve_jax_prepare if self.is_jax else sr.solve_numpy_prepare
        self._prepare_fn_m   = sr.solve_jax_prepare_modified_ratios if self.is_jax else sr.solve_numpy_prepare_modified_ratios
            
        #! store the jitted functions
        if self.is_jax:
            self._gradient_fn_j     = jax.jit(self._gradient_fn)
            self._loss_c_fn_j       = jax.jit(self._loss_c_fn)
            self._deriv_c_fn_j      = jax.jit(self._deriv_c_fn)
            self._covariance_fn_j   = jax.jit(self._covariance_fn)
            self._prepare_fn_j      = jax.jit(self._prepare_fn)
            self._prepare_fn_m_j    = jax.jit(self._prepare_fn_m)
        else:
            self._gradient_fn_j     = self._gradient_fn
            self._loss_c_fn_j       = self._loss_c_fn
            self._deriv_c_fn_j      = self._deriv_c_fn
            self._covariance_fn_j   = self._covariance_fn
            self._prepare_fn_j      = self._prepare_fn
            self._prepare_fn_m_j    = self._prepare_fn_m
    
    def set_solver(self, solver: Callable):
        '''
        Set the solver for the TDVP equation.
        
        Parameters
        ----------
        solver : Callable
            The solver function to be used for the TDVP equation.
        '''
        self.sr_solve_lin = solver
        self._init_solver_lin()
    
    def _init_solver_lin(self):
        """
        Initializes the linear solver for the stochastic reconfiguration (SR) method.
        This method sets up the appropriate solver function (`sr_solve_lin_fn`) based on the type of solver specified
        by `sr_solve_lin_t` and the solver instance `sr_solve_lin`. It determines whether to form the full matrix or use
        matrix-vector products, and configures the solver function with the correct backend and options.
        Raises:
            ValueError: If SR is enabled (`use_sr` is True) but no solver (`sr_solve_lin`) is set.
        Attributes Set:
            self.form_matrix (bool):
                Indicates whether to form the full matrix or use matrix-vector products.
            self.sr_solve_lin_fn (callable):
                The configured solver function for SR linear systems.
        """
        
        if self.sr_solve_lin is None:
            self.sr_solve_lin = solvers.PseudoInverseSolver(backend=self.backend, sigma=self.sr_pinv_cutoff)
        
        if isinstance(self.sr_solve_lin, str) or isinstance(self.sr_solve_lin, solvers.SolverType):
            self.sr_solve_lin = solvers.choose_solver(solver_id=self.sr_solve_lin, sigma=self.sr_pinv_cutoff)
        
        if self.sr_solve_lin is not None:
            if self.sr_solve_lin_t == solvers.SolverForm.GRAM.value:
                self.form_matrix    = False
            elif self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value:
                self.form_matrix    = False
            else:
                self.form_matrix    = True
                
            #! set the solver function
            self.sr_solve_lin_fn = self.sr_solve_lin.get_solver_func(
                    backend_module  = self.backend,
                    use_matvec      = self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value,
                    use_fisher      = self.sr_solve_lin_t == solvers.SolverForm.GRAM.value,
                    use_matrix      = self.form_matrix,
                    sigma           = self.sr_diag_shift,
                )
            
        elif self.use_sr and self.sr_solve_lin is None:
            raise ValueError('The solver is not set. Please set the solver or use the default one.') 
    
    def set_preconditioner(self, precond: Callable):
        '''
        Set the preconditioner for the TDVP equation.
        
        Parameters
        ----------
        precond : Callable
            The preconditioner function to be used for the TDVP equation.
        '''
        self.sr_precond = precond
        self._init_preconditioner()
    
    def _init_preconditioner(self):
        """
        Initializes the preconditioner function for the stochastic reconfiguration (SR) solver
        based on the selected solver form.

        Depending on the value of `self.sr_solve_lin_t`, this method assigns the appropriate
        preconditioner application function from `self.sr_precond` to `self.sr_precond_fn`:
            - If `GRAM`, uses `get_apply_gram()`
            - If `MATVEC`, uses `get_apply()`
            - If `MATRIX`, uses `get_apply_mat()`
        Raises a ValueError if the solver form is unrecognized or if the preconditioner is not set.

        Raises:
            ValueError: If the preconditioner is not set or the solver form is invalid.
        """
        if      isinstance(self.sr_precond, str) or                                 \
                isinstance(self.sr_precond, precond.PreconditionersTypeSym) or      \
                isinstance(self.sr_precond, precond.PreconditionersTypeNoSym) or    \
                isinstance(self.sr_precond, precond.PreconditionersType) or         \
                hasattr(self.sr_precond, 'value') and isinstance(self.sr_precond.value, int):
            print('Using preconditioner: ', self.sr_precond)
            self.sr_precond = precond.choose_precond(precond_id=self.sr_precond, backend=self.backend)
        
        if self.sr_precond is not None:
            if self.sr_solve_lin_t == solvers.SolverForm.GRAM.value:
                self.sr_precond_fn = self.sr_precond.get_apply_gram()
            elif self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value:
                self.sr_precond_fn = self.sr_precond.get_apply()
            elif self.sr_solve_lin_t == solvers.SolverForm.MATRIX.value:
                self.sr_precond_fn = self.sr_precond.get_apply_mat()
            else:
                raise ValueError('The preconditioner is not set. Please set the preconditioner or use the default one.')  
    
    def set_useminsr(self, use_minsr: bool):
        '''
        Set the use of minimum stochastic reconfiguration (minsr) method.
        
        Parameters
        ----------
        use_minsr : bool
            Whether to use the minimum stochastic reconfiguration (minsr) method.
        '''
        self.use_minsr = use_minsr
        if self.use_minsr:
            self._covariance_fn = sr.covariance_jax_minsr if self.is_jax else sr.covariance_np_minsr
        else:
            self._covariance_fn = sr.covariance_jax if self.is_jax else sr.covariance_np

    def set_diag_shift(self, diag_shift: float):
        '''
        Set the diagonal shift for the TDVP equation.
        
        Parameters
        ----------
        diag_shift : float
            The diagonal shift to be used for the TDVP equation.
        '''
        self.sr_diag_shift = diag_shift
        self._init_solver_lin()

    ###################
    
    def set_rhs_prefact(self, rhs_prefact: Union[float, complex]):
        '''
        Set the right-hand side prefactor for the TDVP equation.
        This is used to scale the right-hand side of the TDVP equation.
        Parameters
        ----------
        rhs_prefact : Union[float, complex]
            The right-hand side prefactor to be used for the TDVP equation.
        '''
        self.rhs_prefact = rhs_prefact
    
    def set_snr_tol(self, snr_tol: float):
        '''
        Set the signal-to-noise ratio tolerance for the TDVP equation.
        
        Parameters
        ----------
        snr_tol : float
            The signal-to-noise ratio tolerance to be used for the TDVP equation.
        '''
        self.sr_snr_tol = snr_tol
    
    def set_pinv_tol(self, pinv_tol: float):
        '''
        Set the pseudo-inverse tolerance for the TDVP equation.
        This is used to determine the cutoff for the pseudo-inverse calculation.
        Parameters
        ----------
        pinv_tol : float
            The pseudo-inverse tolerance to be used for the TDVP equation.
        '''
        self.sr_pinv_tol = pinv_tol
    
    ###################
    #! GETTERS
    ###################
    
    def get_loss_centered(self, loss, loss_m = None):
        '''
        Get the centered loss.
        
        Parameters
        ----------
        loss : Array
            The loss to be centered.
        loss_m : Optional[Array]
            The mean loss to be used for centering.
        
        Returns
        -------
        Array
            The centered loss.
        '''
        return self._loss_c_fn_j(loss, loss.mean(axis = 0) if not loss_m else loss_m)
    
    def get_deriv_centered(self, deriv, deriv_m = None):
        '''
        Get the centered derivative.
        
        Parameters
        ----------
        deriv : Array
            The derivative to be centered.
        
        Returns
        -------
        Array
            The centered derivative.
        '''
        deriv_m = deriv.mean(axis = 0) if not deriv_m else deriv_m
        return self._deriv_c_fn_j(deriv, deriv_m)
    
    ##################
    #! TDVP
    ##################
    
    def _get_tdvp_standard_inner(self, loss, log_deriv, betas = None, r_psi_low_ov_exc = None, r_psi_exc_ov_low = None):
        '''
        Get the standard TDVP loss and derivative.
        
        Parameters
        ----------
        loss : Array
            The loss to be used for the TDVP equation.
        log_deriv : Array
            The logarithm of the derivative to be used for the TDVP equation.
        betas : Optional[Array]
            The betas to be used for the excited states.
        r_psi_low_ov_exc : Optional[Array]
            The ratios of the wavefunction for the low-lying excited states.
        r_psi_exc_ov_low : Optional[Array]
            The ratios of the wavefunction for the excited states.
        Returns
        -------
        Tuple[Array, Array]
            The standard TDVP loss and derivative.
        '''
        
        #! centered loss and derivative
        if betas is None:
            (loss_c, var_deriv_c, var_deriv_c_h, self._n_samples, self._full_size) = self._prepare_fn_j(loss, log_deriv)
        else:
            (loss_c, var_deriv_c, var_deriv_c_h, self._n_samples, self._full_size) = self._prepare_fn_m_j(loss, log_deriv, 
                                                                                            betas, r_psi_low_ov_exc, r_psi_exc_ov_low)        
        return loss_c, var_deriv_c, var_deriv_c_h
    
    def get_tdvp_standard(self, 
                        loss, 
                        log_deriv,
                        betas               : Optional[Array] = None,
                        r_psi_low_ov_exc    : Optional[Array] = None,
                        r_psi_exc_ov_low    : Optional[Array] = None):
        '''
        Get the standard TDVP loss and derivative.
        
        Parameters
        ----------
        loss : Array
            The loss to be used for the TDVP equation.
        log_deriv : Array
            The logarithm of the derivative to be used for the TDVP equation.
        minsr : Optional[bool]
            Whether to use the minimum stochastic reconfiguration (minsr) method.
        Returns
        -------
        Tuple[Array, Array]
            The standard TDVP loss and derivative.
        '''
        #! state information
        self._e_local_mean  = self.backend.mean(loss, axis=0)
        self._e_local_std   = self.backend.std(loss, axis=0)

        with self._time('prepare', self._get_tdvp_standard_inner, loss, log_deriv, betas, r_psi_low_ov_exc, r_psi_exc_ov_low) as prepare:
            loss_c, var_deriv_c, var_deriv_c_h = prepare
            
        # for minsr, it is unnecessary to calculate the force vector, however, do it anyway for now
        with self._time('gradient', self._gradient_fn_j, var_deriv_c_h, loss_c, self._n_samples) as gradient:
            self._f0 = gradient
        self._s0 = None
        if self.form_matrix:
            # the function knows if it's minsr or not
            with self._time('covariance', self._covariance_fn_j, var_deriv_c, var_deriv_c_h, self._n_samples) as covariance:
                self._s0 = covariance
        return self._f0, self._s0, (loss_c, var_deriv_c, var_deriv_c_h)

    ##################
    #! SOLVERS
    ##################
    
    def _solve_prepare_matvec(self, mat_s: Array, mat_sp: Array):
        """
        Prepares the covariance matrix and loss vector for the linear system to be solved.
        This method is used to prepare the input data for the linear solver, including
        centering the covariance matrix and loss vector.
        Parameters
        ----------
        mat_s : Array
            The covariance matrix (centered). [N_samples, N_variational]
        mat_sp : Array
            The covariance matrix (centered - complex conjugate). [N_variational, N_samples]
        Returns
        -------
        Tuple[Array, Array]
            The prepared covariance matrix and loss vector.
        """
        def _matvec(v, sigma):
            inter = self.backend.matmul(mat_s, v)
            return self.backend.matmul(mat_sp, inter) / v.shape[0] + sigma * v
        return jax.jit(_matvec) if self.is_jax else _matvec
    
    def _solve_prepare_s_and_loss(self, vd_c: Array, vd_c_h: Array, loss_c: Array, forces: Array):
        """
        Prepares the covariance matrix and loss vector for the linear system to be solved.
        This method is used to prepare the input data for the linear solver, including
        centering the covariance matrix and loss vector.
        Parameters
        ----------
        vd_c : Array
            The covariance matrix (centered). [N_samples, N_variational]
        vd_c_h : Array
            The covariance matrix (centered - complex conjugate). [N_variational, N_samples]
        loss_c : Array
            The loss vector (centered). [N_samples,]
        forces : Array
            The forces vector (centered). [N_variational,]
        Returns
        -------
        Tuple[Array, Array]
            The prepared covariance matrix and loss vector.
        """
        if self.use_minsr:
            # when using the minimum stochastic reconfiguration (minsr) method,
            # the equation transforms to first calculating the matrix 
            # :math:`T=\bar O \bar O^\\dagger` and then solving
            # :math:`d\\theta = \\bar O^\\dagger T^{-1} \\bar E_{loc}`
            # where :math:`\\bar O` is the centered covariance matrix and :math:`\\bar E_{loc}` is the centered loss vector
            # therefore, one reduces the computation, usefully in the limit where N_param >> N_samples
            # afterall, the solution needs to be multiplied by the covariance matrix (conjugate transpose)
            # :math:`\\tilde {d\\theta} = T^{-1} \\bar E_{loc} \\rightarrow d\\theta = O^\\dagger \\tilde {d\\theta}`
            return (vd_c_h, vd_c), loss_c
        # otherwise, standard TDVP equation is solved
        return (vd_c, vd_c_h), forces

    def _solve_choice(  self, 
                        vec_b           : Array,
                        solve_func      : Callable,
                        mat_s           : Optional[Array] = None,
                        mat_s_p         : Optional[Array] = None,
                        mat_a           : Optional[Array] = None,
                    ):
        """
        Solves a linear system using a specified solver form and configuration.
        Depending on the value of `self.sr_solve_lin_t`, this method dispatches to the appropriate
        solver function with the required arguments. Supports different solver forms such as GRAM and MATRIX,
        and handles special cases like the minimum stochastic reconfiguration (minsr) method.
        Parameters
        ----------
        mat_s : Optional[Array]
            The first matrix to be used in the linear system. Required for GRAM and MATVEC solver forms.
        mat_s_p : Optional[Array]
            The second matrix to be used in the linear system. Required for GRAM and MATVEC solver forms.
        mat_a : Optional[Array]
            The matrix to be used in the linear system. Required for MATRIX solver form.
        vec_b : Array
            The right-hand side vector of the linear system.
        solve_func : Callable
            The function to be used for solving the linear system.
        Returns
        -------
        solution : Array or None
            The solution to the linear system, or None if the solver form is not recognized.
        Raises
        ------
        ValueError
            If the matrix solver is selected with the minimum stochastic reconfiguration (minsr) method,
            which is not implemented.
        """
        
        solution = None
        if self.sr_solve_lin_t == solvers.SolverForm.GRAM.value:
            solution = solve_func(s             =   mat_s,
                                s_p             =   mat_s_p,
                                b               =   vec_b,
                                x0              =   self._x0,
                                precond_apply   =   self.sr_precond_fn,
                                maxiter         =   self.sr_maxiter,
                                tol             =   self.sr_pinv_tol)
        elif self.sr_solve_lin_t == solvers.SolverForm.MATRIX.value:
            if self.use_minsr:
                raise ValueError('The matrix solver is not implemented for the minimum stochastic reconfiguration (minsr) method.')
            solution = solve_func(a             =   mat_a,
                                b               =   vec_b,
                                x0              =   self._x0,
                                precond_apply   =   self.sr_precond_fn,
                                maxiter         =   self.sr_maxiter,
                                tol             =   self.sr_pinv_tol) 
        elif self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value:
            solution = solve_func(matvec        =   self._solve_prepare_matvec(mat_s, mat_s_p),
                                b               =   vec_b,
                                x0              =   self._x0,
                                precond_apply   =   self.sr_precond_fn,
                                maxiter         =   self.sr_maxiter,
                                tol             =   self.sr_pinv_tol)
        return solution
    
    def _solve_handle_x0(self, vec_b: Array, use_old_result: bool):
        """
        Handles the initial guess for the linear solver.
        If `use_old_result` is True, it uses the previous solution as the initial guess.
        Otherwise, it initializes the guess to zero.
        Parameters
        ----------
        vec_b : Array
            The right-hand side vector of the linear system.
        use_old_result : bool
            Whether to use the previous solution as the initial guess.
        Returns
        -------
        Array
            The initial guess for the linear solver.
        """
        if use_old_result:
            self._x0 = self._solution
        if self._x0 is None or self._x0.shape != vec_b.shape:
            self._x0 = self.backend.zeros_like(vec_b)
        return self._x0
    
    def solve(self, 
            e_loc                   : Array,
            log_deriv               : Array,
            # for the excited states
            betas                   : Optional[Array] = None,
            r_psi_low_ov_exc        : Optional[Array] = None,
            r_psi_exc_ov_low        : Optional[Array] = None,
            **kwargs
        ):
        
        #! obtain the loss and covariance without the preprocessor
        self._f0, self._s0, (tdvp)  = self.get_tdvp_standard(e_loc,
                                        log_deriv, 
                                        betas               = betas,
                                        r_psi_low_ov_exc    = r_psi_low_ov_exc,
                                        r_psi_exc_ov_low    = r_psi_exc_ov_low)
        
        f = self._f0                # the force vector
        s = self._s0                # the covariance matrix, if formed
        
        #! handle the solver
        solve_func                  = self.sr_solve_lin_fn
        loss_c, vd_c, vd_c_h        = tdvp

        #! handle the preprocessor
        (mat_s, mat_s_p), vec_b     = self._solve_prepare_s_and_loss(vd_c, vd_c_h, loss_c, f)
        
        #! handle the initial guess - use the previous solution
        with self._time('x0', self._solve_handle_x0, vec_b, kwargs.get('use_old_result', False)) as x0:
            self._x0 = x0

        #! prepare the rhs
        vec_b                       = vec_b * self.rhs_prefact
        
        #! precondition the S matrices?
        if True:
            #!TODO:
            pass
    
        #! solve the linear system
        with self._time('solve', self._solve_choice, vec_b=vec_b, mat_s=mat_s, mat_s_p=mat_s_p, mat_a=s, solve_func=solve_func) as solve:
            solution = solve
        
        if self.use_minsr and solution is not None:
            solution = self.backend.matmul(vd_c_h, solution)
        
        #! save the solution
        self._solution = solution
        return solution
    
    def __call__(self, net_params, t, *, est_fn, configs, configs_ansatze, probabilities, **kwargs):
        '''
        Call the TDVP class to compute the time evolution of the quantum state.
        
        Parameters
        ----------
        net_params : Array
            The network parameters to be used for the TDVP equation.
        t : float
            The time to be used for the TDVP equation.
        est_fn : Callable
            The function to be used for estimating the TDVP equation.
        configs : Array
            The configurations to be used for the TDVP equation.
        configs_ansatze : Array
            The ansatz configurations to be used for the TDVP equation.
        probabilities : Array
            The probabilities to be used for the TDVP equation.
        
        Returns
        -------
        Array
            The time-evolved quantum state.
        '''
        #! get the loss and derivative
        (loss, mean_loss, std_loss), log_deriv, (shapes, sizes, iscpx) = est_fn(net_params, t, configs, configs_ansatze, probabilities, **kwargs)
        
        #! set the meta information
        self.meta       = TDVPStepInfo(
            mean_energy     = mean_loss,
            std_energy      = std_loss,
            failed          = False,
            sr_converged    = False,
            sr_executed     = False,
            sr_iterations   = 0,
            times           = self.timings
        )
        
        if not self.use_sr:
            return loss, self.meta, (shapes, sizes, iscpx)
        
        #! obtain the solution
        try:
            solution: solvers.SolverResult  = self.solve(loss, log_deriv, **kwargs)
                        
            self.meta = TDVPStepInfo(
                mean_energy     = self._e_local_mean,
                std_energy      = self._e_local_std,
                failed          = False,
                sr_converged    = solution.converged,
                sr_executed     = True,
                sr_iterations   = solution.iterations,
                times           = self.timings
            )
            
            return solution.x, self.meta, (shapes, sizes, iscpx)

        except Exception as e:
            print(f"Error during TDVP solve: {e}")
            self.meta = TDVPStepInfo(
                mean_energy     = self._e_local_mean,
                std_energy      = self._e_local_std,
                failed          = True,
                sr_converged    = False,
                sr_executed     = False,
                sr_iterations   = 0,
                times           = self.timings
            )
        return None, self.meta, (shapes, sizes, iscpx)
    
    #########################
    #! Representation
    #########################
    
    def __repr__(self):
        return f'TDVP(backend={self.backend_str},use_sr={self.use_sr},use_minsr={self.use_minsr},rhs_prefact={self.rhs_prefact},sr_snr_tol={self.sr_snr_tol},sr_pinv_tol={self.sr_pinv_tol},sr_diag_shift={self.sr_diag_shift},sr_maxiter={self.sr_maxiter})'
    
    def __str__(self):
        return f'TDVP(backend={self.backend_str},use_sr={self.use_sr},use_minsr={self.use_minsr},rhs_prefact={self.rhs_prefact},sr_snr_tol={self.sr_snr_tol},sr_pinv_tol={self.sr_pinv_tol},sr_diag_shift={self.sr_diag_shift},sr_maxiter={self.sr_maxiter})'

    def __len__(self):
        return len(self._solution) if self._solution is not None else 0
    
    def __getitem__(self, key):
        if self._solution is not None:
            return self._solution[key]
        else:
            raise ValueError('The solution is not available. Please run the TDVP solver first.')
    
    #########################
    #! PROPERTIES
    #########################
    
    @property
    def solution(self):
        return self._solution
    
    @property
    def forces(self):
        return self._f0
    
    @property
    def covariance(self):
        return self._s0
    
    @property
    def loss_mean(self):
        return self._e_local_mean
    
    @property
    def loss_std(self):
        return self._e_local_std
    
    @property
    def n_samples(self):
        return self._n_samples
    
    @property
    def full_size(self):
        return self._full_size

###############
#! END OF FILE
###############