'''
file    : QES/NQS/tdvp.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
'''

import numpy as np
import numba

from typing import Callable, Optional, Union, NamedTuple
from general_python.algebra.utils import get_backend, JAX_AVAILABLE, Array

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
        self._e_local_var    = None
        self._solution       = None     # solution of the TDVP equation
        self._f0             = None     # force vector obtained from the covariance of loss and derivative
        self._s0             = None     # Fisher matrix obtained from the covariance of derivatives
        self._n_samples      = None     # number of samples
        self._full_size      = None     # full size of the covariance matrix
        self._x0             = sr_lin_x0
        
    ###################
    #! SETTERS
    ###################
    
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
                print('Using Gram solver')
                self.form_matrix = False
            elif self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value:
                print('Using matrix-vector solver')
                self.form_matrix = False
            else:
                print('Using matrix solver')
                self.form_matrix = True
            self.sr_solve_lin_fn = self.sr_solve_lin.get_solver_func(
                backend_module  = self.backend,
                use_matvec      = self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value,
                use_fisher      = self.sr_solve_lin_t == solvers.SolverForm.GRAM.value,
                use_matrix      = self.form_matrix,
                sigma           = self.sr_diag_shift
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
    
    def get_loss_centered(self, loss):
        '''
        Get the centered loss.
        
        Parameters
        ----------
        loss : Array
            The loss to be centered.
        
        Returns
        -------
        Array
            The centered loss.
        '''
        return sr.loss_centered_jax(loss, loss.mean(axis=0)) if self.is_jax else sr.loss_centered(loss, loss.mean(axis=0))
    
    def get_deriv_centered(self, deriv):
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
        return sr.derivatives_centered_jax(deriv, deriv.mean(axis=0)) if self.is_jax else sr.derivatives_centered(deriv, deriv.mean(axis=0))
    
    def get_tdvp_standard(self, 
                        loss, 
                        log_deriv,
                        betas               : Optional[Array] = None,
                        r_psi_low_ov_exc    : Optional[Array] = None,
                        r_psi_exc_ov_low    : Optional[Array] = None,
                        minsr               : Optional[bool]  = False):
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
        self._e_local_var   = self.backend.var(loss, axis=0)
        minsr               = self.use_minsr if minsr is None else minsr
        
        if self.is_jax:
            
            #! centered loss and derivative
            if betas is None:
                (loss_c, var_deriv_c, 
                var_deriv_c_h, self._n_samples,
                self._full_size) = sr.solve_jax_prepare(loss, log_deriv)
            else:
                (loss_c, var_deriv_c, 
                var_deriv_c_h, self._n_samples,
                self._full_size) = sr.solve_jax_prepare_modified_ratios(loss, log_deriv, betas, r_psi_low_ov_exc, r_psi_exc_ov_low)
            
            # for minsr, it is unnecessary to calculate the force vector, do it anyway for now
            self._f0 = sr.gradient_jax(var_deriv_c_h, loss_c, self._n_samples)
            
            #! no minsr
            if not minsr:
                if self.form_matrix:
                    self._s0 = sr.covariance_jax(var_deriv_c, var_deriv_c_h, self._n_samples)
                else:
                    self._s0 = None
            else:
                if self.form_matrix:
                    self._s0 = sr.covariance_jax_minsr(var_deriv_c, var_deriv_c_h, self._n_samples)
                else:
                    self._s0 = None
        else:
            #!TODO: add numpy version
            self._f0        = None
            self._s0        = None
            self._n_samples = None
            self._full_size = None
            
        return self._f0, self._s0, (loss_c, var_deriv_c, var_deriv_c_h, self._n_samples, self._full_size)

    ##################
    #! SOLVERS
    ##################
    
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
                                            r_psi_exc_ov_low    = r_psi_exc_ov_low,
                                            minsr               = self.use_minsr)
        f = self._f0
        s = self._s0
        
        # jax block until ready
        if hasattr(f, 'block_until_ready'):
            f.block_until_ready()
        if hasattr(self._s0, 'block_until_ready'):
            self._s0.block_until_ready()
            
        #! handle the solver
        solve_func                  = self.sr_solve_lin_fn
        loss_c, vd_c, vd_c_h, _, _  = tdvp
        
        #! handle the preprocessor
        if self.use_minsr:
            mat_s                   = vd_c_h
            mat_s_p                 = vd_c
            vec_b                   = loss_c
        else:
            mat_s                   = vd_c
            mat_s_p                 = vd_c_h
            vec_b                   = f
        
        #! handle the initial guess - use the previous solution
        use_old_result              = kwargs.get('use_old_result', False)
        if use_old_result:
            self._x0 = self._solution
        
        #! handle the initial guess
        if self._x0 is None or self._x0.shape != vec_b.shape:
            self._x0 = jnp.zeros_like(vec_b)
        
        #! prepare the rhs
        vec_b = vec_b * self.rhs_prefact
        
        #! precondition the S matrices?
        if True:
            #!TODO:
            pass
    
        #! solve the linear system
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
            solution = solve_func(a             =   s,
                                b               =   f,
                                x0              =   self._x0,
                                precond_apply   =   self.sr_precond_fn,
                                maxiter         =   self.sr_maxiter,
                                tol             =   self.sr_pinv_tol) 
        elif self.sr_solve_lin_t == solvers.SolverForm.MATRIX.value:
            
            def matvec(v):
                inter = jnp.matmul(mat_s, v)
                return jnp.matmul(mat_s_p, inter) / v.shape[0]
            
            if self.is_jax:
                matvec = jax.jit(matvec)
            
            solution = solve_func(matvec        =   matvec,
                                b               =   vec_b,
                                x0              =   self._x0,
                                precond_apply   =   self.sr_precond_fn,
                                maxiter         =   self.sr_maxiter,
                                tol             =   self.sr_pinv_tol)
            
        if self.use_minsr:
            solution = jnp.matmul(vd_c_h, solution)
        
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
            sr_iterations   = 0
        )
        
        if not self.use_sr:
            return loss, self.meta, (shapes, sizes, iscpx)
        
        #! obtain the solution
        try:
            solution: solvers.SolverResult  = self.solve(loss, log_deriv, **kwargs)
                        
            self.meta = TDVPStepInfo(
                mean_energy     = self._e_local_mean,
                std_energy      = self._e_local_var,
                failed          = False,
                sr_converged    = solution.converged,
                sr_executed     = True,
                sr_iterations   = solution.iterations
            )
            
            return solution.x, self.meta, (shapes, sizes, iscpx)

        except Exception as e:
            print(f"Error during TDVP solve: {e}")
            self.meta = TDVPStepInfo(
                mean_energy     = self._e_local_mean,
                std_energy      = self._e_local_var,
                failed          = True,
                sr_converged    = False,
                sr_executed     = False,
                sr_iterations   = 0
            )
        return None, self.meta, (shapes, sizes, iscpx)
        
    #########################
    #! Representation
    #########################
    
    def __repr__(self):
        return f'TDVP(backend={self.backend_str}, use_sr={self.use_sr}, use_minsr={self.use_minsr}, rhs_prefact={self.rhs_prefact}, sr_snr_tol={self.sr_snr_tol}, sr_pinv_tol={self.sr_pinv_tol}, sr_diag_shift={self.sr_diag_shift}, sr_maxiter={self.sr_maxiter})'
    
###############