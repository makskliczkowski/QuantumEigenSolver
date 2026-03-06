A) Public API inventory (Python)
- Package entry (`general_python/algebra/__init__.py`) lazy exports:
- utils/backend state: `backend_mgr`, `get_backend`, `get_global_backend`, `ACTIVE_BACKEND_NAME`, `ACTIVE_NP_MODULE`, `ACTIVE_RANDOM`, `ACTIVE_SCIPY_MODULE`, `ACTIVE_JIT`, `ACTIVE_JAX_KEY`, `ACTIVE_INT_TYPE`, `ACTIVE_FLOAT_TYPE`, `ACTIVE_COMPLEX_TYPE`.
- solver selection API: `SolverType`, `choose_solver`.
- backend ops API: `BackendOps`, `get_backend_ops`, `default_ops`.
- preconditioner factory API: `choose_precond`.
- submodule aliases: `LinalgModule`, `backend_linalg`, `solvers`, `preconditioners`, `ode`, `ran_wrapper`, `ran_matrices`, `eigen`, `utilities`, `utils`.
- `backend_linalg.py` public functions:
- `change_basis`, `change_basis_matrix`, `outer`, `kron`, `kron_sparse`, `inner`, `ket_bra`, `bra_ket`, `overlap`, `overlap_diagonal`, `trace`, `hilbert_schmidt_norm`, `frobenius_norm`, `identity`, `identity_sparse`, `to_dense`, `to_sparse`, `eig`, `eigh`, `eigsh`, `givens_rotation`.
- `solvers/__init__.py` public API:
- `SolverForm`, `choose_solver`; class exports expected by users through factory path.
- `solvers/backend_ops.py` public API:
- `BackendOps` class helpers, `get_backend_ops`.
- `preconditioners.py` public API:
- enums `PreconditionersType`, `PreconditionersTypeSym`, `PreconditionersTypeNoSym`.
- classes `Preconditioner`, `IdentityPreconditioner`, `JacobiPreconditioner`, `CholeskyPreconditioner`, `SSORPreconditioner`, `IncompleteCholeskyPreconditioner`, `ILUPreconditioner`.
- helper `preconditioner_idn`, factory `choose_precond`.
- `ran_wrapper.py` public API:
- RNG helpers: `initialize`, `set_global_seed`, `default_dtype`, `handle_rng`, `uniform`, `normal`, `exponential`, `poisson`, `gamma`, `beta`, `randint_np`, `randint`, `choice_np`, `choice`, `shuffle`, `shuffle_indices`.
- class `RandomRegistry`.
- sub-namespace `RMT` with `random_matrix`, `goe`, `gue`, `coe`, `cue`, `cre`, `random_vector`, `list_capabilities`, `get_info`.
- `ran_matrices.py` public API:
- `CUE_QR`, `goe`, `gue`, `coe`, `cre`, `cue`, class `RMT` with `random_matrix`, `list_capabilities`.
- `ode.py` public API:
- `IVP`, `Euler`, `Heun`, `AdaptiveHeun`, `RK`, `ScipyRK`, `OdeTypes`, `choose_ode`.
- `eigen` public API across files:
- solver/factory helpers: `choose_eigensolver`, `decide_method`, `estimate_hilbert_memory_gb`.
- exact helpers: `full_diagonalization`.
- parameters helpers: `get_lanczos_parameters`, `get_lanczos_memory_estimate_gb`.
- result classes: `EigenSolver`, `EigenResult`.
- solver class names: `ExactEigensolver`, `ExactEigensolverScipy`, `LanczosEigensolver`, `LanczosEigensolverScipy`, `ArnoldiEigensolver`, `ArnoldiEigensolverScipy`, `BlockLanczosEigensolver`, `BlockLanczosEigensolverScipy`.
- `utilities` public API:
- `PfaffianAlgorithms`, `Pfaffian`, `Hafnian`, `hafnian`.

B) Julia API mapping
- Target umbrella path:
- `juqusolver/src/Algebra.jl`.
- Submodule files under `juqusolver/src/Algebra/`:
- `Utils.jl`, `BackendLinalg.jl`, `Preconditioners.jl`, `Solvers.jl`, `BackendOps.jl`, `RanWrapper.jl`, `RanMatrices.jl`, `Ode.jl`, `Eigen.jl`, `Utilities.jl`.
- Package-level exports from `Algebra`:
- direct parity names: `backend_mgr`, `get_backend`, `get_global_backend`, `ACTIVE_*`, `SolverType`, `choose_solver`, `BackendOps`, `get_backend_ops`, `default_ops`, `choose_precond`.
- module aliases: `LinalgModule`, `backend_linalg`, `solvers`, `preconditioners`, `ode`, `ran_wrapper`, `ran_matrices`, `eigen`, `utilities`, `utils`.
- Type constraints:
- numerical kernels return concrete arrays/scalars (`Vector{Float64}`, `Matrix{ComplexF64}`, `Float64`) where deterministic.
- avoid `Any` in hot kernels (`inner`, `overlap`, iterative solver loops, random matrix generation).

C) Behavior spec
- Preserve user-facing API intent with deterministic Julia semantics.
- No Julia-side backend switching to alternate runtimes; one Julia implementation path.
- `choose_solver` and `choose_precond` accept string/enum/integer ids and return configured objects.
- linear algebra helpers preserve shape conventions and conjugation rules from Python naming.
- random wrappers use explicit seeding paths for reproducibility.
- eigen factory returns stable deterministic solver choice for fixed inputs.
- utility kernels (`pfaffian`, `hafnian`) preserve mathematical definitions for small-to-moderate matrix sizes and reject invalid shape constraints.

D) Test plan
- Umbrella tests:
- alias modules and exported symbols exist.
- backend state constants and `get_backend/get_global_backend` return consistent objects.
- backend linalg tests:
- `outer`, `inner`, `ket_bra`, `bra_ket`, `trace`, `frobenius_norm`, `eig/eigh` fixtures.
- solver tests:
- `choose_solver` returns configured direct/cg solvers.
- direct solve and iterative solve converge on SPD fixture with tolerance.
- preconditioner tests:
- identity and Jacobi apply paths; factory id parsing.
- random tests:
- deterministic draws under fixed seeds.
- GOE/GUE symmetry and dtype checks.
- ODE tests:
- Euler/Heun/RK one-step consistency on simple ODE.
- eigen tests:
- exact solver matches `eigen` reference ordering rule.
- factory decision deterministic.
- utilities tests:
- pfaffian on 2x2 and 4x4 skew-symmetric fixtures.
- hafnian on 2x2/4x4 symmetric fixtures.

E) Performance and typing gate
- Hot entrypoint 1:
- `Algebra.BackendLinalg.inner` type-stable scalar return (`Float64`/`ComplexF64`) without `Any`.
- Hot entrypoint 2:
- `Algebra.Solvers.cg_solve` type-stable `SolverResult` and allocation-controlled inner loop.
- Hot entrypoint 3:
- `Algebra.RanMatrices.CUE_QR` type-stable `Matrix{ComplexF64}`.
- Verification:
- `@code_warntype` for above entrypoints has concrete body returns.
- one `@btime` for `cg_solve` and one for `CUE_QR` with allocations reported.
