1) Files changed/added
- Implementation:
- `juqusolver/src/Algebra.jl`
- `juqusolver/src/Algebra/Utils.jl`
- `juqusolver/src/Algebra/BackendLinalg.jl`
- `juqusolver/src/Algebra/Preconditioners.jl`
- `juqusolver/src/Algebra/Solvers.jl`
- `juqusolver/src/Algebra/BackendOps.jl`
- `juqusolver/src/Algebra/RanWrapper.jl`
- `juqusolver/src/Algebra/RanMatrices.jl`
- `juqusolver/src/Algebra/Ode.jl`
- `juqusolver/src/Algebra/Eigen.jl`
- `juqusolver/src/Algebra/Utilities.jl`
- Integration:
- `juqusolver/src/QuantumEigenSolver.jl`
- `juqusolver/test/general_python_init_test.jl`
- Tests/docs/examples:
- `juqusolver/test/general_algebra_test.jl`
- `juqusolver/examples/general_algebra_example.jl`
- `juqusolver/docs/src/general_algebra.md`
- `README.md`

2) Test summary
- Module command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("test/general_algebra_test.jl")'`
- Result:
- `Algebra Umbrella`: 12/12 pass
- `Algebra Utils`: 9/9 pass
- `BackendLinalg Parity`: 11/11 pass
- `Preconditioners and Solvers`: 11/11 pass
- `BackendOps`: 4/4 pass
- `Random Wrapper and Matrices`: 7/7 pass
- `ODE Solvers`: 9/9 pass
- `Eigen API`: 17/17 pass
- `Pfaffian and Hafnian`: 5/5 pass
- Full regression command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("test/common_test.jl"); include("test/general_maths_test.jl"); include("test/general_lattices_test.jl"); include("test/general_python_init_test.jl"); include("test/general_physics_test.jl"); include("test/general_algebra_test.jl")'`
- Result:
- all testsets pass.
- Example command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("examples/common_example.jl"); include("examples/general_maths_example.jl"); include("examples/general_lattices_example.jl"); include("examples/general_python_init_example.jl"); include("examples/general_physics_example.jl"); include("examples/general_algebra_example.jl")'`
- Result:
- all examples run successfully; algebra example prints solver convergence, CUE unitarity error, eigenvalues, and RK4 step result.
- Package load command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'using QuantumEigenSolver; println(QuantumEigenSolver.list_available_modules())'`
- Result:
- package precompiles and loads; module list includes `algebra`.

3) Type stability summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Algebra.jl"); using .Algebra; using InteractiveUtils; A=[4.0 1.0;1.0 3.0]; b=[1.0,2.0]; @code_warntype Algebra.backend_linalg.inner(ComplexF64[1+im,2], ComplexF64[2-im,3]); @code_warntype Algebra.solvers.cg_solve(A, b; eps=1e-10, maxiter=100); @code_warntype Algebra.ran_matrices.CUE_QR(8; simple=false, rng=MersenneTwister(2));'`
- Result:
- `backend_linalg.inner` body return type: `ComplexF64`.
- `solvers.cg_solve` kwcall body return type: `SolverResult{Vector{Float64}}`.
- `ran_matrices.CUE_QR` kwcall body return type: `Matrix{ComplexF64}`.

4) Benchmark summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Algebra.jl"); using .Algebra; using BenchmarkTools; using Random; A=[4.0 1.0;1.0 3.0]; b=[1.0,2.0]; @btime Algebra.solvers.cg_solve($A, $b; eps=1e-10, maxiter=100); rr=MersenneTwister(2); @btime Algebra.ran_matrices.CUE_QR(8; simple=false, rng=$rr);'`
- Result:
- `cg_solve`: `160.375 ns (20 allocations: 832 bytes)`.
- `CUE_QR`: `3.724 μs (25 allocations: 11.45 KiB)`.
