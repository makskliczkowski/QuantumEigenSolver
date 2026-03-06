1) Files changed/added
- Spec:
- `porting/specs/qes_operator.md`
- Implementation:
- `juqusolver/src/Algebra/QESOperator.jl`
- `juqusolver/src/Algebra/Operator/PhaseUtils.jl`
- `juqusolver/src/Algebra/Operator/Sign.jl`
- `juqusolver/src/Algebra/Operator/Hardcore.jl`
- `juqusolver/src/Algebra/Operator/SpinlessFermions.jl`
- `juqusolver/src/Algebra/Operator/SpinHalf.jl`
- `juqusolver/src/Algebra/Operator/SpinOne.jl`
- `juqusolver/src/Algebra/Operator/Catalog.jl`
- `juqusolver/src/Algebra/Operator/Anyon.jl`
- `juqusolver/src/Algebra/Operator/Loader.jl`
- Tests:
- `juqusolver/test/qes_operator_test.jl`
- `juqusolver/test/runtests_all.jl`
- Example:
- `juqusolver/examples/qes_operator_example.jl`
- `juqusolver/examples/run_all_examples.jl`
- Docs:
- `juqusolver/docs/src/qes_operator.md`
- `juqusolver/docs/src/index.md`
- README:
- `README.md`

2) Test summary
- Module command:
- `~/.juliaup/bin/julia --project=juqusolver -e 'include("juqusolver/test/qes_operator_test.jl")'`
- Result:
- `QES Operator Surface`: 12/12 pass
- `QES Operator Phase And Sign Helpers`: 7/7 pass
- `QES Spin Half Invariants`: 8/8 pass
- `QES Spin One Invariants`: 9/9 pass
- `QES Fermion And Hardcore Invariants`: 13/13 pass
- `QES Operator Loader And Catalog`: 9/9 pass
- `QES Operator Sparse Matrix Build`: 6/6 pass
- Full regression command:
- `~/.juliaup/bin/julia --project=juqusolver -e 'include("juqusolver/test/runtests_all.jl")'`
- Result:
- full suite passes including common, maths, lattices, physics, algebra, qes_algebra, qes_operator, qes_models, qes_core, qes_solvers.
- Examples command:
- `~/.juliaup/bin/julia --project=juqusolver -e 'include("juqusolver/examples/run_all_examples.jl")'`
- Result:
- full examples bundle passes; operator example writes:
- `juqusolver/tmp/data_loader/qes_operator_sigz_nnz.txt`
- Package load command:
- `~/.juliaup/bin/julia --project=juqusolver -e 'include("juqusolver/src/QuantumEigenSolver.jl"); using .QuantumEigenSolver; println("ok")'`
- Result:
- package loads successfully.

3) Type stability summary
- Command:
- `~/.juliaup/bin/julia --project=juqusolver -e 'include("juqusolver/src/QuantumEigenSolver.jl"); using .QuantumEigenSolver; using InteractiveUtils; O = QuantumEigenSolver.Algebra.QESOperator; @code_warntype O.operators_spin.sigma_x_int_np(UInt64(13), 8, Int[0,2,3]); @code_warntype O.operators_spinless_fermions.c_dag_int_np(UInt64(13), 8, Int[0,2,3]; prefactor=1.0); @code_warntype O.operators_hardcore.hardcore_create_int(UInt64(13), 8, Int[0,2,3], pi/3);'`
- Result:
- `sigma_x_int_np` body return is concrete `Tuple{Vector{UInt64},Vector{ComplexF64}}`.
- `c_dag_int_np` keyword dispatch returns concrete `Tuple{Vector{UInt64},Vector{ComplexF64}}`.
- `hardcore_create_int` body return is concrete `Tuple{Vector{UInt64},Vector{ComplexF64}}`.

4) Benchmark summary
- Command:
- `~/.juliaup/bin/julia --project=juqusolver -e 'include("juqusolver/src/QuantumEigenSolver.jl"); using .QuantumEigenSolver; using BenchmarkTools; O = QuantumEigenSolver.Algebra.QESOperator; st = UInt64(0x1234); sites = Int[0,2,3]; @btime O.operators_spin.sigma_x_int_np($st, 16, $sites); @btime O.operators_spinless_fermions.c_dag_int_np($st, 16, $sites; prefactor=1.0); @btime O.operators_hardcore.hardcore_create_int($st, 16, $sites, $(pi/3));'`
- Result:
- `sigma_x_int_np`: `75.387 ns (6 allocations: 192 bytes)`
- `c_dag_int_np`: `100.169 ns (10 allocations: 352 bytes)`
- `hardcore_create_int`: `111.441 ns (11 allocations: 368 bytes)`
- Sparse check:
- `operator_matrix(...; sparse_out=true)` returns `SparseMatrixCSC` for spin-1/2 and spin-1 operators.
