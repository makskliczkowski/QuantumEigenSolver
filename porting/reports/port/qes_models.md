1) Files changed/added
- Spec:
- `porting/specs/qes_models.md`
- Docs/example updates:
- `juqusolver/docs/src/qes_models.md`
- `juqusolver/docs/src/honeycomb_kitaev_diagonalization.md`
- `juqusolver/examples/qes_models_example.jl`
- `juqusolver/examples/honeycomb_kitaev_diagonalization_example.jl`
- `juqusolver/test/qes_models_test.jl`
- README note update:
- `README.md`
- Implementation note:
- `juqusolver/src/PhysicsModels/*.jl` already matched the expanded parity contract from the updated spec.
- Follow-up optimization/polish:
- `juqusolver/src/PhysicsModels/Common.jl`

2) Test summary
- Module command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("test/qes_models_test.jl")'`
- Result:
- `QES Models Surface`: 3/3 pass
- `QES Models Interacting Spin Dispatch and Invariants`: 15/15 pass
- `QES Models Lattice Integration`: 8/8 pass
- `QES Models Honeycomb Kitaev Diagonalization`: 5/5 pass
- `QES Models Interacting Random Families`: 11/11 pass
- `QES Models Interacting Fermionic`: 10/10 pass
- `QES Models Noninteracting`: 27/27 pass
- `QES Models Dispatch Errors and Validation`: 9/9 pass
- Full regression command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("test/common_test.jl"); include("test/general_maths_test.jl"); include("test/general_lattices_test.jl"); include("test/general_physics_test.jl"); include("test/general_algebra_test.jl"); include("test/qes_algebra_test.jl"); include("test/qes_models_test.jl"); include("test/general_python_init_test.jl"); include("test/qes_core_test.jl")'`
- Result:
- all testsets pass.
- Example command:
- `cd juqusolver && ~/.juliaup/bin/julia --project examples/qes_models_example.jl`
- Result:
- example runs successfully and prints dimensions, Hermiticity checks, deterministic seed checks, and many-body PLRB size.
- Additional example command:
- `cd juqusolver && ~/.juliaup/bin/julia --project examples/honeycomb_kitaev_diagonalization_example.jl`
- Result:
- exact diagonalization on periodic honeycomb lattice runs successfully and writes `juqusolver/tmp/data_loader/honeycomb_kitaev_spectrum.txt`.
- Full examples command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("examples/common_example.jl"); include("examples/general_maths_example.jl"); include("examples/general_lattices_example.jl"); include("examples/general_python_init_example.jl"); include("examples/general_physics_example.jl"); include("examples/general_algebra_example.jl"); include("examples/qes_algebra_example.jl"); include("examples/qes_models_example.jl"); include("examples/qes_core_example.jl")'`
- Result:
- all examples run successfully and artifacts are written to `juqusolver/tmp/<type>`.
- Package load command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'using QuantumEigenSolver; println(QuantumEigenSolver.list_available_modules()); println(isdefined(QuantumEigenSolver, :PhysicsModels)); println(QuantumEigenSolver.PhysicsModels.choose_model("dummy"; ns=2) isa QuantumEigenSolver.PhysicsModels.DummyHamiltonian)'`
- Result:
- package precompiles and loads; `PhysicsModels` is available and dummy factory path works.

3) Type stability summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/QuantumEigenSolver.jl"); using .QuantumEigenSolver; using InteractiveUtils; @code_warntype QuantumEigenSolver.PhysicsModels.choose_model("tfim"; ns=8, j=1.0, hx=0.5, hz=0.1); @code_warntype QuantumEigenSolver.PhysicsModels.intr.Spin.HeisenbergKitaev(ns=8, K=(0.3,0.2,0.1), J=0.7, Gamma=0.2, dlt=1.0, hx=0.1, hy=0.05, hz=0.0); @code_warntype QuantumEigenSolver.PhysicsModels.nintr.SYK2(ns=128, seed=13);'`
- Result:
- `PhysicsModels.choose_model("tfim"; ...)` returns typed union across model families and `DummyHamiltonian` with no `Any`.
- `HeisenbergKitaev(...)` returns concrete `HeisenbergKitaev{Nothing}`.
- `SYK2(...)` returns concrete `SYK2`.

4) Benchmark summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/QuantumEigenSolver.jl"); using .QuantumEigenSolver; using BenchmarkTools; @btime QuantumEigenSolver.PhysicsModels.choose_model("tfim"; ns=8, j=1.0, hx=0.5, hz=0.1); @btime QuantumEigenSolver.PhysicsModels.nintr.SYK2(ns=128, seed=13);'`
- Result:
- `choose_model("tfim"; ns=8, ...)`: `94.041 us (24 allocations: 2.00 MiB)`.
- `SYK2(ns=128, seed=13)`: `83.917 us (35 allocations: 660.62 KiB)`.
- `choose_model("aubry_andre"; ns=32, lx=8, ly=4, ...)`: `196.875 us (68 allocations: 67.14 KiB)` after forward-neighbor fallback allocation polish.
