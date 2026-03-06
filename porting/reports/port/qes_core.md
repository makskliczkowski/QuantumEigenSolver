1) Files changed/added
- Implementation:
- `juqusolver/src/Common/Globals.jl`
- `juqusolver/src/Common/Registry.jl`
- `juqusolver/src/Common/Session.jl`
- Integration:
- `juqusolver/src/Common.jl`
- `juqusolver/src/QuantumEigenSolver.jl`
- Tests/docs/examples:
- `juqusolver/test/common_test.jl`
- `juqusolver/test/qes_core_test.jl`
- `juqusolver/examples/qes_core_example.jl`
- `juqusolver/docs/src/qes_core.md`
- `README.md`
- Constraint cleanup requested by user:
- `juqusolver/src/Algebra.jl`
- `juqusolver/src/Algebra/Utils.jl`
- `juqusolver/src/Algebra/BackendOps.jl`
- `juqusolver/src/Algebra/Eigen.jl`
- `juqusolver/src/Physics/Spectral.jl`
- `juqusolver/test/general_algebra_test.jl`

2) Test summary
- Module command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("test/qes_core_test.jl")'`
- Result:
- `QES Globals API`: 7/7 pass
- `QES Seed Scope`: 5/5 pass
- `QES Registry API`: 5/5 pass
- `QES Session API`: 6/6 pass
- Full regression command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("test/common_test.jl"); include("test/general_maths_test.jl"); include("test/general_lattices_test.jl"); include("test/general_physics_test.jl"); include("test/general_algebra_test.jl"); include("test/general_python_init_test.jl"); include("test/qes_core_test.jl")'`
- Result:
- all testsets pass.
- Example command:
- `cd juqusolver && ~/.juliaup/bin/julia --project examples/qes_core_example.jl`
- Result:
- example runs successfully; session start/stop logs appear and registry/globals outputs are deterministic.
- Full examples command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("examples/common_example.jl"); include("examples/general_maths_example.jl"); include("examples/general_lattices_example.jl"); include("examples/general_python_init_example.jl"); include("examples/general_physics_example.jl"); include("examples/general_algebra_example.jl"); include("examples/qes_core_example.jl")'`
- Result:
- all examples run successfully and artifacts are saved under `juqusolver/tmp/<type>`.
- Package load command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'using QuantumEigenSolver; println(QuantumEigenSolver.list_available_modules()); println(length(QuantumEigenSolver.list_capabilities()))'`
- Result:
- package precompiles and loads; module list is `["algebra", "common", "lattices", "maths", "physics"]`.

3) Type stability summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/QuantumEigenSolver.jl"); using .QuantumEigenSolver; using InteractiveUtils; @code_warntype QuantumEigenSolver.qes_next_key(); @code_warntype QuantumEigenSolver.list_modules(); @code_warntype QuantumEigenSolver.run(backend="julia", seed=11, precision="float64");'`
- Result:
- `qes_next_key` body return type: `UInt64`.
- `list_modules` body return type: `Vector{Dict{String,String}}`.
- `run` keyword call body return type: `QESSession`.

4) Benchmark summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/QuantumEigenSolver.jl"); using .QuantumEigenSolver; using BenchmarkTools; QuantumEigenSolver.qes_reseed(11); @btime QuantumEigenSolver.qes_next_key(); @btime QuantumEigenSolver.list_modules();'`
- Result:
- `qes_next_key`: `3.000 ns (0 allocations: 0 bytes)`.
- `list_modules`: `1.554 us (50 allocations: 5.42 KiB)`.
