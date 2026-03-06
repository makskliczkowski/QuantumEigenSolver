1) Files changed/added
- Implementation note:
- `juqusolver/src/Physics*.jl` already satisfies the current `general_physics` parity spec in this pass.
- No source changes were required after parity re-audit.
- Fresh verification artifacts updated:
- `porting/reports/port/general_physics.md`
- `porting/reports/review/general_physics.md`

2) Test summary
- Module command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("test/general_physics_test.jl")'`
- Result:
- `Physics Umbrella`: 9/9 pass
- `Physics Operators Parsing`: 5/5 pass
- `Physics Entropy Kernels`: 6/6 pass
- `Physics Density Matrix`: 8/8 pass
- `Physics Thermal and Statistical`: 9/9 pass
- `Physics Eigenlevels`: 13/13 pass
- `Physics Spectral`: 9/9 pass
- `Physics Response`: 10/10 pass
- `Physics Single Particle Correlators`: 6/6 pass
- Example command:
- `cd juqusolver && ~/.juliaup/bin/julia --project examples/general_physics_example.jl`
- Result:
- example runs successfully and reports deterministic finite outputs for density-matrix, entropy, thermal, spectral, response, and single-particle paths.
- Package load command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'using QuantumEigenSolver; println(QuantumEigenSolver.list_available_modules()); println(isdefined(QuantumEigenSolver, :Physics)); println(isdefined(QuantumEigenSolver, :PhysicsModels));'`
- Result:
- package loads successfully; `Physics` and `PhysicsModels` are defined.

3) Type stability summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Physics.jl"); using .Physics; using InteractiveUtils; psi = ComplexF64[1/sqrt(2),0,0,1/sqrt(2)]; ord = Tuple(0:1); @code_warntype Physics.density_matrix.rho_numpy(psi, 1, 2; local_dim=2, order=ord); @code_warntype Physics.spectral.Core.greens_function_diagonal(0.25, [-1.0,0.0,1.0]; eta=0.05); @code_warntype Physics.sp.correlation_matrix.corr_from_statevector(psi, 2; mode="slater", order=2);'`
- Result:
- `rho_numpy` body return type: `Matrix{ComplexF64}`.
- `greens_function_diagonal` body return type: `Vector{ComplexF64}` for scalar omega.
- `corr_from_statevector` body return type: `Matrix{ComplexF64}`.
- No `Any` in hot entrypoint body returns.

4) Benchmark summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Physics.jl"); using .Physics; using BenchmarkTools; psi = ComplexF64[1/sqrt(2),0,0,1/sqrt(2)]; ev = randn(64); @btime Physics.spectral.Core.greens_function_diagonal(0.25, $ev; eta=0.05); @btime Physics.sp.correlation_matrix.corr_from_statevector($psi, 2; mode="slater", order=2);'`
- Result:
- `greens_function_diagonal`: `143.207 ns (3 allocations: 2.16 KiB)`.
- `corr_from_statevector`: `123.589 ns (16 allocations: 688 bytes)`.
