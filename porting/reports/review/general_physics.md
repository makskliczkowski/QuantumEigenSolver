passes gate
- API drift evidence:
- `Physics` exports expected umbrella aliases: `density_matrix`, `eigenlevels`, `entropy`, `operators`, `statistical`, `thermal`, `spectral`, `response`, `sp`, `single_particle`.
- `single_particle === sp` invariant holds.
- Submodule APIs required by spec are present:
- density-matrix kernels (`rho_numpy`, `rho`, `schmidt`, spectra, site reductions).
- operators parser and spectral window helpers.
- entropy, thermal, statistical, eigenlevels, spectral core/wrappers, response, and SP correlator namespaces.
- No Julia-side JAX modules or JAX-only wrappers are present.

- Behavior and invariants evidence:
- all `general_physics` testsets pass.
- Hermiticity, trace, normalization, and finite-value checks pass across representative density-matrix, thermal, spectral, response, and SP paths.
- deterministic module capability listing and aliasing behavior are preserved.

- Type stability evidence:
- `@code_warntype` body returns are concrete for:
- `Physics.density_matrix.rho_numpy`.
- `Physics.spectral.Core.greens_function_diagonal`.
- `Physics.sp.correlation_matrix.corr_from_statevector`.
- no `Any` in reported hot-entrypoint body returns.

- Allocation evidence:
- `greens_function_diagonal`: `143.207 ns (3 allocations: 2.16 KiB)`.
- `corr_from_statevector`: `123.589 ns (16 allocations: 688 bytes)`.
- allocations are bounded and consistent with current correctness-first API contract.

- Integration evidence:
- `test/general_physics_test.jl` passes end-to-end.
- `examples/general_physics_example.jl` runs successfully.
- `using QuantumEigenSolver` succeeds with `Physics` and `PhysicsModels` available.
