A) Public API inventory (Python)
- Source module: `Python/QES/__init__.py`, `qes_globals.py`, `registry.py`, `session.py`.
- Session API:
- `QESSession(backend='numpy', seed=42, precision='float64', num_threads=None)`.
- `QESSession.start() -> QESSession`.
- `QESSession.stop() -> None`.
- `run(backend='numpy', seed=42, precision='float64', num_threads=None) -> QESSession` (context-manager entrypoint).
- Global singleton API:
- `get_logger(**kwargs) -> Logger`.
- `get_backend_manager() -> BackendManager`.
- `get_numpy_rng() -> numpy.random.Generator`.
- `reseed_all(seed) -> BackendManager`.
- `next_jax_key() -> Any`.
- `split_jax_keys(n) -> Any`.
- Top-level wrapper API:
- `qes_reseed(seed) -> None`.
- `qes_next_key() -> Any`.
- `qes_split_keys(n) -> Any`.
- `qes_seed_scope(seed, touch_numpy_global=False, touch_python_random=False)` context manager yielding seed suite.
- Registry API:
- `list_modules(include_submodules=True) -> List[Dict[str,str]]`.
- `describe_module(name_or_path) -> str`.

B) Julia API mapping
- Target files:
- `juqusolver/src/Common/Globals.jl` (module `GlobalsModule`).
- `juqusolver/src/Common/Registry.jl` (module `RegistryModule`).
- `juqusolver/src/Common/Session.jl` (module `SessionModule`).
- Common umbrella integration:
- `juqusolver/src/Common.jl` includes/reexports:
- `get_backend_manager`, `get_logger`, `get_numpy_rng`, `reseed_all`.
- `qes_reseed`, `qes_next_key`, `qes_split_keys`, `qes_seed_scope`.
- `list_modules`, `describe_module`.
- `QESSession`, `run`.
- Top-level package integration:
- `juqusolver/src/QuantumEigenSolver.jl` reexports the same qes-core names.
- Type constraints:
- singleton backend manager stored as concrete mutable struct with `MersenneTwister` rng.
- `list_modules` returns `Vector{Dict{String,String}}`.
- `qes_split_keys` returns `Vector{UInt64}` deterministically.

C) Behavior spec
- Deterministic behavior:
- reseeding via `reseed_all` and `qes_reseed` must make subsequent RNG draws reproducible.
- `qes_seed_scope` must temporarily reseed and restore prior manager seed/rng after scope exit.
- Session behavior:
- `run(; kwargs...)` returns started `QESSession`.
- `run(f::Function; kwargs...)` starts session, executes `f(session)`, always stops in `finally`.
- Registry behavior:
- `list_modules` returns stable sorted module list with keys `name`, `path`, `description`.
- `describe_module` resolves curated module names and full dotted paths with fallback string.
- JAX handling for Julia:
- no Julia-side JAX backend module split.
- key helpers are consolidated deterministic RNG helpers (`qes_next_key`, `qes_split_keys`).

D) Test plan
- Unit tests for qes core:
- singleton accessors return stable objects (`get_logger`, `get_backend_manager`).
- reseed reproducibility with `get_numpy_rng`.
- `qes_next_key` returns `UInt64`; `qes_split_keys(n)` length and uniqueness checks.
- `qes_seed_scope` restores previous manager seed/rng state.
- `list_modules` returns dictionaries with required keys; `describe_module` known/unknown behavior.
- session tests:
- `run(; ...)` starts a session and applies backend/seed config.
- `run(do ...)` pattern stops session automatically on completion.
- regression:
- include qes-core tests in full suite with existing foundation/algebra/physics tests.

E) Performance and typing gate
- Hot entrypoint 1:
- `Common.GlobalsModule.qes_next_key` type-stable return `UInt64`.
- Hot entrypoint 2:
- `Common.RegistryModule.list_modules` concrete return `Vector{Dict{String,String}}`.
- Hot entrypoint 3:
- `Common.SessionModule.run(; ...)` returns concrete `QESSession`.
- Verification:
- `@code_warntype` for the three entrypoints reports concrete body return types (no `Any` in body return).
- one `@btime` for `qes_next_key` and one for `list_modules`.
