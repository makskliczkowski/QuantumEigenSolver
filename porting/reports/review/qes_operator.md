passes gate
- API drift evidence:
- `QESOperator` exports required parity surface:
  - enums: `SymmetryGenerators`, `GlobalSymmetries`, `LocalSpaceTypes`, `OperatorTypeActing`
  - container/helpers: `Operator`, `create_operator`, `create_add_operator`, `ensure_operator_output_shape_numba`, `operator_identity`, `operator_matrix`
  - loaders/introspection: `get_operator_module`, `help`, `list_submodules`, `list_operators`
  - spin-1/2 operators: `sig_x`, `sig_y`, `sig_z`, `sig_p`, `sig_m`, `sig_pm`, `sig_mp`, `sig_k`, `sig_z_total`, `sig_xy`, `sig_yx`, `sig_yz`, `sig_zy`, `sig_zx`, `sig_xz`, `sig_xyz`
  - spin-1 operators: `s1_x`, `s1_y`, `s1_z`, `s1_z2`, `s1_plus`, `s1_minus`, `s1_pm`, `s1_mp`
  - fermion operators: `c`, `cdag`, `ck`, `ckdag`, `n`
  - hardcore kernels and sign helpers are exported.
- `operators_spin`, `operators_spin_1`, `operators_spinless_fermions`, `operators_hardcore`, `operators_anyon`, `catalog`, `operator_loader`, `phase_utils`, `sign` aliases are present.

- Physics correctness evidence:
- spin-1/2 invariants pass:
  - Pauli-X involution on integer basis.
  - Pauli-Y phase magnitude/unitarity checks.
  - Pauli-Z diagonal sign checks.
  - mixed correlator `sig_xy` equals sequential local action.
- spin-1 invariants pass:
  - raising/lowering edge rules.
  - branching structure for `Sx`.
  - two-site `S+S-` transition check.
- fermion/hardcore/anyon invariants pass:
  - `cdag` then `c` roundtrip on local site.
  - number operator occupancy checks.
  - occupancy constraints for hardcore create/annihilate.
  - anyon phase magnitude check.

- Type stability evidence:
- `@code_warntype` reports concrete tuple return types for hot entrypoints:
  - `operators_spin.sigma_x_int_np`
  - `operators_spinless_fermions.c_dag_int_np`
  - `operators_hardcore.hardcore_create_int`
- no `Any`-typed returns in these hot kernels.

- Allocation evidence:
- benchmark results:
  - `sigma_x_int_np`: `75.387 ns (6 allocations: 192 bytes)`
  - `c_dag_int_np`: `100.169 ns (10 allocations: 352 bytes)`
  - `hardcore_create_int`: `111.441 ns (11 allocations: 368 bytes)`
- allocations are bounded and deterministic for current vector-return parity API.

- Sparse and integration evidence:
- `operator_matrix(...; sparse_out=true)` returns sparse matrices (`SparseMatrixCSC`) for both spin-1/2 and spin-1 tests.
- `qes_operator` tests pass in isolation.
- full regression suite and full examples suite pass with `qes_operator` included.
