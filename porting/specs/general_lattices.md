A) Public API inventory (Python `general_python/lattices/__init__.py`)
- Core enums/types:
- `BoundaryFlux`, `Lattice`, `LatticeBackend`, `LatticeBC`, `LatticeDirection`, `LatticeType`.
- Concrete lattices:
- `SquareLattice`, `HexagonalLattice`, `HoneycombLattice`, `TriangularLattice`, `GraphLattice`.
- Registry/factory:
- `register_lattice`, `available_lattices`, `choose_lattice`.
- Symmetry objects:
- `HighSymmetryPoints`, `HighSymmetryPoint`, `KPathResult`, `StandardBZPath`.
- Visualization API:
- `plot_bonds`, `plot_lattice_structure`, `plot_real_space`, `plot_reciprocal_space`, `plot_brillouin_zone`.
- `LatticePlotter`, `format_lattice_summary`, `format_vector_table`, `format_real_space_vectors`, `format_reciprocal_space_vectors`, `format_brillouin_zone_overview`.
- Regions API:
- `RegionType`, `LatticeRegionHandler`, `Region`, `KitaevPreskillRegion`, `LevinWenRegion`, `HalfRegions`, `DiskRegion`, `PlaquetteRegion`, `CustomRegion`, `KPRegion`, `LWRegion`.
- `get_predefined_region`, `list_predefined_regions`.
- Utility/testing exports:
- `tools`, `run_lattice_tests`.

B) Julia API mapping
- Umbrella module:
- `juqusolver/src/Lattices.jl`.
- Submodules:
- `juqusolver/src/Lattices/Tools.jl`
- `juqusolver/src/Lattices/Core.jl`
- `juqusolver/src/Lattices/Registry.jl`
- `juqusolver/src/Lattices/Symmetry.jl`
- `juqusolver/src/Lattices/Visualization.jl`
- `juqusolver/src/Lattices/Regions.jl`
- Export parity target:
- all names listed in section A are available from `Lattices`.
- internal helper names may exist but must not change behavior of section-A symbols.
- Type/structure constraints:
- concrete fields in lattice structs (no `Any` in hot objects).
- immutable structs for lattice data where feasible.
- deterministic construction and neighbor generation.

C) Behavior specification
- Boundary handling:
- `handle_dim` infers valid dimension from `(lx, ly, lz)` and rejects non-positive extents.
- `handle_boundary_conditions` accepts enum/string aliases and supports twisted flux input.
- Factory behavior:
- `choose_lattice` accepts string or `LatticeType`, normalizes dimension/boundary, returns correct concrete lattice.
- `register_lattice` preserves uniqueness unless `overwrite=true`; aliases map to the same lattice type.
- Geometry invariants:
- `ns` matches expected lattice-site count.
- neighbor indices are valid and deterministic.
- adjacency output is deterministic and consistent with neighbor lists.
- Symmetry/reciprocal behavior:
- high-symmetry points and BZ path interpolation are deterministic for fixed lattice/path inputs.
- Region behavior:
- region constructors produce valid `A/B/C` partitions and deterministic predefined-region lookups.

D) Test plan
- API surface:
- assert required Python-level names are exported from `Lattices`.
- Core tools/factory:
- boundary parsing, dimension handling, registry aliasing, lattice construction for all built-in lattices.
- Geometry:
- coordinates shape, neighbor cardinality sanity, graph adjacency behavior.
- Symmetry:
- high-symmetry collection availability and BZ path data shape/labels.
- Visualization:
- formatter outputs non-empty and structured text.
- plot facade returns deterministic data containers with expected fields.
- Regions:
- handler-based region generation (`half`, `disk`, `sublattice`, `custom`) and predefined-region lookup.
- Regression:
- run `common`, `maths`, and `lattices` tests together.

E) Performance and typing gates
- Gate 1:
- `site_index(::SquareLattice, ...)` type-stable `Int` return and zero allocations.
- Gate 2:
- `get_nei(::AbstractLattice, ...)` returns concrete `Vector{Int}` without `Any`.
- Gate 3:
- symmetry path container creation (`bz_path_data`) returns concrete `KPathResult`.
- Verification commands:
- `@code_warntype` on `site_index`, `get_nei`, and `bz_path_data`.
- `@btime` benchmark for `site_index` and `get_nei`.
- Acceptance:
- no `Any` in hot-path outputs.
- zero allocations for `site_index` and `get_nei` steady-state calls.
