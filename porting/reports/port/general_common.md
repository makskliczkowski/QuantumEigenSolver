1) Files changed/added
- Existing implementation retained:
- `juqusolver/src/Common.jl`
- `juqusolver/src/Common/Binary.jl`
- `juqusolver/src/Common/Directories.jl`
- `juqusolver/src/Common/Plot.jl`
- `juqusolver/src/Common/Datah.jl`
- `juqusolver/src/Common/HDF5Man.jl`
- `juqusolver/src/Common/LazyEntry.jl`
- `juqusolver/src/Common/Plotters/DataLoader.jl`
- `juqusolver/src/Common/Flog.jl`
- `juqusolver/src/Common/Memory.jl`
- `juqusolver/src/Common/Timer.jl`
- `juqusolver/test/common_test.jl`
- `juqusolver/examples/common_example.jl`
- `juqusolver/docs/src/common.md`
- Cross-module package load fix:
- `juqusolver/Project.toml`
- `juqusolver/Manifest.toml`

2) Test summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("test/common_test.jl")'`
- Result:
- `Common Umbrella Module`: 3/3 pass
- `Common Reexports`: 12/12 pass
- `New Common Surface Exported`: 23/23 pass
- `Plotter Cycles and Save Helpers`: 9/9 pass
- `DataHandler and Data Loader`: 11/11 pass
- `HDF5Manager Availability Path`: 3/3 pass
- Example command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("examples/common_example.jl")'`
- Result:
- example runs and writes deterministic artifacts under `juqusolver/tmp/plot`, `juqusolver/tmp/data_loader`, and `juqusolver/tmp/hdf5`, including SVG output from `Plotter.save_fig`.

3) Type stability summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Common.jl"); using .Common; using InteractiveUtils; @code_warntype Common.mask_from_indices([0,2,3]); @code_warntype Common.indices_from_mask(UInt64(13)); @code_warntype Common.Binary.int2base(13, 8; spin=true, spin_value=1.0);'`
- Result:
- `mask_from_indices` body type: `UInt64` (no `Any`)
- `indices_from_mask` body type: `Vector{Int64}` (no `Any`)
- `int2base` kwcall body type: `Vector{Float64}` (no `Any`)

4) Benchmark summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Common.jl"); using .Common; using BenchmarkTools; x = collect(0:31); @btime Common.mask_from_indices($x);'`
- Result:
- `11.512 ns (0 allocations: 0 bytes)`
