1) Files changed/added
- Existing implementation retained:
- `juqusolver/src/Maths.jl`
- `juqusolver/src/Maths/MathUtils.jl`
- `juqusolver/src/Maths/Random.jl`
- `juqusolver/src/Maths/Statistics.jl`
- `juqusolver/test/general_maths_test.jl`
- `juqusolver/examples/general_maths_example.jl`
- `juqusolver/docs/src/general_maths.md`
- Cross-module package load fix:
- `juqusolver/Project.toml`
- `juqusolver/Manifest.toml`

2) Test summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("test/general_maths_test.jl")'`
- Result:
- `Maths Umbrella Module`: 7/7 pass
- `Maths Random CUE`: 3/3 pass
- `Maths Utilities`: 11/11 pass
- `Maths Statistics`: 6/6 pass
- `Maths Deferred Symbols Not Exported`: 3/3 pass
- Example command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("examples/general_maths_example.jl")'`
- Result:
- example runs and prints module metadata, CUE shape, nearest index/value, and binned averages.

3) Type stability summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Maths.jl"); using .Maths; using InteractiveUtils; using Random; data = [1.0 2.0 3.0 4.0; 2.0 3.0 4.0 5.0]; grid = [0.0 1.0 2.0 3.0; 0.0 1.0 2.0 3.0]; centers = [1.0, 2.0]; @code_warntype Maths.random.CUE_QR(4; simple=false, rng=MersenneTwister(1)); @code_warntype Maths.statistics.bin_avg(data, grid, centers; delta=0.1, cutoff_num=1); @code_warntype Maths.math_utils.find_nearest_idx([0.1, 1.2, 3.0, 4.0], 2.5);'`
- Result:
- `CUE_QR` kwcall body type: `Matrix{ComplexF64}` (no `Any`)
- `bin_avg` kwcall body type: `Vector{Float64}` (no `Any`)
- `find_nearest_idx` body type: `Int64` (no `Any`)

4) Benchmark summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Maths.jl"); using .Maths; using BenchmarkTools; x = randn(1024); @btime Maths.math_utils.find_nearest_idx($x, 0.2);'`
- Result:
- `1.083 μs (0 allocations: 0 bytes)`
