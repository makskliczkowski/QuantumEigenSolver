1) Files changed/added
- Implementation:
- `juqusolver/src/Algebra/QESSymmetries.jl`
- `juqusolver/src/Algebra/QESHilbert.jl`
- Spec/docs/tests/examples:
- `porting/specs/qes_algebra.md`
- `juqusolver/test/qes_algebra_test.jl`
- `juqusolver/examples/qes_algebra_example.jl`
- `juqusolver/docs/src/qes_algebra.md`
- `README.md`

2) Test summary
- Module command:
- `~/.juliaup/bin/julia --project=juqusolver -e 'include("juqusolver/test/qes_algebra_test.jl")'`
- Result:
- `QES Algebra Export Surface`: 17/17 pass
- `QES Hilbert Config and Construction`: 19/19 pass
- `QES Symmetry Compact Mapping Invariants`: 13/13 pass
- `QES Hamiltonian Registry Semantics`: 10/10 pass
- `QES Global Registry and Backend Helper Wrappers`: 8/8 pass
- Full regression command:
- `~/.juliaup/bin/julia --project=juqusolver -e 'include("juqusolver/test/common_test.jl"); include("juqusolver/test/general_maths_test.jl"); include("juqusolver/test/general_lattices_test.jl"); include("juqusolver/test/general_physics_test.jl"); include("juqusolver/test/general_algebra_test.jl"); include("juqusolver/test/qes_algebra_test.jl"); include("juqusolver/test/general_python_init_test.jl"); include("juqusolver/test/qes_core_test.jl"); include("juqusolver/test/qes_models_test.jl")'`
- Result:
- all testsets pass.
- Example command:
- `cd juqusolver && ~/.juliaup/bin/julia --project examples/qes_algebra_example.jl`
- Result:
- example runs successfully; translation `k=0` sector gives reduced size (`ns=4`, `nh=6`), reduced/full roundtrip error is near machine precision, and mixed-generator `Translation_x + Reflection` sector is constructed.
- Full examples command:
- `~/.juliaup/bin/julia --project=juqusolver -e 'include("juqusolver/examples/common_example.jl"); include("juqusolver/examples/general_maths_example.jl"); include("juqusolver/examples/general_lattices_example.jl"); include("juqusolver/examples/general_python_init_example.jl"); include("juqusolver/examples/general_physics_example.jl"); include("juqusolver/examples/general_algebra_example.jl"); include("juqusolver/examples/qes_algebra_example.jl"); include("juqusolver/examples/qes_core_example.jl"); include("juqusolver/examples/qes_models_example.jl")'`
- Result:
- all examples run and continue writing artifacts under `juqusolver/tmp/<type>`.

3) Type stability summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Algebra.jl"); using .Algebra; using InteractiveUtils; cfgm = Algebra.HilbertConfig(ns=4, symmetry_generators=[Algebra.SymmetrySpec(:Translation_x, 0), Algebra.SymmetrySpec(:Reflection, 1)], gen_mapping=true); hsm = Algebra.from_config(Algebra.HilbertSpace, cfgm); vm = randn(ComplexF64, hsm.nhfull); @code_warntype Algebra.from_config(Algebra.HilbertSpace, cfgm); @code_warntype Algebra.QESHilbert.transform_to_reduced_space(hsm, vm); reg = Algebra.HamiltonianRegistry{Algebra.Hamiltonian}(); Algebra.register(reg, "toy", (hcfg, p) -> Algebra.Hamiltonian(hilbert_space=Algebra.HilbertSpace(ns=Int(get(p, :ns, 2))), name=String(get(p, :name, "H"))); description="toy", default_kwargs=Dict{Symbol,Any}(:ns => 2)); hcfg = Algebra.HamiltonianConfig(kind="toy", parameters=Dict{Symbol,Any}(:name => "H1")); @code_warntype Algebra.instantiate(reg, hcfg);'`
- Result:
- `from_config(::Type{HilbertSpace}, cfg)` body return type: `HilbertSpace`.
- `transform_to_reduced_space(hs, vfull)` body return type: `Vector{ComplexF64}`.
- `instantiate(::HamiltonianRegistry{Hamiltonian}, cfg)` body return type: `Hamiltonian`.

4) Benchmark summary
- Command:
- `cd juqusolver && ~/.juliaup/bin/julia --project -e 'include("src/Algebra.jl"); using .Algebra; using BenchmarkTools; cfgm = Algebra.HilbertConfig(ns=4, symmetry_generators=[Algebra.SymmetrySpec(:Translation_x, 0), Algebra.SymmetrySpec(:Reflection, 1)], gen_mapping=true); hsm = Algebra.from_config(Algebra.HilbertSpace, cfgm); vm = randn(ComplexF64, hsm.nhfull); @btime Algebra.from_config(Algebra.HilbertSpace, $cfgm); @btime Algebra.QESHilbert.transform_to_reduced_space($hsm, $vm);'`
- Result:
- `from_config(HilbertSpace, cfgm)`: `13.708 us (317 allocations: 21.08 KiB)`.
- `transform_to_reduced_space(hsm, vm)`: `39.609 ns (2 allocations: 160 bytes)`.
