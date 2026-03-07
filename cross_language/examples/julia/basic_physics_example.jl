include("../../../juqusolver/src/QuantumEigenSolver.jl")
using .QuantumEigenSolver
using LinearAlgebra

function main()
    println("--- Julia Physics Example ---")

    bits    = QuantumEigenSolver.Common.Binary.int2base(13, 6; spin=true, spin_value=1.0)
    println("int2base(13,6): ", bits)

    lat     = QuantumEigenSolver.Lattices.SquareLattice(lx=2, ly=3, dim=2)
    println("square lattice Ns: ", lat.ns)

    vals    = [-2.0, -1.25, -0.2, 0.7, 1.5, 2.4, 3.1]
    gr      = QuantumEigenSolver.Physics.Eigenlevels.gap_ratio(vals; fraction=0.8, use_mean_lvl_spacing=true)
    println("gap ratio mean/std: ", gr["mean"], " ", gr["std"])

    psi     = ComplexF64[0.2, -0.3, 0.1, 0.25, -0.4, 0.15, 0.5, -0.2, 0.05, 0.1, -0.15, 0.2, -0.1, 0.3, -0.25, 0.35]
    psi   ./= norm(psi)
    s       = QuantumEigenSolver.Physics.Eigenlevels.entropy_vonNeuman(psi, 4, 2; TYP="SCHMIDT")
    println("entropy: ", s)

    A = ComplexF64[
        2.0 0.5 0.0 0.0;
        0.5 1.0 0.25 0.0;
        0.0 0.25 1.5 0.75;
        0.0 0.0 0.75 3.0
    ]
    res = QuantumEigenSolver.Algebra.EigenModule.choose_eigensolver(QuantumEigenSolver.Algebra.EigenModule.ExactEigensolver; A=A, hermitian=true)
    println("lowest eigenvalue: ", sort(real.(res.eigenvalues))[1])
end

main()
