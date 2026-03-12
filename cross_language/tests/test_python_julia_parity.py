import math

import numpy as np

from parity_utils import run_julia_lines, run_python_lines


JULIA_IMPORT = 'include("juqusolver/src/QuantumEigenSolver.jl"); using .QuantumEigenSolver;'


def _norm(name: str) -> str:
    return name.replace("_", "").replace("!", "").lower()


def _module_public_python(module: str):
    code = (
        f"import {module} as m; "
        "import inspect; "
        "names=[n for n in dir(m) if not n.startswith('_')]; "
        "print('\\n'.join(names))"
    )
    return set(run_python_lines(code))


def _module_public_julia(modpath: str):
    code = (
        JULIA_IMPORT
        + f" mod={modpath}; "
        + "for n in names(mod, all=false); println(String(n)); end"
    )
    return set(run_julia_lines(code))


def test_api_inventory_required_symbols_exist_on_both_sides():
    checks = [
        (
            "QES.general_python.common.binary",
            "QuantumEigenSolver.Common.Binary",
            ["int2base"],
        ),
        (
            "QES.general_python.physics.eigenlevels",
            "QuantumEigenSolver.Physics.Eigenlevels",
            ["entropy_vonNeuman", "gap_ratio"],
        ),
        (
            "QES.general_python.algebra.eigen.factory",
            "QuantumEigenSolver.Algebra.EigenModule",
            ["choose_eigensolver"],
        ),
        (
            "QES.Algebra.Operator.impl.operators_spin",
            "QuantumEigenSolver.Algebra.OperatorsCore",
            ["sig_x", "sig_z", "sig_z_total"],
        ),
        (
            "QES.general_python.lattices.square",
            "QuantumEigenSolver.Lattices",
            ["SquareLattice"],
        ),
    ]

    for pymod, jlmod, required in checks:
        py_names = {_norm(n) for n in _module_public_python(pymod)}
        jl_names = {_norm(n) for n in _module_public_julia(jlmod)}
        for sym in required:
            ns = _norm(sym)
            assert ns in py_names, f"Python missing required symbol {sym} in {pymod}"
            assert ns in jl_names, f"Julia missing required symbol {sym} in {jlmod}"


def test_int2base_parity():
    py = run_python_lines(
        "from QES.general_python.common.binary import int2base; "
        "out=int2base(13,6,spin=True,spin_value=1.0); "
        "print(','.join(str(float(x)) for x in out))"
    )[0]
    jl = run_julia_lines(
        JULIA_IMPORT
        + " out = QuantumEigenSolver.Common.Binary.int2base(13, 6; spin=true, spin_value=1.0);"
        + " println(join(out, ','))"
    )[0]

    pyv = np.array([float(x) for x in py.split(',')], dtype=float)
    jlv = np.array([float(x) for x in jl.split(',')], dtype=float)
    assert np.allclose(pyv, jlv)


def test_gap_ratio_parity():
    vals = [-2.0, -1.25, -0.2, 0.7, 1.5, 2.4, 3.1]
    arr = ",".join(str(x) for x in vals)

    py = run_python_lines(
        "import numpy as np; "
        "from QES.general_python.physics.eigenlevels import gap_ratio; "
        f"r=gap_ratio(np.array([{arr}],dtype=float), fraction=0.8, use_mean_lvl_spacing=True); "
        "print(f\"{float(r['mean'])},{float(r['std'])}\")"
    )[0]

    jl = run_julia_lines(
        JULIA_IMPORT
        + f" vals = Float64[{arr}];"
        + " r = QuantumEigenSolver.Physics.Eigenlevels.gap_ratio(vals; fraction=0.8, use_mean_lvl_spacing=true);"
        + " println(string(r[\"mean\"], ',', r[\"std\"]))"
    )[0]

    py_mean, py_std = [float(x) for x in py.split(',')]
    jl_mean, jl_std = [float(x) for x in jl.split(',')]

    assert math.isclose(py_mean, jl_mean, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(py_std, jl_std, rel_tol=1e-9, abs_tol=1e-9)


def test_entropy_vonneumann_parity():
    psi = np.array([0.2, -0.3, 0.1, 0.25, -0.4, 0.15, 0.5, -0.2, 0.05, 0.1, -0.15, 0.2, -0.1, 0.3, -0.25, 0.35], dtype=float)
    psi = psi / np.linalg.norm(psi)
    arr = ",".join(str(float(x)) for x in psi)

    py = run_python_lines(
        "import numpy as np; "
        "from QES.general_python.physics.eigenlevels import entropy_vonNeuman; "
        f"psi=np.array([{arr}],dtype=float);"
        "print(float(entropy_vonNeuman(psi, 4, 2, TYP='SCHMIDT')))"
    )[0]

    jl = run_julia_lines(
        JULIA_IMPORT
        + f" psi = ComplexF64[{arr}];"
        + " println(QuantumEigenSolver.Physics.Eigenlevels.entropy_vonNeuman(psi, 4, 2; TYP=\"SCHMIDT\"))"
    )[0]

    assert math.isclose(float(py), float(jl), rel_tol=1e-9, abs_tol=1e-9)


def test_exact_eigensolver_parity():
    a = np.array(
        [
            [2.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.25, 0.0],
            [0.0, 0.25, 1.5, 0.75],
            [0.0, 0.0, 0.75, 3.0],
        ],
        dtype=float,
    )
    data_jl = ";".join(" ".join(str(float(x)) for x in row) for row in a)
    data_py = ",".join("[" + ",".join(str(float(x)) for x in row) + "]" for row in a)

    py = run_python_lines(
        "import numpy as np; "
        "from QES.general_python.algebra.eigen.factory import choose_eigensolver; "
        f"A=np.array([{data_py}],dtype=float);"
        "res=choose_eigensolver(method='exact', A=A, hermitian=True);"
        "vals=np.sort(np.real(np.array(res.eigenvalues)));"
        "print(','.join(str(float(v)) for v in vals))"
    )[0]

    jl = run_julia_lines(
        JULIA_IMPORT
        + f" A = ComplexF64[{data_jl}];"
        + " res = QuantumEigenSolver.Algebra.EigenModule.choose_eigensolver(QuantumEigenSolver.Algebra.EigenModule.ExactEigensolver; A=A, hermitian=true);"
        + " vals = sort(real.(res.eigenvalues));"
        + " println(join(vals, ','))"
    )[0]

    pyv = np.array([float(x) for x in py.split(',')], dtype=float)
    jlv = np.array([float(x) for x in jl.split(',')], dtype=float)
    assert np.allclose(pyv, jlv, rtol=1e-10, atol=1e-10)


def test_square_lattice_site_count_parity():
    py = run_python_lines(
        "from QES.general_python.lattices.square import SquareLattice; "
        "L=SquareLattice(lx=2,ly=3,dim=2); print(int(L.Ns))"
    )[0]
    jl = run_julia_lines(
        JULIA_IMPORT
        + " L = QuantumEigenSolver.Lattices.SquareLattice(lx=2, ly=3, dim=2);"
        + " println(Int(L.ns))"
    )[0]
    assert int(py) == int(jl)


def test_quadratic_occupation_helper_parity():
    py = run_python_lines(
        "import numpy as np; "
        "from QES.Algebra.Quadratic.hamil_quadratic_utils import occ_to_indices, occ_to_mask, energy_from_mask; "
        "idx=occ_to_indices(np.array([True, False, True, False]), 4); "
        "mask=occ_to_mask([1,3], 4); "
        "eps=np.array([-1.0,-0.2,0.5,1.25], dtype=float); "
        "eng=energy_from_mask(np.uint64(5), eps); "
        "print(','.join(map(str, idx))); "
        "print(int(mask)); "
        "print(float(eng))"
    )

    jl = run_julia_lines(
        JULIA_IMPORT
        + " idx = QuantumEigenSolver.Algebra.occ_to_indices(Bool[true, false, true, false], 4);"
        + " mask = QuantumEigenSolver.Algebra.occ_to_mask([1, 3], 4);"
        + " eps = Float64[-1.0, -0.2, 0.5, 1.25];"
        + " eng = QuantumEigenSolver.Algebra.energy_from_mask(UInt64(5), eps);"
        + " println(join(idx, ','));"
        + " println(Int(mask));"
        + " println(eng);"
    )

    assert py[0] == jl[0]
    assert int(py[1]) == int(jl[1])
    assert math.isclose(float(py[2]), float(jl[2]), rel_tol=1e-10, abs_tol=1e-10)


def test_quadratic_thermal_scan_parity():
    eps = [-1.5, -0.25, 0.4, 1.2]
    temps = [0.25, 0.5, 1.0, 2.0]
    arr_eps = ",".join(str(x) for x in eps)
    arr_t = ",".join(str(x) for x in temps)

    py = run_python_lines(
        "import numpy as np; "
        "from QES.Algebra.Properties.quadratic_thermal import quadratic_thermal_scan; "
        f"eps=np.array([{arr_eps}], dtype=float); "
        f"T=np.array([{arr_t}], dtype=float); "
        "scan=quadratic_thermal_scan(eps, T, particle_type='fermion', particle_number=2.0); "
        "print(','.join(str(float(x)) for x in scan['mu'])); "
        "print(','.join(str(float(x)) for x in scan['N'])); "
        "print(','.join(str(float(x)) for x in scan['C_V'])); "
        "print(','.join(str(float(x)) for x in scan['S']))"
    )

    jl = run_julia_lines(
        JULIA_IMPORT
        + f" eps = Float64[{arr_eps}];"
        + f" T = Float64[{arr_t}];"
        + " scan = QuantumEigenSolver.Physics.thermal.quadratic_thermal_scan(eps, T; particle_type=\"fermion\", particle_number=2.0);"
        + " println(join(scan[\"mu\"], ','));"
        + " println(join(scan[\"N\"], ','));"
        + " println(join(scan[\"C_V\"], ','));"
        + " println(join(scan[\"S\"], ','));"
    )

    for py_line, jl_line in zip(py, jl):
        pyv = np.array([float(x) for x in py_line.split(",")], dtype=float)
        jlv = np.array([float(x) for x in jl_line.split(",")], dtype=float)
        assert np.allclose(pyv, jlv, rtol=1e-6, atol=1e-6)


def test_quadratic_bogoliubov_energy_parity():
    py = run_python_lines(
        "import numpy as np; "
        "from QES.Algebra.hamil_quadratic import QuadraticHamiltonian; "
        "K=np.array([[0.2,-1.0,0.0],[-1.0,-0.3,-0.5],[0.0,-0.5,0.1]], dtype=np.complex128); "
        "D=np.array([[0.0,0.15,0.0],[-0.15,0.0,0.25],[0.0,-0.25,0.0]], dtype=np.complex128); "
        "H=QuadraticHamiltonian.from_bdg_matrices(hermitian_part=K, antisymmetric_part=D, constant=0.1, dtype=np.complex128); "
        "W, eps, c = H.diagonalizing_bogoliubov_transform(); "
        "print(','.join(str(float(x)) for x in np.real(np.array(eps)))); "
        "print(float(c)); "
        "print(W.shape[0]); print(W.shape[1])"
    )

    jl = run_julia_lines(
        JULIA_IMPORT
        + " K = ComplexF64[0.2 -1.0 0.0; -1.0 -0.3 -0.5; 0.0 -0.5 0.1];"
        + " D = ComplexF64[0.0 0.15 0.0; -0.15 0.0 0.25; 0.0 -0.25 0.0];"
        + " H = QuantumEigenSolver.Algebra.QuadraticHamiltonian(; hermitian_part=K, antisymmetric_part=D, constant=0.1);"
        + " W, eps, c = QuantumEigenSolver.Algebra.diagonalizing_bogoliubov_transform(H);"
        + " println(join(eps, ','));"
        + " println(c);"
        + " println(size(W, 1)); println(size(W, 2));"
    )

    py = py[-4:]
    py_eps = np.array([float(x) for x in py[0].split(",")], dtype=float)
    jl_eps = np.array([float(x) for x in jl[0].split(",")], dtype=float)
    assert np.allclose(py_eps, jl_eps, rtol=1e-8, atol=1e-8)
    assert math.isclose(float(py[1]), float(jl[1]), rel_tol=1e-8, abs_tol=1e-8)
    assert int(py[2]) == int(jl[2])
    assert int(py[3]) == int(jl[3])
