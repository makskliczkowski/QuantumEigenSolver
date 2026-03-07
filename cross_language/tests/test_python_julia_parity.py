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
