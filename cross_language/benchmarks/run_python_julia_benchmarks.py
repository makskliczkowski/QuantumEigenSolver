import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "pyqusolver" / "Python"


def _resolve_julia_bin() -> str:
    env_bin = os.environ.get("JULIA_BIN", "").strip()
    if env_bin:
        return env_bin
    return str(
        Path.home()
        / ".julia"
        / "juliaup"
        / "julia-1.12.5+0.aarch64.apple.darwin14"
        / "Julia-1.12.app"
        / "Contents"
        / "Resources"
        / "julia"
        / "bin"
        / "julia"
    )


JULIA_BIN = _resolve_julia_bin()
REPORT_DIR = ROOT / "cross_language" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _py_env():
    env = os.environ.copy()
    pp = str(PYTHON_SRC)
    env["PYTHONPATH"] = pp if not env.get("PYTHONPATH") else pp + os.pathsep + env["PYTHONPATH"]
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    return env


def _bench_python(code: str, repeats: int = 5) -> float:
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=_py_env(), check=True, capture_output=True, text=True)
        vals.append(time.perf_counter() - t0)
    return statistics.median(vals)


def _bench_julia(code: str, repeats: int = 5) -> float:
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        subprocess.run([JULIA_BIN, "--project=juqusolver", "-e", code], cwd=str(ROOT), check=True, capture_output=True, text=True)
        vals.append(time.perf_counter() - t0)
    return statistics.median(vals)


def main():
    julia_import = 'include("juqusolver/src/QuantumEigenSolver.jl"); using .QuantumEigenSolver; '

    benches = [
        {
            "name": "binary_int2base",
            "python": (
                "from QES.general_python.common.binary import int2base; "
                "out=int2base(123456, 20, spin=True, spin_value=1.0); "
                "print(float(out[0]))"
            ),
            "julia": (
                julia_import
                + "out=QuantumEigenSolver.Common.Binary.int2base(123456,20;spin=true,spin_value=1.0); println(out[1])"
            ),
        },
        {
            "name": "physics_gap_ratio",
            "python": (
                "import numpy as np; "
                "from QES.general_python.physics.eigenlevels import gap_ratio; "
                "vals=np.linspace(-3.0,3.0,256); r=gap_ratio(vals,fraction=0.8,use_mean_lvl_spacing=True); "
                "print(float(r['mean']))"
            ),
            "julia": (
                julia_import
                + "vals=collect(range(-3.0,3.0;length=256)); r=QuantumEigenSolver.Physics.Eigenlevels.gap_ratio(vals; fraction=0.8, use_mean_lvl_spacing=true); println(r[\"mean\"]);"
            ),
        },
        {
            "name": "algebra_exact_eigen",
            "python": (
                "import numpy as np; "
                "from QES.general_python.algebra.eigen.factory import choose_eigensolver; "
                "n=64; A=np.random.default_rng(7).standard_normal((n,n)); A=0.5*(A+A.T); "
                "res=choose_eigensolver(method='exact', A=A, hermitian=True); "
                "print(float(np.real(res.eigenvalues[0])))"
            ),
            "julia": (
                julia_import
                + "using Random, LinearAlgebra; n=64; rng=MersenneTwister(7); A=randn(rng, n, n); A=0.5*(A+A');"
                + "res=QuantumEigenSolver.Algebra.EigenModule.choose_eigensolver(QuantumEigenSolver.Algebra.EigenModule.ExactEigensolver; A=A, hermitian=true); println(real(res.eigenvalues[1]));"
            ),
        },
        {
            "name": "quadratic_thermal_scan",
            "python": (
                "import numpy as np; "
                "from QES.Algebra.Properties.quadratic_thermal import quadratic_thermal_scan; "
                "eps=np.linspace(-2.0, 2.0, 64); T=np.linspace(0.25, 2.0, 24); "
                "scan=quadratic_thermal_scan(eps, T, particle_type='fermion', particle_number=32.0); "
                "print(float(scan['C_V'][0]))"
            ),
            "julia": (
                julia_import
                + "eps=collect(range(-2.0, 2.0; length=64)); T=collect(range(0.25, 2.0; length=24)); "
                + "scan=QuantumEigenSolver.Physics.thermal.quadratic_thermal_scan(eps, T; particle_type=\"fermion\", particle_number=32.0); println(scan[\"C_V\"][1]);"
            ),
        },
    ]

    rows = []
    for b in benches:
        t_py = _bench_python(b["python"])
        t_jl = _bench_julia(b["julia"])
        speedup = t_py / t_jl if t_jl > 0 else float("inf")
        rows.append({"name": b["name"], "python_s": t_py, "julia_s": t_jl, "py_over_jl": speedup})

    md = ["# Python vs Julia Benchmarks", "", "| case | python [s] | julia [s] | py/jl |", "|---|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['python_s']:.6f} | {r['julia_s']:.6f} | {r['py_over_jl']:.3f} |")

    md_path = REPORT_DIR / "python_julia_benchmarks.md"
    json_path = REPORT_DIR / "python_julia_benchmarks.json"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n".join(md))
    print(f"\nwritten: {md_path}")
    print(f"written: {json_path}")


if __name__ == "__main__":
    main()
