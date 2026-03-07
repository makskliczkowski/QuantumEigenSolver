import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def _env():
    env = os.environ.copy()
    pp = str(PYTHON_SRC)
    env["PYTHONPATH"] = pp if not env.get("PYTHONPATH") else pp + os.pathsep + env["PYTHONPATH"]
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    return env


def main():
    julia_bin = _resolve_julia_bin()
    cmds = [
        [sys.executable, "-m", "pytest", "cross_language/tests/test_python_julia_parity.py", "-q"],
        [sys.executable, "cross_language/benchmarks/run_python_julia_benchmarks.py"],
        [sys.executable, "cross_language/examples/python/basic_physics_example.py"],
        [julia_bin, "--project=juqusolver", "cross_language/examples/julia/basic_physics_example.jl"],
    ]

    for cmd in cmds:
        print("\n$", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), env=_env(), check=True)


if __name__ == "__main__":
    main()
