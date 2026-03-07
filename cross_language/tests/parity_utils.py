import os
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT            = Path(__file__).resolve().parents[2]
PYTHON_SRC      = ROOT / "pyqusolver" / "Python"
JULIA_PROJECT   = ROOT / "juqusolver"

# --------------------------------------------------------------------------
#! JULIA
# --------------------------------------------------------------------------

def _resolve_julia_bin() -> str:
    env_bin = os.environ.get("JULIA_BIN", "").strip()
    if env_bin:
        return env_bin
    # Use direct Julia binary to avoid juliaup launcher lockfile writes in sandboxed subprocesses.
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

# --------------------------------------------------------------------------
#! PYTHON
# --------------------------------------------------------------------------

def python_env() -> dict:
    env = os.environ.copy()
    pp = str(PYTHON_SRC)
    env["PYTHONPATH"] = pp if not env.get("PYTHONPATH") else pp + os.pathsep + env["PYTHONPATH"]
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    return env

# --------------------------------------------------------------------------
#! RUNNERS
# --------------------------------------------------------------------------

def run_julia(code: str) -> subprocess.CompletedProcess:
    cmd = [JULIA_BIN, "--project=juqusolver", "-e", code]
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True)

def run_julia_lines(code: str) -> List[str]:
    out = run_julia(code).stdout.strip()
    if not out:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]

def run_python(code: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-c", code]
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True, env=python_env())

def run_python_lines(code: str) -> List[str]:
    out = run_python(code).stdout.strip()
    if not out:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]

# --------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------