# Cross-Language Parity Suite (Python vs Julia)

This directory contains repository-level checks for the consistency between:

- `pyqusolver` (`pyqusolver/Python`)
- `juqusolver` (`juqusolver`)

Location is intentionally at repository root level for shared validation.

## Contents

- `tests/test_python_julia_parity.py`
  - API inventory checks for key mapped modules.
  - Numerical parity checks for core methods.
- `benchmarks/run_python_julia_benchmarks.py`
  - Cross-language benchmark snapshots for representative operations.
  - Writes reports to `cross_language/reports/`.
- `examples/python/basic_physics_example.py`
- `examples/julia/basic_physics_example.jl`
  - Same conceptual example in both languages.
- `run_all.py`
  - Runs tests, benchmarks, and both examples.

## Run

From repository root:

```bash
python cross_language/run_all.py
```

Or run only parity tests:

```bash
PYTHONPATH=pyqusolver/Python pytest cross_language/tests/test_python_julia_parity.py -q
```

Or run only cross-language benchmarks:

```bash
python cross_language/benchmarks/run_python_julia_benchmarks.py
```

## Notes

- `JULIA_BIN` can be set to point at a specific Julia binary:
  - `JULIA_BIN=/path/to/julia python cross_language/run_all.py`
- Benchmarks are compact snapshots for regression tracking, not full scaling studies.
- Numerical parity checks compare deterministic small problems to avoid backend drift.
- This suite is designed for computation-focused parity validation rather than exploratory usage.
- Python runners set `MPLCONFIGDIR=/tmp/mplconfig` and `XDG_CACHE_HOME=/tmp` by default in this harness.
