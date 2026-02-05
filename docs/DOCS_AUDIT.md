# Docs Audit

Date: 2026-02-05

## Scope and inventory

This repository contains two separate Sphinx documentation sets:

- `docs/` (repo-level docs). Uses `docs/conf.py`, `docs/Doxyfile`, and includes
  C++ API material via Breathe.
- `pyqusolver/Python/docs/` (Python package docs). Uses
  `pyqusolver/Python/docs/conf.py` with MyST and nbsphinx.

Key content that exists:

- Repo-level guides: `docs/introduction.rst`, `docs/installation.rst`,
  `docs/usage.rst`, `docs/api.rst`, `docs/cpp_api.rst`, `docs/contributing.rst`.
- Python package guides: `pyqusolver/Python/docs/getting_started.md`,
  `pyqusolver/Python/docs/design_principles.md`,
  `pyqusolver/Python/docs/installation.md`, `pyqusolver/Python/docs/examples.md`,
  `pyqusolver/Python/docs/tutorials.md`, and API pages in
  `pyqusolver/Python/docs/api/`.

## How docs are built

Local build for repo-level docs:

- Install Python package and doc dependencies (from repo root):
  - `python -m pip install -e pyqusolver/Python[docs]`
- Generate Doxygen XML (for Breathe):
  - `cd docs && doxygen Doxyfile`
- Build Sphinx:
  - `cd docs && sphinx-build -b html . _build/html`

Local build for Python package docs:

- Install from `pyqusolver/Python`:
  - `cd pyqusolver/Python && python -m pip install -e .[docs]`
- Build Sphinx:
  - `cd pyqusolver/Python/docs && make html`

Read the Docs build:

- Root config: `.readthedocs.yaml`
  - Python 3.11, Ubuntu 22.04
  - Installs `pyqusolver/Python[docs]`
  - Runs `doxygen` in `docs/`
  - Builds Sphinx with `docs/conf.py`

There is also a separate RTD config in `pyqusolver/Python/docs/readthedocs.yaml`
that is not referenced by the root build. It can drift from the main RTD setup.

## Missing or outdated items

- `pyqusolver/Python/docs/index.rst` previously referenced pages that do not
  exist (quickstart, concepts, modules, physics_background). These topics are
  not currently documented.
- `pyqusolver/Python/docs/api.rst` did not exist, so the API toctree had no
  landing page.
- Repo-level `docs/api.rst` is narrative only and does not use autodoc, so it
  can drift from the actual API surface.
- `docs/requirements.txt` exists but is not tracked; RTD relies on
  `pyqusolver/Python[docs]`, so local builds should avoid depending on the
  untracked file.
- Some module-level packages did not state input/output, dtype/shape,
  stability, or determinism assumptions, which makes API usage ambiguous.

## Notes for future work

- If you want a single documentation entry point, decide whether to keep both
  Sphinx trees or consolidate content. Right now they are separate by design.
- Consider adding minimal "quickstart" and "concepts" pages if those topics
  are important for onboarding.
