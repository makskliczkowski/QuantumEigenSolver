You are porting Python to Julia with priorities in this strict order:

- Physics correctness
- Deterministic behavior and reproducibility
- Type stability, speed, and memory efficiency
- Minimal API surface (do not add features)
- Do not add new public functions unless required to preserve the existing public API.
- Keep comments/docstrings ASCII only. No unicode symbols, no decorative characters.
- Match the existing Python comment/doc style: concise, technical, old-school.
- Prefer concrete parametric types. Avoid Any, avoid abstract-typed struct fields in hot code.
- Prefer multiple dispatch over runtime backend switching.
- numba overhead with multiple function creation when unnecessary shall be replaced with fast, no dispatch julia versions...

## Every module must include

- unit tests
- at least one example
- a short docs page

Make sure that all functionality is implmented before proceeding - do not leave anything behind. Make sure different parts
of code for a single module file are well separated so that we know what to look at and the code is readable.
MAKE COMMENTS WITH DESCRIPTIONS.

## Hard constraints

- ASCII only in comments/docstrings (no unicode, no decorative symbols).
- Keep comment style similar to the Python codebase: technical, concise, old-school.
- No unnecessary functions, no feature additions, no refactors that change behavior unless explicitly required for correctness.
- Concrete parametric types; avoid Any; avoid abstract fields in performance-critical structs.
- Hot loops must be allocation-free where feasible.
- Preserve similar file and folder structure!

## Every ported module must include

a) Julia implementation
b) module unit tests (only if the module is big enough, we don't need tests for each of the modules!)
c) one minimal runnable example
d) a short docs page in bullet points
e) update root README: short bullet list plus link to that docs page

## Constraints

- Never run destructive shell commands. Never delete files outside the repo root.
- Never delete files outside the repo root.
- Never rewrite git history.

## Use my style of indentation - example

"""
    int2base(n::Integer, size::Integer; spin::Bool=BACKEND_DEF_SPIN, spin_value::AbstractFloat=BACKEND_REPR)

Convert integer to binary/spin array.
"""
function int2base(n::Integer, size::Integer; spin::Bool=BACKEND_DEF_SPIN, spin_value::AbstractFloat=BACKEND_REPR)
    val_true    = float(spin_value)
    val_false   = spin ? -val_true : 0.0

    out         = Vector{typeof(val_true)}(undef, size)
    nn          = UInt64(n)

    @inbounds for i in 1:size
        pos     = size - i
        bit     = (nn >> UInt64(pos)) & 1
        out[i]  = bit == 1 ? val_true : val_false
    end
    return out
end

## Write docstrings for each function you create

Even when it's some template specification, override, etc. Tell what do the parameters represent!

Deliverables must preserve public API semantics as defined in the module prompt.
Make sure in the implementation that nothing is omitted for later, that's why we do step by step.

## Always check AGENTS.md before performing repository edits

## Workflows must be executed via skills defined in .agents/skills

Do not invent alternative workflows.

## Use a unified package QuantumEigenSolver with submodules

- Common
- Lattices
- Algebra
- PhysicsModels
- Solvers
- NQS
- MachineLearning

Avoid Python-style deeply nested folders. Use flat-ish module files plus small subfolders only when needed for clarity. Do not forget documentation for functions and modules and files!

Use multiple dispatch + concrete parametric structs. Avoid abstract fields in performance-critical structs.

Keep the public API consistent with Python where it matters for user-facing calls, but do not preserve internal Python indirections that harm performance.

## DOCSTRINGS SHOULD BE WRITTEN IN VSCODE RECOGNIZABLE PATTERN

Example:
'''
Compute expectation value.

Parameters
----------

psi :: Vector{ComplexF64}
H   :: SparseMatrixCSC

Returns
-------

Float64
'''

## BACKEND CONSOLIDATION

When Python has multiple numerical backends, Julia should keep one implementation path with consistent API behavior.
Do not keep backend-specific wrapper modules when one Julia implementation is sufficient.
