API Reference
=============

The documented Python API is the installable ``QES`` package under
``pyqusolver/Python``. The root package is intentionally lazy: ``import QES``
sets global backend defaults and exposes stable entry points without importing
the full JAX/NQS stack.

Top-Level Package
-----------------

.. automodule:: QES
   :members:
   :undoc-members:
   :show-inheritance:

Algebra
-------

.. automodule:: QES.Algebra
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: QES.Algebra.hilbert
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: QES.Algebra.hamil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: QES.Algebra.Operator.operator
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

.. automodule:: QES.Algebra.Model
   :members:
   :undoc-members:
   :show-inheritance:

Solvers
-------

.. automodule:: QES.Solver
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: QES.Solver.MonteCarlo.montecarlo
   :members:
   :undoc-members:
   :show-inheritance:

Neural Quantum States
---------------------

.. automodule:: QES.NQS
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: QES.NQS.nqs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: QES.NQS.src.nqs_entropy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: QES.NQS.src.nqs_spectral
   :members:
   :undoc-members:
   :show-inheritance:

Shared Utilities
----------------

.. automodule:: QES.general_python
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: QES.general_python.lattices
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: QES.general_python.physics.spectral.spectral_backend
   :members:
   :undoc-members:
   :show-inheritance:
