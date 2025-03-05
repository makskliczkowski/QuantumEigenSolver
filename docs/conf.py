import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'QuantumEigenSolver'
author = 'Maksymilian Kliczkowski'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

html_theme = 'sphinx_rtd_theme'
