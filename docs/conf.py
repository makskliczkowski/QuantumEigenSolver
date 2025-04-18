# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root directory to sys.path so that Sphinx can find the modules
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project     = 'Quantum Eigensolver'
copyright   = '2025, Maksymilian Kliczkowski'
author      = 'Maksymilian Kliczkowski'

# -- General configuration ---------------------------------------------------

# The master toctree document
master_doc = 'index'

# Add any Sphinx extension module names here.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns to ignore when looking for source files.
exclude_patterns = ['_build', 'build', 'Thumbs.db', '.DS_Store']
