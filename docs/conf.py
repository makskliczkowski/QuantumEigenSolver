# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root directory and Python package to sys.path
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../Python'))
sys.path.insert(0, os.path.abspath('../Python/QES'))

# -- Project information -----------------------------------------------------

project     = 'QES - Quantum EigenSolver'
copyright   = '2025, Maksymilian Kliczkowski'
author      = 'Maksymilian Kliczkowski'
release     = '0.1.0'

# -- General configuration ---------------------------------------------------

# The master toctree document
master_doc  = 'index'

# Add any Sphinx extension module names here.
extensions = [
    'sphinx.ext.autodoc',           # Core library for html generation from docstrings
    'sphinx.ext.viewcode',          # Add links to highlighted source code
    'sphinx.ext.napoleon',          # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',       # Link to other project's documentation
    'sphinx.ext.mathjax',           # Math support
    'sphinx_rtd_theme',             # Read the Docs theme
]

# Add any paths that contain templates here, relative to this directory.
templates_path      = ['_templates']

# List of patterns to ignore when looking for source files.
exclude_patterns    = ['_build', 'build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options  = {
    'canonical_url'                 : '',
    'analytics_id'                  : '',
    'logo_only'                     : False,
    'display_version'               : False,
    'prev_next_buttons_location'    : 'bottom',
    'style_external_links'          : False,
    'vcs_pageview_mode'             : '',
    'style_nav_header_background'   : 'grey',
    # Toc options
    'collapse_navigation'           : True,
    'sticky_navigation'             : True,
    'navigation_depth'              : 4,
    'includehidden'                 : True,
    'titles_only'                   : False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for autodoc ----------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members'               : True,
    'member-order'          : 'bysource',
    'special-members'       : '__init__',
    'undoc-members'         : True,
    'exclude-members'       : '__weakref__'
}

# Generate autosummary automatically
# autosummary_generate                = True

# -- Options for Napoleon (Google/NumPy style docstrings) -------------------

napoleon_google_docstring               = True
napoleon_numpy_docstring                = True
napoleon_include_init_with_doc          = False
napoleon_include_private_with_doc       = False
napoleon_include_special_with_doc       = True
napoleon_use_admonition_for_examples    = False
napoleon_use_admonition_for_notes       = False
napoleon_use_admonition_for_references  = False
napoleon_use_ivar                       = False
napoleon_use_param                      = True
napoleon_use_rtype                      = True
napoleon_preprocess_types               = False
napoleon_type_aliases                   = None
napoleon_attr_annotations               = True

# -- Options for intersphinx extension --------------------------------------

intersphinx_mapping = {
    'python'        : ('https://docs.python.org/3/', None),
    'numpy'         : ('https://numpy.org/doc/stable/', None),
    'scipy'         : ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib'    : ('https://matplotlib.org/stable/', None),
    'pandas'        : ('https://pandas.pydata.org/docs/', None),
    'jax'           : ('https://jax.readthedocs.io/en/latest/', None),
    'flax'          : ('https://flax.readthedocs.io/en/latest/', None),
}

# -- Mock imports for dependencies that might not be available ---------------

autodoc_mock_imports = [
    'jax',
    'jaxlib', 
    'flax',
    'optax',
    'physics-tenpy',
    'tenpy',
    'pyutils',
    'scienceplots',
    'IPython',
    'h5py',
    'scikit-learn',
    'sklearn',
    'scikit-image',
    'numba',
    'requests',
    'tqdm',
    'sympy',
    'matplotlib',
    'numpy',
    'scipy',
    'pandas',
    'tensorflow',
    'torch',
    'QES.general_python.common.flog',
    'QES.general_python.algebra.utils',
    'QES.general_python.ml.schedulers',
]

# Disable autodoc strict mode to avoid import failures
autodoc_inherit_docstrings = False
autodoc_preserve_defaults = True

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True

#! ---------------------------------------------------------------------------
#! End of Sphinx configuration
#! ---------------------------------------------------------------------------
