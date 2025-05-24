"""
QES (Quantum Eigen Solver) Python Package

This package provides tools and utilities for quantum eigenvalue problem solving, 
leveraging modern scientific computing libraries. It includes support for numerical 
and symbolic computation, machine learning, optimization, and visualization.

Dependencies:
    - numpy: Numerical operations and array handling
    - scipy: Scientific computing and numerical operations
    - matplotlib: Plotting and visualization
    - pandas: Data manipulation and analysis
    - sympy: Symbolic mathematics
    - jax, jaxlib: High-performance numerical computing on CPU and GPU
    - flax: Neural network support in JAX
    - optax: Optimization library for JAX
    - scikit-learn: Machine learning utilities
    - scikit-image: Image processing utilities
    - tqdm: Progress bars
    - requests: HTTP requests

Version:
    0.1.0

Author:
    Maksymilian Kliczkowski
Date:
    2025-02-01
Organization:
    Wroclaw University of Science and Technology, Poland
License:
    This package is distributed under the Creative Commons Attribution 4.0 International License.
"""

import os
import re
from setuptools import setup, find_packages

# Read the README file for long description
def read_readme():
    """Read README.md file and return its content."""
    readme_path = os.path.join(os.path.dirname(__file__), '..', '..', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "QES (Quantum Eigen Solver) Python Package for quantum eigenvalue problem solving."

# Read version from __init__.py
def get_version():
    """Extract version from __init__.py file."""
    init_path = os.path.join(os.path.dirname(__file__), '__init__.py')
    if os.path.exists(init_path):
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
            version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", content)
            if version_match:
                return version_match.group(1)
    return '0.1.0'

# Core dependencies required for basic functionality
INSTALL_REQUIRES = [
    'numpy>=1.20.0',        # For numerical operations and array handling
    'scipy>=1.7.0',         # For scientific computing and numerical operations
    'matplotlib>=3.3.0',    # For plotting and visualization
    'pandas>=1.3.0',        # For data manipulation and analysis
    'sympy>=1.8',           # For symbolic mathematics
    'tqdm>=4.60.0',         # For progress bars
    'requests>=2.25.0',     # For HTTP requests
    'numba>=0.55.0',        # For just-in-time compilation
]

# Optional dependencies for enhanced functionality
EXTRAS_REQUIRE = {
    'jax': [
        'jax>=0.4.0',           # For JAX support on CPU and GPU
        'jaxlib>=0.4.0',        # For JAX support on CPU and GPU
        'flax>=0.6.0',          # For neural network support in JAX
        'optax>=0.1.0',         # For optimization in JAX
    ],
    'ml': [
        'scikit-learn>=1.0.0',  # For machine learning utilities
        'scikit-image>=0.18.0', # For image processing utilities
    ],
    'hdf5': [
        'h5py>=3.1.0',          # For HDF5 file format support
    ],
    'dev': [
        'pytest>=6.0.0',        # For testing
        'pytest-cov>=2.12.0',   # For coverage testing
        'black>=21.0.0',        # For code formatting
        'flake8>=3.9.0',        # For linting
        'mypy>=0.910',          # For type checking
        'sphinx>=4.0.0',        # For documentation generation
        'sphinx-rtd-theme>=0.5.0',  # For documentation theme
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'nbsphinx>=0.8.0',      # For Jupyter notebook integration
        'pandoc>=1.12.0',       # For document conversion
    ],
}

# All optional dependencies combined
EXTRAS_REQUIRE['all'] = [dep for deps in EXTRAS_REQUIRE.values() for dep in deps]

setup(
    # Basic package information
    name                            =   'QES',
    version                         =   get_version(),
    author                          =   'Maksymilian Kliczkowski',
    author_email                    =   'maksymilian.kliczkowski@pwr.edu.pl',
    description                     =   'Quantum Eigen Solver: Comprehensive framework for quantum eigenvalue problem solving',
    long_description                =   read_readme(),
    long_description_content_type   =   'text/markdown',

    # URLs and metadata
    url                             =   'https://github.com/makskliczkowski/QuantumEigenSolver',
    project_urls                    =   {
        'Documentation'     : 'https://github.com/makskliczkowski/QuantumEigenSolver/wiki',
        'Source'            : 'https://github.com/makskliczkowski/QuantumEigenSolver',
        'Tracker'           : 'https://github.com/makskliczkowski/QuantumEigenSolver/issues',
    },
    
    # License and classifiers
    license                         =   'CC-BY-4.0',
    classifiers                     =   [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Natural Language :: English',
    ],
    keywords                        =   'quantum physics eigenvalue solver monte carlo variational neural networks jax',

    # Package discovery and data
    packages                        =   find_packages(exclude=['tests*', 'docs*', 'examples*']),
    include_package_data            =   True,
    package_data                    =   {
        'QES': [
            '*.md',
            '*.txt',
            '*.rst',
            'log/*',
            'general_python/tests/data/*',
        ],
    },
    
    # Dependencies
    python_requires                 =   '>=3.8',
    install_requires                =   INSTALL_REQUIRES,
    extras_require                  =   EXTRAS_REQUIRE,

    # Additional options
    zip_safe                        =   False,
    platforms                       =   ['any'],
)
