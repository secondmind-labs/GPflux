# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import warnings

# Point to root source dir for API doc, relative to this file:
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'GPflux'
copyright = '2021, Secondmind'
author = 'Secondmind'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    'sphinx.ext.mathjax',  # Render math via Javascript
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    'sphinx_autodoc_typehints', # Automatically document param types (less noise in class signature)
    'nbsphinx',  # Integrate Jupyter Notebooks and Sphinx
    'IPython.sphinxext.ipython_console_highlighting'
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc (.inv file)
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    # Unfort. doesn't work yet! See https://github.com/mr-ubik/tensorflow-intersphinx/issues/1
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://raw.githubusercontent.com/mr-ubik/tensorflow-intersphinx/master/tf2_py_objects.inv"
    ),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
add_module_names = False # Remove namespaces from class/method signatures 

# Add any paths that contain Jinja2 templates here, relative to this directory.
templates_path = ['_templates']

#https://sphinxguide.readthedocs.io/en/latest/sphinx_basics/settings.html
# -- Options for LaTeX -----------------------------------------------------
latex_elements = {
'preamble': r'''
\usepackage{amsmath,amsfonts,amssymb,amsthm}
''',
}

# -- Options for HTML output -------------------------------------------------

# Pydata theme
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo-company.png"
html_theme_options = { "show_prev_next": False}
html_css_files = ['pydata-custom.css']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
