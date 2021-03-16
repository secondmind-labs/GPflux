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
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "GPflux"
copyright = (
    "Copyright 2021 The GPflux Contributors\n"
    "\n"
    "Licensed under the Apache License, Version 2.0\n"
)
author = "The GPflux Contributors"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------

default_role = "any"  # try and turn all `` into links
add_module_names = False  # Remove namespaces from class/method signatures


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.mathjax",  # Render math via Javascript
    "IPython.sphinxext.ipython_console_highlighting",  # syntax-highlighting ipython interactive sessions
]


### Automatic API doc generation
extensions.append("autoapi.extension")
autoapi_dirs = ["../gpflux"]
autoapi_add_toctree_entry = False
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "private-members",
    "special-members",
    "imported-members",
    "show-inheritance",
]
# autoapi_member_order = "bysource"  # default
# autoapi_member_order = "groupwise"  # by type then alphabetically


### intersphinx: Link to other project's documentation (see mapping below)
extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "tensorflow": (
       "https://www.tensorflow.org/api_docs/python",
       "tf2_py_objects.inv"
    ),
    "tensorflow_probability": (
       "https://www.tensorflow.org/probability/api_docs/python",
       "tfp_py_objects.inv"
    ),
    "gpflow": ("https://gpflow.readthedocs.io/en/master/", None),
}

### todo: to-do notes
extensions.append("sphinx.ext.todo")
todo_include_todos = True  # pre-1.0, it's worth actually including todos in the docs

### nbsphinx: Integrate Jupyter Notebooks and Sphinx
extensions.append("nbsphinx")
nbsphinx_allow_errors = True  # Continue through Jupyter errors

### sphinxcontrib-bibtex
extensions.append("sphinxcontrib.bibtex")
bibtex_bibfiles = ["refs.bib"]


# Add any paths that contain Jinja2 templates here, relative to this directory.
templates_path = ["_templates"]

# https://sphinxguide.readthedocs.io/en/latest/sphinx_basics/settings.html
# -- Options for LaTeX -----------------------------------------------------
latex_elements = {
    "preamble": r"""
\usepackage{amsmath,amsfonts,amssymb,amsthm}
""",
}

# -- Options for HTML output -------------------------------------------------

# Pydata theme
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_css_files = ["pydata-custom.css"]

# theme-specific options. see theme docs for more info
html_theme_options = {
    "show_prev_next": False,
    "github_url": "https://github.com/secondmind-labs/gpflux",
}

# If True, show link to rst source on rendered HTML pages
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
