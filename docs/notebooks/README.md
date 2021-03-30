# Creating a new Jupyter Notebook

To make Notebooks easier to work with, they are stored as `.py` files in the
`docs/notebooks` directory within the GPflux repository, and only converted to
actual Notebook files (`.ipynb`) when needed (for example, by Sphinx when
building Readthedocs HTML). Please do not commit `.ipynb` files to the
repository.

## Creating a new Notebook as a .py file

Insert the following at the top of the file:

    # -*- coding: utf-8 -*-
    # ---
    # jupyter:
    #   jupytext:
    #     cell_markers: '"""'
    #     formats: ipynb,py:percent
    #     text_representation:
    #       extension: .py
    #       format_name: percent
    #       format_version: '1.3'
    #       jupytext_version: 1.3.3
    #   kernelspec:
    #     display_name: Python 3
    #     language: python
    #     name: python3
    # ---

Make sure a text (ie. comment) cell is encapsulated as follows:

    # %% [markdown]
    """
    <your multi-line text goes here>
    """

Make sure a code cell is encapsulated as follows:

    # %%
    # Optional code comment
    def YourCodeGoesHere():
       pass

To test it as a Notebook, make sure Jupytext is installed (`pip install jupytext`) and run:

    jupytext --to notebook <your-new-file>.py

This creates `<your-new-file>.ipynb`. Run `jupyter-notebook <your-new-file>.ipynb`
in the normal way to make sure it formats and executes correctly in the IPython environment.

When ready, commit `<your-new-file>.py`. You can delete `<your-new-file>.ipynb`.

## Creating a new Notebook as a regular Notebook

You can run `jupyter-notebook` and press the **New** button to create a
Notebook using the Jupyter UI. When ready, save and exit, and run the following
command to convert `<your-new-file>.ipynb` file to `<your-new-file>.py` (with
text cells formatted as multi-line comments):

    jupytext --update-metadata '{"jupytext": {"cell_markers": "\"\"\""}}' --to py:percent <your-new-file>.ipynb

When ready, commit `<your-new-file>.py`. You can delete `<your-new-file>.ipynb`.

## Including your Notebook in GPflux's Sphinx-built documentation

Within the GPflux repository, open [docs/tutorials.rst](../tutorials.rst) and
insert a TOC entry for the Notebook, for example:

    .. toctree::
       :maxdepth: 1

       notebooks/keras_integration
       notebooks/<your-new-file>

Then, build the docs as detailed in [docs/README.md](../README.md).
