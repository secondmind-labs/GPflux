# GPflux documentation

The API reference (generated from docstrings) and Jupyter Notebook tutorials are automatically built by a Github action (see .github/workflows/deploy.yaml) on commit to `develop` and published to [https://prowler-io.github.io/gpflux/index.html](https://prowler-io.github.io/gpflux/index.html). TODO: update URL!

## Jupyter Notebooks

If you want to run the Jupyter Notebook tutorials interactively, install additional dependencies in the `docs` directory:

`pip install -r docs_requirements.txt`

...and then run the appropriate Notebook:

`jupyter-notebook notebooks/<name-of-notebook>`

If you want to create a new Notebook tutorial for inclusion in the doc set, see `notebooks/README.md`.

## API reference

If you want to build the documentation locally:

1) Make sure you have a Python 3.7 virtualenv and `gpflux` is installed as per the instructions in `../README.md`)

3) In the `docs` directory, install dependencies:

   `pip install -r docs_requirements.txt`
   
   If pandoc does not install via pip, or step 3) fails with a 'Pandoc' error, download and install Pandoc separately from `https://github.com/jgm/pandoc/releases/` (e.g. `pandoc-<version>-amd64.deb` for Ubuntu), and try running step 2) again.

4) Compile the documentation:

   `make html`

5) Run a web server:

   `python -m http.server`

6) Check documentation locally by opening (in a browser):

   http://localhost:8000/_build/html/
