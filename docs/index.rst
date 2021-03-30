.. Copyright 2021 The GPflux Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

..
   Note: Items in the toctree form the top-level navigation.

.. toctree::
   :hidden:

   GPflux <self>
   Benchmarks <notebooks/benchmarks>
   Tutorials <tutorials>
   API Reference <autoapi/gpflux/index>


Welcome to GPflux
==================

GPflux is a research toolbox dedicated to Deep Gaussian processes (DGP) :cite:p:`damianou2013deep`, the hierarchical extension of Gaussian processes (GP) created by feeding the output of one GP into the next.

GPflux uses the mathematical building blocks from `GPflow <http://www.gpflow.org/>`_ :cite:p:`gpflow2020` and marries these with the powerful layered deep learning API provided by `Keras <https://www.tensorflow.org/api_docs/python/tf/keras>`_.
This combination leads to a framework that can be used for:

- researching (deep) Gaussian process models (e.g., :cite:p:`salimbeni2017doubly, dutordoir2018cde, salimbeni2019iwvi`), and
- building, training, evaluating and deploying (deep) Gaussian processes in a modern way, making use of the tools developed by the deep learning community.


Getting started
---------------

We have provided multiple `Tutorials <tutorials>` showing the basic functionality of the toolbox, and have a comprehensive `API Reference <autoapi/gpflux/index>`.

As a quick teaser, here's a snippet from the `intro notebook <notebooks/intro>` that demonstrates how a two-layer DGP is built and trained with GPflux for a simple one-dimensional dataset:


.. code-block:: python

   # Layer 1
   Z = np.linspace(X.min(), X.max(), X.shape[0] // 2).reshape(-1, 1)
   kernel1 = gpflow.kernels.SquaredExponential()
   inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z.copy())
   gp_layer1 = gpflux.layers.GPLayer(
      kernel1, inducing_variable1, num_data=X.shape[0], num_latent_gps=X.shape[1]
   )

   # Layer 2
   kernel2 = gpflow.kernels.SquaredExponential()
   inducing_variable2 = gpflow.inducing_variables.InducingPoints(Z.copy())
   gp_layer2 = gpflux.layers.GPLayer(
      kernel2,
      inducing_variable2,
      num_data=X.shape[0],
      num_latent_gps=X.shape[1],
      mean_function=gpflow.mean_functions.Zero(),
   )

   # Initialise likelihood and build model
   likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
   two_layer_dgp = gpflux.models.DeepGP([gp_layer1, gp_layer2], likelihood_layer)

   # Compile and fit
   model = two_layer_dgp.as_training_model()
   model.compile(tf.optimizers.Adam(0.01))
   history = model.fit({"inputs": X, "targets": Y}, epochs=int(1e3), verbose=0)

The model described above produces the fit shown in Fig 1. For comparison, in Fig. 2 we show the fit on the same dataset by a vanilla single-layer GP model.

.. list-table::

   *  - .. figure:: ./_static/two_layer_fit.png
            :alt: Fit on the Motorcycle dataset of a two-layer deep Gaussian process.
            :width: 90%

            Fig 1. Two-Layer Deep GP

      - .. figure:: ./_static/single_layer_fit.png
            :alt: Fit on the Motorcycle dataset of a single-layer Gaussian process.
            :width: 90%

            Fig 2. Single-Layer GP


Installation
------------

Latest release from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^

To install GPflux using the latest release from PyPI, run

.. code::

   $ pip install gpflux

The library supports Python 3.7 onwards, and uses `semantic versioning <https://semver.org/>`_.

Latest development release from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a check-out of the develop branch of the `GPflux GitHub repository <https://github.com/secondmind-labs/gpflux>`_, run

.. code::

   $ pip install -e .


Join the community
------------------

GPflux is an open source project. We welcome contributions. To submit a pull request, file a bug report, or make a feature request, see the `contribution guidelines <https://github.com/secondmind-labs/gpflux/blob/develop/CONTRIBUTING.md>`_.

We have a public  `Slack workspace <https://join.slack.com/t/secondmind-labs/shared_invite/zt-mjkavx5e-LfePbVegb9lXRA_ZUqTyMA>`_. Please use this invite link if you'd like to join, whether to ask short informal questions or to be involved in the discussion and future development of GPflux.


Bibliography
------------

.. bibliography::
   :all:
