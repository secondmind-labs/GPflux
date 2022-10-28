# GPflux

<!-- TODO: -->
<!-- [![PyPI version](https://badge.fury.io/py/gpflux.svg)](https://badge.fury.io/py/gpflux) -->
<!-- [![Coverage Status](https://codecov.io/gh/secondmind-labs/GPflux/branch/develop/graph/badge.svg?token=<token>)](https://codecov.io/gh/secondmind-labs/GPflux) -->
[![Quality checks and Tests](https://github.com/secondmind-labs/GPflux/actions/workflows/quality-check.yaml/badge.svg)](https://github.com/secondmind-labs/GPflux/actions/workflows/quality-check.yaml)
[![Docs build](https://github.com/secondmind-labs/GPflux/actions/workflows/deploy.yaml/badge.svg)](https://github.com/secondmind-labs/GPflux/actions/workflows/deploy.yaml)

[Documentation](https://secondmind-labs.github.io/GPflux/) |
[Tutorials](https://secondmind-labs.github.io/GPflux/tutorials.html) |
[API reference](https://secondmind-labs.github.io/GPflux/autoapi/gpflux/index.html) |
[Slack](https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw)

## What does GPflux do?

GPflux is a toolbox dedicated to Deep Gaussian processes (DGP), the hierarchical extension of Gaussian processes (GP).

GPflux uses the mathematical building blocks from [GPflow](http://www.gpflow.org/) and marries these with the powerful layered deep learning API provided by [Keras](https://www.tensorflow.org/api_docs/python/tf/keras).
This combination leads to a framework that can be used for:

- researching new (deep) Gaussian process models, and
- building, training, evaluating and deploying (deep) Gaussian processes in a modern way — making use of the tools developed by the deep learning community.


## Getting started

In the [Documentation](https://secondmind-labs.github.io/GPflux/), we have multiple [Tutorials](https://secondmind-labs.github.io/GPflux/tutorials.html) showing the basic functionality of the toolbox, a [benchmark implementation](https://secondmind-labs.github.io/GPflux/notebooks/benchmarks.html) and a comprehensive [API reference](https://secondmind-labs.github.io/GPflux/autoapi/gpflux/index.html).


## Install GPflux

This project is assuming you are using `python3`.

#### For users

To install the latest (stable) release of the toolbox from [PyPI](https://pypi.org/), use `pip`:
```bash
$ pip install gpflux
```
#### For contributors

To install this project in editable mode, run the commands below from the root directory of the `GPflux` repository.
```bash
make install
```
Check that the installation was successful by running the tests:
```bash
make test
```
You can have a peek at the [Makefile](Makefile) for the commands.


## The Secondmind Labs Community

### Getting help

**Bugs, feature requests, pain points, annoying design quirks, etc:**
Please use [GitHub issues](https://github.com/secondmind-labs/GPflux/issues/) to flag up bugs/issues/pain points, suggest new features, and discuss anything else related to the use of GPflux that in some sense involves changing the GPflux code itself. We positively welcome comments or concerns about usability, and suggestions for changes at any level of design. We aim to respond to issues promptly, but if you believe we may have forgotten about an issue, please feel free to add another comment to remind us.

### Slack workspace

We have a public [Secondmind Labs slack workspace](https://secondmind-labs.slack.com/). Please use this [invite link](https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw) and join the #gpflux channel, whether you'd just like to ask short informal questions or want to be involved in the discussion and future development of GPflux.


### Contributing

All constructive input is very much welcome. For detailed information, see the [guidelines for contributors](CONTRIBUTING.md).


### Maintainers

GPflux was originally created at [Secondmind Labs](https://www.secondmind.ai/labs/) and is now actively maintained by (in alphabetical order)
[Vincent Dutordoir](https://vdutor.github.io/) and
[ST John](https://github.com/st--/).
**We are grateful to [all contributors](CONTRIBUTORS.md) who have helped shape GPflux.**

GPflux is an open source project. If you have relevant skills and are interested in contributing then please do contact us (see ["The Secondmind Labs Community" section](#the-secondmind-labs-community) above).

We are very grateful to our Secondmind Labs colleagues, maintainers of [GPflow](https://github.com/GPflow/GPflow), [Trieste](https://github.com/secondmind-labs/trieste) and [Bellman](https://github.com/Bellman-devs/bellman), for their help with creating contributing guidelines, instructions for users and open-sourcing in general.


## Citing GPflux

To cite GPflux, please reference our [arXiv paper](https://arxiv.org/abs/2104.05674) where we review the framework and describe the design. Sample Bibtex is given below:

```
@article{dutordoir2021gpflux,
    author = {Dutordoir, Vincent and Salimbeni, Hugh and Hambro, Eric and McLeod, John and
        Leibfried, Felix and Artemev, Artem and van der Wilk, Mark and Deisenroth, Marc P.
        and Hensman, James and John, ST},
    title = {GPflux: A library for Deep Gaussian Processes},
    year = {2021},
    journal = {arXiv:2104.05674},
    url = {https://arxiv.org/abs/2104.05674}
}
```


## License

[Apache License 2.0](LICENSE)
