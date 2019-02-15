# `coremdlr`

This repo has been split from the original `corebreakout` package to provide just the modeling components of the project. It is assumed that datasets have already been created or are available from `corebreakout` (which is now the just segmentation + labeling tools).


## Goals for Functionality

`coremdlr` aims to facilitate research on the modeling of lithology and facies in core + log datasets using both deep learning / computer vision and traditional machine learning approaches.

## Requirements

Base requirements:

-   `Python 3.x`
-   `Numpy` + `Scipy` + `Matplotlib`
-   `scikit-image`
-   `scikit-learn`
-   `tensorflow.keras`

Individual scripts and notebooks may require some other libraries.

## Setup Notes

Submodules in this package use absolute imports. You should be able to
use `pip` to do a local develop mode install:

```bash
$ pip install -e path/to/coremdlr
```

This is the recommended option, because it will automatically try to
install any dependencies you don't already have -- but alternatively,
you can add the root directory add to your `PYTHONPATH`:

```bash
$ cd path/to/coremdlr
$ export PYTHONPATH=$PYTHONPATH:`pwd`
```

## Data

TODO

## Code & Module Organization

TODO

