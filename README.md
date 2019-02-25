# `coremdlr`

This repo has been split from the original `corebreakout` package to provide just the modeling components of the project. It is assumed that datasets have already been created or are available from `corebreakout` (which is now the just segmentation + labeling tools).

## Functionality

`coremdlr` aims to facilitate research on the modeling of lithology and facies in core photo + well log datasets using both deep learning / computer vision and traditional machine learning approaches.

## Requirements

Base requirements:

-   `Python 3.x`
-   `Numpy` + `Scipy` + `Matplotlib`
-   `scikit-image`
-   `scikit-learn`
-   `tensorflow.keras`

Individual scripts and notebooks may require some other libraries.

## Setup Notes

Submodules in this package use absolute imports. You should be able to use `pip` to do a local develop mode install:

```bash
$ pip install -e path/to/coremdlr
```

This is the recommended option since it will automatically try to install any dependencies you don't already have. But alternatively, you can add the root directory add to your `PYTHONPATH`:

```bash
$ cd path/to/coremdlr
$ export PYTHONPATH=$PYTHONPATH:`pwd`
```

## Code & Module Organization

**`coremdlr.datasets`** : loading / preprocessing / generating data

**`coremdlr.models`** : heirarchical set of generic model classes

**`coremdlr.experiments`** : experiment scripts/notebooks 

**`coremdlr.networks`** : `tf.keras` network creation functions

**`coremdlr.layers/.ops`** : `tf.keras` custom layers and tensor operations

**`coremdlr.config`** : configuration of default paths / dataset args / viz properties

**`coremdlr.viz`** : core data and analysis (e.g., conf. matrix) plotting

## Data

Current data consists of 12 North Sea wells. Need to work on standardizing label scheme.

