# `coremdlr`

*Note, this repo is part of research that has been submitted to **[Frontiers Earth Science](https://www.frontiersin.org/journals/earth-science)**, the paper name and DOI will be included once released!*

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

Pre-processing of UK core image data, check out **[CoreBreakout](https://joss.theoj.org/papers/10.21105/joss.01969.pdf)**.

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

**`coremdlr.config`** : default paths / labels / dataset args / viz properties

**`coremdlr.datasets`** : loading / preprocessing / generating data

**`coremdlr.experiments`** : modeling experiment scripts / notebooks 

**`coremdlr.models`** : hierarchical set of generic model classes

**`coremdlr.networks`** : `tf.keras` network creation functions

**`coremdlr.layers/.ops`** : `tf.keras` custom layers and tensor operations

**`coremdlr.viz`** : core data and analysis (e.g., conf. matrix) plotting

## Data

Current data consists of 12 UK Contiential Shelf wells from Q204 and Q205. Please check out the data folder for more information and licensing. 
