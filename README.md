# `coremdlr`

*Note, this repo is part of research that has been submitted to **[Frontiers Earth Science](https://www.frontiersin.org/journals/earth-science)**, the paper name and DOI will be included once released!*

This repo contains code for modeling lithology and facies in core photo + well log datasets, using both deep learning / computer vision and traditional machine learning approaches.

For pre-processing of UK core image data, check out **[CoreBreakout](https://joss.theoj.org/papers/10.21105/joss.01969.pdf)**.


## Requirements

Base requirements:

-   `Python 3.x`
-   `Numpy` + `Scipy` + `Matplotlib`
-   `scikit-image`
-   `scikit-learn`
-   `tensorflow.keras`

Individual scripts and notebooks may require some other libraries. The requirements list is currently incomplete - aiming to fix that.


## Installation

Submodules in this package use absolute imports. You should be able to use `pip` to do a local develop mode install:

```bash
$ pip install -e path/to/coremdlr
```


## Organization

The `coremdlr` module contains a number of submodules:

**`coremdlr.config`** : settings and default paths / labels / dataset args / viz properties

**`coremdlr.datasets`** : loading / preprocessing / generating data

**`coremdlr.models`** : hierarchical set of generic model classes

**`coremdlr.networks`** : `tf.keras` network construction functions

**`coremdlr.layers/.ops`** : `tf.keras` custom layers and tensor operations

**`coremdlr.viz`** : plotting data and analysis (e.g., confusion matrices)

---

The final experiments for the paper mostly took place in `experiments`, and more specifically the `notebooks_*` subdirectories.

---

The `notebooks` folder contains some random notebooks and a `figures` subdirectory in which paper figures were generated.


## Data

Current data consists of 12 UK Contiential Shelf wells from Q204 and Q205. Please check out the data folder for more information and licensing.
