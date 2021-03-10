==========================================
``coremdlr``: The Core Sample Image Tool
==========================================

**Current Task(s)**:

- In the process of rewriting some of the ``CoreColumn`` code to make better and more flexible (optionally labeled) figures
    - see ``coremdlr/viz_utils.py`` (rewriting function from ``github.com/seg/2016-ml-contest``)
- Now have some labels on Brittany's core to play with for the facies modeling problem 
- Manually labeling segmentation data for training. (Hopefully getting OS + training set up on dual GTX system soon.)
- Need to write some unit tests -- will probably save us some headaches down the road to start now.

Goals for Functionality
=======================

``coremdlr`` aims to facilitate geological research in reservoir characterization by providing:

- **A segmentation pipeline for core sample images**: separating sample material from background across a full directory of images and stacking them into a single column, keeping track of the depth of each pixel row. Currently, this segmentation is carried out in a relatively hard-coded way, which was written to produce accurate segmentations for the boxes with the black ``CoreLab`` template. I am working to build a model that generalizes this part of the tool to operate on core images with a variety of backgrounds and templates.
- **A log curve of grayscale intensity vs. depth**: particularly in unconventional (?) reservoirs, grayscale intensity can serve as a proxy for grain size. This is a useful measure for characterizing the layers and depositional *facies* in the core.
- **Automatic detection and (probabilistic) classification of facies**. Building a reliable model for automatic labeling of processed core images could be a productivity multiplier for geologists, even without being robust enough to trust in completely uncorrected analysis.

Requirements
============

Base requirements:

- ``Python 3.x``
- ``Numpy`` + ``Scipy`` + ``Matplotlib``
- ``scikit-image``

Models requirements:

- ``tensorflow``
- (``keras``, ``scikit-learn``)

Particular scripts and notebooks may require some other libraries as well.

Setup Notes
===========

Submodules in this package use absolute imports. You should be able to use ``pip`` to do a local develop mode install::

    $ pip install -e path/to/coremdlr

This is the recommended option, because it will automatically try to install any dependencies you don't already have -- but alternatively, you can add the root directory should be added to your ``PYTHONPATH``::

    $ export PYTHONPATH=$PYTHONPATH:`pwd`

Data
====

**Note**: Most data is not included on this repo, but some data directories are shown for code clarity. Contact me directly to get all of the necessary data. (Eventually, current-best model weights will probably be uploaded directly to the repo.)

**Current data**:
    - ``CoreLab``: 45 images (~500 feet of core). Core has styrofoam backing and cropped box is oriented at the bottom right of a black template. Thin artificial breaks and plugs present, but no large gaps.
        - Visible light, ``jpg``, size = ``650x841``
        - Visible light and UV, ``tif``, size = ``2550x3300``
    - ``Apache``: 40 images, templated similar to ``CoreLab``
        - Visible light and UV, ``jpg``, size = ``2550x3300``
    - ``E997``: 30 images, cardboard box, "natural" black template
        - Visible light, ``jpg``, size = ``2789x3942``
    - ``Shell``: 51 images, wood box, "natural" black template
        - Visible light, ``jpg``, size = ``2370x3423``
    - ``Conoco``: 105 images, metal(?) box, off-white/blue template
        - Visible light, ``jpg``, size = ``2226x2796``
    - ``StatOil``: 23 images, wood backing strips, off-white template 
        - Visible light, ``jpg``, size = ``1768x3197``

Currently I'm generating some additional segmentation training data from each set using GIMP photo editor. Saving painted images with ``_BlueLabel.jpg`` extension.

Code & Module Organization
==========================

``coremdlr/``:  base classes and utilities. Looking to add ``test`` scripts soon.

``scripts/``: files intended to be executed from the command-line

``notebooks/``: jupyter notebooks for prototyping, demos, etc. (Some are a little bit of a mess.)

``models_segmentation/``: feature engineering and model building for the image segmentation problem

``models_facies/``: feature engineering and model building for the facies classification problem

``data/``: root directory for segmentation input and output data + labeled training images

- **Note**: ``models_`` directories may have their own subdirectories for reformatted or augmented data

Image Segmentation
==================

Current solution is rigidly designed for CoreLab-like black image templates. Possible generalized solutions:

- Superpixel classification/labeling
    - Superpixel hyperparameters are difficult to tune, but they may at least make it easier to manually label new training data (e.g., via an image editor bucket tool)
- Some experimentation with ``tensorflow`` and ``keras`` models is being performed with GPU training via the Google Cloud Compute Engine. (Hopefully soon we can train locally on GTX machine.)
    - `jakeret/tf_unet <https://github.com/jakeret/tf_unet>`_
      - Trying this one first. It has a nice clean interface and should hopefully be more than good enough for binary segmentation.
    - `divamgupta/image-segmentation-keras <https://github.com/divamgupta/image-segmentation-keras>`_
    - `HongyangGao/PixelDCN <https://github.com/HongyangGao/PixelDCN>`_

Have acquired additional datasets of raw core photos, and am working on segmentating these images to use as additional training data.

Facies Modeling
===============

Have a little bit of training data on the ``Apache`` core now. Can begin playing around as we gather more data. Possibilities:

- Scale-invariant (scale-space pyramid) template matching / correlation filter methods (on curve)
  - Is there enough information in the curve? I kind of doubt it, but we can play around with it.
- Break point detection (for region proposal) + various classification models (on curve and/or image)
  - Break point detection might take some clever tuning, but it'd be nice to have options for classification models.
- ``TensorFlow`` Object Detection API (customized region proposal network?) (on image)
  - This would be the SoTA way, but would also require a lot more training data and is computationally expensive (at least to train -- evaluation of a learned model would be plenty fast.)

