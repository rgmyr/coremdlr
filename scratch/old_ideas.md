# `coremdlr/models_facies`:  Facies Modeling

**Labeled data**:

- Have a little bit of manual label data on the `Apache` core.
- Have April's labels on ~415 feet of the original `CoreLab` core.
- Working on new set of British cores

## Current Ideas

See `Papers/texture` folder on Dropbox.

Since most image features are low-level (e.g., small shapes and textures), I'm looking into texture-specific methods for image feature extraction (and fine-grained classification, which has a lot of the same issues). Until recently, state of the art performance on texture classification datasets was achieved by using classification methods (usually SVM) on *improved Fisher vector* encodings (or *improvements* on IFV) of CNN-extracted features (usually pre-trained). It should be trivial to use encodings as input to a recurrent model, or directly as a part of a longer feature vector that includes well log data. 

Current state-of-the-art on most standard texture datasets looks to be a CVPR 2018 paper: [Deep Texture Manifold for Ground Terrain Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf). The same group released a [PyTorch implementation of the Texture Encoding Layer](http://hangzh.com/PyTorch-Encoding/experiments/texture.html) from their previous Paper.

### CNN Features

There is a lot of literature showing that local features extracted from convolutional layers of a CNN are useful for fine-grained and texture-based image classification (fully connected layer features are useful for more generic object classification).

**Encoding by pooling**: [The Treasure beneath Convolutional Layers: Cross-convolutional-layer Pooling for Image Classification](https://arxiv.org/abs/1411.7466)

- Think of a convolutional layer activation of size `H x W x D` as subdividing the input image into `H x W` regions, each represented by a `D`-dimensional vector of local features, or a **feature vector in a spatial unit** 
- Fully connected layers transform spatial feature maps into a global (whole image) feature map -- so using fully connected activations as local features introduces significant domain mismatch
- Cross-convolutional-layer pooling for a layer `t` with `D_t` channels creates a `(D_t+1, D_t)`-dimensional global feature representation. For each activation map (channel) `k` in layer `t+1`, we sum over the `D_t`-dimensional local feature vectors of `t` weighted the values in `k` corresponding to the spatial location of each feature vector. This results in `D_t+1` pooled feature vectors of size `D_t`.
- This is justified by the general observation that individual feature/activation maps are usually sparse and tend to indicate semantically meaningful regions of interest
    - I don't know whether this is the case for purely textural images -- plotting a few random activation maps from each conv block in `MobileNetV2` showed some maps highlighting *semantically* meaningful, but possibly irrelevant regions like artificial cracks and core edges.
    - However, we can look at a lot more of the maps and play with some other weighting schemes or transforms of activation maps (Gaussian weighting, zeroing the highest X% of activations and rescaling, just using some "good" subset of channels, etc.) to find something that reliabily highlights regions we know are meaningful for this problem. 
- PCA is generally used to reduce dimensionality of image representations (or just decorelate them)
- Multi-resolution representations are constructed be resizing image, and spatial resolution can be cheaply increased by optionally breaking images into blocks which are each passed through CNN


### Improved Fisher Vectors

Using the activation-guided pooling method above could be tricky, for the reasons mentioned above. Many papers instead encode local features with Fisher vectors, which is a kind of local pooling.

FV encoding involves fitting a generative model with `k` modes to the distribution of feature descriptors having some dimension `d >> k` (*e.g.*, CNN conv features, SIFT descriptors). Traditionally the model is taken to be a Gaussian Mixture Model, which associates a descriptor `x` to each of the `k` modes with strengths given by the posterior probabilities. Using this strength, the mean and covariance deviance between `x` and each mode `k` can be computed. The FV encoding of `x` is just the vector containing all of these mean and covariance deviance values (`size=2*k`).

*Improved* Fisher vectors improve classification performance by:

- **Power normalization** (`h(z) -> sign(z)*|z|**a`, where `0<=a<=1`): applying a non-linear additive kernel helps with the problem of increased sparsity with increased `k` (making dot product / L2 norm less reliable). Original paper uses `a=0.5` (square root).
- **Normalization**: L2 normalization performs better than simple normalization by number of encoded features.

#### VLFeat

A fast `C` implementation of (improved) [Fisher vector](http://www.vlfeat.org/api/fisher-fundamentals.html) encoding is [available in the VLFeat library](http://www.vlfeat.org/overview/encodings.html), which is callable in Python using the [cyvlfeat](https://github.com/menpo/cyvlfeat) wrappers. The library and wrappers can be installed simultaneously via `conda`:

```
$ conda install -c menpo cyvlfeat
```

Standard usage is to fit a Gaussian Mixture Model to estimate clustered distributions on the raw features, then use the resulting means/covariances/priors to do the Fisher encoding. Something like below, probably with `improved=True`:

```python
import cyvlfeat

X_train.shape == (n_samples, n_features)
x.shape == (1, n_features)

# showing default params
means, covars, priors, LL, posteriors = cyvlfeat.gmm(X_train, n_clusters=10,
                                           max_num_iterations=100, covariance_bound=None,
                                           init_mode='rand', init_priors=None, 
                                           init_means=None, init_covars=None,
                                           n_repetitions=1, verbose=False)

# showing default params, need to use transposes
encoded_x = cyvlfeat.fisher(x.T, means.T, covariances.T, priors.T, normalized=False, 
                           square_root=False, improved=False, fast=False, verbose=False)

# k = 2*n_dim*n_components
encoded_x.shape ==  (k, 1)
```
I've copied `gmm` and fisher` function docstrings at the bottom of this README. 

### (H)SCFVC

For usual vector dimensionalities (200+), there is a related improvement called (Hybrid) Sparse Coding Fisher-like Vector Coding. We can start with regular IFV, and maybe try this as an extension later. See paper in Dropbox.

## Submodule Contents

### Data Creation

Some notebooks and utilities for segmenting/stacking core images and creating labels from associated tables (csv) of facies. Once a better segmentation model gets trained, much this code should be packaged in a base module.

### Notebooks

- `ApacheFirstTrainingSet`: playing with the `xml`-exported labels on the Apache core (used Labelme tool). Grabbed sections of the curve to look at the shapes of the different facies classes used, as well as the corresponding sections of the image. Implemented breakpoint marking on the image (in `viz_utils.py` now, I think?) -- we were considering trying to predict `breakpoints` then do classification on candidate sections between nearby breakpoints.
- Various training notebooks, which will be cleaned up as the set of possible approaches is narrowed
    
### `analysis_images`

Histograms of facies (specific and broad class) distributions, and the curve and image sections corresponding to the labeled sections of the `Apache` core mentioned above.

## Keras CNN Notes


#### Callbacks

Always use at least the `ModelCheckpoint` callback, which saves current best model (by whatever metric specified), rather than just getting the model from the final training epoch. The `EarlyStopping` callback is also a very useful.

#### Multi-GPU Models

For `MobileNetV2` models, multi-GPU speedup is negligible. For `Xception` models, it can definitely help. This is a snippet showing how to build a multi-GPU model in `keras`:

```python
from keras import Model
from keras.utils.training_utils import multi_gpu_model

# fill in Model/Layers setup here

model_build = Model(input=some_input, output=some_prediction)

try:
    model = multi_gpu_model(model_build)
    print("Model Built. Using multiple (data parallel) GPUs.")
except:
    model = model_build
    print("Model Built. Using single GPU.")
    pass

# train model
# save model_build
```

Important to save `model_build` (**not** the distributed `model`) after training.


## `cyvlfeat`

```python
#cyvlfeat.gmm
def gmm(X, n_clusters=10, max_num_iterations=100, covariance_bound=None,
        init_mode='rand', init_priors=None, init_means=None, init_covars=None,
        n_repetitions=1, verbose=False):
    """Fit a Gaussian mixture model

    Parameters
    ----------
    X : [n_samples, n_features] `float32/float64` `ndarray`
        The data to be fit. One data point per row.
    n_clusters : `int`, optional
        Number of output clusters.
    max_num_iterations : `int`, optional
        The maximum number of EM iterations.
    covariance_bound : `float` or `ndarray`, optional
        A lower bound on the value of the covariance. If a float is given
        then the same value is given for all features/dimensions. If an
        array is given it should have shape [n_features] and give the
        lower bound for each feature.
    init_mode: {'rand', 'kmeans', 'custom'}, optional
        The initialization mode:

          - rand: Initial mean positions are randomly  chosen among
                  data samples
          - kmeans: The K-Means algorithm is used to initialize the cluster
                    means
          - custom: The intial parameters are provided by the user, through
                    the use of ``init_priors``, ``init_means`` and
                    ``init_covars``. Note that if those arguments are given
                    then the ``init_mode`` value is always considered as
                    ``custom``

    init_priors : [n_clusters,] `ndarray`, optional
        The initial prior probabilities on each components
    init_means : [n_clusters, n_features] `ndarray`, optional
        The initial component means.
    init_covars : [n_clusters, n_features] `ndarray`, optional
        The initial diagonal values of the covariances for each component.
    n_repetitions : `int`, optional
        The number of times the fit is performed. The fit with the highest
        likelihood is kept.
    verbose : `bool`, optional
        If ``True``, display information about computing the mixture model.

    Returns
    -------
    priors : [n_clusters] `ndarray`
        The prior probability of each component
    means : [n_clusters, n_features] `ndarray`
        The means of the components
    covars : [n_clusters, n_features] `ndarray`
        The diagonal elements of the covariance matrix for each component.
    ll : `float`
        The found log-likelihood of the input data w.r.t the fitted model
    posteriors : [n_samples, n_clusters] `ndarray`
        The posterior probability of each cluster w.r.t each data points.
    """

# cyvlfeat.fisher
def fisher(x, means, covariances, priors, normalized=False, square_root=False,
           improved=False, fast=False, verbose=False):
    """
    Computes the Fisher vector encoding of the vectors ``x`` relative to
    the diagonal covariance Gaussian mixture model with ``means``,
    ``covariances``, and prior mode probabilities ``priors``.
    By default, the standard Fisher vector is computed.
    Parameters
    ----------
    x : [D, N]  `float32` `ndarray`
        One column per data vector (e.g. a SIFT descriptor)
    means :  [F, N]  `float32` `ndarray`
        One column per GMM component.
    covariances :  [F, N]  `float32` `ndarray`
        One column per GMM component (covariance matrices are assumed diagonal,
        hence these are simply the variance of each data dimension).
    priors :  [F, N]  `float32` `ndarray`
        Equal to the number of GMM components.
    normalized : `bool`, optional
        If ``True``, L2 normalize the Fisher vector.
    square_root : `bool`, optional
        If ``True``, the signed square root function is applied to the return
        vector before normalization.
    improved : `bool`, optional
        If ``True``, compute the improved variant of the Fisher Vector. This is
        equivalent to specifying the ``normalized`` and ``square_root` options.
    fast : `bool`, optional
        If ``True``, uses slightly less accurate computations but significantly
        increase the speed in some cases (particularly with a large number of
        Gaussian modes).
    verbose: `bool`, optional
        If ``True``, print information.
    Returns
    -------
    enc : [k, 1] `float32` `ndarray`
        A vector of size equal to the product of
        ``k = 2 * the n_data_dimensions * n_components``.
    """
```
















