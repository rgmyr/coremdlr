# Dataset Docs

**See docstrings for specific usage details.**

User should only need to directly construct a `FaciesDataset` object. The `WellLoader` class for individual wells exists mostly for code separation/readability and programmer convenience. That said, it can be useful to inspect a single `WellLoader` instance if there are any data issues with unclear origins.

### `FaciesDataset`

Dataset container for `image`/`pseudoGR`/`logs` data. Instantiates `WellLoader`s for each well in `wells` and `test_wells`, splits into `train`/`test` sets of sections aggregated and labeled at `label_resolution`.


### `WellLoader`

Object responsible for loading an individual well, along with any features from that well.


### `FeatureScaler`


### `PseudoExtractor`
