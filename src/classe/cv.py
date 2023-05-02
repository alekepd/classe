"""Provides classes for creating collective variables.
"""

from copy import deepcopy
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.decomposition import PCA
from umap import UMAP

_default_cv_options = {
    "pca_n_components": 20,
    "umap_n_neighbors": 8,
    "umap_n_components": 2,
}

PCA_PRE = "pca_"
UMAP_PRE = "umap_"

DEFAULT_KNR_NNEIGH = 7


# identity function
def _identity(var):
    return var


class PCAUMAP:
    """Dimensional reduction object that combines PCA preprocessing with
    UMAP dimensional reduction.

    Implements the sklearn fit/transform API.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """Initializes object.

        Arguments
        ---------
        kwargs: dictionary
            Collection of arguments of form {function_option:value}l
            options prefixed with "pca_" are stripped of prefix and
            passed to PCA, options prefixed with "umap_" are stripped
            of prefix and passed to UMAP.
        """

        args = deepcopy(_default_cv_options)

        pca_params = {}
        umap_params = {}

        keys = list(args.keys())
        for key in keys:
            if key.find(PCA_PRE) == 0:
                derived_key = key.removeprefix(PCA_PRE)
                value = args.pop(key)
                pca_params.update({derived_key: value})
            if key.find(UMAP_PRE) == 0:
                derived_key = key.removeprefix(UMAP_PRE)
                value = args.pop(key)
                umap_params.update({derived_key: value})
        if len(args) > 0:
            raise ValueError(f"Unrecognized argument: {args}")

        self.pca = PCA(**pca_params)
        self.umap = UMAP(**umap_params)

    def fit(self, feats):
        """See sklearn's PCA.fit method."""

        self.pca.fit(feats)
        pca_out = self.pca.transform(feats)
        self.umap.fit(pca_out)

    def transform(self, feats):
        """See sklearn's PCA.transform method."""

        pca_out = self.pca.transform(feats)
        return self.umap.transform(pca_out)


class TransferCV:
    """Reduces a dataset using dimensional reduction, and then trains a
    regressor to reproduce this reduction. Useful when dimensional reduction
    must be applied outside the training set. Implements sklearn
    fit/transform syntax, and add transfer_transform which convenient for
    SHAP analysis.
    """

    def __init__(
        self,
        transfer_featurizer=_identity,
        reducer=None,
        regressor=None,
    ):
        """Initializes object.

        Arguments
        ---------
        transfer_featurizer: callable (default: _identity)
            When calling transfer_transform, the data is first transformed using
            this callable. In practice, it is likely the "shaper" function which
            is outputted during classification analysis, but can be anything.
        reducer: object implementing sklearn fit/transform API (default: PCAUMAP
                 instance)
            Function used for original dimensional reduction.
        regressor: object implementing sklearn fit/predict API (default: KNR
                 instance)
            Function used to emulate the output of reducer.
        """

        if reducer is None:
            reducer = PCAUMAP()
        if regressor is None:
            regressor = KNR(n_neighbors=DEFAULT_KNR_NNEIGH)

        self.featurizer = transfer_featurizer
        self.reducer = reducer
        self.regressor = regressor
        self.ref_cv_vals = None

    def fit(self, data):
        """See sklearn's PCA.fit method.

        Note that the self.featurizer (transfer_featurizer) is not applied here.
        """

        self.reducer.fit(data)
        self.ref_cv_vals = self.reducer.transform(data)
        self.regressor.fit(data, self.ref_cv_vals)

    def transfer_transform(self, data):
        """Similar to the transform method, but applies the transfer_featurizer
        set at __init__ before transforming.

        Primarily useful if performing SHAP analysis and wish to predict on new
        configurations which are described using structural variables. In this
        case, you may provide the shaper (output during comparison) as this
        featurizer and then apply the trained object directly on the structural
        features.
        """

        pretransformed = self.featurizer(data)
        return self.regressor.predict(pretransformed)

    def transform(self, data):
        """See sklearn's PCA.fit transform."""

        return self.regressor.predict(data)
