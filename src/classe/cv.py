from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.decomposition import PCA
from umap import UMAP


def _identity(var):
    return var


class PCAUMAP:
    """Dimensional reduction object that combine's PCA post processing with
    UMAP dimensional reduction.

    Implements the sklearn fit/transform API.
    """

    default_pca_params = {"n_components": 5}
    default_umap_params = {"n_neighbors": 5, "n_components": 2}

    def __init__(
        self,
        pca_params=None,
        umap_params=None,
    ):
        """Initializes object.

        Arguments
        ---------
        pca_params: dictionary
            passed to sklearn PCA initialization as option.
        umap_params: dictionary
            passed to umap-learn UMAP initialization as options.
        """
        if pca_params is None:
            pca_params = PCAUMAP.default_pca_params
        if umap_params is None:
            umap_params = PCAUMAP.default_umap_params
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
            regressor = KNR(n_neighbors=5)

        self.featurizer = transfer_featurizer
        self.reducer = reducer
        self.regressor = regressor
        self.ref_cv_vals = None

    def fit(self, data):
        """See sklearn's PCA.fit method."""
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
