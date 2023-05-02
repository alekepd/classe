"""Provides routines for comparing two data frames characterizing trajectories
using classification and shap values.
"""

import pandas as pd
import lightgbm as lgbm
import numpy as np
import shap

kb = 1.987204259 * (10**-3)  # kcal/(mol K)

# keys for values in returned dictionary
TABLE_KEY = "table"
FEATCOL_KEY = "feature_names"
SHAPCOL_KEY = "shap_names"
SHAPER_KEY = "shaper"
OTHERCOL_KEY = "other"

# column names for output table
INTRAIN_CKEY = "in_train"
DELTAU_CKEY = "delta_u"
FAKELABEL_CKEY = "fake_label"
TREEOUT_CKEY = "tree_output"

# trajectory position array shapes: n_frames, n_atoms, n_dim
# feature array shapes: n_cases, n_features
# indices of tables matter for identifying samples

_default_lgbm_options = {
    "num_leaves": 5,
    "objective": "binary",
    "metric": ["binary", "l2", "binary_error"],
    "boosting": "dart",
    "n_iter": 500,
    "max_bin": 6150,
    "feature_fraction": 0.80,
    "drop_rate": 0.5,
    "max_drop": 1000,
    "bagging_freq": 50,
    "bagging_fraction": 0.15,
}


# prefix used for naming shap columns based on feature columns
SHAP_PRE = "s-"


def shapify_name(name):
    """Takes a variable name and returns a version for its corresponding SHAP
    value.
    """

    return SHAP_PRE + name


def deshapify_name(name):
    """Transforms a shap'ed variable name back to its original."""

    if len(name) < 4:
        raise ValueError("Not a shaped variable name; too short.")
    if name.find("s-") != 0:
        raise ValueError("Not a shaped variable name; bad prefix.")
    return name[2:]


_test_string = "abcd-efg-92.g_h"
assert deshapify_name(shapify_name(_test_string)) == _test_string


def compare_tables(
    table_0,
    table_1,
    temperature,
    train_fraction=0.5,
    lgbm_options=None,
    return_label_lists=True,
):
    """Create a shap table from feature tables. Performs tree classification
    between data in table_0 and table_1, and then extracts SHAP values.

    (Free) energy values are in kcal/(mol * K).

    Arguments
    ---------
    table_0: panda DataFrame
        Structural (featurized) data from ensemble 0 (0-indexed). Columns should
        only contain feature values, index should be frame number.
    table_1: panda DataFrame
        Structural data (featurized) from ensemble 1 (0-indexed). Columns should
        only contain feature values, index should be frame number.
    temperature: positive real
        Temperature of the system in Kelvin. Converted to J via module variable
        `kb` when calculating delta_u.
    train_fraction: float (between 0 and 1, default: 0.5)
        Fraction of data to use as training set. Remainder is use as validation
        set (score shown during training).

        NOTE: If using DART (this is default), although the validation loss is
        shown during training LGBM does not do automatic stopping (so pay
        attention to make sure you aren't overfitting).
    lgbm_options: dictionary (passed to lgbm.train as options,
                  default: _default_lgbm_options module variable)
        This dictionary is passed to lgbm.train as options.
    return_label_lists: boolean (default: True)
        Modifies what is returned. See Return.

    Return
    ------
    if return_label_lists a dictionary with the following keys (all strings):
        table
            panda DataFrame which contains classification output, shap values,
            and input features
        feature_names
            List of input feature column names
        shap_names
            List of shap column names
        other_names
            List of other column names
    else:
        only the shap data frame

    The shap data frame has many columns. Features are included with names
    unchanged; shap values are the same as features but prefixed with "s-".
    delta_u uses the trained model to estimate the pointwise difference in the
    PMFs/log densities of the provided data tree_output notes the original
    output of the model (between 0 and 1).  fake_label is primarily for
    debugging and notes the label that was used internally for a given example.
    """

    if lgbm_options is None:
        lgbm_options = _default_lgbm_options

    kbt = temperature * kb

    feature_names = table_0.columns

    # create full and balanced dataframes. We train on the balanced one,
    # then use model to create labels for full frame.
    frame, frame_b = combine_tables(table_0, table_1, force_balance="both")

    # mark who is in train vs test
    mask = np.full(len(frame_b), False)
    to_change = np.random.choice(
        list(range(len(mask))), np.int(len(mask) * train_fraction), replace=False
    )
    mask[to_change] = True
    frame_b[INTRAIN_CKEY] = mask

    # make lgbm dataset objects
    frame_train = frame_b[frame_b[INTRAIN_CKEY]]
    frame_test = frame_b[~frame_b[INTRAIN_CKEY]]
    ds_train = lgbm.Dataset(frame_train[feature_names], frame_train[FAKELABEL_CKEY])
    ds_test = lgbm.Dataset(
        frame_test[feature_names], frame_test[FAKELABEL_CKEY], reference=ds_train
    )

    # train model
    trees = lgbm.train(lgbm_options, train_set=ds_train, valid_sets=[ds_test])

    # associate train labels from subsetted data to original frame
    frame[INTRAIN_CKEY] = frame_b[INTRAIN_CKEY]
    frame[INTRAIN_CKEY].fillna(False)

    # get tree output for full data frame
    frame[TREEOUT_CKEY] = trees.predict(frame[feature_names])

    nonfeature_names = list(set(frame.columns) - set(feature_names))

    # get shap value generator (not values themselves)
    explainer = shap.TreeExplainer(trees)

    # get shap values
    frame_shap = pd.DataFrame(
        kbt * (explainer.shap_values(frame[feature_names])[0]), dtype=np.float32
    )

    # rename and put label columns
    frame_shap.columns = [shapify_name(ob) for ob in feature_names]
    shap_names = frame_shap.columns
    frame_shap.index = frame.index
    frame_shap = pd.concat([frame, frame_shap], axis=1)
    frame_shap[DELTAU_CKEY] = -kbt * np.log(
        frame_shap[TREEOUT_CKEY] / (1 - frame_shap[TREEOUT_CKEY])
    )

    if return_label_lists:
        agg = {
            TABLE_KEY: frame_shap,
            FEATCOL_KEY: feature_names,
            SHAPCOL_KEY: shap_names,
            OTHERCOL_KEY: nonfeature_names,
            SHAPER_KEY: lambda x: explainer.shap_values(x)[0],
        }
        return agg
    return frame_shap


def combine_tables(table_0, table_1, force_balance=False, add_label=FAKELABEL_CKEY):
    """Nicely concatenate two data tables, optionally balancing number of
    records coming from each argument.

    Arguments
    ---------
    table_0: panda DataFrame
        table of first (featurized) trajectory. Index should be the frame number.
        Columns are features.
    table_1: panda DataFrame
        table of second (featurized) trajectory. Index should be the frame number.
        Columns are features.
    force_balance: boolean or 'both' (default: False)
        If true, the classes are balanced (so that there is an approximately equal
        number of samples from table_0 and table_1). Index values correctly
        identify the records used.
    add_label: string or None (default: FAKELABEL_CKEY)
        If not None, then a new column is added using this argument as its name.
        This column contains a 0 or 1 based on whether a record came from
        table_0 or table_1.

    Returns
    -------
    If force_balance is truthy, tuple with first element the entire combined
    DataFrame and second element the balanced frame. If false, only the
    combined frame.

    Returned frame has a hierarchical index, with outermost later using 0 and 1
    to signify which frame a given record comes from, and the inner index
    preserves the indices used in the original tables.
    """

    # make combined panda
    safe_t0 = table_0.copy()
    safe_t1 = table_1.copy()
    if add_label is not None:
        safe_t0[add_label] = 0
        safe_t1[add_label] = 1
    frame = pd.concat([safe_t0, safe_t1], keys=[0, 1])

    # random sample rows to make the balanced table
    nsamples = frame[FAKELABEL_CKEY].value_counts().min()
    grouped = frame.groupby(FAKELABEL_CKEY, as_index=False)
    frame_b = grouped.apply(lambda x, n: x.sample(n), n=nsamples)
    frame_b.index = frame_b.index.droplevel(0)

    if force_balance == "both":
        return (frame, frame_b)
    if force_balance is True:
        return frame_b
    if force_balance is False:
        return frame
    raise ValueError("force_balance argument unclear.")


def compare_tables_cv(table_0, table_1, n_folds=5, lgbm_options=None):
    """Performs cross validation using the lightgbm library. Useful for determining
    the number of trees to use. Note that only the fold-averaged losses (and their
    standard deviations) as a function of tree number are returned.

    Arguments
    ---------
    table_0: panda DataFrame
        table of first (featurized) trajectory. Index should be the frame number.
        Columns are features.
    table_1: panda DataFrame
        table of second (featurized) trajectory. Index should be the frame number.
        Columns are features.
    n_folds: positive integer (default: 5)
        Number of cross validation folds
    lgbm_options: dictionary
        Options passed to lightgbm

    Return
    ------
    Dictionary of metrics as a function iteration. See lightgbm's cv function for
    more information.

    NOTE
    ----
    This procedure evaluates the performance of a model building procedure where all
    parameters but number of trees is constant. A single run can tell you the
    correct number of trees to use, but if you want to optimize over other
    hyperparameters it will require multiple calls. Furthermore, no trained model is
    returned.
    """

    if lgbm_options is None:
        lgbm_options = _default_lgbm_options

    feature_names = table_0.columns

    frame_b = combine_tables(table_0, table_1, force_balance=True)
    dataset = lgbm.Dataset(frame_b[feature_names], frame_b[FAKELABEL_CKEY])

    # train model cv
    trees = lgbm.cv(lgbm_options, train_set=dataset, nfold=n_folds, stratified=False)

    return trees


def summarize_shap_table(table, column_names, quantile=None):
    """Provide summary statistics for a shap table.

    Arguments
    ---------
    table: panda DataFrame
        Data to summarize. Subsetted using column_names.
    column_names: list of strings or panda index
        Selection of columns to summarize
    quantile: float between 0 and 1 or None (default: None)
        If not None, then the quantile over the absolute values is returned;
        else, the mean is.

    NOTE: No balancing of classes is done.

    Return
    ------
    Mean or quantile of the absolute values of each column
    """

    frame = table[column_names]
    if quantile is None:
        return frame.abs().apply(lambda x: x.mean(), axis=0)
    return frame.abs().apply(lambda x: x.quantile(quantile), axis=0)
