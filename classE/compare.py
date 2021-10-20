import pandas as pd
import lightgbm as lgbm
import numpy as np
import shap

kb = 1.380649*(10**-23) #boltzmann constant J/K
kbt = kb*350

# trajectory position array shapes: n_frames, n_atoms, n_dim
# feature array shapes: n_cases, n_features

# indices of tables matter for identifying samples

_default_lgbm_options = {'num_leaves':6,
                         'objective':'binary',
                         'metric':['binary','l2','binary_error'],
                         'boosting':'dart',
                         'feature_fraction':0.5,
                         'n_iter':100}

def compare_distance_tables(table_0,table_1,
                            train_fraction=0.5,force_balance=False,
                            lgbm_options=_default_lgbm_options,
                            return_label_lists=True):
    """Create a shap table using structure tables. Performs tree classification
    between data in table_0 and table_1, and then extracts SHAP values.

    Arguments
    ---------
    table_0: panda DataFrame
        Structural data from ensemble 0 (0-indexed). Columns should only contain
        feature values, index should be frame number.
    table_1: panda DataFrame
        Structural data from ensemble 1 (0-indexed). Columns should only contain
        feature values, index should be frame number.
    train_fraction: float (between 0 and 1, default: 0.5)
        Fraction of data to use as training set. Remainder is use as validation
        set (score shown during training). 

        NOTE: If using DART (this is default), although the validation loss is
        shown during training LGBM does not do automatic stopping (so pay
        attention to make sure you aren't overfitting).
    force_balance: boolean (default: False)
        If true, before training the input tables are made to have the same
        number of samples via downsampling.
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
        only the shap df

    shap df has many columns. Feature are included with names unchanged; shap
    values are the same as features but prefixed with "s-". delta_u uses the
    trained model to estimate the pointwise difference in the PMFs/log densities
    of the provided data (note that this uses the module variable kbt);
    tree_output notes the original output of the model (between 0 and 1).
    fake_label is primarily for debugging and notes the label that was used
    internally for a given example.
    """

    feature_names = table_0.columns

    #make combined panda
    safe_t0 = table_0.copy()
    safe_t1 = table_1.copy()
    safe_t0['fake_label'] = 0
    safe_t1['fake_label'] = 1
    df = pd.concat([safe_t0,safe_t1],keys=[0,1])

    #makes classes balanced if desired
    if force_balance:
        nsamples = df['fake_label'].value_counts().min()
        grouped = df.groupby('fake_label',as_index=False)
        df_b = grouped.apply(lambda x,n: x.sample(n),n=nsamples)
        df_b.index = df_b.index.droplevel(0)
    else:
        df_b = df

    #mark who is in train vs test
    mask = np.full(len(df_b),False)
    to_change = np.random.choice(list(range(len(mask))),
                                 np.int(len(mask)*train_fraction))
    mask[to_change] = True
    df_b['in_train'] = mask

    #make lgbm dataset objects
    df_train = df_b[df_b['in_train']]
    df_test = df_b[~df_b['in_train']]
    ds_train = lgbm.Dataset(df_train[feature_names],df_train['fake_label'])
    ds_test = lgbm.Dataset(df_test[feature_names],
                           df_test['fake_label'],
                           reference=ds_train)

    #train model
    trees = lgbm.train(lgbm_options,
                       train_set=ds_train,
                       valid_sets=[ds_test])

    #associate train labels from subsetted data to original frame
    df['in_train'] = df_b['in_train']
    df['in_train'].fillna(False)

    #get tree output
    df['tree_output'] = trees.predict(df[feature_names])

    nonfeature_names = list(set(df.columns)-set(feature_names))

    #get shap value generator (not values themselves)
    explainer = shap.TreeExplainer(trees)

    #get shap values
    df_shap = pd.DataFrame(explainer.shap_values(df[feature_names])[0],
                           dtype=np.float32)

    #rename and put label columns
    df_shap.columns = ["s-"+ob for ob in feature_names]
    shap_names = df_shap.columns
    df_shap.index = df.index
    df_shap = pd.concat([df, df_shap], axis=1)
    df_shap['delta_u'] = \
        kbt*np.log(df_shap['tree_output']/(1-df_shap['tree_output']))

    if return_label_lists:
        d = {"table":df_shap,
             "feature_names":feature_names,
             "shap_names":shap_names,
             "other_names":nonfeature_names
             }
        return d
    else:
        return df_shap

def summarize_shap_table(table,column_names):
    """Provide summary statistics for a shap table.

    Arguments
    ---------
    table: panda DataFrame
        Data to summarize. Subsetted using column_names.
    column_names: list of strings or panda index
        Selection of columns to summarize

    NOTE: No balancing of classes is done.

    Return
    ------
    Mean absolute value of each column
    """

    df = table[column_names]
    return df.abs().apply(lambda x: x.mean(),axis=0)

