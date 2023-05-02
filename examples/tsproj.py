#!/usr/bin/env python3
"""Command line tool for analyzing binary numpy trajectories using
boosted gradient trees, pairwise distances, SHAP values, and UMAP.

Example usage:

    tsproj.py --traj1 <traj file 1> --traj2 <traj file 2> \
    --temp <temperature> [other options]

will compare <traj file 1> to <traj file 2>, and write the results to a csv.

Many additional options are possible. For example:

    - adding --cv will calculate KNN-PCA-UMAP collective variables for a subset
      of the data.

    - adding --proj-traj <traj file 3> with --cv will project <traj file 3> onto
      the collective variables made using the first two trajectories.

A more complex invocation could look like:

    tsproj.py -1 extr_dda.npy  -2 ref_dda.npy --temp 350 --lz4 --cv \
    --proj_traj model_dda.npy --stats --lgbm_options n_iter 100 \
    --lgbm_options num_leaves 5 \
    --box_diam 200 --cv_max_train 10000

    which would compare extr_dda.npy to ref_dda.npy to each other at 350K.
    Distances would be calculated in a periodic cubic box of 200. Collective
    variables would be generated for extr_dda.npy and model_dda.py, and
    additionally applied to model_dda.py. Certain optiosn for lgbm are
    overridden. Files are compressed using lz4 before writing (requires lz4 to
    be installed)

Call command using -h for more information on possible arguments.
"""

from pathlib import Path
import argparse
from platform import node
from textwrap import dedent
import numpy as np

import pandas as pd
from datetime import datetime
from copy import deepcopy
import classe.featurization as feat
import classe.compare as cc
import classe.cv as cv
from numbers import Number

# Names of command line options
TRAJ1_KEY = "traj1"
TRAJ2_KEY = "traj2"
BOXDIAM_KEY = "box_diam"
PROJTRAJ_KEY = "proj_traj"
MAXDIST_KEY = "max_dist"
BATCHSIZE_KEY = "batch_size"
LGBMOPT_KEY = "lgbm_options"
TRAINF_KEY = "train_frac"
CV_KEY = "cv"
CVOPT_KEY = "cv_options"
CVMAXMAP_KEY = "cv_max_map"
CVMAXTRAIN_KEY = "cv_max_train"
STATSTAB_KEY = "stats"
NSTATS_KEY = "n_stats"
TEMP_KEY = "temp"
LZ4_KEY = "lz4"
GFILESUFF_KEY = "file_suffix"

# Extensions and comment characters to use in csv writing
COMMENT_PF = "# "
LZ4_SUFF = "lz4"
CSV_SUFF = "csv"

# Suffixes used to create written files
SHAP_FILE_TAG = ""
SHAP_SUMMARY_FILE_TAG = "_stats"
SHAP_CV_FILE_TAG = "_cv"

# Names for columns written solely in this command
INCVTRAIN_CKEY = "in_cv_train"


def pr(key):
    """Adds prefix for making command line options."""

    return "--" + key


def parse_args():
    """Parses command line arguments. Returns dictionary."""

    desc = """
        Command line tool for basic analysis of trajectory files.
        Produces and saves SHAP values, SHAP projected coordinates, and
        optionally projects a system onto the derived SHAP coordinates.
        Uses a pairwise distance featurization. Trajectory files must be
        provided as numpy binary files of shape (n_frames, n_atoms, n_dims).
        Example usage: tsproj.py --traj1 <traj file 1> --traj2 <traj file 2>
    """

    parser = argparse.ArgumentParser(
        prog="tsproj",
        description=dedent(desc),
    )
    parser.add_argument(
        "-1",
        pr(TRAJ1_KEY),
        action="store",
        required=True,
        type=Path,
        help="First trajectory to analyze. Compared to traj 2.",
    )
    parser.add_argument(
        "-2",
        pr(TRAJ2_KEY),
        action="store",
        required=True,
        type=Path,
        help="Second trajectory to analyze. Compared to traj 1.",
    )
    parser.add_argument(
        pr(BOXDIAM_KEY),
        action="store",
        default=None,
        required=False,
        type=float,
        help="If set, pairwise distances are corrected for cubic periodic "
        "boundary conditions of this diameter.",
    )
    parser.add_argument(
        pr(TEMP_KEY),
        action="store",
        required=True,
        type=float,
        help="Temperature of system in Kelvin.",
    )
    parser.add_argument(
        pr(CV_KEY),
        action="store_true",
        required=False,
        help="Whether to generate and save PCA-UMAP-KNR (CV) variables.",
    )
    parser.add_argument(
        pr(CVOPT_KEY),
        action="append",
        nargs=2,
        help="Additional arguments to CV generation. Must of be the form "
        "'argument' 'value'.",
    )
    parser.add_argument(
        pr(CVMAXMAP_KEY),
        action="store",
        type=int,
        default=int(5e6),
        required=False,
        help="Maximum number of points map using trained CV.",
    )
    parser.add_argument(
        pr(CVMAXTRAIN_KEY),
        action="store",
        type=int,
        default=int(2e3),
        required=False,
        help="Maximum number of points used to create CV map.",
    )
    parser.add_argument(
        pr(PROJTRAJ_KEY),
        action="append",
        required=False,
        default=[],
        type=Path,
        help="Tertiary trajectory files to project onto derived variables (CVs).",
    )
    parser.add_argument(
        pr(MAXDIST_KEY),
        action="store",
        default=np.Inf,
        type=float,
        help="Maximum pairwise distance to consider per trajectory frame "
        "(larger distances are clipped).",
    )
    parser.add_argument(
        pr(BATCHSIZE_KEY),
        action="store",
        default=50000,
        type=int,
        help="Number of frames to transform to distances in each batch. Larger "
        "values are faster, but require more memory.",
    )
    parser.add_argument(
        pr(LGBMOPT_KEY),
        action="append",
        nargs=2,
        help="Additional arguments to LGBM. Must of be the form 'argument' 'value'.",
    )
    parser.add_argument(
        pr(TRAINF_KEY),
        action="store",
        type=float,
        default=0.8,
        help="Fraction of data to use for LGBM training (between 0 and 1 inclusive).",
    )
    parser.add_argument(
        pr(STATSTAB_KEY),
        action="store_true",
        help="If specified, write table of summarized shap values.",
    )
    parser.add_argument(
        pr(NSTATS_KEY),
        action="store",
        type=int,
        default=7,
        help="Number of structural variables to correlate with and summarize "
        "shap values for.",
    )
    parser.add_argument(
        pr(LZ4_KEY),
        action="store_true",
        help="Write csv files using lz4 compression. Requires lz4 library.",
    )
    parser.add_argument(
        pr(GFILESUFF_KEY),
        action="store",
        type=str,
        default="",
        help="Suffix (before extension) for all saved files. Defaults to no suffix.",
    )
    return vars(parser.parse_args())


def numberify(val):
    if isinstance(val, Number):
        return val
    try:
        if "." in val:
            return float(val)
        else:
            return int(val)
    except (TypeError, ValueError):
        return val


def get_traj(filename):
    """Produces a numpy array from a filename. Produced numpy must be of the
    shape (n_frames,n_atoms,n_dims).
    """

    return np.load(filename)


def get_featurizer(batch_size, distance_max, box_diam):
    """Creates callable featurizer that will be used for analysis. Here uses
    pairwise distances, but could be modified.
    """

    def f(traj):
        return feat.make_distance_table(
            traj,
            batch_size=batch_size,
            distance_max=distance_max,
            box_diam=box_diam,
            sort_key=int,
        )

    return f


def save_filename_stem(filename_1, filename_2):
    """Creates filename core that is used in the libary to name the resulting CV
    files.
    """

    return filename_1.parent / Path(filename_1.stem + "_vs_" + filename_2.stem)


def path_cat(core, suff):
    """Adds suffix to Path object filename.

    Arguments
    ---------
    core (Path):
        Path object to modify
    suff (string):
        Suffix to add

    Return
    ------
    Path object with suff added to end of filename
    """

    return core.parent / (core.name + suff)


def get_filename_map(name_0, name_1, other_names):
    """Creates lookup table mapping names to integers and integers to names.

    Used for writing csv's with real file names. We assume that 0 and 1 are used in the
    output of the classification routines; later indices are used consistently from this
    map.
    """

    fmap = {name_0: 0, name_1: 1, 0: name_0, 1: name_1}
    for ind, name in enumerate(other_names, 2):
        fmap.update({ind: name, name: ind})
    return fmap


def write_table_csv(
    table,
    filename,
    cmdline_options=None,
    lgbm_params=None,
    cv_params=None,
    lz4=False,
    frame_ckey="frame",
    filename_ckey="filename",
    filename_map=None,
):
    """Writes a DataFrame to a csv file with a header and optional lz4 compression.

    lz4 compression may speed up writing for large csvs and reduces disk usage.
    For its use the lz4 package must be installed.

    Arguments
    ---------
    table (pd.DataFrame):
        DataFrame to write to file. If filename_map is not None or False, index
        should be a MultiIndex with two levels: outer level gives the source
        file, inner level gives frame number. See frame_ckey, filename_ckey,
        and filename_map.
    filename (Path)
        Core of location to write file to. Suffixes (.csv or csv.lz4) are added.
    cmdline_options:
        Object representing command line options. Typically a dictionary. Added
        to header using repr.
    lgbm_params (dict):
        Object representing command lgbm options. Typically a dictionary. Added
        to header using repr.
    cv_params (dict):
        Object representing collective variable options. Typically a dictionary.
        Added to header using repr.
    lz4 (boolean):
        If true, lz4 compression is used to compress the csv as it is written
        to disk.  Requires lz4.
    frame_ckey (string):
        Name of column used to represent frame information extracted from index
        of DataFrame (see filename_map).
    filename_ckey (string):
        Name of column used to represent filename information extracted from index
        of DataFrame (see filename_map).
    filename_map (None, dict, or False)
        When saving, the index of table is usually turned into columns under
        the names given by frame_ckey and filename_ckey.

        If filename_map is not None or False, it is used to translate the
        filename index value to a nicer value (e.g., going from an integer to a
        filename) in the form filename_map[index value].

        If filename_map is set to False, the index values are not touched at
        all. This allows for tables which do not have file-frame indices to be
        saved using the same function.
    """

    filename = path_cat(filename, "." + CSV_SUFF)

    if filename_map is not False:
        _filename_ckey = "_" + filename_ckey
        # index into columns
        neatened_table = table.sort_index().reset_index(
            names=[_filename_ckey, frame_ckey]
        )
        # map names
        if filename_map is not None:
            new_column = neatened_table[_filename_ckey].map(filename_map)
        else:
            new_column = neatened_table[_filename_ckey]
        neatened_table.insert(0, filename_ckey, new_column)
        neatened_table.drop(_filename_ckey, axis=1, inplace=True)
    else:
        neatened_table = table

    if lz4:
        import lz4.frame

        def method(fn):
            if fn.suffix != LZ4_SUFF:
                fn = path_cat(fn, "." + LZ4_SUFF)
            return lz4.frame.open(
                fn, "wt", compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC
            )

    else:

        def method(fn):
            return open(fn, "w")

    with method(filename) as f:
        # write metadata and comments
        f.write(COMMENT_PF + datetime.today().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write(COMMENT_PF + node() + "\n")
        f.write(COMMENT_PF + "CMDLINE args: " + repr(cmdline_options) + "\n")
        f.write(COMMENT_PF + "LGBM args: " + repr(lgbm_params) + "\n")
        if cv_params is not None:
            f.write(COMMENT_PF + "CV args: " + repr(cv_params) + "\n")
        neatened_table.to_csv(f, index=False)


def make_lgbm_args(args):
    """Creates arguments for lgbm. Combines default options from library with
    given options.
    """

    default_args = deepcopy(cc._default_lgbm_options)
    if args is not None:
        transformed_args = dict(args)
        default_args.update(transformed_args)
    for key, val in default_args.items():
        default_args[key] = numberify(val)
    return default_args


def make_cv_args(args):
    """Creates arguments for UMAP variable creation. Combines default options
    from library with given options.
    """

    default_args = deepcopy(cv._default_cv_options)
    if args is not None:
        transformed_args = dict(args)
        default_args.update(transformed_args)
    for key, val in default_args.items():
        default_args[key] = numberify(val)

    return default_args


def safe_sample(table, n):
    """Randomly samples without repleacement a DataFrame with sane corner cases.

    If n is smaller than 1, 1 sample is returned. If more random samples than
    rows are asked for, all rows are returned (none are duplicated).
    """

    safe_n = max(n, 1)
    if len(table) > safe_n:
        return table.sample(safe_n, replace=False)
    else:
        return table


def top_variables(shap_table, n, return_stats=False):
    """Returns the names of the _features_ that correspond to best n shap values
    in shap_table. Best is determined by maximal median absolute values.

    Returns
    -------
    If return_stats, return value is (variable_names, stats) where both are
    lists of strings or floats;  Else only variable_names.
    """

    stats = shap_table.apply(lambda x: np.median(np.abs(x)))
    stats.sort_values(ascending=False, inplace=True)
    if n < len(stats):
        best_shaps = stats[:n]
    else:
        best_shaps = stats
    names = [cc.deshapify_name(x) for x in best_shaps.index]
    if return_stats:
        return (names, list(best_shaps))
    else:
        return names


def add_index_mask(table, index, mask_column):
    """Adds column to table specifying which rows have index values included in
    index.

    True signifies included. Operates in place.
    """

    table[mask_column] = False
    table.loc[index, mask_column] = True


def make_transfer_cv_table(feat_tab, transfer_cv, desc_features, cv_prefix="CV_"):
    """Creates table of values mapped onto pre-created collective variables and
    annotates with important features and shap values.

    Arguments
    ---------
    feat_tab (pd.DataFrame):
        features to map onto collective variables and summarize
    transfer_cv (cv.PCAUMAP):
        pre-created CV mapper
    desc_features (list of strings):
        list of feature names in feat_tab that are important
    cv_prefix (string):
        String prefixed when creating names of CV columns

    Return
    ------
    pd.Dataframe with columns for collective variables, the n most important
    features, and the n most important feature's shap values. Index matches that
    of feat_tab
    """

    # instead transfer_transform we featurize manually so we can save features
    # and then transform
    shap_tab = pd.DataFrame(transfer_cv.featurizer(feat_tab))
    shap_tab.index = feat_tab.index
    shap_tab.columns = [cc.shapify_name(x) for x in feat_tab.columns]

    cvs = pd.DataFrame(transfer_cv.transform(shap_tab))
    cvs.index = feat_tab.index
    cvs.columns = [cv_prefix + str(x) for x in cvs.columns]

    subbed_feat_tab = feat_tab.loc[:, desc_features]
    subbed_shap_tab = shap_tab.loc[:, [cc.shapify_name(x) for x in desc_features]]
    return pd.concat([cvs, subbed_feat_tab, subbed_shap_tab], axis=1)


def main():
    options = parse_args()

    # make featurizer
    feater = get_featurizer(
        batch_size=options[BATCHSIZE_KEY],
        distance_max=options[MAXDIST_KEY],
        box_diam=options[BOXDIAM_KEY],
    )

    # create feature table using pairwise distances
    feat_1 = feater(get_traj(options[TRAJ1_KEY]))
    feat_2 = feater(get_traj(options[TRAJ2_KEY]))

    # run classe analysis
    lgbm_params = make_lgbm_args(options[LGBMOPT_KEY])
    analysis = cc.compare_tables(
        feat_1,
        feat_2,
        temperature=options[TEMP_KEY],
        train_fraction=options[TRAINF_KEY],
        lgbm_options=lgbm_params,
    )

    save_stem = save_filename_stem(options[TRAJ1_KEY], options[TRAJ2_KEY])

    # this makes assumptions on how 0 and 1 are returned as index labels.
    # these relationships are documented in the main code.
    filename_map = get_filename_map(
        options[TRAJ1_KEY].name, options[TRAJ2_KEY].name, options[PROJTRAJ_KEY]
    )
    write_table_csv(
        analysis[cc.TABLE_KEY],
        filename=path_cat(save_stem, SHAP_FILE_TAG + options[GFILESUFF_KEY]),
        lz4=options[LZ4_KEY],
        lgbm_params=lgbm_params,
        cmdline_options=options,
        filename_map=filename_map,
    )

    # find most important variables
    shap_cols = analysis[cc.SHAPCOL_KEY]
    top_features, top_shap_stats = top_variables(
        analysis[cc.TABLE_KEY][shap_cols], n=options[NSTATS_KEY], return_stats=True
    )

    # write summary table if required
    if options[STATSTAB_KEY]:
        summary_tab = pd.DataFrame(top_shap_stats, index=top_features).T
        write_table_csv(
            summary_tab,
            filename=path_cat(
                save_stem, SHAP_SUMMARY_FILE_TAG + options[GFILESUFF_KEY]
            ),
            lz4=options[LZ4_KEY],
            lgbm_params=lgbm_params,
            cmdline_options=options,
            filename_map=False,
        )

    # if CVs were asked for
    if options[CV_KEY]:
        table = analysis[cc.TABLE_KEY]
        # get points that were not part of the validation set
        # these are not test, but also are still class-balanced.
        val = table.loc[~table[cc.INTRAIN_CKEY].fillna(True), :]

        # design cv using validation points
        cv_params = make_cv_args(options[CVOPT_KEY])
        dimmer = cv.PCAUMAP(**cv_params)
        CV = cv.TransferCV(transfer_featurizer=analysis[cc.SHAPER_KEY], reducer=dimmer)

        # train cv
        subbed = safe_sample(val[shap_cols], options[CVMAXTRAIN_KEY])
        CV.fit(subbed)

        # used for reporting at end
        cv_train_index = subbed.index

        # if we need to project a tertiary trajs
        extraps = []
        # we can only map so many points, so split our budget
        per_frame_map_budget = options[CVMAXMAP_KEY] // (len(options[PROJTRAJ_KEY]) + 1)
        # for filename in enumerate(options[PROJTRAJ_KEY]
        for traj_file in options[PROJTRAJ_KEY]:
            feat_ex = feater(get_traj(traj_file))
            subbed = safe_sample(feat_ex, per_frame_map_budget)
            extrap = make_transfer_cv_table(
                subbed,
                transfer_cv=CV,
                desc_features=top_features,
            )

            # make index a multiindex with outer value of 2 representing that
            # these samples came from the third traj
            label_number = filename_map[traj_file]
            index = pd.MultiIndex.from_frame(
                pd.DataFrame({0: label_number, 1: extrap.index})
            )
            extrap.index = index
            extraps.append(extrap)
        feat_cols = analysis[cc.FEATCOL_KEY]
        # map more of original points using CV
        subbed = safe_sample(table[feat_cols], per_frame_map_budget)
        interp = make_transfer_cv_table(
            subbed,
            transfer_cv=CV,
            desc_features=top_features,
        )

        data = pd.concat([interp] + extraps)

        # mark while samples are knn-extrapolated
        add_index_mask(data, cv_train_index, INCVTRAIN_CKEY)

        # add delta u
        data[cc.DELTAU_CKEY] = table[cc.DELTAU_CKEY]

        write_table_csv(
            data,
            filename=path_cat(save_stem, SHAP_CV_FILE_TAG + options[GFILESUFF_KEY]),
            lz4=options[LZ4_KEY],
            lgbm_params=lgbm_params,
            cmdline_options=options,
            cv_params=cv_params,
            filename_map=filename_map,
        )


if __name__ == "__main__":
    main()
