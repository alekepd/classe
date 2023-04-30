"""Provides tools for transforming numpy trajectories into pandas.DataFrames
for future analysis.

This module contains both legacy functions for calculating molecular distances,
as well as a more advanced framework for molecular featurization.

Legacy functions
----------------
Function to create DataFrames of distances directly from arrays:
make_distance_table. See function documentation for usage.

Advanced framework
------------------

process transforms a filename or numpy trajectory into a DataFrame of features.
mprocess performs a similar task, but operates on lists of numpy files,
parallelizes their featurization across processors, and then neatly aggregates
the results into a single DataFrame.

Other elements of this module include functions which form the actual
featurization, like rog (radius of gyration).

NOTE: Many featurization functions may have computation that is redundant to
each other. The aim is to provide a parallel application interface which
maintains information about the source of each calculated feature.

Example
-------

Example of using the advanced framework:

from glob import glob
import mdtraj
import featurize as ft

#the cg simulations we analyze are at the resolution of full_cg_atom_names, so
#atomistic reference data sometimes has to be mapped to this resolution.
#However, we are performing much of the resolution at cg_atom_names (which
#here is carbon alpha resolution).

full_cg_atom_names = ['O', 'N', 'CA', 'CB', 'C']
cg_atom_names = ['CA']

#some of the methods need a set of reference coordinates.
#we first extract the atomistic coordinates
ref_traj = mdtraj.load("filename.pdb")

#Then we map to the carbon alpha resolution
ref_traj = ft.atom_filter(ref_traj,atom_list=['CA'])

#We may transform the coordinates to angstrom units from nm (this
#is because of mdtraj units vs the unites typical trajectory files are in.
#YMMV.
ref_pdb_coords = ref_traj.xyz[0,:,:]*10.0

#mprocess takes a list of featurization functions, which we will create and put in
#featers.
featers = []
featers.append(ft.ncurry(ft.gen_dssp_codes,atom_names=full_cg_atom_names))
featers.append(ft.ncurry(ft.rog,atom_names=cg_atom_names))
cut_f = ft.ncurry(ft.sigmoid,
                  shift=6.5,
                  scale=5e-1)
native_f = ft.ncurry(ft.per_atom_frac_native_contacts,
                     reference_pos=ref_pdb_coords,
                     contact_f=cut_f,
                     atom_names=cg_atom_names,
                     dont_count_diagonal=2,
                     normalize_counts=False,
                     )

files = glob("./files/*coords*.npy")

#this is call that analyzes in parallel the files with our featers functions
out=ft.mprocess(files,
                pdb_filename="./source.pdb",
                featers=featers,
                stride=5,
                joblib=True)

#save the resulting DataFrame to a csv file
out.to_csv("test.csv",index=False)
"""

import re
import collections
import pandas as pd
from sklearn.metrics import pairwise_distances as sk_pairwise_distances
import numpy as np
import mdtraj as md
import joblib as jb


def process(array_filename, pdb_filename, featers, stride=1, self_label=False):
    """Goes from the filename of a numpy trajectory and pdb to a panda.Dataframe of
    features, along with indices of timesteps.

    Arguments
    ---------
    array_filename (numpy.ndarray, string, or a single-element dictionary)
        In the case of filename of a .npy or .npz trajectory file, or a
        numpy.ndarray, the resulting array will be passed to the elements
        of @featers. In the case of a dictionary, the length 1 argument
        must have the form of {<name>:<array>}, where <name> is used in
        situations where a filename would be (e.g., self_label).

        Should include an array of dimension 2-4. The array is then transformed
        to be of dimension 4.  If the original dimension is less than 4,
        (in the case of dimension=3) the n_replica index is assumed to be
        omitted; otherwise (in the case of dimension=2) the n_replica and
        n_frame indices are assumed to be omitted. Omitted indices are
        added with a size of 1.

        npz files are assumed to have the coordinates stored under the
        'coords' key.
    pdb_filename (string)
        Name of pdb file which will be directly passed to elements of
        featers
    featers (list of callables)
        List, each element of which is a callable which must only take two
        arguments: array, and pdb_filename. Additionally, the function must
        return a 2-tuple. The first element of the list with the first
        element a list of 2-d arrays, each of which corresponds to a replica.
        These arrays must be of shape (n_frames, signal_dimension). The
        second entry is a list of strings, each of which is a column name
        for each signal dimension in the first tuple element.

        Note that even featers with scalar output _should_ have a
        2-dimensional output shape and (one-element) list of column names,
        although internal panda calls seem to not cause issues if the
        output array is one dimensional.
    stride (positive integer)
        Every stride frames are extracted from pos before processing. Note
        that `local_timestep` correctly will tell you which frame
        came from each array timestep.
    self_label (boolean)
        If true, a homogeneous column is made with column name "filename"
        and array_filename is used as its content, unless array_filename is
        a numpy.ndarray; in this case the column is filled with None

    Return
    ------
    pandas.DataFrame with (possibly) the following columns:
        ['local_timestep',
         'replica',
         ... feature names ...
         'filename'
        ]
    """
    npz_pattern = re.compile(r"^.*\.npz$")
    npy_pattern = re.compile(r"^.*\.npy$")
    if isinstance(array_filename, np.ndarray):
        rpos = array_filename
        self_name = None
    elif isinstance(array_filename, collections.Mapping):
        self_name = list(array_filename.keys())[0]
        rpos = array_filename[self_name]
    elif npz_pattern.match(array_filename):
        rpos = np.load(array_filename)["coords"]
        self_name = array_filename
    elif npy_pattern.match(array_filename):
        rpos = np.load(array_filename)
        self_name = array_filename
    else:
        raise ValueError("Unknown file type.")
    if len(rpos.shape) == 2:
        # the functions are designed to work with trajectories with
        # multiple replicas which have an extra index
        rpos = rpos[None, None, :, :]
    elif len(rpos.shape) == 3:
        rpos = rpos[None, :, :, :]

    n_steps = rpos.shape[1]
    n_replicas = rpos.shape[0]
    rpos = rpos[:, ::stride, :, :]
    proto_tables = []
    for rep in range(n_replicas):
        tab = pd.DataFrame(np.arange(n_steps), columns=["local_timestep"])
        tab = tab.iloc[::stride, :]
        tab["replica"] = rep
        proto_tables.append([tab])
    for feater in featers:
        data, names = feater(rpos, pdb_filename)
        for datum, proto_table in zip(data, proto_tables):
            to_add = pd.DataFrame(datum, columns=names)
            to_add.index = proto_table[0].index
            # table[names] = to_add
            proto_table.append(to_add)
    tables = [pd.concat(pt, axis=1) for pt in proto_tables]
    full_table = pd.concat(tables, axis=0)
    if self_label is True:
        full_table.insert(loc=0, column="filename", value=self_name)
    elif self_label is not False:
        full_table.insert(loc=0, column="filename", value=self_label)

    return full_table


def mprocess(array_filenames, add_r_step_index=True, joblib=False, n_jobs=-2, **kwargs):
    """Dispatch process function across multiple filenames. Options for parallel
    execution via joblib.

    Arguments
    ---------
    array_filenames (list of strings or dictionary of string:array pairs):
        List of all filenames on which to run process. If a dictionary, this
        is assumed to be of the form 'name':array, and each dictionary
        key:value pair individually passed to process as a single item
        dictionary.

        For information on how these arrays are processed, see process.

        NOTE: if processing multiple already-loaded arrays, it is likely
        best to pass using the dictionary method. Else, organization of the
        eventual aggregate DataFrame will likely be mixed up, as the process
        call will not assign (unique) names to the data given by each individual
        array.
    add_r_step_index (boolean):
        If true, each replica (which typically spans multiple files) is
        indexed according to the sequence of its frames under
        "contig_time_index". This indexing is agnostic to any information on
        striding that is performed by process.
    joblib (boolean):
        If true, jobs are parallelized using joblib. This parallelization
        can make debugging harder.
    n_jobs (integer):
        Number of threads for joblib to use. Passed to joblib.Parallel
    kwargs
        Additional arguments are curried into the process call before
        dispatch. It is likely you want to pass pdb_filename this way.

    Return
    ------
    pandas.DataFrame with the following columns (depending on arguments):
    ['local_timestep', 'replica', ... feature columns...
     'filename', 'contig_time_index']

    Example execution:
        import featurize
        import glob
        files=glob("epoch_19/*.npy")
        out=featurize.mprocess(files,pdb_filename="epoch_19/source.pdb",
                               featers=[featurize.gen_dssp_codes,
                                        featurize.rog],
                               joblib=True,
                               stride=1)

    """
    if isinstance(array_filenames, collections.Mapping):
        array_filenames = [{key: ob} for key, ob in array_filenames.items()]
    if joblib:
        cur_p = ncurry(process, self_label=True, **kwargs)
        results = jb.Parallel(n_jobs=n_jobs)(
            jb.delayed(cur_p)(i) for i in array_filenames
        )
    else:
        results = []
        for file_name in array_filenames:
            subtable = process(file_name, self_label=True, **kwargs)
            results.append(subtable)
    table = pd.concat(results, axis=0)
    # the following steps cannot be performed with a duplicate axis
    table.reset_index(inplace=True, drop=True)
    if add_r_step_index:
        table.sort_values(["replica", "filename", "local_timestep"], inplace=True)
        sap = table.groupby("replica").apply(get_local_index)
        if isinstance(sap.index, pd.MultiIndex):
            sap.index = sap.index.droplevel(0)
        else:
            sap = sap.iloc[0, :]
        table.insert(loc=0, column="contig_time_index", value=sap)
        table.sort_values(["replica", "contig_time_index"], inplace=True)
    # we do this _again_ because the previous steps jumbled up the axis and we
    # want to return a table with a clean axis
    table.reset_index(inplace=True, drop=True)
    return table


def HackyMDTrajTrajectory(array, pdb_filename, *args, pdb_atom_names=None, **kwargs):
    """Hacks a MDTrajTrajectory object together. Positions are set by
    @array and toplogy is set via an initial mdtraj object created from
    @pdb_name. @pdb_atom_names can filter the pdb (but not the coordinates)
    to make it match @array.

    Arguments
    ---------
    array (numpy.ndarray):
        xyz for the eventual mdtraj object. Should have shape
        (n_frames,n_atoms,n_dims).
    pdb_filename (string or pathname):
        location of pdb file.
    pdb_atom_names (iterable of strings or None, default None)
        Names to use to filter pdb derived traj before adding
        @array as positions.
    *args,**kwargs:
        passed to mdtraj.load when loading pdb

    Return
    ------
    Mdtraj.Trajectory

    NOTE: array should be of shap (n_frames,n_atoms,n_dim)
        This means that the position arrays with multiple
        parallel trajectories are not appropriate.
    NOTE: wipes coordinates in pdb when substituting in array.
    """
    traj = md.load(pdb_filename, *args, **kwargs)
    if pdb_atom_names is not None:
        traj = atom_filter(traj, atom_list=pdb_atom_names)
    traj.xyz = array
    traj.time = np.arange(len(array))
    if traj.unitcell_lengths is not None:
        traj.unitcell_lengths = np.repeat(
            traj.unitcell_lengths, repeats=len(array), axis=0
        )
    if traj.unitcell_angles is not None:
        traj.unitcell_angles = np.repeat(
            traj.unitcell_angles, repeats=len(array), axis=0
        )
    return traj


# featurization functions act on rpos,pdb_filename and return a list
# of feature sequences in the order of the replicas in the array
# AND a list of feature names.
# rpos has shape (n_replicas,n_steps,n_atoms,n_dims)


def gen_dssp_codes(
    rpos, pdb_filename, atom_names=None, simplified=True, scale_coords=(1 / 10.0)
):
    """Generates dssp codes for each trajectory in @rpos.

    Arguments
    ---------
    rpos (numpy.ndarray):
        numpy ndarray that describes Cartesian positions of multiple
        trajectories. Has shape (n_replicas,n_frames,n_sites,n_dims).
    pdb_filename (string or pathname):
        location of pdb file. Should match rpos _before_ any filtering via
        atom_names is done.
    atom_names (list of strings):
        Names to use to filter traj) before calculating DSSP codes.
        Included for consistency with other fertilization functions, but not
        particularly helpful for this method.
    simplified (boolean):
        Whether to return simplified DSSP codes.
    scale_coords (float):
        factor to multiple positions by before passing to mdtraj DSSP
        calculation. Useful to account for units (typical mlcg trajectories
        should be multiplied by (1/10.0) due to Angstrom to nm conversion.

    Return
    ------
    tuple; first element is a list, each element is a 2d array of
    dssp codes x timestep for a single replica. Second is names of the
    columns for the dssp time series'.

    """
    # note that we scale by 10
    trajs = [HackyMDTrajTrajectory(sl * scale_coords, pdb_filename) for sl in rpos]
    if atom_names is not None:
        trajs = [atom_filter(traj, atom_names) for traj in trajs]
    dssps = [md.compute_dssp(st, simplified=simplified) for st in trajs]
    names = ["dssp-" + str(x) for x in trajs[0].topology.residues]
    return (dssps, names)


def atom_filter(traj, atom_list, return_mask=False):
    """Filters a subsetted mdtraj object based on atom names, or returns a mask
    which does so.

    Arguments
    ---------
    traj (mdtraj.Trajectory):
        mdtraj.Trajectory instance to be filtered
    atom_list (list of strings):
        Used to filter traj by atom names.
    return_mask (boolean):
        if true, instead of returning trajectory return boolean array that
        serves as a mask for indexing future arrays.

    Return
    ------
    mdtraj.Trajectory or boolean array (see return_mask)
    """
    booleans = [atom.name in atom_list for atom in traj.topology.atoms]
    if return_mask:
        return np.array(booleans)
    # this returns a 1 element tuple so note the terminal index
    inds = np.array(booleans, dtype=int).nonzero()[0]
    return traj.atom_slice(inds)


def rmsd(rpos, pdb_filename, reference_pos, atom_names=None):
    """Calculates root mean square deviation

    Arguments
    ---------
    rpos (numpy.ndarray):
        numpy ndarray that describes Cartesian positions of multiple
        trajectories. Has shape (n_replicas,n_frames,n_sites,n_dims).
    pdb_filename (string or pathname):
        location of pdb file. Should match rpos _before_ any filtering via
        atom_names is done.
    atom_names (list of strings):
        Names to use to filter traj (made via ) before adding
        @array as positions.

    Return
    ------
    tuple; first element is a list, each element is a 2d array of
    dssp codes x timestep for a single replica (first dimension is 1).
    Second is a (one element) list of the names of the columns for
    the feature time series.
    """
    trajs = [HackyMDTrajTrajectory(sl, pdb_filename) for sl in rpos]
    ref_traj = HackyMDTrajTrajectory(
        reference_pos[None, :, :], pdb_filename, pdb_atom_names=atom_names
    )
    if atom_names is not None:
        trajs = [atom_filter(traj, atom_list=atom_names) for traj in trajs]
        # ref_traj = atom_filter(ref_traj, atom_list=atom_names)
    rmsds = [md.rmsd(target=t, reference=ref_traj, frame=0) for t in trajs]
    return (rmsds, ["rmsd"])


def rog(rpos, pdb_filename, atom_names=None):
    """Calculates radius of gyration using a pairwise formula.

    Arguments
    ---------
    rpos (numpy.ndarray):
        numpy ndarray that describes Cartesian positions of multiple
        trajectories. Has shape (n_replicas,n_frames,n_sites,n_dims).
    pdb_filename (string or pathname):
        location of pdb file. Should match rpos _before_ any filtering via
        atom_names is done.
    atom_names (list of strings):
        Names to use to filter traj (made via ) before adding
        @array as positions.

    Return
    ------
    tuple; first element is a list, each element is a 2d array of
    dssp codes x timestep for a single replica (first dimension is 1).
    Second is a (one element) list of the names of the columns for
    the feature time series.
    """

    if atom_names is not None:
        traj = md.load(pdb_filename)
        mask = atom_filter(traj, atom_list=atom_names, return_mask=True)
        rpos = rpos[:, :, mask, :]
    n_sites = rpos.shape[2]
    rogs = [
        np.sqrt((traj_distances_array(pos) ** 2).sum(axis=1) / (n_sites**2))
        for pos in rpos
    ]
    return (rogs, ["rog"])


def per_atom_frac_native_contacts(
    rpos,
    pdb_filename,
    reference_pos,
    contact_f,
    aggregate=False,
    non_native=False,
    atom_names=None,
    normalize_counts=True,
    dont_count_diagonal=False,
    distance_named_args={},
):
    """Generates the number of native (or nonnative) contacts each
    residue makes along each step of the trajectory. Has options to aggregate
    contacts over all residues.

    Arguments
    ---------
    rpos (numpy.ndarray):
        numpy ndarray that describes Cartesian positions of multiple
        trajectories. Has shape (n_replicas,n_frames,n_sites,n_dims).
    pdb_filename (string or pathname):
        location of pdb file. Should match rpos _before_ any filtering via
        atom_names is done.
    reference_pos (numpy.ndarray):
        reference trajectory frame used to calculate which contacts are
        native. Likely contains the folded structure.

        NOTE: Not filtered with atom_names! Filter it before you pass it.
    contact_f (callable):
        Function which transforms distances to a "contact" (a
        number between 0 and 1, 1 implying contact). This is rounded to the
        nearest int when deciding whether something is a "native contact".
    aggregate (boolean):
        Whether to aggregate the number of native contacts over each frame
        when returning values. If false, then the native contacts for each
        residue is individually returned.
    non_native (boolean):
        If true, then instead of returning native contacts, non-native
        contacts are returned.
    atom_names (list of strings):
        Names to use to filter traj before calculating native contacts. Not
        applied ot reference_pos.
    normalize_counts (boolean):
        If true, then instead of returning the sum of contact_f for each
        residue (or aggregated residues), this value is divided by the value
        for the native system coordinate frame. Note that the native system
        is judged using the rounded version of contact_f, but each
        trajectory frame is not.
    dont_count_diagonal (boolean or positive integer):
        if a positive integer, then contacts between two atoms
        @dont_count_diagonal away from each other are ignored. True is
        equivalent to 0.

        NOTE: 0, not 1,  corresponds to no self contacts.
    distance_named_args (dict):
        Passed as arguments via ** to distance function.

    Return
    ------
    tuple; first element is a list, each element is a 2d array of
    contact statistics for a single replica. Second is names of the
    columns for the time series'.
    """

    if atom_names is not None:
        traj = md.load(pdb_filename)
        mask = atom_filter(traj, atom_list=atom_names, return_mask=True)
        rpos = rpos[:, :, mask, :]
    if dont_count_diagonal is True:
        dont_count_diagonal = 0
    if normalize_counts and non_native:
        print(
            "Normalizing counts when using non-native contacts is not "
            "understood. Setting normalize_counts=False and continuing."
        )
        normalize_counts = False
    curried_dists = ncurry(
        traj_distances_array, return_matrix=True, **distance_named_args
    )
    reference_contacts = contact_f(curried_dists(reference_pos[None, :, :])).round()
    dists_g = (curried_dists(pos) for pos in rpos)
    if non_native:
        reference_contacts = 1.0 - reference_contacts
    contacts = [reference_contacts * contact_f(dc) for dc in dists_g]
    if dont_count_diagonal is not False:
        mask = 1.0 - thick_diag(reference_contacts.shape[-1], dont_count_diagonal)
        contacts = [c * mask[None, :, :] for c in contacts]
        reference_contacts *= mask
    if aggregate:
        total_native_contacts = np.sum(reference_contacts)
        if normalize_counts:
            agged = [
                np.sum(ctraj, axis=(1, 2)) / total_native_contacts for ctraj in contacts
            ]
        else:
            agged = [np.sum(ctraj, axis=(1, 2)) for ctraj in contacts]
        return (agged, ["ag_frac_native_contacts"])
    collapsed_reference_contacts = reference_contacts.sum(axis=(2,))
    if normalize_counts:
        collapsed_contacts = [
            st.sum(axis=(2,)) / collapsed_reference_contacts for st in contacts
        ]
    else:
        collapsed_contacts = [st.sum(axis=(2,)) for st in contacts]
    atoms = atom_filter(md.load(pdb_filename), atom_list=atom_names).topology.atoms
    names = ["frac_native_contacts_" + str(name) for name in atoms]
    return (collapsed_contacts, names)


def traj_distances_skl(pos_array, return_tuple_labels=False, particle_names=None):
    """Transforms position trajectory into array of nonrepeated distances.
    Uses scikit-learn, very slow.

    Arguments
    ---------
    pos_array: nd-array, shape: n_frames, n_atoms, n_dim
        position array of the molecular trajectory
    return_tuple_labels: boolean, default: False
        If true, then a list of strings labelling the distance columns is also
        returned.
    particle_names: list of strings or None, default None
        If not None, this is used as the names of the particles when satisfying
        return_tuple_labels. If None, non-negative integers are used. No effect
        if return_tuple_labels is False.

    Return
    ------
    nd-array which has the nonrepeated distances as columns. If
    return_tuple_labels, a tuple is returns with said array (element 0)
    as well as the names of the columns (element 1).

    NOTE: This function is for debugging purposes. It does not have any
    advantage over traj_distances_array and is much slower.
    """

    distance_list = []
    n_particles = pos_array.shape[1]
    mat_mask = np.triu_indices(n_particles, 1)  # no diagonal
    for frame in pos_array:
        distances = sk_pairwise_distances(frame)[mat_mask]
        distance_list.append(distances)
    distance_array = np.stack(distance_list)
    if return_tuple_labels:
        if particle_names is None:
            particle_names = [str(ob) for ob in list(range(n_particles))]
        labels = []
        for index_0, index_1 in zip(*mat_mask):
            name_0 = particle_names[index_0]
            name_1 = particle_names[index_1]
            label = sorted([name_1, name_0])
            labels.append(tuple(label))
        return (distance_array, labels)
    return distance_array


def traj_distances_array(
    pos_array,
    return_tuple_labels=False,
    particle_names=None,
    array_labels=False,
    indexing_seqs=None,
    batch_size=50000,
    return_matrix=False,
    box_diam=None,
):
    """Transforms position trajectory into array of nonrepeated distances.
    Uses array operations, relatively fast.

    Arguments
    ---------
    pos_array: nd-array, shape: n_frames, n_atoms, n_dim
        position array of the molecular trajectory
    return_tuple_labels: boolean, default: False
        If true, then a list of strings labelling the distance columns is also
        returned.
    particle_names: list of strings or None, default None
        If not None, this is used as the names of the particles when satisfying
        return_tuple_labels. If None, non-negative integers are used. No effect
        if return_tuple_labels is False.
    indexing_seqs (length 2 sequence):
        two element sequence that will be passed in style
        [:,element_0,element_1] to index the distance array. Useful if the
        ordering of extracted distances should be controlled.
    batch_size:
        Number of distances to calculate per iteration. Larger numbers result
        in faster computation but require more memory.
    return_matrix (boolean):
        If true, distances are not extracted from the matrix, and instead a
        array of shape (n_frames,n_sites,n_sites) is returned. If false,
        distances are extracted so there are no duplicate distances and no
        self distances; return shape is (n_frames,n_distances_per_frame).

        NOTE: if return_matrix and return_inds are both specified then
        hypothetically valid indices are returned, but no indexing if
        performed.
    box_diam (None or positive float):
        If not None, distances are calculated in the context of a cubic periodic
        box of diameter box_diam.


    Return
    ------
    nd-array which has the nonrepeated distances as columns. If
    return_tuple_labels, a tuple is returns with said array (element 0)
    as well as the names of the columns (element 1).
    """

    if (indexing_seqs or return_tuple_labels) and return_matrix:
        raise ValueError(
            "Setting (indexing_seqs or return_tuple_labels) "
            "and return_matrix does not make sense "
            "and is not allowed."
        )
    n_frames = pos_array.shape[0]
    n_particles = pos_array.shape[1]
    n_chunks = max(np.floor(n_frames / batch_size), 1)
    chunks = np.array_split(np.arange(n_frames), n_chunks)
    if return_matrix:
        distance_array = np.empty((n_frames, n_particles, n_particles))
        for chunk in chunks:
            abs_disps = abs(pos_array[chunk, None, :, :] - pos_array[chunk, :, None, :])
            if box_diam is not None:
                disps = np.where(
                    abs_disps < (box_diam / 2), abs_disps, box_diam - abs_disps
                )
            distance_matrix = np.linalg.norm(disps, axis=-1)
            distance_array[chunk, :, :] = distance_matrix
    else:
        n_distances = n_particles * (n_particles - 1) // 2
        distance_array = np.empty((n_frames, n_distances))
        if indexing_seqs is None:
            indices_0, indices_1 = np.triu_indices(n_particles, k=1)
        else:
            indices_0, indices_1 = indexing_seqs
        for chunk in chunks:
            abs_disps = abs(pos_array[chunk, None, :, :] - pos_array[chunk, :, None, :])
            if box_diam is not None:
                abs_disps = np.where(
                    abs_disps < (box_diam / 2), abs_disps, box_diam - abs_disps
                )
            distance_matrix = np.linalg.norm(abs_disps, axis=-1)
            distance_array[chunk, :] = distance_matrix[:, indices_0, indices_1]
    if return_tuple_labels:
        if particle_names is None:
            particle_names = [str(ob) for ob in list(range(n_particles))]
        if array_labels:
            labels = (indices_0, indices_1)
        else:
            labels = []
            for index_0, index_1 in zip(indices_0, indices_1):
                name_0 = particle_names[index_0]
                name_1 = particle_names[index_1]
                label = sorted([name_1, name_0])
                labels.append(tuple(label))
        return (distance_array, labels)
    return distance_array


def make_distance_table(
    pos_array, distance_method=traj_distances_array, distance_max=np.Inf, **kwargs
):
    """Creates a distance panda from position array.

    Arguments
    ---------
    pos_array: nd-array, shape: n_frames, n_atoms, n_dim
        Position array of the molecular trajectory
    distance_method:
        Callable that calculates distance array; see traj_distances_array for
        more information.
    distance_max:
        Distances larger than distance_max are set to distance max.
    kwargs:
        Passed to distance_method

    Return
    ------
    Panda of labeled distance features for each frame.
    """

    distance_array, dist_tups = distance_method(
        pos_array, return_tuple_labels=True, **kwargs
    )

    distance_array[distance_array > distance_max] = distance_max
    tab = pd.DataFrame(distance_array)
    dist_names = [ob[0] + "-" + ob[1] for ob in dist_tups]
    tab.columns = dist_names
    return tab


def sigmoid(arg, shift=0.0, scale=1.0, clamp_min=-30, clamp_max=30):
    """Simple sigmoid function.

    Arguments
    ---------
    shift (float):
        shifts x values by that amount.
    scale (float):
        scales steepness by that amount (large numbers are less steep).
    clamp_min (float):
        x values less than this are clamped to this
    clamp_max (float):
        x values larger than this are clamped to this

    Return
    ------
    float

    NOTE: has issues with overflow, which will throw warnings. These just cause
    the function to (e.g.) return 0 instead of a very small number, so can often
    be ignored. This can be helped using the clamp arguments, but the effect is
    partial.
    """

    clamped = np.clip(arg, a_min=clamp_min, a_max=clamp_max)
    return 1.0 / (1.0 + np.exp((clamped - shift) / scale))


def ncurry(func, **kwargs):
    """Curry a function using named arguments. That is:
    for f(x,y), ncurry(f,y=a) returns a function g, where
    g(b) = f(x=b,y=a). Useful when creating a featurization
    function with certain options set.
    """

    def curried_f(*sub_args):
        return func(*sub_args, **kwargs)

    return curried_f


def label_mod(func, mod, append=False, **kwargs):
    """Takes a featurization function and returns a new featurization function
    where the column names have a string prepended or appended.

    Arguments
    ---------
    mod (string):
        String to add to each column name.
    append (boolean):
        If True, then @mod is appended to each column. If false, it is
        prepended.
    kwargs:
        Remaining arguments are passed as default arguments to the original
        featurizer

    Return
    ------
    feater whose output labels are prepended or appended with the given string
    """

    def modded_f(*sub_args):
        out = func(*sub_args, **kwargs)
        if append:
            new_labels = [string + mod for string in out[1]]
        else:
            new_labels = [mod + string for string in out[1]]
        return (out[0], new_labels)

    return modded_f


def get_local_index(frame):
    """Returns a pandas.Series which indexes the elements a given
    pandas.Dataframe 1:length. The series has the same internal index as the
    analyzed Dataframe.

    Arguments
    ---------
    df (pandas.DataFrame):
        DataFrame to analyze

    Return
    ------
    pandas.Series
    """

    ser = pd.Series(np.arange(frame.shape[0]))
    ser.index = frame.index
    return ser


def thick_diag(size, padding=1):
    """Returns a matrix where the diagonal and offset diagonal are set to 1 and
    other elements are set to zero.

    Arguments
    ---------
    size (positive integer):
        Returned matrix has size (size,size)
    padding (positive integer):
        Number of off diagonals to fill. 0 returns a diagonal matrix.

    Return
    ------
    matrix of size (size,size)
    """

    mat = np.ones((size, size), dtype=int)
    mat = mat - np.tril(mat, -(padding + 1)) - np.triu(mat, (padding + 1))
    return mat
