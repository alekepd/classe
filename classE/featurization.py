import pandas as pd
from sklearn.metrics import pairwise_distances
import numpy as np

def make_distance_table(pos_array,particle_names=None):
    """Creates a distance panda from position array.

    Arguments
    ---------
    pos_array: nd-array, shape: n_frames, n_atoms, n_dim
        position array of the molecular trajectory

    Return
    ------
        Panda of labeled distance features for each frame.
    """

    distance_array,dist_tups = traj_distances(pos_array,
                                              return_tuple_labels=True,
                                              particle_names=particle_names)
    tab = pd.DataFrame(distance_array)
    dist_names = [ob[0]+'-'+ob[1] for ob in dist_tups]
    tab.columns = dist_names
    return tab

def traj_distances_skl(pos_array,return_tuple_labels=False,particle_names=None):
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
    """
    distance_list = []
    n_particles = pos_array.shape[1]
    mat_mask = np.triu_indices(n_particles,1) #no diagonal
    for frame in pos_array:
        distances = pairwise_distances(frame)[mat_mask]
        distance_list.append(distances)
    distance_array = np.stack(distance_list)
    if return_tuple_labels:
        if particle_names is None:
            particle_names = [str(ob) for ob in list(range(n_particles))]
        labels = []
        for p0,p1 in zip(*mat_mask):
            name_0 = particle_names[p0]
            name_1 = particle_names[p1]
            label = sorted([name_1,name_0])
            labels.append(tuple(label))
        return (distance_array,labels)
    else:
        return distance_array

