# classE

A package to analyze molecular dynamics trajectories using classification.

Molecular dynamics trajectories are often characterized using a low dimensional
free energy surface. The difference in two simulations can be analyzed by
generating a free energy surface for each, and then comparing them. The `classe`
method allows one to numerically find the difference between the free energy
surfaces of two simulations without calculating the surface for each simulation
individually. Furthermore, it works in higher dimensions than those accessible
when creating a free energy surface for each simulation individually.

`classe` can be used to compare the free energy surfaces created when mapping
atomistic simulations to high dimensional representations (e.g., considering a
protein and the carbon-alpha resolution). This allows the output of typical
particulate coarse-grained to be compared to atomistic reference data to
estimate the difference between the coarse-grained force-field and the manybody
potential of mean force. The current implementation not only provides and
estimate of difference in free energy surfaces, but also provides a
decomposition using SHAP values to identify which features (typically
interatomic distances) contribute to the estimated error.

## Installation

`classe` can be be installed by cloning this repo, changing to its root
directory, and calling `pip install .` or `pip install .[cv]`. The former only
install the libraries needed for error analysis, while the latter installs a larger
set of packages allowing for the creation of collective variables (as well as
error analysis).

## Example usage
```python
import classe.featurization as feat
import classe.compare as cc
import numpy as np

#get molecular trajectories (shape n_frames,n_atoms,n_dims)
#data is not included in the repository
mod_array=np.load("data/short_mod_array.npy")
ref_array=np.load("data/short_ref_array.npy")

#create feature table using pairwise distances
mod=feat.make_distance_table(mod_array,batch_size=50000)
ref=feat.make_distance_table(ref_array,batch_size=50000)

#run classE analysis
results=cc.compare_tables(mod,ref,temperature=350)
```
`results` contains a table with the SHAP values and estimate error, along with many
other useful quantities. 

Each trajectory frame is assigned a shap value for each feature, and is not low
dimensional. It can be summarized using UMAP and KNR via built in tools, which
also allow the learned CVs to be applied to new molecular configurations.
```python
import classe.cv as cv

#extract the data not in the training set
tab=results['table'].loc[~results['table']['in_train'],:]
tabstruct=tab[results['feature_names']]
tabshap=tab[results['shap_names']]

CV=cv.TransferCV(transfer_featurizer=results['shaper'])

#train CV using shap data
CV.fit(tabshap.sample(int(1e4)))

#apply CV to subset of data using structural features
coords=CV.transfer_transform(tabstruct.sample(int(1e4)))
```
A executable python tool demonstrating typical analysis is found at 
`examples/tsproj.py`.
