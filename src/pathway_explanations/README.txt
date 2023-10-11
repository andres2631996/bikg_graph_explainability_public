Short summary of source code structure in repository:

- "explainer.py" contains the general functions to deal with the pipeline, 
calling all data processing functions in "data.py", community processing functions
in "pathways.py", mask processing functions in "masks.py", GNN model 
functions in "model.py", kernel functions in "kernels.py", and 
surrogate model functions in "wlm.py"

- "data.py" contains functions dealing with graph data handling (both node features and edge indexes)
These functions mostly deal with the conversion of heterogeneous graph dictionaries into
concatenated graph tensors with node type and edge type tensors. They also include
the extraction of computational graph. They include as well the conversion of homogenized 
graph data back into heterogeneous dictionaries. 

- "pathways.py" contains functions to work with graph communities of nodes or edges
They handle the filtering of communities in a computational graph, the 
conversion of heterogeneous graph dictionaries of communities into
concatenated tensor types, the conversion of communities of element names to
communities of element indexes, the handling of binary masks with active and
inactive communities for graph perturbation. They also process the aggregation
of node-wise or edge-wise configuration values into community-wise
configuration values

- "masks.py" handles perturbation masks of nodes or edges. It includes functions
for creating masks of internal permutations, and of external permutations, as
well as fully random masks

- "model.py" contains functions to deal with the GNN model to be explained.
It includes both inference functions as well as pre-processing and post-processing
methods to operate with the trained GNN model, most of these last dealing with
heterogeneous graph output

- "kernels.py" contains functions for computing or approximating the kernels

- "wlm.py" contains all necessary functions to operate the surrogate weighted 
linear regression model (loss functions, optimizer and scheduler loaders, training and
validation regimes, callers to GNN model inference and kernel computation...)

These functions contain test replicates with names "test_data.py", "test_explainer.py", etc.
for code testing in "../../tests/". The exact functions for testing are contained.

- The script "test_utils.py" contains utility functions to run the tests of the remaining
scripts