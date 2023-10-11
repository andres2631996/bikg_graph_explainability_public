<!--
<p align="center">
  <img src="https://github.com/AZ-AI/bikg-graphexplainability/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  graph_interpretability
</h1>

<p align="center">
    <a href="https://github.com/AZ-AI/bikg-graphexplainability/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/AZ-AI/bikg-graphexplainability/workflows/Tests/badge.svg" />
    </a>
    <a href="https://pypi.org/project/graph_interpretability">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/graph_interpretability" />
    </a>
    <a href="https://pypi.org/project/graph_interpretability">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/graph_interpretability" />
    </a>
    <a href="https://github.com/AZ-AI/bikg-graphexplainability/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/graph_interpretability" />
    </a>
    <a href='https://graph_interpretability.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/graph_interpretability/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/AZ-AI/bikg-graphexplainability/branch/main">
        <img src="https://codecov.io/gh/AZ-AI/bikg-graphexplainability/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com/AZ-AI/bikg-graphexplainability/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

Library that runs community-aware explanations of GNN architectures on top of homogeneous or heterogeneous graphs in PyTorch Geometric


### Maturity level
![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)


## 💪 Getting Started

We deeply recommend to have a GPU installed if you are going to analyse graphs with thousands of nodes and millions of edges (or even larger)

Create a new Conda environment with `Python 3.8.12`. **Note:** This is specific version of Python is needed for running Pytorch Geometric on the SCP.

```bash
conda create -n bikg_graphexplainability python=3.8.12
conda activate bikg_graphexplainability
```

Then install extra packages with (might take a while):

If you have GPU available with CUDA 10.2:

```bash
./dev_setup.sh
```

Else, if you only have CPU, you can install the environment with:

```bash
./dev_setup_cpu.sh
```


### Example of use
The code snippet below shows an example of use. 

We offer a toy example proof in 
a notebook in **examples/toy_example.ipynb**

```
from pathway_explanations.explainer import Explainer

# Load graph data, graph communities, and GNN model

# Load hyperparameters for explanation pipeline, for example:

hyperparams = {
                        "seed": seed,
                        "interpret_samples": 20,
                        "epochs": 50,
                        "optimizer": "adam",
                        "lr": 0.01,
                        "lr_patience": 10,
                        "l1_lambda": 1e-4
                    }

# Run actual explanation pipeline

pipeline = Explainer(
        node_features,
        edge_index,
        arch,
        hyperparams,
        names,
        communities,
        community_names,
        query_type,
        problem,
        node_types,
        edge_types
    )

config_val_df, pathway_df = pipeline.run(query_node, repeats)
```
All the actions that the pipeline can complete are stored in the "Explainer" object. 
The most important action to be executed with this object is "run", which will 
provide us with the node or edge explanations and the community explanations. 

**Variables for "Explainer" object:**

* "node_features" : torch.tensor or dict of torch.tensor (REQUIRED)
Stores the node features from the input graph. If the graph is 
homogeneous, it is a tensor. If the graph is heterogeneous, it is 
a dict of tensors, with the node type name strings as keys and the 
respective tensors as values. For example:

```
# Heterogeneous node features
node_features = {"node_type1": tensor1, ... ,"node_typeN": tensorN}
```
**The different tensors in "node_features" as dict
can have a different number of features (second dimensions)**


* "edge_index" : torch.tensor or dict of torch.tensor (REQUIRED)
Stores the edge indexes from the input graph. If the graph is homogeneous,
it is a tensor. If the graph is heterogeneous, it is 
a dict of tensors, with the edge type name tuples as keys and the 
respective tensors as values. For example:

```
# Heterogeneous edge indexes
edge_index = {("node_type1","relation1","node_type2"): tensor1, 
              ... ,
              ("node_typeN_1","relationM","node_typeN"): tensorN}
```
**In heterogeneous graphs, the node type names in the tuple keys
of the edge index dictionary should match the node type names
in the string keys of the node feature dictionary. **


* "arch": torch.nn.Module (REQUIRED)
GNN model object. It supports any kind of GNN architecture supported by
PyTorch Geometric, as the pipeline is GNN-agnostic.
**If "arch" is loaded from a file, the pipeline will work best with
architectures saved with the "pth.tar" format**

* hyperparams: dict (REQUIRED)
Hyperparameter set for the pipeline. An example can be found in 
"config/configs.json"
```
hyperparams = {
              "seed": 1,
              "interpret_samples": 20,
              "epochs": 50,
              "optimizer": "adam",
              "lr": 0.01,
              "lr_patience": 10,
              "l1_lambda": 1e-4
          }
```
- "seed" is an int with the random seed for pipeline initialization
- "interpret_samples" is an int with the number of perturbations to be completed per training epoch of the surrogate model
- "epochs" is an int with the number of epochs to train the surrogate model
- "optimizer" is a str with the optimizer type to train the surrogate model (for now, only "adam" is supported)
- "lr" is a float with the learning rate to train the surrogate model
- "lr_patience" is an int with the number of epochs to be waited before applying early stopping, if the training loss does not decrease **It should be lower than "epochs"**
- "l1_lambda" is a float with the L1 regularization constant to train the surrogate model




* "names": list of str or dict of list of str (OPTIONAL, BUT RECOMMENDED)
Names of graph elements to be explained. In a node or graph prediction setup, the names are node names. In an edge prediction setup, the names are edge names.
**For homogeneous graphs, it is a list which should match the first dimension of "node_features" tensor for node or graph prediction tasks, or the second dimension of "edge_index" tensor for edge prediction tasks**

**For heterogeneous graphs, it is a dict which should match the first dimension of the tensors in "node_features" dict for node or graph prediction tasks, or the second dimension of all tensors in "edge_index" dicts for edge prediction tasks. In this dict, the keys should be the same as for "node_features" or "edge_index", respectively**. For example:

```
# Graph names in heterogeneous setup

# For node or graph prediction
names = {"node_type1": [name1_1,...,name1_I],...,"node_typeN": [name1_N,...,nameN_J]}

# For edge prediction
names = {("node_type1","relation1","node_type2"): [name1_1,...,name1_I], 
        ... ,
        ("node_typeN_1","relationM","node_typeN"): [name1_N,...,nameN_J]}
```
**If "names" is not provided, the pipeline will take the numeric node or edge indexes as names**


* "communities": list of str or int, or dict of list of str or int (OPTIONAL, BUT RECOMMENDED)
For homogeneous graphs: In node or graph prediction tasks, "communities" contains the node names (if it is a list of str) or the node indexes (if it is a list of int) grouped by communities. For edge prediction tasks, it contains respectively the edge names or edge indexes.

```
# Homogeneous graph communities

# For node or graph prediction
communities = [["node1",...,"nodeI"],...,["nodeJ",...,"nodeK"]] # With node names
communities = [[1,...,I],...,[J,...,K]] # With node indexes

# For edge prediction
communities = [["edge1",...,"edgeI"],...,["edgeJ",...,"edgeK"]] # With edge names
communities = [[1,...,I],...,[J,...,K]] # With edge indexes
```

For heterogeneous graphs: In node or graph prediction tasks, "communities" contains the node names of each node type as key in the dict (if the lists contain str) or the node indexes (if the lists contain int). For edge prediction tasks, it contains respectively the edge names or edge indexes for each edge type. **The pipeline assumes that communities in heterogeneous graphs are only formed by nodes or edges of the same type. The dict keys should match the keys of "node_features" for node or graph prediction tasks and the keys of "edge_index" for edge prediction tasks**

```
# Heterogeneous graph communities

# For node or graph prediction
communities = {"node_type1" : ["node1",...,"nodeI"],...,"node_typeN" : ["nodeJ",...,"nodeK"]] # With node names
communities = {"node_type1" : [1,...,I],..., "node_typeN" : [J,...,K]} # With node indexes

# For edge prediction
communities = {("node_type1","relation1","node_type2") : ["edge1",...,"edgeI"],
              ...,
              ("node_typeN_1","relationM","node_typeN") : ["edgeJ",...,"edgeK"]] # With edge names
communities = {("node_type1","relation1","node_type2") : [1,...,I],
              ...,
              ("node_typeN_1","relationM","node_typeN") : [J,...,K]] # With edge indexes
```
**If no communities are inputted, the pipeline samples random permutations of nodes or edges, with a behavior similar to SHAP in a graph dataset**


* "community_names" : list of str or int or dict of list of str or int (OPTIONAL, BUT RECOMMENDED)
Respective names of communities in "communities". **It should have the same structure as "communities"**

For homogeneous graphs, "community_names" is a list of str with the community names (or indexes, in case of having list of ints).

For heterogeneous graphs, "community_names" is a dict with lists of str with the community names for each node type (for node or graph prediction tasks) or for each edge type (for edge prediction tasks). The lists can contain ints if they contain indexes instead of node or edge names.

```
# Heterogeneous graph community names

# For node or graph prediction
community_names = {"node_type1" : ["community1",...,"communityN"], ..., "node_typeN" : ["communityJ",...,"communityK"]] # With node names
community_names = {"node_type1" : [1,...,I], ..., "node_typeN" : [J,...,K]} # With node indexes

# For edge prediction
community_names = {("node_type1","relation1","node_type2") : ["community1",...,"communityI"],
                  ...,
                  ("node_typeN_1","relationM","node_typeN") : ["communityJ",...,"communityK"]] # With edge names
community_names = {("node_type1","relation1","node_type2") : [1,...,I],
                  ...,
                  ("node_typeN_1","relationM","node_typeN") : [J,...,K]] # With edge indexes
```

**In case that "community_names" is not given, the pipeline will construct community names with numeric indexes**

* query_type: str or tuple (OPTIONAL)
In heterogeneous graphs, it is the name of the node or edge type that is explained. In homogeneous graphs, this variable is not required.
For node or graph prediction tasks, "query_type" is a str with the node type to be explained, while for edge prediction tasks, "query_type" is a tuple with the the edge type to be explained, so that:

´´´
query_type = ("source_node_type","relation","target_node_type")
´´´
The node or edge types should be present as keys in "node_features" or "edge_index" dicts, respectively

* problem: str (OPTIONAL, BUT RECOMMENDED)
Problem to be solved ("node_prediction" for node prediction, "edge_prediction" or "link_prediction" for edge prediction, and "graph_prediction" for graph prediction). If not given, it is assumed to be "node_prediction"

* node_types : torch.tensor of int (OPTIONAL)
Tensor with node types in "heterogeneous graphs with node_features and edge_index being inputted as torch.tensor instead of dict". It is not required for homogeneous graphs or heterogeneous graphs denoted with dict variables. **Its length should match the first dimension of the "node_features" tensor** 

* edge_types : torch.tensor of int (OPTIONAL)
Tensor with edge types in "heterogeneous graphs with node_features and edge_index being inputted as torch.tensor instead of dict". It is not required for homogeneous graphs or heterogeneous graphs denoted with dict variables. **Its length should match the second dimension of the "edge_index" tensor**

**Inputs for the "run" method**
* "query_name" : str or int (REQUIRED)
Name of the node, edge or graph being explained. It can also be an int, if it is an index. **It should be present in the "names" variable for the "Explainer" object**
* "repeats" : int (OPTIONAL)
Number of model initializations to be used in the explanation pipeline. **If not given, it is 1**

**The outputs from the pipeline are:**
* config_val_df : pd.DataFrame
Dataframe with the node configuration values for node or graph prediction problems or edge configuration values for edge prediction problems. It contains a "names" column with the node or edge names from "names" variable, a "score" column with the node or edge scores, and a "std" column with the variation of scores between different initializations, if the variable "repeats" is larger than 1.

* pathway_df : pd.DataFrame
Dataframe with the aggregated community scores. It contains a "names" column with the community names from "community_names" variable, and a "score" column with the respective community scores

## Testing

To execute tests from these pipeline, run in command line:

```
pytest tests/
```
**CAUTION: THE TEST FOR comp_graph function in test_data.py script may present some issues**

## Auxiliary files
- config/configs.json: contains a recommended configuration file with explainer pipeline hyperparameters (these parameters may need to be modified)
- test_data/gcn_homo_1hop_lungCancer.pth.tar: trained homogeneous model on a lung cancer gene node classification use case, used in test suites in tests/folder
- test_data/gcn_homo_1hop_lungCancer.pth.tar: trained heterogeneous model on a lung cancer gene node classification use case, used in test suites in tests/folder
- Dockerfile: file to run tests in a GitHub runner (to be further developed in the future)

## 🚀 Installation

<!-- Uncomment this section after your first ``tox -e finish``
The most recent release can be installed from
[PyPI](https://pypi.org/project/graph_interpretability/) with:

```bash
$ pip install graph_interpretability
```
-->

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/AZ-AI/bikg-graphexplainability.git
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/AZ-AI/bikg-graphexplainability/blob/main/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

<!--
### 📖 Citation

Citation goes here!
-->

<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

-->

<!--
### 💰 Funding

This project has been supported by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
-->

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>


The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/AZ-AI/bikg-graphexplainability.git
$ cd bikg-graphexplainability
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/AZ-AI/bikg-graphexplainability/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/AZ-AI/bikg-graphexplainability.git
$ cd bikg-graphexplainability
$ tox -e docs
$ open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/graph_interpretability/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.
</details>
