import os, sys
import torch
from torch import nn
import numpy as np
import itertools

from torch_geometric.nn import GCNConv, Linear, HeteroConv, GATConv


class GCN_homo(torch.nn.Module):  # Homogeneous GCN
    def __init__(self, node_features):
        super().__init__()

        # Load hyperparameters
        seed = 0
        hidden_channels = [16]
        out_neurons = 1
        conv_layers = [16]
        fc_layers = [16, 16, 32]

        assert isinstance(seed, int) or isinstance(seed, float), "Seed is not numeric"
        assert isinstance(hidden_channels, list), "Number of hidden channels is not list"
        assert isinstance(out_neurons, int) or isinstance(
            out_neurons, float
        ), "Number of output neurons is not numeric"
        assert isinstance(conv_layers, list), "Convolutional layer sizes is not list"
        assert isinstance(fc_layers, list), "Fully-connected layer sizes is not list"

        assert (
            conv_layers[-1] == fc_layers[0]
        ), "The size of the last convolutional layer must match the size of the first fully-connected layer"

        seed, out_neurons = int(seed), int(out_neurons)

        torch.manual_seed(seed)

        conv_list, fc_list = [], []

        for enum_conv_layer, conv_layer in enumerate(conv_layers):  # Setup convolutional backbone
            assert isinstance(conv_layer, int) or isinstance(
                conv_layer, float
            ), "Size of convolutional layer is not numeric"
            conv_layer = abs(int(conv_layer))

            if enum_conv_layer == 0:
                conv = GCNConv(node_features, conv_layer)
            else:
                conv = GCNConv(conv_layers[enum_conv_layer - 1], conv_layer)

            conv_list.append(conv)
            conv_list.append(nn.ReLU())

        self.conv = nn.ModuleList(conv_list)

        for enum_fc_layer, fc_layer in enumerate(fc_layers):  # Setup linear backbone

            assert isinstance(fc_layer, int) or isinstance(
                fc_layer, float
            ), "Size of convolutional layer is not numeric"

            if enum_fc_layer == (len(fc_layers) - 1):  # Last FC layer
                lin = Linear(fc_layers[enum_fc_layer], out_neurons)
                act = nn.Sigmoid()
            else:
                lin = Linear(fc_layers[enum_fc_layer], fc_layers[enum_fc_layer + 1])
                act = nn.ReLU()
            fc_list.append(lin)
            fc_list.append(act)

        self.fc = nn.ModuleList(fc_list)

    def forward(self, x, edge_index):

        for enum_c, c in enumerate(self.conv):  # Convolutional backbone
            if enum_c % 2 == 0:  # Even layer: convolution
                x = c(x, edge_index)
            else:  # Odd layer: activation function
                x = c(x)

        for l in self.fc:  # Fully-connected backbone
            x = l(x)

        return x


class GCN(torch.nn.Module):  # Heterogeneous GCN
    def __init__(self, node_features):
        super().__init__()

        # Load hyperparameters
        seed = 0
        hidden_channels = [2]
        out_neurons = 1
        conv_layers = [2]
        fc_layers = [2, 2, 4]
        node_types = ["0", "1"]
        relations = ["a", "b"]
        edge_types = [("0", "a", "1"), ("1", "b", "0")]

        assert isinstance(seed, int) or isinstance(seed, float), "Seed is not numeric"
        assert isinstance(hidden_channels, list), "Number of hidden channels is not numeric"
        assert isinstance(out_neurons, int) or isinstance(
            out_neurons, float
        ), "Number of output neurons is not numeric"
        assert isinstance(node_types, list), "Node types is not list"
        assert isinstance(conv_layers, list), "Convolutional layer sizes is not list"
        assert isinstance(fc_layers, list), "Fully-connected layer sizes is not list"

        assert (
            conv_layers[-1] == fc_layers[0]
        ), "The size of the last convolutional layer must match the size of the first fully-connected layer"

        for node_type in node_types:
            assert isinstance(node_type, str), "Node type is not string"

        seed, out_neurons = int(seed), int(out_neurons)

        torch.manual_seed(seed)

        conv_list, fc_list = [], []

        for enum_conv_layer, conv_layer in enumerate(conv_layers):  # Setup convolutional backbone
            hetero_dict = {}  # Heterogeneous dictionary where to store all heterogeneous relations

            assert isinstance(conv_layer, int) or isinstance(
                conv_layer, float
            ), "Size of convolutional layer is not numeric"

            conv_layer = abs(int(conv_layer))

            # for relation in relations:
            for conv_layer in conv_layers:
                conv = HeteroConv(
                    {
                        edge_type: GATConv((-1, -1), conv_layer, add_self_loops=False)
                        for edge_type in edge_types
                    },
                    aggr="sum",
                )

            # conv = HeteroConv(hetero_dict, aggr="sum")  # GCN on top of heteregeneous relations
            conv_list.append(conv)
            conv_list.append(nn.ReLU())

        self.conv = nn.ModuleList(conv_list)

        for enum_fc_layer, fc_layer in enumerate(fc_layers):  # Setup linear backbone

            assert isinstance(fc_layer, int) or isinstance(
                fc_layer, float
            ), "Size of convolutional layer is not numeric"

            if enum_fc_layer == (len(fc_layers) - 1):  # Last FC layer
                lin = Linear(fc_layers[enum_fc_layer], out_neurons)
                act = nn.Sigmoid()
            else:
                lin = Linear(fc_layers[enum_fc_layer], fc_layers[enum_fc_layer + 1])
                act = nn.ReLU()
            fc_list.append(lin)
            fc_list.append(act)

        self.fc = nn.ModuleList(fc_list)

    def forward(self, x_dict, edge_index_dict):

        for enum_c, c in enumerate(self.conv):  # Convolutional backbone

            if enum_c % 2 == 0:  # Even layers: convolution
                if enum_c == 0:
                    x = c(x_dict, edge_index_dict)
                else:
                    x = c(x[list(x.keys())[0]], edge_index_dict)
            else:  # Odd layers: activation function
                x = {key: c(x) for key, x in x.items()}

        for enum_l, l in enumerate(self.fc):  # Fully-connected backbone
            if enum_l == 0:
                x = l(x[list(x.keys())[0]])
            else:
                x = l(x)

        return x


def sort_feature(feat, node_types, names):
    """
    Sort node feature tensor according to node names

    Params
    ------
    feat : torch.tensor
        Feature matrix to be sorted
    node_types : torch.tensor
        Initial node types to be sorted
    names : list of str
        Node names used for sorting, assume names are integers

    Returns
    -------
    sort_feat : torch.tensor
        Sorted feature matrix
    sort_node_types : torch.tensor
        Sorted node types (None for homogeneous graph)
    sort_names : list of str
        Sorted node names

    """

    try:
        names_ind = torch.tensor([int(name) for name in names], dtype=int, device=feat.device)
    except:
        print("There are non-numeric names in the node names")

    argsort = torch.argsort(names_ind)
    sort_names = [names[i] for i in argsort]

    sort_feat = feat[argsort]

    sort_node_types = None
    if node_types is not None:
        sort_node_types = node_types[argsort]

    return sort_feat, sort_node_types, sort_names


def sort_edge_index(edge_index, edge_type):
    """
    Sort edge index and edge types according to the node index
    in the edge index matrix

    Params
    ------
    edge_index : torch.tensor
        Edge index to be sorted
    edge_type : torch.tensor
        Edge type to be sorted

    Returns
    -------
    sorted_edge_index : torch.tensor
        Sorted edge index
    sorted_edge_type : torch.tensor
        Sorted edge type (None for homogeneous graph)

    """

    argsort = torch.argsort(edge_index)[0]
    sorted_edge_index = edge_index[:, argsort]

    sorted_edge_type = None
    if edge_type is not None:
        sorted_edge_type = edge_type[argsort]

    return sorted_edge_index, sorted_edge_type


def sort_numerical_pathways(pathways):
    """
    Transform all elements of pathways into integers
    and sort those integers inside

    Params
    ------
    pathways : list of lists of str
        Pathway set


    Returns
    -------
    out_pathways : list of lists of int
        Sorted out pathways, in numerical format

    """
    out_pathways = []
    for i in range(len(pathways)):
        out_pathway = [int(element) for element in pathways[i]]
        out_pathway.sort()
        out_pathways.append(out_pathway)

    return out_pathways


def check_suitability_external_mask(pathways, pathway_ind, external_mask):
    """
    Check if external mask given is suitable.
    All communities except the internal one (given by pathway_ind),
    should be True or False in each perturbation of the
    external mask, without taking into account those graph
    elements that lie in the internal community of interest

    Params
    ------
    pathways : list of int
        Communities to be analyzed
    pathway_ind : int
        Pathway index that is just internally analyzed
    external_mask : torch.tensor
        External mask

    """
    # Iterate through all the pathways
    assert (pathway_ind >= 0) and (
        pathway_ind < len(pathways)
    ), "Wrong pathway index: it should be zero or positive and lower than the number of pathways"

    for i, pathway in enumerate(pathways):

        if i != pathway_ind:  # Go through all external communities

            # Store here a copy of all pathways to have here all the pathways except the one under analysis
            pathways_copy = pathways.copy()
            del pathways_copy[i]
            # Store all the other pathways in here
            other_pathways = np.array(list(itertools.chain.from_iterable(pathways_copy))).astype(
                int
            )

            pathway_to_analyze = pathway  # Generate a copy of the pathway to be analyzed

            # Check if there is any element in any other pathway in
            # common with the internal community, and
            # do not take it into account, since if the element is
            # active in the other pathway, it will cause a collision
            _, _, ind = np.intersect1d(
                other_pathways, np.array(pathway_to_analyze).astype(int), return_indices=True
            )

            if len(ind) > 0:
                # Some common elements have been found,
                # remove them from the pathway to be analyzed
                pathway_to_analyze = np.array(pathway_to_analyze).astype(int)
                pathway_to_analyze = np.delete(pathway_to_analyze, ind).tolist()

            # Sampled communities
            sampled_pathways = external_mask[:, pathway_to_analyze]
            # Sum sampled pathways: the sum has to be zero (all False)
            # or the number of elements (all True)
            sum_sampled_pathways = torch.sum(sampled_pathways, 1)
            unique_sum = torch.unique(sum_sampled_pathways)

            # Equal conditions
            device = external_mask.device

            # See if we have both all True and all False in all perturbations
            equal_both = torch.equal(
                unique_sum,
                torch.tensor([0, len(pathway_to_analyze)], device=device, dtype=torch.long),
            )
            # See if we have all False in all perturbations
            equal_zero = torch.equal(unique_sum, torch.tensor([0], device=device, dtype=torch.long))
            # See if we have all True in all perturbations
            equal_one = torch.equal(
                unique_sum, torch.tensor([len(pathway_to_analyze)], device=device, dtype=torch.long)
            )

            assert equal_both or equal_zero or equal_one


def load_GCN_model(num_features, file, trained=True, eval=True, hetero=False):
    """
    Load homogeneous GCN model from state file
    It loads the checkpoint "gcn_homo_1hop_lungCancer.pth.tar"
    in "test_data" folder, if trained == True

    Params:
    ------
    num_features : int
        Number of input features for the model
    file : str or None
        File where model is stored
        If None, the trained model is not loaded
    trained : bool (default: True)
        If there is a trained model (True) or not (False)
    eval : bool (default: True)
        If the model needs to undergo evaluation (True) or not (False)
    hetero : bool (default: False)
        If the model deals with a heterogeneous graph (True) or homogeneous (False)


    Returns
    -------
    arch : torch geometric model
        Loaded model

    """
    if hetero:  # R-GCN
        arch = GCN(num_features)
    else:  # GCN
        arch = GCN_homo(num_features)  # Assume the model takes as input 84 hidden features

    if file is not None:
        if trained and os.path.exists(file) and "pth.tar" in file:
            state_dict = torch.load(file)["model"]
            arch.load_state_dict(state_dict)
            if eval:
                arch.eval()

    return arch


def parameter_unpacker(param):
    """
    Extract values from parameter generator
    attached to a certain PyTorch model

    Params
    ------
    param : PyTorch model parameter
        Initial generator

    Returns
    -------
    values : list of torch.tensor
        List of actual model parameters in all model
        layers

    """
    values = [p.data for p in param]  # Store names and tensors in this dict

    return values


def generate_random_tensor(shape, device):
    """
    Generate random tensor for testing set_seed
    function of explainer.py script

    Params:
    ------
    shape : tuple
        Shape of random tensor to be generated
    device : torch.device
        Computational device

    Returns:
    -------
    tensor : torch.tensor
        Random tensor generated

    """
    tensor = torch.rand(size=shape, device=device)
    return tensor
