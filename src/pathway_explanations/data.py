import torch
import numpy as np
import os, sys
import pandas as pd
import itertools

from torch_geometric.utils.subgraph import k_hop_subgraph
import torch.nn.functional as F
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    # Import only if GPU is available
    import cupy as cp


from pathway_explanations.pathways import Pathways


class Data:
    """
    Class to manipulate graph data for
    pathway explanations

    Params
    ------
    feat : torch.tensor of float or dict of torch.tensor
        Feature matrix
        For heterogeneous graphs, it is a dict
    edge_index : torch.tensor of float or dict of torch.tensor
        Edge indices
        For heterogeneous graphs, it is a dict

    """

    def __init__(self, feat, edge_index):
        self.feat = feat
        self.edge_index = edge_index

    def preprocess_hetero_graph(self):
        """
        Convert heterogeneous graph to a graph with common node features
        matrix and edge index matrix, including a vector for
        edge type and a vector for node type

        i.e:
        node_features = {"type1":tensor1,...,"typeN":tensorN}
        convert to:
        homo_node_features = [tensor1|...|tensorN] and
        node_types = [0000|...|NNNNNN]

        (same mechanics for edge index)

        """
        # Homogeneous graph
        hetero_node_types = None
        hetero_edge_types = None
        node_types = None
        edge_types = None
        node_pointers = None
        edge_pointers = None
        padded_dims = None

        feat_homo = self.feat
        edge_index_homo = self.edge_index

        if isinstance(self.edge_index, dict) and isinstance(self.feat, dict):
            # Heterogeneous graph (heterogeneous graphs are defined with dictionaries)
            # Extract number of relations for heterogeneous graph
            hetero_node_types = list(self.feat.keys())
            hetero_edge_types = list(self.edge_index.keys())
            relations = len(hetero_edge_types)

            (
                feat_homo,
                edge_index_homo,
                node_types,
                edge_types,
                node_pointers,
                edge_pointers,
                padded_dims,
            ) = self.hetero2homo()

        return (
            hetero_node_types,
            hetero_edge_types,
            feat_homo,
            edge_index_homo,
            node_types,
            edge_types,
            node_pointers,
            edge_pointers,
            padded_dims,
        )

    def hetero2homo(self):
        """
        Convert heterogeneous graph to homogeneous graph,
        keeping track of different node and edge types

        i.e:
        node_features = {"type1":tensor1,...,"typeN":tensorN}
        convert to:
        homo_node_features = [tensor1|...|tensorN] and
        node_types = [0000|...|NNNNNN]

        (same mechanics for edge index)

        Params
        ------

        Results
        -------
        feat_homo : torch.tensor
            Joint homogeneous feature matrix
        edge_index_homo : torch.tensor
            Joint homogeneous edge index matrix
        node_types : torch.tensor
            All node types
        edge_types : torch.tensor
            All edge types
        node_pointers : list of int
            Where node feature matrices begin for different node types
        edge_pointers : list of int
            Where edge index matrices begin for different edge indexes
        padded_dims : tuple of int
            If node features have different sizes for each type,
            they need to be padded to be set into a homogeneous
            graph. How much each node type has been padded

        """
        # Concatenate node feature tensors and extract node types
        feat_homo, node_types, padded_dims, node_pointers = self.concatenate_hetero_features()

        # Concatenate edge index tensors
        edge_index_homo, edge_types, edge_pointers = self.concatenate_hetero_edge_indices(
            node_pointers
        )

        return (
            feat_homo,
            edge_index_homo,
            node_types,
            edge_types,
            node_pointers,
            edge_pointers,
            padded_dims,
        )

    @staticmethod
    def homo2hetero(matrix, element_type, element_type_names, padded_dims=None):
        """
        Convert homogeneous graph with element types and element type
        names into a heterogeneous graph

        i.e:
        node_features = [tensor1|...|tensorN] and
        node_types = [0000|...|NNNNNN]
        convert to:
        hetero_node_features = {"type1":tensor1,...,"typeN":tensorN}

        (same mechanics for edge index)


        Params
        ------
        matrix : torch.tensor
            Homogeneous data matrix
        element_type : torch.tensor
            Type of elements in homogeneous
        element_type_names : list of str or tuple
            Names of element type
        padded_dims : tuple of int
            Padded dimensions for node features,
            if they have been previously padded
            (default: None)

        Returns
        -------
        dict_hetero : dict
            Heterogeneous dataset

        """

        # Extract the amount of element types
        unique_types = torch.unique(element_type)

        # Loop through the different element types
        # and build node features for heterogeneous graph
        dict_hetero = {}

        # assert unique_types.shape[0] == len(
        #    element_type_names
        # ), "The unique element types from the homogeneous graph do not match the length of element names"
        cont = 0

        if isinstance(element_type_names[0], tuple):
            # Analysis of node types involved in edge analysis
            all_node_types = []
            for element_name in element_type_names:
                source = element_name[0]  # Node type in source
                target = element_name[-1]  # Node type in target
                if not (source in all_node_types):
                    all_node_types.append(source)
                if not (target in all_node_types):
                    all_node_types.append(target)

        for enum_element_name, element_name in enumerate(element_type_names):
            ind_type = torch.where(element_type == enum_element_name)[0]

            if isinstance(element_type_names[0], str):
                # Node feature analysis
                dict_hetero[element_name] = matrix[ind_type]

                if padded_dims is not None:
                    if padded_dims[cont] > 0:
                        # Part of the features were padded when constructing the
                        # homogeneous graph. Undo this padding
                        dict_hetero[element_name] = dict_hetero[element_name][
                            :, : (-padded_dims[cont])
                        ]

            elif isinstance(element_type_names[0], tuple):
                # Edge index analysis
                dict_hetero[element_name] = matrix[:, ind_type].long()

                # If some index is empty, assign at least the first two members of the type
                # if ind_type.shape[0] == 0:
                #    dict_hetero[element_name] = matrix[:, :2].long()

            cont += 1

        return dict_hetero

    @staticmethod
    def hetero2homo_names(names):
        """
        When converting heterogeneous graph to homogeneous graph,
        set all names in a common list

        i.e:
        names = {"type1":["name1_type1",...,"nameM_type1"],
                ...,
                "typeN":["nameN_type1",...,"nameI_typeN"]}

        convert to

        homo_names = ["name1_type1",...,"nameM_type1"|...|"nameN_type1",...,"nameI_typeN"]
        name_types = [00000|...|NNNNNNNN]

        Params:
        ------

        names : dict or list of str
            Element names to be explained

        Returns
        -------
        homo_names : list of str
            Homogeneous element names
        type_names : torch.tensor
            Types of different names analysed

        """
        homo_names = names
        type_names = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(names, dict):
            # Heterogeneous graph
            dict_names = list(names.values())
            homo_names = list(itertools.chain.from_iterable(dict_names))
            type_names = [
                torch.zeros((len(name_list)), device=device) + i
                for i, name_list in enumerate(dict_names)
            ]
            type_names = torch.cat(type_names)

        return homo_names, type_names

    def comp_graph(self, ind, n_hops, problem, names, node_types=None, edge_types=None):
        """
        Extract k-hop computational graph

        i.e:

        node_features and edge_index should contain nodes within k-hops of
        query node or edge

        Params:
        ------
        ind : int
            Index of element of interest
        n_hops : int
            Number of hops for computational graph
        problem : str or tuple of str
            Node or edge type to be explained, only for
            heterogeneous graphs (default: None)
        names : list of str
            List of element names (default: None)
        node_types : torch.tensor
            Types of nodes in heterogeneous graphs (default: None)
        edge_types : torch.tensor
            Types of edges in heterogeneous graphs (default: None)

        Returns:
        -------
        sub_feat : torch.tensor or dict of torch.tensor
            Feature matrix for computational graph
            Dict in case of heterogeneous graph
        sub_edge_index : torch.tensor or dict of torch.tensor
            Edge index matrix for computational graph
            Dict in case of heterogeneous graph
        sub_names : list of str
            Names of elements of interest in computational graph
        sub_ind : torch.tensor
            Index of element of interest for computational graph
        sub_node_types : torch.tensor
            Types of nodes of heterogeneous graph in computational graph (default: None)
        sub_edge_types : torch.tensor
            Types of edges of heterogeneous graph in computational graph (default: None)

        """

        # Include one more hop than the coverage of the black box, else
        # the predictions for the node of interest change due how PyG
        # works internally
        n_hops += 1

        # Computational graph extraction
        node_mask, sub_edge_index, sub_ind, edge_mask = k_hop_subgraph(
            ind, n_hops, self.edge_index, relabel_nodes=True
        )

        # If the resulting edge index is empty, leave at least the node
        # self connected
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if edge_mask.sum() == 0:
            sub_edge_index = torch.tensor([[sub_ind], [sub_ind]], dtype=torch.long, device=device)

        # Filter mask
        sub_feat = self.feat[node_mask]

        # Set mapped edges to be unique
        names_array = np.array(names, dtype=str)

        # Obtain node types and edge types for heterogeneous graphs
        sub_node_types, sub_edge_types = None, None

        if node_types is not None:
            sub_node_types = node_types[node_mask]
        if edge_types is not None:
            ind_filter = torch.where(edge_mask == True)[0]
            sub_edge_types = edge_types[ind_filter]

        if ("node" in problem) or ("graph" in problem):
            sub_names = names_array[node_mask.cpu().numpy()].tolist()
        elif "edge" in problem:
            sub_names = names_array[ind_filter].tolist()

        return sub_feat, sub_edge_index, sub_names, sub_ind, sub_node_types, sub_edge_types

    def element_size(self, problem):
        """
        Obtain number of elements to explain in problem, for
        training weighted linear regression model

        i.e.
        for node or graph prediction, number of nodes is first dimension of node feature tensor
        for edge prediction, number of edges is second dimension of edge index tensor

        Params
        ------
        problem : str
            Type of problem to be analysed

        Returns
        -------
        size : int
            Size of elements of interest to be explained

        """
        if "edge" in problem:
            size = self.edge_index.shape[1]
        else:
            size = self.feat.shape[0]

        return size

    def build_edge_mask(self, mask):
        """
        Build edge mask for graph topology perturbation

        i.e.
        mask contains zeros in node indexes that are deactivated
        (mask=[0,0,1,1,0,...])
        locate node indexes that are zeroed
        (zero_node_indexes = [0,1,4,...])
        remove any edge index in the edge index matrix with a
        zero node index participating
        (if edge_index_0 = [0,6] --> REMOVE, if edge_index_1 = [2,3] --> KEEP)
        Build then edge_mask with zeros in the edges to be removed

        Params
        ------
        mask : torch.tensor
            Binary mask indicating which nodes to
            keep active or inactive

        Returns
        -------
        edge_mask : torch.tensor
            Masked edges based on masked nodes from
            input, for graph topology perturbation
        edge_indices : torch.tensor
            Concatenated edge indices

        """

        # Get indexes of active nodes
        ones_ind = torch.where(mask.flatten() == True)[0]

        # Obtain original edge indexes by concatenation
        edge_indexes = torch.hstack(
            [self.edge_index.int() + (i * mask.shape[1]) for i in range(mask.shape[0])]
        )

        # Obtain locations of active nodes
        # within source node coordinates
        # and target node coordinates
        if torch.cuda.is_available():
            source_mask = cp.in1d(
                cp.asarray(edge_indexes[0]).astype(int), cp.asarray(ones_ind).astype(int)
            )
            target_mask = cp.in1d(
                cp.asarray(edge_indexes[1]).astype(int), cp.asarray(ones_ind).astype(int)
            )
        else:
            source_mask = np.in1d(
                np.asarray(edge_indexes[0]).astype(int), np.asarray(ones_ind).astype(int)
            )
            target_mask = np.in1d(
                np.asarray(edge_indexes[1]).astype(int), np.asarray(ones_ind).astype(int)
            )

        # Keep only node locations where both sources
        # and targets are conserved
        edge_mask = torch.as_tensor(source_mask * target_mask, device=mask.device)
        del source_mask, target_mask

        return edge_mask, edge_indexes

    def perturb_node(self, mask, edge_type=None):
        """
        Provide a perturbation of the node topology,
        given a binary mask

        i.e:
        Get edge_mask with zeroes in edges to be removed, from build_edge_mask
        and remove those edges from the edge index

        edge_mask = [1,0,0,1...]
        Edge indexes in positions 1 and 2 are removed
        edge_indexes = [edge_index0,edge_index1,edge_index2,edge_index3,...]
        converts to
        final_edge_indexes = [edge_index0,edge_index3,...]

        (If there is a tensor with edge types, it is also processed)


        Params
        ------
        mask : torch.tensor
            Binary mask indicating which nodes to
            keep active or inactive
        edge_type : torch.tensor
            Edge type in heterogeneous graph (default: None)
        Returns
        -------
        perturb_edge_index : torch.tensor
            Perturbed edge index, resulting from the
            mask perturbations
        """

        # Obtain edge mask
        edge_mask, edge_indexes = self.build_edge_mask(mask)

        # Filter edge indices
        perturb_edge_index = edge_indexes[:, edge_mask]

        perturb_edge_type = None
        if edge_type is not None:
            # Heterogeneous graph
            # Filter edge types
            edge_types = torch.hstack([edge_type.int()] * (mask.shape[0]))
            perturb_edge_type = edge_types[edge_mask]

        return perturb_edge_index, perturb_edge_type

    def perturb_edge(self, mask, edge_type=None):
        """
        Provide a perturbation of the edge topology,
        given a binary mask

        i.e:
        mask contains zeroes in edges to be removed
        mask = [1,0,0,1...]
        The code removes those edges

        Edge indexes in positions 1 and 2 are removed
        edge_indexes = [edge_index0,edge_index1,edge_index2,edge_index3,...]
        converts to
        final_edge_indexes = [edge_index0,edge_index3,...]

        (If there is a tensor with edge types, it is also processed)


        Params
        ------
        mask : torch.tensor
            Binary mask indicating which nodes to
            keep active or inactive
        edge_type : torch.tensor
            Edge type in heterogeneous graph (default: None)

        Returns
        -------
        perturb_edge_index : torch.tensor
            Perturbed edge index, resulting from the
            mask perturbations
        perturb_edge_type : torch.tensor
            Perturbed edge type, resulting from the
            mask perturbations

        """

        mask_flatten = mask.flatten()

        # Shift edge indexes by perturbation index * number of nodes
        edge_indexes = torch.hstack(
            [self.edge_index.int() + i * self.feat.shape[0] for i in range(mask.shape[0])]
        )

        # Keep those edges where the mask is active
        active_ind = torch.where(mask_flatten == True)[0]
        perturb_edge_index = edge_indexes[:, active_ind]

        # Obtain edge types for perturbed graph, in the case of heterogeneous
        perturb_edge_type = None
        if edge_type is not None:
            edge_type = torch.hstack([edge_type.int()] * mask.shape[0])
            perturb_edge_type = edge_type[active_ind]

        return perturb_edge_index, perturb_edge_type

    def concat_features(self, n, node_type=None):
        """
        For graph topology perturbation purposes,
        concatenate node features as many times as
        perturbations have been created

        i.e:
        if we have node_features and n=3
        convert to:
        concat_node_features = [node_features|node_features|node_features]

        Params
        ------
        n : int
            Number of perturbations that have been created
        node_type : torch.tensor
            Node type in heterogeneous graph (default: None)

        Returns
        -------
        concat_feat : torch.tensor
            Concatenated feature matrices
        concat_type : torch.tensor
            Concatenated node types

        """
        concat_feat = torch.vstack([self.feat] * n)

        concat_type = None
        if node_type is not None:
            # Heterogeneous graph
            concat_type = torch.hstack([node_type] * n)

        return concat_feat, concat_type

    def perturbator(self, mask, problem, node_type=None, edge_type=None):
        """
        Complete graph topology perturbation for configuration
        value approximation

        i.e:
        initial node features are concatenated as many times as perturbations
        to generate in a batch

        node_features
        converts to
        final_node_features = [node_features|node_features|...]

        while edge_index is perturbed
        - if mask contains information on nodes
            . get zeroed node indexes --> get edges with zeroed node indexes
            --> remove those edges
        - if mask contains information on edges
            . get zeroed edge indexes --> remove those edges

        Params
        ------
        mask : torch.tensor
            Perturbation mask
        problem : str
            Graph element analysed
        node_type : torch.tensor
            Node type for heterogeneous graph (default: None)
        edge_type : torch.tensor
            Edge type for heterogeneous graph (default: None)

        Returns
        -------
        concat_feat : torch.tensor
            Concatenated node features
        concat_node_type : torch.tensor
            Concatenated node types
        perturbed_edge_index : torch.tensor
            Perturbed edge index
        perturbed_edge_type : torch.tensor
            Perturbed edge type

        """
        # Release un-needed memory in the GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Provide perturbations of graph topology
        # Concatenate node features
        concat_feat, concat_node_type = self.concat_features(mask.shape[0], node_type)

        # Perturb edges according to mask
        if "edge" in problem:
            perturbed_edge_index, perturbed_edge_type = self.perturb_edge(mask, edge_type)
        else:
            perturbed_edge_index, perturbed_edge_type = self.perturb_node(mask, edge_type)

        return concat_feat.float(), concat_node_type, perturbed_edge_index, perturbed_edge_type

    @staticmethod
    def config_val_dataframe(config_val_mean, config_val_std, names):
        """
        Provide dataframe with configuration values and names of
        graph elements to be explained

        i.e.
        mean results, std results, names convert to

        Dataframe:
        name1 | mean1 | std1
        ...
        nameN | meanN | stdN

        Params
        ------
        config_val_mean : torch.tensor
            Individual graph element configuration value,
            averaged over a number of repetitions
        config_val_std : torch.tensor
            Standard deviation of ndividual graph element
            configuration value, averaged over a number
            of repetitions
        names : list of str
            Individual graph element names

        Returns
        -------
        config_val_df : pd.DataFrame
            Final dataframe with information on configuration values
            and graph element names

        """
        config_val_df = {
            "name": names,
            "config_value_mean": config_val_mean.cpu().detach().numpy(),
            "config_value_std": config_val_std.cpu().detach().numpy(),
        }

        config_val_df = pd.DataFrame(config_val_df)
        config_val_df = config_val_df.set_index("name")
        config_val_df = config_val_df.sort_values(by=["config_value_mean"], ascending=False)

        return config_val_df

    def concatenate_hetero_features(self):
        """
        Process node feature tensors from heterogeneous graphs

        i.e:
        node_features = {"type1":tensor1,...,"typeN":tensorN}
        convert to:
        homo_node_features = [tensor1|...|tensorN] and
        node_types = [0000|...|NNNNNN]

        Returns
        -------
        concat_feat_tensors : torch.tensor
            Concatenate feature tensors
        node_types : torch.tensor
            Node types of the concatenated feature tensors
        padded_dimensions : tuple of int
            If node features have different sizes for each type,
            they need to be padded to be set into a homogeneous
            graph. How much each node type has been padded
        node_pointers : list of int
            Where does each node feature tensor begin,
            when it is concatenated

        """
        # Extract node feature tensor and names of types
        node_type_names = list(self.feat.keys())
        feat_tensors = list(self.feat.values())
        device = feat_tensors[0].device

        # Concatenate all node feature tensors
        # Assume all node types have the same number of features
        # TODO: allow for different feature number in different node types
        # For now, apply zero padding on the shorter tensors
        padded_feat_tensors, padded_dimensions, node_pointers = pad_feat_tensors(feat_tensors)

        concat_feat_tensors = torch.vstack(padded_feat_tensors)

        # Obtain node types
        node_types = torch.hstack(
            [
                torch.zeros((padded_feat_tensors[i].shape[0]), device=device) + i
                for i in range(len(padded_feat_tensors))
            ]
        )

        return concat_feat_tensors, node_types, padded_dimensions, node_pointers

    def concatenate_hetero_edge_indices(self, node_pointers):
        """
        For heterogeneous graphs, add to the edge index matrix an accumulated
        value from the node feature matrix, depending on the node types

        i.e:
        edge_indexes = {("node_type1","relation1","node_type2"):tensor1,
                        ...,
                        ("node_typeN_1","relationM","node_typeN"):tensorM}
        convert to:
        homo_edge_indexes = [tensor1|...|tensorN] and
        edge_types = [0000|...|NNNNNN]


        Params
        ------
        edge_index : dict
            Edge index information for heterogeneous graph
        edge_index_tensors : list of torch.tensor
            Edge index tensors in heterogeneous graph
        edge_relations : list of tuples
            Edge relations in heterogeneous graph
        pointers : list of int
            Where does each node feature tensor begin, when it is concatenated
        Returns
        -------
        concat_edge_index_tensors : list of torch.tensor
            Mapped edge index tensor according to the node feature matrices in
            the heterogeneous graph
        edge_index_types : list of torch.tensor
            Edge index types
        pointers: list of int
            Where does each tensor begin, when it is concatenated
        """
        # Extract edge index relations and tensors
        edge_relations = list(self.edge_index.keys())
        edge_tensors = list(self.edge_index.values())
        node_type_names = list(self.feat.keys())

        device = edge_tensors[0].device

        # Be careful with the node type the source and target belong to
        # Convert rows from edge index tensors
        mapped_edge_indexes = []
        edge_index_types = []
        pointers = []

        # Edge index tensor counter
        i = 0

        # Edge counter
        pointer = 0

        for edge_relation, edge_tensor in zip(edge_relations, edge_tensors):
            pointers.append(pointer)
            source_ind = node_type_names.index(edge_relation[0])
            target_ind = node_type_names.index(edge_relation[-1])

            # Quantity that has to be added to
            # each row of the edge index, based on the node type
            adding = torch.tensor(
                [[node_pointers[source_ind], node_pointers[target_ind]]], device=device
            ).T

            mapped_edge_indexes.append(torch.add(edge_tensor, adding))

            # mapped_edge_indexes.append(edge_tensor)

            # Store edge types in here
            edge_index_types.append(torch.zeros((edge_tensor.shape[-1]), device=device) + i)

            # Update edge counter
            i += 1
            pointer += edge_tensor.shape[-1]

        # Concatenate edge indexes and edge types
        concat_edge_indexes = torch.hstack(mapped_edge_indexes)
        edge_index_types = torch.hstack(edge_index_types)

        return concat_edge_indexes, edge_index_types, pointers


def pad_feat_tensors(feat_tensors):
    """
    Produce padded feature tensors so that
    all tensors have same number of features
    and a homogeneous graph can be built

    i.e.
    feat_tensors from node_features (a dict of node features in homogeneous graph)
    feat_tensors = [tensor1 (X features),...,tensorN (Y features)]
    Checks the largest feature size in the N tensors, then converts to:
    padded_tensors = [tensor1 (padded to MAX features found),
                        ...,
                        tensorN (padded to MAX features found)]

    Params
    ------
    feat_tensors : list of torch.tensors
        Feature tensors for different node types

    Returns
    -------
    padded_tensors : list of torch.tensors
        Padded feature tensors for different node types
    padded_dimensions : list of int
        How much each tensor has been padded
    pointers: list of int
        Where does each tensor begin, when it is concatenated

    """

    feat_numbers = [feat_tensor.shape[1] for feat_tensor in feat_tensors]
    max_feat_number = max(feat_numbers)  # Largest feature number on all node types

    padded_tensors = []
    padded_dimensions = []
    pointers = []

    pointer = 0

    for feat_tensor in feat_tensors:
        diff = max_feat_number - feat_tensor.shape[1]
        padded_dimensions.append(diff)

        pointers.append(pointer)

        pointer += feat_tensor.shape[0]

        if diff > 0:
            padded_tensor = F.pad(feat_tensor, (0, diff))
            padded_tensors.append(padded_tensor)
        else:
            padded_tensors.append(feat_tensor)

    return padded_tensors, padded_dimensions, pointers
