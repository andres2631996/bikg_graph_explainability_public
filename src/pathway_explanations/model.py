import torch
import os, sys
import torch_geometric
from torch_geometric.utils.subgraph import get_num_hops
import inspect


from pathway_explanations.data import Data


class Model:
    """
    Class to manipulate graph data for
    pathway explanations

    Params
    ------
    arch : PyG model
        Model to be explained
    params : dict
        Hyperparameters

    """

    def __init__(self, arch):
        self.arch = arch

    def get_hops(self, num_relations=0):
        """
        Obtain number of hops of model
        If model is heterogeneous, the number of hops
        are divided by the number of relations in the
        graph

        i.e.
        if model has X convolutional operators (GCN, GAT, GIN, APPNP,...)
        get hops=X

        Params
        ------
        num_relations : int
            Number of relations in edge indices, if
            graph is heterogeneous (default: 0)

        Returns
        -------
        hops : int
            Number of hops

        """
        # Obtain model hops
        num_hops = get_num_hops(self.arch)

        # If model is heterogeneous, the number of hops
        # is divided by the number of relations in the
        # graph
        if num_relations > 0:
            num_hops //= num_relations

        return num_hops

    def infer(self, feat, edge_index, node_types=None, edge_types=None):
        """
        Perform inference on a graph sample

        i.e.
        Take the graph, defined by the node features and the edge indexes
        (and node types and edge types for heterogeneous case),
        and run inference on a previously trained model

        Params
        ------
        feat : torch.tensor of float or dict of torch.tensor
            Feature matrix
            For heterogeneous graphs, it is a dict
        edge_index : torch.tensor of float or dict of torch.tensor
            Edge indices
            For heterogeneous graphs, it is a dict
        node_types : torch.tensor
            Types of nodes in heterogeneous graph
            Default: None
        edge_types : torch.tensor
            Types of edges in heterogeneous graph
            Default: None


        Returns
        -------
        out : torch.tensor
            Predicted output
            (node-wise, edge-wise, or graph-wise,
            depending on task at hand)

        """

        # self.arch = self.arch.eval()

        # Release un-needed memory in the GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Number of input arguments to architecture
        # Determine if we need to input node types and edge types as well
        args = inspect.getargspec(self.arch.forward).args

        with torch.no_grad():
            if len(args) == 3:
                # Node features and edge indexes as arguments
                out = self.arch(feat, edge_index)
            elif (len(args) == 5) and (node_types is not None) and (edge_types is not None):
                # Include node types and edge types in architecture (+ node features and edge indexes)
                out = self.arch(feat, edge_index, node_types, edge_types)

        del feat, edge_index

        return out

    def predict_hetero_output(
        self,
        feat,
        edge_index,
        node_types,
        edge_types,
        node_type_names,
        edge_type_names,
        num_perturbs,
        num_nodes,
        sub_ind=None,
        padded_dims=None,
        problem="node_prediction",
    ):
        """
        Function to run model inference on heterogeneous graph
        with more than one node type or edge type. The heterogeneous
        graph is homogenized (with tensors as node features and edge indexes,
        plus node types and edge types), so the node features need to be converted back
        into a dictionary of node types, while the edge indexes need to
        be converted back to a dictionary of edge types. The converted graph
        can then be used for model inference

        i.e.
        homogenized_features = [features_type1|...|features_typeN]
        homogenized_edge_indexes = [edge_index_type1|...|edge_index_typeM]

        converts to:

        features = {"node_type1":features_type1,...,"node_typeN":features_typeM}
        edge_indexes = {("node_type1","relation1","node_type2"):edge_index_type1,
                        ...,
                        ("node_typeM_1","relationM","node_typeM"):edge_index_typeM}
        The indexes in edge_indexes may be subtracted the accumulated shift from
        previous node types, if there is more than one node type

        Then,
        model output = GNN(features,edge_indexes)

        Params:
        ------
        feat : torch.tensor
            Homogenized node features
        edge_index : torch.tensor
            Homogenized edge index
        node_types : torch.tensor
            Indexes with node types for heterogeneous graph
        edge_types : torch.tensor
            Indexes with edge types for heterogeneous graph
        node_type_names : list of str
            Corresponding names of node types for heterogeneous graph
        edge_type_names : list of tuple of str
            Corresponding names of edge types for heterogeneous graph
        num_perturbs : int
            Number of perturbations for explanation pipeline
        num_nodes : int
            Number of graph elements to be explained
        sub_ind : int (default: None)
            Index of element to be explained in heterogeneous graph
        padded_dims : tuple of int (default: None)
            Padded dimensions in case that original node features in
            heterogeneous graph were not the same across node types
        problem : str
            Problem that is being modelled
            ("node_prediction","edge_prediction","graph_prediction")

        """

        # Extract the amount of element types
        unique_node_types = torch.unique(node_types)
        unique_edge_types = torch.unique(edge_types)

        # Extract heterogeneous feature matrix dictionary
        # Extract heterogeneous edge index dictionary
        hetero_outputs = []

        values = edge_index[0, :]

        for perturb in range(num_perturbs):
            # Feature indexes inside perturbation
            indexes = torch.arange(perturb * (num_nodes), (perturb + 1) * (num_nodes))

            perturb_feat = feat[indexes]
            perturb_feat_types = node_types[indexes]

            # Locate edge indexes involved in perturbation
            largerthan = values >= (perturb * num_nodes)  # Inferior bound for edge indices
            lowerthan = values < ((perturb + 1) * num_nodes)  # Superior bound for edge indices
            prod = lowerthan * largerthan  # Intersection of inferior and superior bounds
            edge_index_ind = torch.where(prod == True)[0]

            perturb_edge_index = edge_index[:, edge_index_ind]
            perturb_edge_index -= indexes[0]
            perturb_edge_index_types = edge_types[edge_index_ind]

            if perturb_edge_index_types.shape[0] == 0:
                hetero_outputs.append(0)
                continue

            # Determine where the different node types begin and start
            if perturb == 0:
                node_pointers = []
                for unique_node_type in unique_node_types:
                    w = torch.where(node_types == unique_node_type)[0]
                    node_pointers.append(w[0])

            # Convert into heterogeneous graph
            data_class = Data(perturb_feat, perturb_edge_index)
            perturb_feat_dict = data_class.homo2hetero(
                perturb_feat, perturb_feat_types, node_type_names, padded_dims
            )

            perturb_edge_index_dict = data_class.homo2hetero(
                perturb_edge_index, perturb_edge_index_types, edge_type_names
            )

            # Iterate through edge index fields
            # Fix shift of different edge index types caused by the presence of different node types
            for edge_type_name in edge_type_names:
                edge_index_matrix = perturb_edge_index_dict[edge_type_name]
                ind_source = node_type_names.index(edge_type_name[0])
                ind_target = node_type_names.index(edge_type_name[-1])
                edge_index_matrix[0] -= node_pointers[ind_source]
                edge_index_matrix[1] -= node_pointers[ind_target]
                perturb_edge_index_dict[edge_type_name] = edge_index_matrix

            out = self.arch(perturb_feat_dict, perturb_edge_index_dict)

            if ("node" in problem) and (sub_ind is not None):
                out = out[sub_ind, 0].item()

            hetero_outputs.append(out)

        hetero_outputs = torch.tensor(hetero_outputs, device=feat.device)

        return hetero_outputs

    @staticmethod
    def hetero2homo_output(hetero_output):
        """
        If output from heterogeneous graph is a dict, convert it
        back into homo by concatenating the different types

        i.e.
        output = {"type1":[out1_1,...,outN_1],...,"typeM":[out1_M,...,outI_M]}
        converts to
        final_output = [out1_1,...,outN_1|...|out1_M,...,outI_M]

        Params
        ------
        hetero_output : dict
            Output from heterogeneous parametric relational model

        Returns
        -------
        output : torch.tensor
            Homogeneous output
        output_types : torch.tensor
            Types of stacked homogeneous outputs

        """
        if isinstance(hetero_output, torch.Tensor):
            # If output is already a tensor, no need for reformatting
            return hetero_output, None

        values = list(hetero_output.values())
        output = torch.vstack(values)
        output_types = torch.hstack(
            [
                torch.zeros((len(value)), device=value.device, dtype=torch.int) + i
                for i, value in enumerate(values)
            ]
        )

        return output, output_types

    @staticmethod
    def extract_node_edge_output(output, ind, n):
        """
        In node and edge prediction tasks, we get a node-wise
        or edge-wise prediction, while we only have to extract
        values for the node or edge of interest that we want to
        explain, so we only have to focus on that

        i.e.

        We have concatenated perturbations from three outputs
        outputs = [output1|...|outputN]
        What we have to do is to take the output in position "i" from each case
        final_outputs = [output1_indexI,output2_indexI,...,outputN_indexI]

        Params
        ------
        output : torch.tensor
            Original output (number of nodes/edges X number of perturbations)
        ind : int
            Index of node or edge to locate
        n : int
            Number of elements to be explained

        Returns
        -------
        individual_output : torch.tensor
            Final perturbed outputs for the individual node/edge

        """

        all_ind = torch.arange(start=ind, end=output.shape[0], step=n)
        individual_output = output[all_ind]

        return individual_output
