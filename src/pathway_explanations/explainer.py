import torch
import numpy as np
import pandas as pd
import random
import sys

from pathway_explanations.data import Data
from pathway_explanations.masks import Mask
from pathway_explanations.model import Model
from pathway_explanations.pathways import Pathways
from pathway_explanations.wlm import LinearRegression, train_model


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Explainer:
    """
    Main class for pathway explainer

    Params:
    ------
    feat : torch.tensor of float or dict of torch.tensor
        Feature matrix
        For heterogeneous graphs, it is a dict
    edge_index : torch.tensor of float or dict of torch.tensor
        Edge indices
        For heterogeneous graphs, it is a dict
    arch : PyG model
        Model to be explained
    params : dict
        Hyperparameters
    names : list of str or dict
        List of element names
        Dict for heterogeneous graph
    pathways : list of ints or str or dict (default: None)
        Group of nodes or node indexes with
        pathway information
        In case of heterogeneous graphs, it is a dict with the
        different node or edge types
        If it is None, approximate Shapley values instead of
        Configuration Values
    pathway_names : list of str or dict
        List of pathway names (default: None)
        Dict in case of heterogenous graph
    element_type : str or tuple of str
        Node or edge type to be explained, only for
        heterogeneous graphs (default: None)
    problem : str
        Type of problem at hand
        (default: "node_prediction", but may be
        "edge_prediction" or "graph_prediction")
    node_types : torch.tensor of int
        Custom node types, in case that heterogeneous graphs
        converted back into homogeneous graphs are used
        Default: None
    edge_types : torch.tensor of int
        Custom edge types, in case that heterogeneous graphs
        converted back into heterogeneous graphs are used
        Default: None

    """

    def __init__(
        self,
        feat,
        edge_index,
        arch,
        params,
        names,
        pathways=None,
        pathway_names=None,
        element_type=None,
        problem="node_prediction",
        node_types=None,
        edge_types=None,
    ):

        # Conduct initial assertions
        self.initial_assertions(
            feat, edge_index, arch, params, names, pathways, pathway_names, element_type, problem
        )

        problem = problem.lower().strip()

        self.feat = feat
        self.edge_index = edge_index
        self.arch = arch
        self.params = params
        self.names = names
        self.pathways = pathways
        self.pathway_names = pathway_names
        self.element_type = element_type
        self.problem = problem
        self.node_types = node_types
        self.edge_types = edge_types

    @staticmethod
    def initial_assertions(
        feat, edge_index, arch, params, names, pathways, pathway_names, element_type, problem
    ):

        """
        Conduct initial datatype assertions

        Params:
        ------
        feat : torch.tensor or dict of torch.tensor
            Node features
        edge_index : torch.tensor or dict of torch.tensor
            Edge index
        arch : torch.nn.Module
            Architecture
        params : dict
            Hyperparameter
        names : list of str or dict of list of str
            Element names
        pathway_names : list of str or dict of list of str
            Graph community names
        element_type : str
            In heterogeneous graphs, node or edge type that is explained
        problem : str
            Problem type ("node_prediction","edge_prediction","graph_prediction")


        """

        #
        if pathways is not None:
            assert isinstance(pathways, list) or isinstance(
                pathways, dict
            ), "Pathways is not list or dict"
        if pathway_names is not None:
            assert isinstance(pathway_names, list) or isinstance(
                pathway_names, dict
            ), "Pathway names is not list or dict"
            assert len(pathway_names) == len(
                pathways
            ), "Length of list with pathway names and list with pathway indexes do not match"

        assert isinstance(feat, torch.Tensor) or isinstance(
            feat, dict
        ), "Feature matrix is not torch tensor or dict"
        assert isinstance(edge_index, torch.Tensor) or isinstance(
            edge_index, dict
        ), "Edge index matrix is not torch tensor or dict"
        assert isinstance(names, list) or isinstance(
            names, dict
        ), "Element names is not list or dict"

        # TODO: allow other data types to be processed, in a future iteration
        # TODO: how to properly assert the datatype of a model object
        # assert isinstance(model,
        assert isinstance(params, dict), "Hyperparameters given is not dictionary"

        assert isinstance(problem, str), "Problem type given is not string"

        if element_type is not None:
            # TODO: assume for now we are only explaining one element of one type in heterogeneous graphs
            assert isinstance(element_type, str) or isinstance(
                element_type, tuple
            ), "Element type is not string (node) nor tuple (edge)"

            if "node" in problem:
                assert isinstance(feat, dict), "Feature given is not a dict of node types"
                node_types = list(feat.keys())
                assert (
                    element_type in node_types
                ), "Node type '{}' is not among input node types in heterogeneous graph".format(
                    element_type
                )
            elif "edge" in problem:
                assert isinstance(
                    edge_index, dict
                ), "Edge index given is not a dict of edge index types"
                edge_types = list(edge_index.keys())
                assert (
                    element_type in edge_types
                ), "Edge type '{}' is not among input node types in heterogeneous graph".format(
                    element_type
                )

    @staticmethod
    def extract_index(element, names=None):
        """
        Obtain index of element of interest

        i.e.
        names = [name1,...,query,...,nameN]
        Get index of query

        Params
        ------
        element : str
            Element name (node/edge/graph) to be explained
        names : list of str
            List of node/edge/graph names (default: None)

        Returns
        -------
        gene_ind : int
            Index of gene of interest

        """
        # TODO: expand to more than one node, in a future iteration
        if names is None:
            # If no names are given, assume the node name is already
            # a node index
            assert isinstance(element, int) or isinstance(
                element, float
            ), "No element names have been given and the node name given is not numeric"
            return int(element)

        assert element in names, "Element name '{}' is not present in the graph".format(element)

        names_array = np.array(names, dtype=str)
        ind = int(np.where(names_array == element)[0])
        return ind

    def filter_hetero_names(self, names, node_type, edge_type, node_type_names, edge_type_names):
        """
        In case we have a heterogeneous graph where the model to
        be explained only works with one type of node or edge,
        get index of sample to study from only that certain
        node type or edge type

        i.e:

        node_features = {"type1": tensor1, ..., "typeN": tensorN}
        names = {"type1":[name1_1,...,query,...],"typeN": [nameN_1,...,nameN_J]]}
        If GNN only works with tensor1, get index of query node or edge
        in tensor1

        Params
        ------
        names : list of str
            Element names to be filtered
        element_type : str or tuple
            Node type or edge type to be explained
        node_type : torch.tensor
            Numeric labels for node type
        edge_type : torch.tensor
            Numeric labels for edge type
        node_type_names : list of str
            Names of node types
        edge_type_names : list of tuple
            Names of edge types

        Returns
        -------
        filtered_names : list of str
            Filtered element names for node type
            or edge type of interest

        """

        names_array = np.array(names, dtype=str)

        if isinstance(self.element_type, str):
            # Node explanations
            ind = node_type_names.index(self.element_type)
            element_index = torch.where(node_type == ind)[0]
            filtered_names = names_array[element_index.cpu().numpy()]

        elif isinstance(self.element_type, tuple):
            # Edge explanations
            ind = edge_type_names.index(self.element_type)
            element_index = torch.where(edge_type == ind)[0]
            filtered_names = names_array[element_index.cpu().numpy()]

        elif self.element_type is None:
            # Take that element with node type 1
            # Heterogeneous graphs that were converted into
            # homogeneous prior to the explanation pipeline
            element_index = torch.where(node_type == 1)[0]
            filtered_names = names_array[element_index.cpu().numpy()]

        return filtered_names.tolist()

    @staticmethod
    def weight_stacking(weights):
        """
        Create a stack of weights from different repetitions

        i.e:
        explanations = [[value1_node1,...,value1_nodeN],...,
                        [valueI_node1,...,valueI_nodeN]]

        Params
        ------
        weights : list of torch.tensor
            Set of weights

        Returns
        -------
        stack_mean : torch.tensor
            Mean weights
        stack_std : torch.tensor
            Standard deviation of weights

        """
        stack = torch.vstack(weights)
        stack_mean = torch.mean(stack, 0)
        stack_std = torch.std(stack, 0, unbiased=False)

        return stack_mean, stack_std

    def run(self, element, times=1):
        """
        Main execution function for pathway explanations

        Params
        ------
        element : str
            Element name (node/edge/graph) to be explained
        times : int
            Number of times for computations to be repeated

        Returns
        -------
        config_val_df : pd.DataFrame
            Node values, sorted descendingly
        pathway_df : pd.DataFrame
            Pathway-wise aggregated values,
            sorted descendingly

        """

        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up deterministic seed for random processes,
        # if the computations are executed an only time
        if times == 1:
            set_seed(self.params["seed"])

        # Define data class
        unprocessed_data_class = Data(self.feat, self.edge_index)

        # Define pathway class
        if self.pathways is not None:
            unprocessed_pathway_class = Pathways(self.pathways, self.pathway_names)

        # Preprocess heterogenous graph to a homogeneous graph
        (
            hetero_node_types,
            hetero_edge_types,
            self.feat,
            self.edge_index,
            node_types,
            edge_types,
            node_pointers,
            edge_pointers,
            padded_dims,
        ) = unprocessed_data_class.preprocess_hetero_graph()

        if node_types is None and self.node_types is not None:
            # Homogeneous graph with custom node types
            node_types = self.node_types.clone()

        if edge_types is None and self.edge_types is not None:
            # Homogeneous graph with custom node types
            edge_types = self.edge_types.clone()

        # Preprocess names of heterogeneous graph to homogeneous graph
        self.names, name_types = unprocessed_data_class.hetero2homo_names(self.names)

        # Preprocess heterogeneous pathways to homogeneous pathways
        if self.pathways is not None:
            (
                self.pathways,
                self.pathway_names,
                pathway_types,
            ) = unprocessed_pathway_class.hetero2homo(self.problem, node_pointers, edge_pointers)
            pathway_class = Pathways(self.pathways, self.pathway_names, pathway_types)

        # Extract computational graph, for node or edge prediction
        # Extract pathways from computational graph

        data_class = Data(self.feat, self.edge_index)

        sub_pathway = None
        sub_pathway_names = None
        sub_pathway_types = None
        sub_node_types = None
        sub_edge_types = None

        if not ("graph" in self.problem):

            # Extract number of hops analyzed by model
            model_class = Model(self.arch)
            relations = 0
            if hetero_edge_types is not None:
                relations = len(hetero_edge_types)

            n_hops = model_class.get_hops(relations)

            # Index extraction
            ind = self.extract_index(element, self.names)

            # Computational graph extraction
            (
                sub_feat,
                sub_edge_index,
                sub_names,
                sub_ind,
                sub_node_types,
                sub_edge_types,
            ) = data_class.comp_graph(ind, n_hops, self.problem, self.names, node_types, edge_types)

            if self.pathways is not None:
                # Computational graph extraction for pathways
                (
                    sub_pathway,
                    sub_pathway_names,
                    sub_pathway_types,
                ) = pathway_class.comp_graph(sub_names)

        else:
            # No need for computational graph extraction for graph prediction
            sub_feat = self.feat.clone()
            sub_edge_index = self.edge_index.clone()
            sub_names = self.names
            sub_ind = self.extract_index(element, sub_names)

            sub_node_types = None
            if node_types is not None:
                # Heterogeneous graph
                sub_node_types = node_types.clone()

            sub_edge_types = None
            if edge_types is not None:
                # Heterogeneous graph
                sub_edge_types = edge_types.clone()

            if self.pathways is not None:
                sub_pathway = self.pathways
                sub_pathway_names = self.pathway_names
                sub_pathway_types = pathway_types

        # If the problem is not graph prediction, extract
        # node index or edge index of interest
        if not ("graph" in self.problem) and (
            self.element_type is not None
            or self.node_types is not None
            or self.edge_types is not None
        ):
            # Heterogeneous graph where the model to be
            # explained only provides outputs for a certain
            # node type or edge type
            # Filter names of heterogenous elements
            sub_names_filtered = self.filter_hetero_names(
                sub_names, sub_node_types, sub_edge_types, hetero_node_types, hetero_edge_types
            )
            sub_ind = self.extract_index(element, sub_names_filtered)

        # Transform pathway structure from string names to numerical indexes
        sub_pathway_inds = None
        if self.pathways is not None:
            sub_pathway_class = Pathways(sub_pathway, sub_pathway_names)
            if isinstance(sub_pathway[0][0], str):
                sub_pathway_inds = sub_pathway_class.names2inds(sub_names)
            elif isinstance(sub_pathway[0][0], int):
                sub_pathway_inds = sub_pathway

            sub_names_array = np.array(sub_names, dtype=str)

        del self.feat, self.edge_index

        # Obtain size of elements to explain
        sub_data_class = Data(sub_feat, sub_edge_index)
        elements = sub_data_class.element_size(self.problem)

        # Train model to obtain element-wise configuration values

        # Run these as many times as specified in the arguments
        config_vals = []

        if isinstance(sub_ind, torch.Tensor):
            sub_ind = sub_ind[0]

        for i in range(times):
            # Generate masks
            mask_loader, _ = Mask(
                sub_feat, sub_edge_index, sub_pathway_inds, self.params, self.problem
            ).mask_generator()

            # Call weighted linear regression model
            wlrm = LinearRegression(elements)
            wlrm = wlrm.to(sub_feat.device)

            # Train weighted linear regression model and get
            # coefficients as a result
            config_val, _, _ = train_model(
                mask_loader,
                self.params,
                sub_feat,
                sub_edge_index,
                wlrm,
                self.arch,
                self.problem,
                sub_ind,
                sub_node_types,
                sub_edge_types,
                hetero_node_types,
                hetero_edge_types,
                padded_dims,
            )
            config_vals.append(config_val[0].squeeze())

            del mask_loader, wlrm

        # Obtain the mean and standard deviation of the configuration
        # values from several runs
        mean_config_val, std_config_val = self.weight_stacking(config_vals)

        config_val_df = sub_data_class.config_val_dataframe(
            mean_config_val, std_config_val, sub_names
        )

        # Aggregate config values into pathway-wise scores
        pathway_df = None
        if self.pathways is not None:
            pathway_df = sub_pathway_class.aggregate(mean_config_val, sub_pathway_inds)

        # return (
        #    config_val_df,
        #    pathway_df,
        #    sub_feat,
        #    sub_edge_index,
        #    sub_pathway_inds,
        #    sub_ind,
        #    sub_pathway_names,
        #    sub_node_types,
        #    sub_edge_types,
        #    sub_pathway_types,
        # )
        return config_val_df, pathway_df
