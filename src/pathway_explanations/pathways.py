import torch
import numpy as np
import os, sys
import pandas as pd
import itertools


class Pathways:
    """
    Class for manipulating graph communities or "pathways"
    for graph explanations

    Params
    ------
    communities : list of lists of str or int
        Set of graph communities
    community_names : list of str
        Set of community names
    community_types : torch.tensor
        Community types in heterogeneous graphs (default: None)

    """

    def __init__(self, communities, community_names, community_types=None):
        self.communities = communities
        self.community_names = community_names
        self.community_types = community_types

        if self.community_names is None:
            # Set community names to community indexes
            self.community_names = np.arange(len(self.communities)).tolist()

    def comp_graph(self, names):
        """
        Provide pathway structure for computational graph

        i.e.
        filter for those pathways that are only in k-hop distance from node or
        edge of interest

        pathways = [pathway1,pathway2,...,pathwayN]

        if pathway2 has no node or edge in computational graph, converts to:
        final_pathways = [pathway1,pathway3,...,pathwayN]

        (pathway names are also processed with the same mechanics)

        Params
        ------
        names : list of str or int
            Node names in computational graph


        Returns
        -------
        sub_pathway : list of lists of str or int
            Pathway structure in computational graph
        sub_pathway_names : list of str
            Set of community names in computational graph
        sub_pathway_types : torch.tensor
            Pathway type for computational graph of heterogeneous graph
        penalties : torch.tensor
            Penalties for partially covered
            pathways in the computational graph, as their effect on
            central node or edge should be lower

        """

        # Check in each pathway the element names in computational graph
        names_array = np.array(names, dtype=str)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sub_pathway = []
        sub_pathway_names = []
        sub_pathway_types = None

        if self.community_types is not None:
            sub_pathway_types = []

        cont_community = 0

        # Check which communities lie in the computational graph
        for community, community_name in zip(self.communities, self.community_names):
            community_array = np.array(community, dtype=str)
            common = np.intersect1d(community_array, names_array)

            # Add pathway to computational graph if it has some overlap
            # with the existing computational graph. If cover is partial, compute a penalty
            if len(common) > 0:
                sub_pathway.append(common.tolist())
                sub_pathway_names.append(community_name)
                if self.community_types is not None:
                    sub_pathway_types.append(self.community_types[cont_community])

            cont_community += 1

        if self.community_types is not None:
            device = self.community_types.device
            sub_pathway_types = torch.tensor(sub_pathway_types, device=device)

        return sub_pathway, sub_pathway_names, sub_pathway_types

    def names2inds(self, names):
        """
        Convert names of nodes in communities into
        numerical indexes

        i.e.

        if pathway_names = [nodeX,nodeY,nodeN], convert to:
            pathway_idxes = [X,Y,N]

        Params
        ------
        names : list of str or int
            Node names in computational graph

        Returns
        -------
        inds : list of lists of int
            Community structure as numerical indexes

        """
        if isinstance(self.communities[0][0], int):
            # Communities are already expressed as integer indexes
            return self.communities

        inds = []
        names_array = np.array(names, dtype=str)
        for community in self.communities:
            community_array = np.array(community, dtype=str)
            _, ind, _ = np.intersect1d(names_array, community_array, return_indices=True)
            inds.append(ind.tolist())

        return inds

    def shift_hetero_pathways(self, pointers):
        """
        Shift numerical pathway names by a pointer when converting
        a heterogeneous graph into a homogeneous graph

        i.e.
        if pathways = [[1,2,3],[7,8,9],...] and index=10
        (because there may be a previous node type with 10 nodes),
        converts to final_pathways = [[11,12,13],[17,18,19],...]

        Params
        ------
        pointers : torch.tensor
            Indexes in homogeneous graph where each
            heterogeneous graph element starts

        """
        keys = list(self.communities.keys())

        for key, pointer in zip(keys, pointers):
            for i in range(len(self.communities[key])):
                aux = np.array(self.communities[key][i]) + pointer
                self.communities[key][i] = aux.tolist()

    def hetero2homo(self, problem, node_pointers=None, edge_pointers=None):
        """
        Convert dict of pathways and pathway names into
        an only list of pathways to deal more easily
        with homogeneous graphs

        i.e.
        if pathways = {"type1":[pathway1_1,...,pathwayN_1],...,"typeI":[pathway1_I,...,pathwayM_I]}
        converts to:
        homo_pathways = [pathway1_1,...,pathwayN_1,...,pathway1_I,...,pathwayM_I]
        the indexes of some pathways may be shifted as the node types accummulate, if
        the pathways contain numerical node or edge indexes

        Params
        ------
        problem : str
            Type of problem to be solved
        node_pointers : torch.tensor
            Where each node feature matrix starts
        edge_pointers : torch.tensor
            Where each edge index matrix starts

        Returns
        -------
        homo_communities : list of lists of int or str
            Homogeneous pathway list
        homo_community_names : list of lists of int or str
            Homogeneous pathway names
        community_types : torch.tensor
            Heterogeneous pathway types

        """

        homo_communities = self.communities
        homo_community_names = self.community_names
        community_types = None

        if isinstance(self.communities, dict):
            # Heterogeneous graph
            keys = list(self.communities.keys())
            values = list(self.communities.values())

            if isinstance(self.communities[keys[0]][0][0], int) or isinstance(
                self.communities[keys[0]][0][0], float
            ):
                # If pathway names are given by integers, we should
                # shift them by the given node pointers or edge pointers

                if problem == "node":
                    self.shift_hetero_pathways(node_pointers)
                elif problem == "edge":
                    self.shift_hetero_pathways(edge_pointers)

            # Join all pathways in an only list
            community_types = []
            homo_communities = []
            homo_community_names = []
            cont_key = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for key, value in zip(keys, values):
                community_types.append(torch.zeros((len(value)), device=device) + cont_key)
                for community in value:
                    homo_communities.append(community)
                cont_key += 1
                homo_community_names.append(self.community_names[key])

            community_types = torch.hstack(community_types)
            homo_community_names = list(itertools.chain.from_iterable(homo_community_names))

        return homo_communities, homo_community_names, community_types

    def mask_generator(self, half_size, size, size_internal, device):
        """
        Provide a mask for pathways with a certain number of perturbations.
        Use antithetic sampling: half of the rows are assigned randomly,
        and the other half are filled with the opposite values

        i.e:

        if we have N pathways, and M perturbations, create a
        binary mask with 1s for pathways to keep and 0s for
        pathways to perturb. This is completed as part of the
        external perturbations for the graph

        Params
        ------
        half_size : int
            Rows to be generated
        size : int
            Full mask size
        size_internal : int
            Rows of final mask dedicated to internal perturbations only
        device : torch.device
            Computational device

        Returns
        -------
        pathway_mask : torch.tensor
            Mask for pathways, with antithetic sampling

        """

        # Build external mask for pathways with antithetic sampling
        # Half built by sampling random communities and other half by sampling the inverse communities
        pathway_mask_half = torch.randint(
            low=0, high=2, size=(half_size, len(self.communities)), device=device, dtype=torch.bool
        )

        pathway_mask = torch.vstack([pathway_mask_half, ~pathway_mask_half])

        if (size - size_internal) % 2 != 0:
            # If we have an odd number of external perturbations
            # include an additional external perturbation

            additional = torch.randint(
                low=0, high=2, size=(1, len(self.communities)), device=device, dtype=torch.bool
            )

            pathway_mask = torch.vstack([pathway_mask, additional])

        return pathway_mask

    def activate_dead_mask(self, pathway_mask, pathway_ind):
        """
        In case that no communities are activated in the external mask computation,
        activate randomly some of these communities

        i.e.
        pathway_mask is a binary mask with 1s for active pathways
        and 0s for inactive pathways, but for some reason, it just
        contains 0s. Consequently:

        pathway_mask = [0,0,0,0,0,0,...] converts to:
        pathway_mask = [0,1,1,0,1,0,...]

        Params
        ------
        pathway_mask : torch.tensor
            Initial inactive mask with pathways
        pathway_ind : int
            Index of pathway being analysed

        Returns
        -------
        fixed_mask : torch.tensor
            Final mask with activated pathways

        """

        # If no external community is active, it is like an internal permutation
        # Allow at least to randomly activate one of the communities for each
        # permutation

        fixed_mask = pathway_mask.clone()  # Store final results in this tensor

        communities_to_activate = torch.randperm(len(self.communities))

        # Do not take into account the internal community analyzed
        communities_to_activate = communities_to_activate[communities_to_activate != pathway_ind]

        if (pathway_mask.shape[0] > len(communities_to_activate)) and (
            len(communities_to_activate) > 0
        ):
            # If there are more permutations to generate than coalitions to choose,
            # repeat the communities to activate several times
            q = pathway_mask.shape[0] // len(communities_to_activate)
            communities_to_activate = torch.cat([communities_to_activate] * (q + 1))
        communities_to_activate = communities_to_activate[: pathway_mask.shape[0]]
        row_inds = torch.arange(pathway_mask.shape[0], device=pathway_mask.device, dtype=torch.long)
        fixed_mask[row_inds, communities_to_activate] = True

        return fixed_mask

    def pathway_mask2node_mask(self, pathway_mask):
        """
        Convert pathway mask with binaries with instructions on
        which pathways to activate and deactivate

        Params
        ------
        pathway_mask : torch.tensor
            Pathway-wise mask

        i.e.
        pathway_mask is a binary vector with 0s for inactive pathways
        and 1s for active pathways (pathway_mask = [1,0,0,1,1,0,0,...])
        If pathway structure is pathways = [[element1_1,...,elementN_1],...,[element1_I,...,elementM_I]]
        Elements in pathways with 1s are kept, while elements in pathways with 0s are removed
        For example, element1_1 to elementN_1 from pathway1 are kept
        With these node indexes, a binary node mask can be built
        with 0s for removed nodes and 1s for preserved nodes


        Returns
        -------
        element_mask : torch.tensor
            Graph element-wise mask built according to the
            active and inactive pathways
        repeat_communities : torch.tensor
            Tiled community structure used to later
            build external masks

        """

        # Extract joint indexes and lengths of pathways
        joint_communities = list(itertools.chain.from_iterable(self.communities))
        len_communities = [len(community) for community in self.communities]

        # Get device
        device = pathway_mask.device
        joint_communities = torch.tensor(joint_communities, device=device)
        len_communities = torch.tensor(len_communities, device=device)

        repeat_communities = torch.tile(joint_communities, (pathway_mask.shape[0], 1))
        repeat_len_communities = torch.tile(len_communities, (pathway_mask.shape[0], 1)).flatten()

        # Extend mask to all node structure
        element_mask = torch.repeat_interleave(pathway_mask.flatten(), repeat_len_communities)

        # Reshape external mask: (number of perturbations x joint community shape)
        element_mask = element_mask.reshape((pathway_mask.shape[0], joint_communities.shape[0]))

        return element_mask, repeat_communities

    def aggregate(self, config_val, community_inds):
        """
        Aggregate configuration values into a community-wise fashion
        and build a dataframe out of it

        i.e:
        if a pathway=[node1,...,nodeN] and the configuration values are:
        config_val = [config_val1,...,config_valN]
        This is converted to:
        config_val_pathway = AGGREGATE(config_val1,...,config_valN)

        Params
        ------
        config_val : torch.tensor
            Configuration values for individual graph elements
        names : list of str
            Sorted names of elements according to their configuration value
        community_inds : list of list of ints
            Community indices
        penalty : list of int
            Penalty for large pathways overlapping with computational graph

        Returns
        -------
        community_df : pd.DataFrame
            Community-wise scores

        """
        # Set up dictionary with community information
        d = {"name": self.community_names}

        d["score"] = [
            torch.mean(config_val[community_ind]).item() for community_ind in community_inds
        ]

        # Convert saved dictionary to dataframe
        df = pd.DataFrame(d)
        df = df.set_index("name")

        # Sort results for all query nodes
        df = df.sort_values(by=["score"], ascending=False)

        return df.dropna()
