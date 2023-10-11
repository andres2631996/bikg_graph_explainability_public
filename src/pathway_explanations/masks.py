import torch
import math
from torch.utils.data import DataLoader
import itertools
import os, sys
from pathway_explanations.pathways import Pathways
from pathway_explanations.data import Data


class Mask(Data):
    """
    Class for manipulating mask data for approximating
    Configuration Values

    Params
    ------
    feat : torch.tensor or dict
        Node feature information
    edge_index : torch.tensor or dict
        Edge index information
    pathways : list of lists of int or str
        Pathways to be considered
    params : dict
        Hyperparams
    problem : str
        Type of problem to be explained with algorithm

    """

    def __init__(self, feat, edge_index, pathways, params, problem):
        super().__init__(feat, edge_index)
        self.pathways = pathways
        self.params = params
        self.problem = problem

    @staticmethod
    def assertions_mask_generator(params):
        """
        Check input parameters for mask generator function

        Params
        ------
        params : dict
            Hyperparameters

        """
        n_perturbs = params["interpret_samples"]
        epochs = params["epochs"]
        # Make sure the number of perturbations per epoch are numeric
        assert isinstance(n_perturbs, int) or isinstance(
            n_perturbs, float
        ), "Number of perturbations in batch is not numeric"
        n_perturbs = abs(n_perturbs)
        # Make sure the number of epochs is numeric, too
        assert isinstance(epochs, int) or isinstance(
            epochs, float
        ), "Number of epochs in batch is not numeric"
        epochs = abs(epochs)

        return n_perturbs, epochs

    def obtain_device(self):
        """
        Extract device in use

        i.e. if mask is in CUDA, return CUDA
        (same for CPU)

        """
        if isinstance(self.feat, dict):
            # Heterogeneous graph
            keys = list(self.feat.keys())
            device = self.feat[keys[0]].device
        else:
            # Homogeneous graph
            device = self.feat.device

        return device

    @staticmethod
    def get_internal_mask(pathway, len_pathways, total_size, device):
        """
        Obtain mask with internal perturbations for configuration
        value approximation

        i.e.
        if pathway=[node1,...,nodeN], the code obtains
        random binary perturbations of the nodes in the pathway
        obtain mask = [1,0,0,...]
        The indexes of those nodes that are 1 are registered as 1
        in a final mask, while the index of nodes that are 0 are set
        to be 0. This is done as part of the internal perturbations

        Params
        ------
        pathway : list of int
            Pathway indexes
        len_pathways : list of int
            Pathway lengths
        total_size : int
            Total mask size
        device : torch.device
            Device where to do computations

        Returns
        -------
        internal_mask : torch.tensor
            Size of internal mask
        size_internal : int
            Number of internal permutations completed

        """

        # Provide a proportional number of perturbations to the
        # pathway size
        fraction = len(pathway) / torch.sum(len_pathways)
        size = math.ceil(fraction * total_size)

        # Rows to perturb for internal community only (external communities are left as zero)
        size_internal = math.ceil(fraction * size)

        # Cap the number of internal coalitions to at least 1
        if size_internal < 3:
            size_internal = 1
            size = 2

        # print(size,size_internal,len(pathway))

        # Obtain final internal mask
        internal_mask = torch.randint(
            low=0, high=2, size=(size, len(pathway)), device=device, dtype=torch.bool
        )

        # print(internal_mask)

        return internal_mask, size_internal

    def get_external_indices(self, full_mask, ind_pathway, size_internal):
        """
        Obtain external mask for approximating configuration
        values

        i.e.
        The code randomly perturbs pathways
        if pathways = [pathway1,...,pathwayN], it gets a random binary mask
        pathway_mask = [1,0,0,...]
        then it takes the node indexes in those active (resp. inactive)
        pathways and sets those indexes to 1 (resp. 0) to build a mask
        with external permutations of graph elements

        Params
        ------
        full_mask : torch.tensor
            Full mask to be modified with external coalition information
        ind_pathway : int
            Pathway index to be analyzed
        size_internal : int
            Rows of final mask dedicated to internal perturbations only
        pathways : list of lists of int or str
            Pathways to be considered


        Returns
        -------
        full_mask : torch.tensor
            Full mask modified with external coalition information

        """

        # Obtain masked pathways to get external communities
        pathway_class = Pathways(self.pathways, None)
        device = full_mask.device
        pathway_mask = pathway_class.mask_generator(
            (full_mask.shape[0] - size_internal) // 2, full_mask.shape[0], size_internal, device
        )

        # Do not activate internal community being analyzed, leave it to be altered by internal permutations
        pathway_mask[:, ind_pathway] = False

        # If no pathways are active, activate randomly some of them
        if (len(self.pathways) - 1 > 0) and (pathway_mask.sum() == 0):
            pathway_mask = pathway_class.activate_dead_mask(pathway_mask, ind_pathway)

        # Convert the pathway-wise mask with information on active and inactive
        # pathways into a node-wise mask
        element_mask, repeat_pathways = pathway_class.pathway_mask2node_mask(pathway_mask)

        # Check where node mask is active
        active_rows = torch.where(element_mask == True)[0] + size_internal

        # Activate those mask elements where node mask is active
        full_mask[active_rows, repeat_pathways[element_mask > 0]] = True

        return full_mask

    @staticmethod
    def mask_loader(mask, pieces):
        """
        Load pieces of mask with all allowed perturbations
        to be used in batch mode during weighted linear
        model training

        i.e.
        if mask contains X perturbations, it is fed to a dataloader
        that iteratively sends batches of a certain size to train
        a surrogate weighted linear regression model

        Params
        ------
        mask : torch.tensor of bool
            Full mask to be loaded in pieces for training
            the weighted linear regression model
        pieces : int
            Number of pieces the output mask should be
            fragmented into


        Returns
        -------
        loader : torch dataloader
            Mask dataloader

        """
        # Determine mask batch size from pieces
        batch_size = mask.shape[0] // pieces

        # Set dataloader
        loader = DataLoader(mask, batch_size=batch_size, num_workers=0)
        return loader

    def shapley_mask(self, size, device):
        """
        If no pathway information is specified, compute mask with
        random perturbations that help approximate Shapley values
        instead of Configuration Values

        i.e.
        Directly obtain random binary perturbations of graph elements to explain
        mask is #perturbations X #graph elements, without any
        community-based perturbations

        Params
        ------
        size : tuple of int
            Mask size, (number of specified perturbations,element size)
        device : torch.device
            Computational device

        Returns
        -------
        mask : torch.tensor
            Binary mask with perturbation information for Shapley value
            computation

        """

        # Build fully-random mask
        mask = torch.randint(low=0, high=2, size=size, device=device, dtype=torch.bool)

        return mask

    def mask_generator(self):
        """
        Create permutation sampling masks for later approximation
        of configuration values. Store the resulting mask in a
        dataloader

        i.e.
        Obtain perturbation mask tensor, with 0s for graph elements
        to be removed and 1s for graph elements to be preserved.
        Graph elements are perturbed according to configuration value
        theory
        (https://www.sciencedirect.com/science/article/pii/S0899825605001247)

        Returns
        ------
        loader : torch dataloader
            Mask dataloader for training config value model
        pathway_rows : torch.tensor or None
            Pathway index telling which pathway is used for
            building internal perturbations in each mask row
            If Shapley values are computed, it is None

        """
        # Extract relevant hyperparameters
        n_perturbs, epochs = self.assertions_mask_generator(self.params)

        # Get device
        device = self.obtain_device()

        # Get number of nodes of interest
        if "edge" in self.problem:
            # Edge prediction
            element_size = self.edge_size.shape[1]
        else:
            # Node or graph prediction
            element_size = self.feat.shape[0]

        if self.pathways is not None:
            # Configuration value approximation

            # Extract lengths of pathways
            len_pathways = [len(pathway) for pathway in self.pathways]
            len_pathways = torch.tensor(len_pathways, device=device)

            masks = []

            # Store here the pathway that each perturbation row uses as internal mask
            # This simplifies later the testing process of this function
            pathway_rows = []
            pathway_sizes = []

            # Sort pathways by length
            argsort = torch.argsort(len_pathways, descending=True)

            # Sort elements of pathways
            sorted_pathways = [self.pathways[argsort[i]] for i in range(argsort.shape[0])]

            cumulative_size = 0

            # Iterate through all pathways
            for enum_pathway, pathway in enumerate(sorted_pathways):
                pathway.sort()

                internal_mask, size_internal = self.get_internal_mask(
                    pathway, len_pathways, n_perturbs * epochs, device
                )

                # Initialize mask for all permutations
                mask = torch.zeros(
                    (internal_mask.shape[0], element_size), device=device, dtype=torch.bool
                )

                # Fill mask with external coalitions
                mask = self.get_external_indices(mask, enum_pathway, size_internal)

                # Integrate internal mask into main mask
                mask[:, pathway] = internal_mask

                pathway_rows.append([argsort[enum_pathway]] * (mask.shape[0]))
                pathway_sizes.append([len(pathway)] * (mask.shape[0]))
                masks.append(mask)

                if (cumulative_size > n_perturbs * epochs) and (element_size > 4000):
                    # Restrict results to most populated pathways to save up memory and time
                    break

                cumulative_size += mask.shape[0]

            # Concatenate generated masks
            mask = torch.vstack(masks)

            # Concatenate pathway indexes as well
            pathway_rows = torch.tensor(
                list(itertools.chain.from_iterable(pathway_rows)), dtype=torch.int, device=device
            )

            pathway_sizes = torch.tensor(
                list(itertools.chain.from_iterable(pathway_sizes)), dtype=torch.int, device=device
            )

        else:
            # Shapley value approximation instead
            mask = self.shapley_mask((n_perturbs * epochs, element_size), device)
            pathway_rows = None

        if (
            (element_size > 4000)
            and (mask.shape[0] > n_perturbs * epochs)
            and (self.pathways is not None)
        ):
            # If there are many nodes to be explained,
            # and there are more rows than the product of initial
            # perturbations and epochs, sample mask rows starting from the most
            # populated pathways
            # up to the product of perturbations and rows
            # To alleviate memory issues

            ind_size = torch.argsort(pathway_sizes, descending=True)
            ind = ind_size[: (n_perturbs * epochs)]

        else:
            # Randomly shuffle perturbation rows of mask
            # Shuffle rows of final tensor
            ind = torch.randperm(mask.size()[0])

        mask = mask[ind]

        if pathway_rows is not None:
            pathway_rows = pathway_rows[ind]

        # Build dataloader
        loader = self.mask_loader(mask, epochs)

        del mask

        return loader, pathway_rows
