import torch
import random
import json


from .test_utils import (
    check_suitability_external_mask,
)

from pathway_explanations.masks import Mask


class TestMask:

    # No need to test the initial assertion function
    # "assertions_mask_generator"
    # As it is just a function to check that all
    # datatypes inputted to the class make sense

    def test_obtain_device(self):
        """
        Function to test obtain_device function in Mask module

        """
        # Generation of device that should be obtained for
        # computations in PyTorch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Node features
        mock_feat = torch.tensor(
            [
                [0.24, 0.56, 0.96, 0.54],
                [0.78, 0.96, 0.12, 0.19],
                [0.85, 0.91, 0.92, 0.13],
                [1.91, 0.98, 0.54, 0.21],
                [0.97, 0.23, 0.0, 0.0],
                [0.21, 0.24, 0.0, 0.0],
                [0.29, 0.37, 0.0, 0.0],
            ],
            device=device,
        )

        # Edge index
        mock_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.long
        )

        # Define default device
        ground_truth = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Run coding pipeline
        # This function is totally independent from the pipeline
        # No need to define extra arguments
        mask_class = Mask(mock_feat, mock_edge_index, None, None, None)
        result = mask_class.obtain_device()

        # Run assertions
        assert ground_truth == result

    def test_get_internal_mask(self):
        """
        Test get_internal_mask function from Mask module

        Params
        ------
        max_perturbs : int
            Maximum number of perturbations to pick at random

        """
        # Generate some mock mask on homogeneous graph elements
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pathways
        mock_pathways = [[3], [1, 2, 3, 4], [5, 7], [7, 8, 0, 4]]
        # Pathway names
        mock_pathway_names = ["1", "2", "3", "4"]
        # Pathway lengths
        mock_pathway_lens = torch.tensor([1, 4, 2, 4], dtype=torch.long, device=device)
        # Perturbations
        mock_perturbs = 3
        # Pathway index
        mock_pathway_ind = 1

        # Get approximate number of rows that the perturbation process
        # will have to generate for internal perturbations of the chosen pathway
        rows = ((mock_pathway_lens[mock_pathway_ind].item()) * mock_perturbs) // (
            mock_pathway_lens.sum().item()
        )

        if rows == 0:
            # If the row fraction is very low, generate at least one row
            rows = 1

        # Generate ground-truth internal mask:
        # process is stochastic, so we can generate it randomly
        ground_truth = torch.randint(
            low=0,
            high=2,
            size=(rows, mock_pathway_lens[mock_pathway_ind]),
            device=device,
            dtype=torch.bool,
        )

        # Run actual coding pipeline
        # No need to define graph edge index nor node feature tensor
        mask_class = Mask(None, None, mock_pathways, None, None)
        result, aux = mask_class.get_internal_mask(
            mock_pathways[mock_pathway_ind], mock_pathway_lens, mock_perturbs, device
        )

        # Run assertions

        # Since the result and the ground truth are stochastic,
        # aim at checking that at least the mask is boolean and
        # that a sufficient number of representative of rows are being represented

        # Check that a sufficient number of rows is being represented
        assert result.shape[0] >= rows

        # Check that the resulting mask is boolean
        assert result.dtype == torch.bool

    def test_get_external_indices(self):
        """
        Test function get_external_indices

        """
        # Generate some mock mask on homogeneous graph elements
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pathways
        mock_pathways = [[3], [1, 2, 3, 4], [5, 7], [7, 8, 0, 4]]
        mock_pathway_names = ["1", "2", "3", "4"]
        # Perturbations
        mock_perturbs = 9
        # Pathway lengths
        mock_pathway_lens = torch.tensor([1, 4, 2, 4], dtype=torch.long, device=device)
        # Pathway index
        mock_pathway_ind = 1

        # Get approximate number of rows that the perturbation process
        # will have to generate for internal perturbations of the chosen pathway
        rows = (mock_pathway_lens[mock_pathway_ind].item() * mock_perturbs) // (
            mock_pathway_lens.sum().item()
        )

        # Ground truth mask
        mock_pathway_mask = torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, True],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, True, False],
                [False, True, False, True],
                [True, True, False, False],
                [True, True, True, False],
                [True, False, False, False],
            ],
            dtype=torch.bool,
            device=device,
        )

        # Generate resulting mask: mask with nodes in pathways
        # activated or deactivated based on the values of mock_pathway_mask
        ground_truth_mask = torch.tensor(
            [
                [False, False, False, False, False, False, False, False, False],
                [True, False, False, False, True, False, False, True, True],
                [False, True, True, True, True, False, False, False, False],
                [False, False, False, False, False, True, False, True, False],
                [False, False, False, False, False, True, False, True, False],
                [True, True, True, True, True, False, False, True, True],
                [False, True, True, True, True, False, False, False, False],
                [False, True, True, True, True, True, False, True, False],
                [False, False, False, True, False, False, False, False, False],
            ],
            dtype=torch.bool,
            device=device,
        )

        # Run reference coding pipeline
        mask_class = Mask(None, None, mock_pathways, None, None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        element_mask = torch.zeros(size=ground_truth_mask.shape, dtype=torch.bool, device=device)
        result = mask_class.get_external_indices(element_mask, mock_pathway_ind, rows)

        # Run assertions

        # Shape assertion
        assert result.shape == ground_truth_mask.shape

        # Datatype asssertion
        assert result.dtype == torch.bool

        # Content assertions
        # In an mask with external community perturbations,
        # all communities except the internal one
        # (given here as mock_pathway_ind)
        # should be all True or all False
        check_suitability_external_mask(mock_pathways, mock_pathway_ind, result)

    def test_mask_loader(self):
        """
        Test mask_loader function in Mask coding module

        """

        # Build up random binary mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ground_truth_mask = torch.tensor(
            [
                [False, False, False, False, False, False, False, False, False],
                [True, False, False, False, True, False, False, True, True],
                [False, True, True, True, True, False, False, False, False],
                [False, False, False, False, False, True, False, True, False],
                [False, False, False, False, False, True, False, True, False],
                [True, True, True, True, True, False, False, True, True],
                [False, True, True, True, True, False, False, False, False],
                [False, True, True, True, True, True, False, True, False],
                [False, False, False, True, False, False, False, False, False],
            ],
            dtype=torch.bool,
            device=device,
        )

        # Define a random number of element pieces for dataloader
        ground_truth_pieces = random.randint(1, ground_truth_mask.shape[0])

        # Run actual coding pipeline

        # No actual inputs are needed to initialize the class
        mask_class = Mask(None, None, None, None, None)
        result_loader = mask_class.mask_loader(ground_truth_mask, ground_truth_pieces)

        # A proper way to check the dataloader is properly working
        # is to concatenate all the loaded masks and see if they
        # add up to the original mask
        result_mask = torch.vstack([sub_mask for sub_mask in result_loader])

        # Run assertions
        # Assertions on the mask pieces from the dataloader
        assert torch.equal(ground_truth_mask, result_mask)

    def test_shapley_mask(self):
        """
        Test shapley_mask function in Mask coding module

        """
        # Build up random binary mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ground_truth_mask = torch.tensor(
            [
                [False, False, False, False, False, False, False, False, False],
                [True, False, False, False, True, False, False, True, True],
                [False, True, True, True, True, False, False, False, False],
                [False, False, False, False, False, True, False, True, False],
                [False, False, False, False, False, True, False, True, False],
                [True, True, True, True, True, False, False, True, True],
                [False, True, True, True, True, False, False, False, False],
                [False, True, True, True, True, True, False, True, False],
                [False, False, False, True, False, False, False, False, False],
            ],
            dtype=torch.bool,
            device=device,
        )

        # Run actual coding pipeline

        # No actual inputs are needed to initialize the class
        mask_class = Mask(None, None, None, None, None)
        result_mask = mask_class.shapley_mask(ground_truth_mask.shape, ground_truth_mask.device)

        # Since result and ground truth are stochastic,
        # their elements will not be the same,
        # but we can at least check the datatypes
        # and the shapes of the tensors

        # Run assertions
        # Shape assertion
        assert ground_truth_mask.shape == result_mask.shape

        # Datatype assertion
        assert result_mask.dtype == torch.bool

    def test_mask_generator(self):
        """
        Test mask_generator function in Mask module

        Params
        ------
        params : dict
            Hyperparameter set for mask generation
        max_feature_length : int
            Length of maximum features
            to create in artificial graph in test

        """
        # Read configs file to get the number of epochs and perturbations
        # currently in use by the pipeline

        with open("config/configs.json", "r") as f:
            params = json.load(f)

        # Work with an artificial graph in this case
        # Generate some mock mask on homogeneous graph elements
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Node feature tensor
        mock_feat = torch.tensor(
            [
                [0.24, 0.56, 0.96, 0.54],
                [0.78, 0.96, 0.12, 0.19],
                [0.85, 0.91, 0.92, 0.13],
                [1.91, 0.98, 0.54, 0.21],
                [0.97, 0.23, 0.0, 0.0],
                [0.21, 0.24, 0.0, 0.0],
                [0.29, 0.37, 0.0, 0.0],
                [0.85, -0.12, 0.65, 0.13],
                [1.91, 0.91, -0.33, 0.21],
            ],
            device=device,
        )

        # Edge index
        mock_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5, 7, 8], [5, 6, 4, 1, 2, 0, 2, 5]], device=device, dtype=torch.long
        )

        # Pathways
        mock_pathways = [[3], [1, 2, 3, 4], [5, 7], [7, 8, 0, 4]]
        # Perturbations
        mock_perturbs = 9
        # Pathway lengths
        mock_pathway_lens = torch.tensor([1, 4, 2, 4], dtype=torch.long, device=device)
        # Pathway index
        mock_pathway_ind = 1

        # Get approximate number of rows that the perturbation process
        # will have to generate for internal perturbations of the chosen pathway
        rows = (mock_pathway_lens[mock_pathway_ind].item() * mock_perturbs) // (
            mock_pathway_lens.sum().item()
        )

        # Ground truth mask
        mock_pathway_mask = torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, True],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, True, False],
                [False, True, False, True],
                [True, True, False, False],
                [True, True, True, False],
                [True, False, False, False],
            ],
            dtype=torch.bool,
            device=device,
        )

        # Run actual coding pipeline
        mask_class = Mask(mock_feat, mock_edge_index, mock_pathways, params, "node")
        # Obtain generated mask loader, and a tensor telling in which row which is the communi
        result, rows = mask_class.mask_generator()

        # The result from the pipeline is a dataloader.
        # We can obtain the full mask by concatenating on the loaded sub-masks
        result_mask = torch.vstack([sub_mask for sub_mask in result])

        # Run assertions
        # Assertions on datatype: must be boolean
        assert result_mask.dtype == torch.bool

        # Assertions on shape: must be at least larger
        # than the number of epochs times perturbations applied
        assert result_mask.shape[0] >= params["interpret_samples"] * params["epochs"]

        # The number of columns of the mask should correspond to the number of nodes
        assert result_mask.shape[1] == mock_feat.shape[0]

        # Assertions on mask structure
        # For configuration value approximation, there should only be one community
        # in each row of the mask with both active and inactive nodes. The nodes
        # in the remaining communities should be either all of them on or all of them off
        # This can be ignored however, if the same graph element is in more than one
        # community at the same time and each community is respectively on or off

        for i, row in enumerate(rows):
            # Iterate through all generated perturbations
            # Check for each row that the conditions for active and inactive
            # communities stated above hold
            check_suitability_external_mask(mock_pathways, row.item(), result_mask[i].unsqueeze(0))


if __name__ == "__main__":
    TestMask.test_obtain_device()
    TestMask.test_get_internal_mask(100)
    TestMask.test_get_external_indices()
    TestMask.test_mask_loader()
    TestMask.test_shapley_mask()
