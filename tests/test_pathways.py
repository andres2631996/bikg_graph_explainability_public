import pandas as pd
import torch
import random
import pytest
from pandas.testing import assert_frame_equal

from .test_utils import (
    sort_numerical_pathways,
)

from pathway_explanations.pathways import Pathways


class TestPathways:
    def test_comp_graph(self):
        """
        Function to test comp_graph function in Pathway module

        """

        # Build artificial communities of nodes, with artificial names
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        communities = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [10, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            [10, 28, 29, 30, 31, 32, 33, 34, 35],
        ]
        community_names = ["west", "north", "south", "east"]

        # Define node around which to make up computational graph
        ind = 10

        # Define the number of hops of the computational graph
        n_hops = random.randint(1, 3)  # From 1 to 3 hops, to be chosen randomly

        # Include one more hop than the coverage of the black box, else
        # the predictions for the node of interest change due how PyG
        # works internally
        n_hops += 1

        # Make up the nodes in the computational graph
        if n_hops == 1:
            comp_graph_nodes = [9, 10, 11, 19, 28]
            ground_truth = [[9, 10], [10, 11], [10, 19], [10, 28]]
            ground_truth_names = ["west", "north", "south", "east"]
            # ground_truth_penalties = torch.tensor([8, 7, 8, 7], device=device)
        elif n_hops == 2:
            comp_graph_nodes = [3, 4, 8, 9, 10, 11, 12, 16, 18, 19, 24, 25, 28, 29, 31]
            ground_truth = [
                [3, 4, 8, 9, 10],
                [10, 11, 12, 16, 18],
                [10, 19, 24, 25],
                [10, 28, 29, 31],
            ]
            ground_truth_names = ["west", "north", "south", "east"]
            # ground_truth_penalties = torch.tensor([6, 4, 5, 5], device=device)
        elif n_hops == 3:
            comp_graph_nodes = [
                0,
                2,
                3,
                4,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                16,
                17,
                18,
                19,
                20,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                34,
            ]
            ground_truth = [
                [0, 2, 3, 4, 7, 8, 9, 10],
                [10, 11, 12, 13, 16, 17, 18],
                [10, 19, 20, 23, 24, 25, 26, 27],
                [10, 28, 29, 30, 31, 32, 34],
            ]
            ground_truth_names = ["west", "north", "south", "east"]
            # ground_truth_penalties = torch.tensor([3, 2, 2, 2], device=device)

        elif n_hops == 4:
            comp_graph_nodes = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
            ]
            ground_truth = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [10, 11, 12, 13, 14, 16, 17, 18],
                [10, 19, 20, 21, 22, 23, 24, 25, 26, 27],
                [10, 28, 29, 30, 31, 32, 33, 34, 35],
            ]
            ground_truth_names = ["west", "north", "south", "east"]
            # ground_truth_penalties = torch.tensor([0, 1, 0, 0], device=device)

        # Execute actual function in the code
        pathway_class = Pathways(communities, community_names)
        result_pathway, result_pathway_names, _ = pathway_class.comp_graph(comp_graph_nodes)

        # Post-process output pathway into int node names that are sorted
        result_pathway = sort_numerical_pathways(result_pathway)

        # Perform assertions
        # Assertion on pathways in computational graph
        assert ground_truth == result_pathway

        # Assertion on pathway names in computational graph
        assert ground_truth_names == result_pathway_names

        # Assertion on penalties
        # assert torch.equal(ground_truth_penalties, result_penalties)

    def test_names2inds(self):
        """
        Function to test names2inds function in Pathway module

        """
        # Create simple artificial data

        node_names = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
            "34",
        ]
        communities = [
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            ["10", "11", "12", "13", "14", "15", "16", "17", "18"],
            ["10", "19", "20", "21", "22", "23", "24", "25", "26", "27"],
            ["10", "28", "29", "30", "31", "32", "33", "34"],
        ]
        community_names = ["west", "north", "south", "east"]

        # Build up ground truth
        # Indexes are sorted according to sorting of string integers
        ground_truth = [
            [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [10, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            [10, 28, 29, 30, 31, 32, 33, 34],
        ]

        # Run code pipeline
        pathway_class = Pathways(communities, community_names)
        result = pathway_class.names2inds(node_names)

        # Run assertions on pathway indexes
        assert ground_truth == result

    def test_shift_hetero_pathways(self):
        """
        Function to test shift_hetero_pathways function in Pathways class

        """
        # Define some mock pathway data

        # Pathways
        mock_hetero_pathways = {"1": [[0, 1, 2, 3, 4], [5, 6, 7]], "2": [[0, 1, 2], [3, 4, 5]]}

        # Pathway names
        mock_hetero_pathway_names = {
            "1": [["0", "1", "2", "3", "4"], ["5", "6", "7"]],
            "2": [["0", "1", "2"], ["3", "4", "5"]],
        }
        # Pathway pointers (index when a certain pathway starts)
        mock_pointers = [0, 8]

        # Build up ground truth output
        ground_truth = {"1": [[0, 1, 2, 3, 4], [5, 6, 7]], "2": [[8, 9, 10], [11, 12, 13]]}

        # Execute actual coding pipeline
        pathway_class = Pathways(mock_hetero_pathways, mock_hetero_pathway_names)
        pathway_class.shift_hetero_pathways(mock_pointers)

        # Execute assertions
        assert mock_hetero_pathways == ground_truth

    def test_hetero2homo(self):
        """
        Function to test hetero2homo function in Pathways class

        """
        # Generate mock heterogeneous pathway data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pathways
        mock_hetero_pathways = {"1": [[0, 1, 2, 3, 4], [5, 6, 7]], "2": [[0, 1, 2], [3, 4, 5]]}

        # Pathway names
        mock_hetero_pathway_names = {
            "1": ["1", "2"],
            "2": ["3", "4"],
        }
        # Pathway pointers (index when a certain pathway starts)
        mock_pointers = [0, 8]

        # Build up ground-truth
        # Pathways
        ground_truth_pathways = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]
        # Pathway names
        ground_truth_names = ["1", "2", "3", "4"]
        # Pathway types
        ground_truth_types = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)

        # Run code pipeline
        pathway_class = Pathways(mock_hetero_pathways, mock_hetero_pathway_names)
        result, result_names, result_types = pathway_class.hetero2homo("node", mock_pointers)

        # Run assertions

        # Assertion on pathways
        assert result == ground_truth_pathways

        # Assertion on pathway names
        assert result_names == ground_truth_names

        # Assertion on pathway types
        assert torch.equal(result_types.int(), ground_truth_types.int())

    def test_mask_generator(self):
        """
        Test mask_generator function in Pathway module

        """
        # Generate some mock mask on homogeneous graph elements
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pathways
        mock_pathways = [[0], [1, 2, 3, 4], [5, 6], [7, 8, 9, 10]]
        mock_pathway_names = ["1", "2", "3", "4"]

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

        # Run code pipeline
        pathway_class = Pathways(mock_pathways, mock_pathway_names)
        result = pathway_class.mask_generator(
            mock_pathway_mask.shape[0] // 2, mock_pathway_mask.shape[0], 0, mock_pathway_mask.device
        )

        # Run assertions on the generated pathway masks
        # Since it is difficult to test the inside of the tensors, as they
        # are stochastic, we just test the shape and the type of the tensors

        # Assertion on shape
        assert mock_pathway_mask.shape == result.shape

        # Assertions on datatype
        assert result.dtype == torch.bool

    def test_activate_dead_mask(self):
        """
        Function to test activate_deadMask function in Pathways class

        """
        # Generate some mock mask on homogeneous graph elements
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pathways
        mock_pathways = [[0], [1, 2, 3, 4], [5, 6], [7, 8, 9, 10]]
        mock_pathway_names = ["1", "2", "3", "4"]

        # Pathway mask: False everywhere except for the pathway of interest
        mock_pathway_ind = 2  # Pathway of interest, it can be active or not in the mask
        mock_pathway_mask = torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, True, False],
                [False, False, True, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, True, False],
                [False, False, False, False],
            ],
            dtype=torch.bool,
            device=device,
        )

        # Run actual coding pipeline
        pathway_class = Pathways(mock_pathways, mock_pathway_names)

        # Obtain result from code pipeline
        result = pathway_class.activate_dead_mask(mock_pathway_mask, mock_pathway_ind)

        # Run assertions

        # Assertion on shape
        assert mock_pathway_mask.shape == result.shape

        # Assertions on datatype
        assert result.dtype == torch.bool

        # Assertions on content: the generated mask should differ with respect to
        # the original mask, as some of the communities in the mask should be
        # now activated
        assert not (torch.equal(mock_pathway_mask, result))

    def test_pathway_mask2node_mask(self):
        """
        Function to test pathway_mask2node_mask function in
        Pathway class

        """
        # Generate some mock mask on homogeneous graph elements
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pathways
        mock_pathways = [[3], [1, 2, 3, 4], [5, 7], [7, 8, 0, 4]]
        mock_pathway_names = ["1", "2", "3", "4"]

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
        ground_truth = torch.tensor(
            [
                [False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, True, True, True, True],
                [False, True, True, True, True, False, False, False, False, False, False],
                [False, False, False, False, False, True, True, False, False, False, False],
                [False, False, False, False, False, True, True, False, False, False, False],
                [False, True, True, True, True, False, False, True, True, True, True],
                [True, True, True, True, True, False, False, False, False, False, False],
                [True, True, True, True, True, True, True, False, False, False, False],
                [True, False, False, False, False, False, False, False, False, False, False],
            ],
            dtype=torch.bool,
            device=device,
        )

        # Execute actual coding pipeline
        pathway_class = Pathways(mock_pathways, mock_pathway_names)

        # Obtain actual code pipeline result
        result, _ = pathway_class.pathway_mask2node_mask(mock_pathway_mask)

        # Run assertions
        # Assertions on resulting mask
        print(ground_truth.shape, result.shape)
        assert torch.equal(ground_truth, result)

    def test_aggregate(self):
        """
        Test function aggregate from Pathway coding module

        """
        # Generate some mock mask on homogeneous graph elements
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pathways
        mock_pathways = [[3], [1, 2, 3, 4], [5, 7], [7, 8, 0, 4]]
        mock_pathway_names = ["1", "2", "3", "4"]
        # Configuration values
        mock_config_val = torch.tensor(
            [0.21, 0.23, 0.95, 0.65, 0.98, -0.21, 0.32, 0.94, -0.34], device=device
        )

        # Build up ground truth output:
        # pathway-wise configuration values
        # from mean values of node-wise configuration values

        # Sort configuration values and names in
        # descending order
        ground_truth_config_val = torch.tensor(
            [0.7025, 0.65, 0.4475, 0.365], dtype=torch.double, device=device
        )
        ground_truth_names = ["2", "1", "4", "3"]
        ground_truth = pd.DataFrame(
            ground_truth_config_val.cpu().detach().numpy(),
            index=ground_truth_names,
            columns=["score"],
        )
        # Set index name to dataframe
        ground_truth.index.name = "name"

        # Execute current coding pipeline
        pathway_class = Pathways(mock_pathways, mock_pathway_names)

        # Obtain result from actual coding pipeline
        # Keep all penalties at a zero level
        result = pathway_class.aggregate(mock_config_val, mock_pathways)

        # Run assertions on pathway dataframes
        assert_frame_equal(ground_truth, result)


if __name__ == "__main__":
    TestPathways.test_comp_graph()
    TestPathways.test_names2inds()
    TestPathways.test_shift_hetero_pathways()
    TestPathways.test_hetero2homo()
    TestPathways.test_mask_generator()
    TestPathways.test_activate_dead_mask(15)
    TestPathways.test_pathway_mask2node_mask()
    TestPathways.test_aggregate()
