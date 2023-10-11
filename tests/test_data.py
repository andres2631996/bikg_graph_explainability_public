import random
import torch
from pandas.testing import assert_frame_equal
import pytest
import pandas as pd

from .test_utils import (
    sort_feature,
    sort_edge_index,
)

from pathway_explanations.data import Data, pad_feat_tensors


class TestData:
    def test_preprocess_hetero_graph_hetero(self):
        """
        Function to test "preprocess_hetero_graph"
        in Data module for heterogeneous graphs

        """

        # Mock data preparation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mock_feat = {
            "1": torch.tensor(
                [
                    [0.24, 0.56, 0.96, 0.54],
                    [0.78, 0.96, 0.12, 0.19],
                    [0.85, 0.91, 0.92, 0.13],
                    [1.91, 0.98, 0.54, 0.21],
                ],
                device=device,
            ),
            "2": torch.tensor([[0.97, 0.23], [0.21, 0.24], [0.29, 0.37]], device=device),
        }
        mock_edge_index = {
            ("1", "a", "2"): torch.tensor([[0, 2, 3], [1, 2, 0]], device=device, dtype=torch.int),
            ("2", "b", "1"): torch.tensor([[2, 0, 1], [1, 2, 0]], device=device, dtype=torch.int),
        }

        # Build-up ground truth

        # Hetero node type names
        ground_truth_node_type_names = ["1", "2"]
        # Hetero edge type names
        ground_truth_edge_type_names = [("1", "a", "2"), ("2", "b", "1")]
        # Concatenated and padded node features
        ground_truth_feat = torch.tensor(
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
        # Concatenated edge indexes
        ground_truth_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.int
        )
        # Node types
        ground_truth_node_types = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1], device=device, dtype=torch.int
        )
        # Edge types
        ground_truth_edge_types = torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=torch.int)
        # Node pointers: where a certain node feature tensor starts
        ground_truth_node_pointers = [0, 4]
        # Edge pointers: where a certain edge index starts
        ground_truth_edge_pointers = [0, 3]
        # Padded dimensions
        ground_truth_padded_dims = [0, 2]

        # Initialize data class
        data_class = Data(mock_feat, mock_edge_index)

        # Run code
        (
            hetero_node_types,
            hetero_edge_types,
            homo_feat,
            homo_edge_index,
            node_types,
            edge_types,
            node_pointers,
            edge_pointers,
            padded_dims,
        ) = data_class.preprocess_hetero_graph()

        print(node_pointers, edge_pointers)

        # Node type names
        assert hetero_node_types == ground_truth_node_type_names
        # Edge type names
        assert hetero_edge_types == ground_truth_edge_type_names
        # Concatenated and padded node feature matrices
        assert torch.equal(homo_feat, ground_truth_feat)
        # Concatenated edge indexes
        assert torch.equal(homo_edge_index, ground_truth_edge_index)
        # Node types
        assert torch.equal(node_types.int(), ground_truth_node_types.int())
        # Edge types
        assert torch.equal(edge_types.int(), ground_truth_edge_types.int())
        # Node pointers
        assert node_pointers == ground_truth_node_pointers
        # Edge pointers
        assert edge_pointers == ground_truth_edge_pointers
        # Padded dimensions
        assert padded_dims == ground_truth_padded_dims

    def test_preprocess_hetero_graph_homo(self):
        """
        Function to test "preprocess_hetero_graph"
        in Data module for homogeneous graphs

        """

        # Build some mock data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # Concatenated edge indexes
        mock_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.int
        )

        # Initialize data class
        data_class = Data(mock_feat, mock_edge_index)

        # Run code
        (
            hetero_node_types,
            hetero_edge_types,
            homo_feat,
            homo_edge_index,
            node_types,
            edge_types,
            node_pointers,
            edge_pointers,
            padded_dims,
        ) = data_class.preprocess_hetero_graph()

        # Run assertions
        assert hetero_node_types is None
        assert hetero_edge_types is None
        assert node_types is None
        assert edge_types is None
        assert node_pointers is None
        assert edge_pointers is None
        assert padded_dims is None
        assert torch.equal(mock_feat, homo_feat)
        assert torch.equal(mock_edge_index, homo_edge_index)

    # For heterogeneous graph, directly run
    # test_preprocess_hetero_graph, as this function is
    # basically a wrapper to hetero2homo

    def test_homo2hetero_nodes(self):
        """
        Test homo2hetero function in Data class,
        for node feature tensors

        """
        # Generate mock homogeneous graph data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Edge indexes
        mock_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.int
        )
        # Node type sizes
        mock_node_types = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.int, device=device)

        # Dimensions to be padded
        mock_padded_dims = [0, 2]

        # Generate ground truth results

        # Ground truth node features
        ground_truth_feat = {
            "1": torch.tensor(
                [
                    [0.24, 0.56, 0.96, 0.54],
                    [0.78, 0.96, 0.12, 0.19],
                    [0.85, 0.91, 0.92, 0.13],
                    [1.91, 0.98, 0.54, 0.21],
                ],
                device=device,
            ),
            "2": torch.tensor([[0.97, 0.23], [0.21, 0.24], [0.29, 0.37]], device=device),
        }

        # Ground truth edge index
        ground_truth_edge_index = {
            ("1", "a", "2"): torch.tensor([[0, 2, 3], [5, 6, 4]], device=device, dtype=torch.int),
            ("2", "b", "1"): torch.tensor([[6, 4, 5], [1, 2, 0]], device=device, dtype=torch.int),
        }

        # Ground truth node type names
        ground_truth_node_type_names = ["1", "2"]

        # Generate random homogeneous graph
        data_class = Data(mock_feat, mock_edge_index)

        # Run original function
        result_dict = data_class.homo2hetero(
            mock_feat, mock_node_types, ground_truth_node_type_names, mock_padded_dims
        )

        # Run assertions
        # Assertion on node type names
        assert list(result_dict.keys()) == ground_truth_node_type_names

        # Assertion on resulting heterogeneous dictionary tensors
        ground_truth_values = list(ground_truth_feat.values())
        result_values = list(result_dict.values())
        for ground_truth_value, result_value in zip(ground_truth_values, result_values):
            assert torch.equal(ground_truth_value, result_value)

    def test_homo2hetero_edges(self):
        """
        Test homo2hetero function in Data class,
        for edge indexes

        """
        # Generate mock homogeneous graph data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Edge indexes
        mock_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.int
        )

        # Edge type sizes
        mock_edge_types = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int, device=device)

        # Generate ground truth results

        # Ground truth edge index
        ground_truth_edge_index = {
            ("1", "a", "2"): torch.tensor([[0, 2, 3], [5, 6, 4]], device=device, dtype=torch.int),
            ("2", "b", "1"): torch.tensor([[6, 4, 5], [1, 2, 0]], device=device, dtype=torch.int),
        }

        # Ground truth edge type names
        ground_truth_edge_type_names = [("1", "a", "2"), ("2", "b", "1")]

        # Generate random homogeneous graph
        data_class = Data(mock_feat, mock_edge_index)

        # Run original function
        result_dict = data_class.homo2hetero(
            mock_edge_index, mock_edge_types, ground_truth_edge_type_names
        )

        # Run assertions
        # Assertion on edge type names
        assert list(result_dict.keys()) == ground_truth_edge_type_names

        # Assertion on resulting heterogeneous dictionary tensors
        ground_truth_values = list(ground_truth_edge_index.values())
        result_values = list(result_dict.values())
        for ground_truth_value, result_value in zip(ground_truth_values, result_values):
            assert torch.equal(ground_truth_value.int(), result_value.int())

    def test_hetero2homo_node_names(self):
        """
        Test hetero2homo_names function in Data class
        on node names

        """
        # Prepare mock data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Node names
        mock_feat_names = {"1": ["0", "1", "2", "3"], "2": ["4", "5", "6"]}

        # Prepare ground truth results
        ground_truth_feat_names = ["0", "1", "2", "3", "4", "5", "6"]

        # Node types
        ground_truth_node_types = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1], dtype=torch.int, device=device
        )

        # Run original function
        # Function is independent of the actual node features and edge indexes
        data_class = Data(None, None)
        result, result_types = data_class.hetero2homo_names(mock_feat_names)

        # Run assertions
        # Assertion on node names
        assert result == ground_truth_feat_names

        # Assertion on node types
        assert torch.equal(result_types.int(), ground_truth_node_types.int())

    def test_hetero2homo_edge_names(self):
        """
        Test hetero2homo_names function in Data class
        on edge names

        """
        # Prepare mock data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Edge names
        mock_edge_names = {("1", "a", "2"): ["0", "1", "2"], ("2", "b", "1"): ["3", "4", "5"]}

        # Prepare ground truth results
        ground_truth_edge_names = ["0", "1", "2", "3", "4", "5"]

        # Edge types
        ground_truth_edge_types = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int, device=device)

        # Run original function
        # Function is independent of the actual node features and edge indexes
        data_class = Data(None, None)
        result, result_types = data_class.hetero2homo_names(mock_edge_names)

        # Run assertions

        # Assertion on edge names
        assert result == ground_truth_edge_names

        # Assertion on edge types
        assert torch.equal(result_types.int(), ground_truth_edge_types.int())

    def test_comp_graph(self, max_feature_length=32):
        """
        Testing of comp_graph function in Data class
        Use an artificial graph easy to extract its
        computational graph, in a node prediction problem

        Params
        ------
        max_feature_length : int
            Maximum length of features

        """
        # Binary flag to determine if graph is heterogeneous or homogeneous
        hetero = bool(random.randint(0, 1))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-defined edge index
        edge_index = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [1, 0],
                [1, 2],
                [1, 5],
                [2, 1],
                [2, 0],
                [2, 4],
                [2, 5],
                [2, 6],
                [2, 7],
                [3, 2],
                [3, 4],
                [3, 9],
                [4, 0],
                [4, 2],
                [4, 3],
                [4, 9],
                [5, 1],
                [5, 2],
                [5, 6],
                [6, 5],
                [6, 2],
                [6, 7],
                [7, 6],
                [7, 2],
                [7, 8],
                [8, 7],
                [8, 9],
                [9, 3],
                [9, 4],
                [9, 8],
                [9, 10],
                [10, 9],
                [10, 11],
                [10, 19],
                [10, 28],
                [11, 10],
                [11, 16],
                [11, 12],
                [11, 18],
                [12, 11],
                [12, 18],
                [12, 13],
                [13, 12],
                [13, 17],
                [13, 14],
                [14, 13],
                [14, 15],
                [16, 11],
                [16, 18],
                [16, 17],
                [16, 15],
                [17, 15],
                [17, 13],
                [17, 18],
                [17, 16],
                [18, 17],
                [18, 12],
                [18, 11],
                [18, 16],
                [19, 10],
                [19, 25],
                [19, 24],
                [20, 25],
                [20, 26],
                [20, 21],
                [21, 20],
                [21, 26],
                [21, 27],
                [21, 22],
                [22, 21],
                [22, 27],
                [22, 23],
                [23, 22],
                [23, 24],
                [24, 23],
                [24, 27],
                [24, 19],
                [24, 27],
                [25, 24],
                [25, 19],
                [25, 20],
                [26, 27],
                [26, 21],
                [26, 20],
                [26, 25],
                [27, 24],
                [27, 26],
                [27, 21],
                [27, 22],
                [28, 10],
                [28, 30],
                [28, 29],
                [29, 28],
                [29, 30],
                [29, 35],
                [30, 31],
                [30, 32],
                [30, 34],
                [30, 35],
                [30, 29],
                [31, 30],
                [31, 28],
                [31, 32],
                [32, 31],
                [32, 33],
                [32, 30],
                [33, 32],
                [33, 34],
                [34, 33],
                [34, 35],
                [34, 30],
                [35, 30],
                [35, 34],
                [34, 29],
            ],
            device=device,
        )

        # Determine size of random features
        feature_size = random.randint(1, max_feature_length)

        # Feature matrix
        features = torch.randn(int(torch.max(edge_index.T[0]).item()) + 1, feature_size)

        # Index to be explained
        ind = 10

        # Determine hops for computational graph extraction
        # (from 1 to 3)
        hops = random.randint(1, 3)

        # Include one more hop than the coverage of the black box, else
        # the predictions for the node of interest change due how PyG
        # works internally
        hops += 1

        # Set up names for nodes in graph
        names = [str(i) for i in range(features.shape[0])]

        # Types of edges, for heterogeneous types
        if hetero:
            node_types = torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    4,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                ],
                device=device,
            )
            edge_types = torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    4,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    4,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                ],
                device=device,
            )
        else:
            node_types, edge_types = None, None

        # Build-up ground-truth
        if hops == 1:
            result_edge_index = torch.tensor(
                [[0, 1, 1, 1, 1, 2, 3, 4], [1, 0, 2, 3, 4, 1, 1, 1]], device=device
            )
            result_feat = features[[9, 10, 11, 19, 28]]
            result_names = ["9", "10", "11", "19", "28"]
            result_node_types = torch.tensor([0, 4, 1, 2, 3], device=device)
            result_edge_types = torch.tensor([4, 4, 4, 4, 4, 4, 4, 4], device=device)

            result_ind = 1

        elif hops == 2:

            result_edge_index = torch.tensor(
                [
                    [
                        0,
                        0,
                        1,
                        1,
                        2,
                        3,
                        3,
                        3,
                        3,
                        4,
                        4,
                        4,
                        4,
                        5,
                        5,
                        5,
                        5,
                        6,
                        6,
                        7,
                        7,
                        8,
                        8,
                        8,
                        9,
                        9,
                        9,
                        10,
                        11,
                        11,
                        12,
                        12,
                        13,
                        14,
                    ],
                    [
                        3,
                        1,
                        0,
                        3,
                        3,
                        4,
                        2,
                        0,
                        1,
                        12,
                        9,
                        3,
                        5,
                        6,
                        4,
                        8,
                        7,
                        8,
                        5,
                        8,
                        5,
                        7,
                        6,
                        5,
                        10,
                        11,
                        4,
                        9,
                        10,
                        9,
                        13,
                        4,
                        12,
                        12,
                    ],
                ],
                device=device,
            )
            result_feat = features[[3, 4, 8, 9, 10, 11, 12, 16, 18, 19, 24, 25, 28, 29, 31]]
            result_names = [
                "3",
                "4",
                "8",
                "9",
                "10",
                "11",
                "12",
                "16",
                "18",
                "19",
                "24",
                "25",
                "28",
                "29",
                "31",
            ]
            result_node_types = torch.tensor(
                [0, 0, 0, 0, 4, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=device
            )
            result_edge_types = torch.tensor(
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        4,
                        0,
                        0,
                        0,
                        4,
                        4,
                        4,
                        4,
                        1,
                        4,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        4,
                        2,
                        2,
                        2,
                        3,
                        4,
                        3,
                        3,
                    ]
                ],
                device=device,
            )

            result_ind = 4

        elif hops == 3:

            result_edge_index = torch.tensor(
                [
                    [
                        0,
                        0,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        3,
                        3,
                        3,
                        3,
                        4,
                        4,
                        5,
                        5,
                        6,
                        6,
                        6,
                        6,
                        7,
                        7,
                        7,
                        7,
                        8,
                        8,
                        8,
                        8,
                        9,
                        9,
                        9,
                        10,
                        10,
                        11,
                        11,
                        11,
                        12,
                        12,
                        12,
                        13,
                        13,
                        13,
                        13,
                        14,
                        14,
                        14,
                        15,
                        15,
                        16,
                        17,
                        17,
                        17,
                        17,
                        18,
                        18,
                        18,
                        19,
                        19,
                        19,
                        20,
                        20,
                        21,
                        21,
                        21,
                        22,
                        22,
                        23,
                        23,
                        23,
                        23,
                        24,
                        24,
                        24,
                        25,
                        25,
                        26,
                        26,
                    ],
                    [
                        3,
                        1,
                        4,
                        3,
                        0,
                        3,
                        6,
                        1,
                        6,
                        2,
                        0,
                        1,
                        5,
                        1,
                        4,
                        6,
                        7,
                        5,
                        2,
                        3,
                        8,
                        6,
                        14,
                        21,
                        11,
                        7,
                        9,
                        13,
                        8,
                        10,
                        13,
                        12,
                        9,
                        13,
                        12,
                        8,
                        11,
                        13,
                        10,
                        8,
                        11,
                        12,
                        9,
                        18,
                        7,
                        17,
                        19,
                        18,
                        17,
                        14,
                        16,
                        20,
                        20,
                        17,
                        14,
                        15,
                        20,
                        18,
                        15,
                        19,
                        17,
                        23,
                        22,
                        7,
                        21,
                        23,
                        22,
                        26,
                        25,
                        24,
                        25,
                        21,
                        23,
                        23,
                        24,
                        22,
                        23,
                    ],
                ],
                device=device,
            )
            result_feat = features[
                [
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
            ]
            result_names = [
                "0",
                "2",
                "3",
                "4",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "16",
                "17",
                "18",
                "19",
                "20",
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
                "34",
            ]

            result_node_types = torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 4, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                device=device,
            )

            result_edge_types = torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    4,
                    0,
                    0,
                    0,
                    4,
                    4,
                    4,
                    4,
                    1,
                    4,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    4,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    4,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                ],
                device=device,
            )
            result_ind = 7

        elif hops == 4:
            result_edge_index = torch.tensor(
                [
                    [
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        3,
                        3,
                        3,
                        4,
                        4,
                        4,
                        4,
                        5,
                        5,
                        5,
                        6,
                        6,
                        6,
                        7,
                        7,
                        7,
                        8,
                        8,
                        9,
                        9,
                        9,
                        9,
                        10,
                        10,
                        10,
                        10,
                        11,
                        11,
                        11,
                        11,
                        12,
                        12,
                        12,
                        13,
                        13,
                        13,
                        14,
                        15,
                        15,
                        15,
                        16,
                        16,
                        16,
                        17,
                        17,
                        17,
                        17,
                        18,
                        18,
                        18,
                        19,
                        19,
                        19,
                        20,
                        20,
                        20,
                        20,
                        21,
                        21,
                        21,
                        22,
                        22,
                        23,
                        23,
                        23,
                        23,
                        24,
                        24,
                        24,
                        25,
                        25,
                        25,
                        25,
                        26,
                        26,
                        26,
                        26,
                        27,
                        27,
                        27,
                        28,
                        28,
                        28,
                        29,
                        29,
                        29,
                        29,
                        29,
                        30,
                        30,
                        30,
                        31,
                        31,
                        31,
                        32,
                        32,
                        33,
                        33,
                        33,
                        34,
                        34,
                        33,
                    ],
                    [
                        1,
                        2,
                        4,
                        0,
                        2,
                        5,
                        1,
                        0,
                        4,
                        5,
                        6,
                        7,
                        2,
                        4,
                        9,
                        0,
                        2,
                        3,
                        9,
                        1,
                        2,
                        6,
                        5,
                        2,
                        7,
                        6,
                        2,
                        8,
                        7,
                        9,
                        3,
                        4,
                        8,
                        10,
                        9,
                        11,
                        18,
                        27,
                        10,
                        15,
                        12,
                        17,
                        11,
                        17,
                        13,
                        12,
                        16,
                        14,
                        13,
                        11,
                        17,
                        16,
                        13,
                        17,
                        15,
                        16,
                        12,
                        11,
                        15,
                        10,
                        24,
                        23,
                        24,
                        25,
                        20,
                        19,
                        25,
                        26,
                        21,
                        20,
                        26,
                        22,
                        21,
                        23,
                        22,
                        26,
                        18,
                        26,
                        23,
                        18,
                        19,
                        26,
                        20,
                        19,
                        24,
                        23,
                        25,
                        20,
                        21,
                        10,
                        29,
                        28,
                        27,
                        29,
                        34,
                        30,
                        31,
                        33,
                        34,
                        28,
                        29,
                        27,
                        31,
                        30,
                        32,
                        29,
                        31,
                        33,
                        32,
                        34,
                        29,
                        29,
                        33,
                        28,
                    ],
                ],
                device=device,
            )

            result_feat = features[
                [
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
                ]
            ]
            result_node_types = torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                ],
                device=device,
            )
            result_edge_types = torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    2,
                    3,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                ],
                device=device,
            )
            result_ind = 10

            result_names = [
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
                "34",
            ]

        # Obtain result from built-in function
        data_class = Data(features, edge_index.T)

        problem = "node"  # Main elements we want to explain are nodes

        # Set hops to 1 less, as the code inside increases in 1 the number of hops later
        (
            sub_feat,
            sub_edge_index,
            sub_names,
            sub_ind,
            sub_node_types,
            sub_edge_types,
        ) = data_class.comp_graph(ind, hops - 1, problem, names, node_types, edge_types)

        # Sort elements in computational graph by name
        # Assume element names are numerical strings

        # Sort nodes and node types
        sub_feat, sub_node_types, sub_names = sort_feature(sub_feat, sub_node_types, sub_names)

        # Sort edges and edge types
        sub_edge_index, sub_edge_types = sort_edge_index(sub_edge_index, sub_edge_types)

        # Run assertions on the obtained results

        # Assertions on feature matrix
        assert torch.equal(result_feat, sub_feat)
        # Assertions on edge index matrix
        # assert torch.equal(result_edge_index, sub_edge_index)
        # Assertions on element of interest index
        assert result_ind == int(sub_ind.item())
        # Assertions on names of elements in computational graph
        assert result_names == sub_names

        if sub_node_types is not None:
            assert torch.equal(result_node_types, sub_node_types)

        # if sub_edge_types is not None:
        #    assert torch.equal(result_edge_types, sub_edge_types)

    def test_element_size_node(self):
        """
        Function to test element_size function in Data class for nodes

        """
        # Generate mock homogeneous graph data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Edge indexes
        mock_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.int
        )

        # Generate ground truth result
        ground_truth = 7

        # Run original function
        data_class = Data(mock_feat, mock_edge_index)
        result = data_class.element_size("node")

        # Run assertions
        assert result == ground_truth

    def test_element_size_edge(self):
        """
        Function to test element_size function in Data class for edges

        """
        # Generate mock homogeneous graph data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Edge indexes
        mock_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.int
        )

        # Generate ground truth result
        ground_truth = 6

        # Run original function
        data_class = Data(mock_feat, mock_edge_index)
        result = data_class.element_size("edge")

        # Run assertions
        assert result == ground_truth

    def test_build_edge_mask(self):
        """
        Function test build_edge_mask function in Data class

        """
        # Build mock data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Node mask
        mock_node_mask = torch.tensor(
            [
                [True, False, True, False, True, False, True],
                [True, True, True, True, False, False, False],
                [False, False, False, False, True, True, True],
            ],
            dtype=torch.bool,
            device=device,
        )

        # Initial edge index
        mock_initial_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.int
        )

        # Number of perturbations
        mock_perturbs = 3
        mock_concat_edge_index = torch.tensor(
            [
                [0, 2, 3, 6, 4, 5, 7, 9, 10, 13, 11, 12, 14, 16, 17, 20, 18, 19],
                [5, 6, 4, 1, 2, 0, 12, 13, 11, 8, 9, 7, 19, 20, 18, 15, 16, 14],
            ],
            device=device,
            dtype=torch.int,
        )

        # Build ground-truth result with edge mask
        ground_truth = torch.tensor(
            [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            device=device,
            dtype=torch.bool,
        )

        # Run actual function in the pipeline
        data_class = Data(mock_feat, mock_initial_edge_index)
        result_edge_mask, result_edge_index = data_class.build_edge_mask(mock_node_mask)

        # Run assertions
        # Assertion for the concatenation of edge indexes in different perturbations
        assert torch.equal(result_edge_index, mock_concat_edge_index)

        # Assertion for the edge index mask generation process
        assert torch.equal(result_edge_mask, ground_truth)

    def test_perturb_node(self):
        """
        Function test perturb_node function in Data class

        """

        # Build mock data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Node mask
        mock_node_mask = torch.tensor(
            [
                [True, False, True, False, True, False, True],
                [True, True, True, True, False, False, False],
                [False, False, False, False, True, True, True],
            ],
            dtype=torch.bool,
            device=device,
        )

        # Edge index
        mock_edge_index = torch.tensor(
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.int
        )

        # Edge type
        mock_edge_types = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int, device=device)

        # Build-up ground truth output
        ground_truth_edge_index = torch.tensor([[2, 4], [6, 2]], device=device, dtype=torch.int)
        ground_truth_edge_types = torch.tensor([0, 1], device=device, dtype=torch.int)

        # Run actual function in the pipeline
        data_class = Data(mock_feat, mock_edge_index)  # No need to specify the node feature matrix
        result_edge_index, result_edge_type = data_class.perturb_node(
            mock_node_mask, mock_edge_types
        )

        # Run assertions

        # Assertion for the perturbation of edge indexes
        assert torch.equal(result_edge_index, ground_truth_edge_index)

        # Assertion for the perturbation of edge types
        assert torch.equal(result_edge_type, ground_truth_edge_types)

    def test_perturb_edge(self):
        """
        Function test perturb_edge function in Data class

        """
        # Build mock data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.int
        )
        mock_concat_edge_index = torch.tensor(
            [
                [0, 2, 3, 6, 4, 5, 7, 9, 10, 13, 11, 12, 14, 16, 17, 20, 18, 19],
                [5, 6, 4, 1, 2, 0, 12, 13, 11, 8, 9, 7, 19, 20, 18, 15, 16, 14],
            ],
            device=device,
            dtype=torch.int,
        )

        # Edge type
        mock_edge_types = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int, device=device)
        mock_concat_edge_types = torch.tensor(
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=torch.int, device=device
        )

        # Edge mask
        mock_edge_mask = torch.tensor(
            [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            device=device,
            dtype=torch.bool,
        )

        # Generate ground-truth output
        ground_truth_edge_index = torch.tensor([[2, 4], [6, 2]], device=device, dtype=torch.int)
        ground_truth_edge_types = torch.tensor([0, 1], device=device, dtype=torch.int)

        # Execute original function
        data_class = Data(mock_feat, mock_edge_index)
        result_edge_index, result_edge_type = data_class.perturb_edge(
            mock_edge_mask, mock_edge_types
        )

        # Assertions

        # Assertion on edge index
        assert torch.equal(ground_truth_edge_index, result_edge_index)

        # Assertion on edge mask
        assert torch.equal(ground_truth_edge_types, result_edge_type)

    def test_concat_features(self):
        """
        Function to test concat_features function

        """
        # Generate mock data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Node types
        mock_node_types = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.int, device=device)

        # Number of perturbations
        perturbs = 3

        # Define ground truth output
        ground_truth = torch.tensor(
            [
                [0.24, 0.56, 0.96, 0.54],
                [0.78, 0.96, 0.12, 0.19],
                [0.85, 0.91, 0.92, 0.13],
                [1.91, 0.98, 0.54, 0.21],
                [0.97, 0.23, 0.0, 0.0],
                [0.21, 0.24, 0.0, 0.0],
                [0.29, 0.37, 0.0, 0.0],
                [0.24, 0.56, 0.96, 0.54],
                [0.78, 0.96, 0.12, 0.19],
                [0.85, 0.91, 0.92, 0.13],
                [1.91, 0.98, 0.54, 0.21],
                [0.97, 0.23, 0.0, 0.0],
                [0.21, 0.24, 0.0, 0.0],
                [0.29, 0.37, 0.0, 0.0],
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
        ground_truth_types = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
            dtype=torch.int,
            device=device,
        )

        # Run actual coding pipeline
        # Execute original function: no need to specify edge index
        data_class = Data(mock_feat, None)
        result_feat, result_type = data_class.concat_features(perturbs, mock_node_types)

        # Assertions
        # Assertions on node feature matrix
        assert torch.equal(ground_truth, result_feat)

        # Assertions on node type matrix
        assert torch.equal(ground_truth_types, result_type)

    # "perturbator" function in Data module just calls functions
    # "concat_features", "perturb_node" and/or "perturb_edge", so
    # it is enough to run the corresponding tests for these functions
    # As no further data manipulations are done by this function

    def test_config_val_dataframe(self):
        """
        Function to test "config_val_dataframe" function in Data module

        Params
        ------
        max_size : int
            Maximum dataframe size

        """
        # Generate mock data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mean configuration values
        mock_config_val_mean = torch.tensor([0.86, 0.54, 0.95, 0.21, -0.26, -0.32, 0.12])

        # Standard deviation configuration values
        mock_config_val_std = torch.tensor([0.02, 0.05, 0.09, 0.12, 0.02, 0.03, 0.07])

        # Names
        mock_names = ["0", "1", "2", "3", "4", "5", "6"]

        # Ground truth output
        ground_truth_dict = {
            "config_value_mean": mock_config_val_mean.cpu().detach().numpy(),
            "config_value_std": mock_config_val_std.cpu().detach().numpy(),
        }
        ground_truth = pd.DataFrame(ground_truth_dict, index=mock_names)

        # Set up index name in dataframe
        ground_truth.index.name = "name"

        # Sort dataframe in decreasing mean value
        ground_truth = ground_truth.sort_values(by=["config_value_mean"], ascending=False)

        # Obtain results from coding pipeline
        data_class = Data(
            None, None
        )  # Actual function is independent from node features and edge index
        result = data_class.config_val_dataframe(
            mock_config_val_mean,
            mock_config_val_std,
            mock_names,
        )

        # Run assertions
        assert_frame_equal(ground_truth, result)

    def test_concatenate_hetero_features(self):
        """
        Function for testing "concatenate_hetero_features" function
        in Data module

        """
        # Mock data preparation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Node features
        mock_feat = {
            "1": torch.tensor(
                [
                    [0.24, 0.56, 0.96, 0.54],
                    [0.78, 0.96, 0.12, 0.19],
                    [0.85, 0.91, 0.92, 0.13],
                    [1.91, 0.98, 0.54, 0.21],
                ],
                device=device,
            ),
            "2": torch.tensor([[0.97, 0.23], [0.21, 0.24], [0.29, 0.37]], device=device),
        }
        # Mock edge index
        mock_edge_index = {
            ("1", "a", "2"): torch.tensor([[0, 2, 3], [5, 6, 4]], device=device, dtype=torch.int),
            ("2", "b", "1"): torch.tensor([[6, 4, 5], [1, 2, 0]], device=device, dtype=torch.int),
        }

        # Ground truth output

        # Node features
        ground_truth_feat = torch.tensor(
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

        # Padded dimensions
        ground_truth_padded_dims = [0, 2]

        # Node types
        ground_truth_node_types = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1], device=device, dtype=torch.int
        )

        # Node pointers
        ground_truth_pointers = [0, 4]

        # Initialize data class
        data_class = Data(mock_feat, mock_edge_index)

        # Obtain result from current coding pipeline
        (
            result_feat,
            result_node_types,
            result_dims,
            result_pointers,
        ) = data_class.concatenate_hetero_features()

        # Run assertions

        # Assertion on concatenated node feature tensors
        assert torch.equal(ground_truth_feat, result_feat)

        # Assertion on generated node types
        assert torch.equal(ground_truth_node_types.int(), result_node_types.int())

        # Assertion on node pointers
        assert ground_truth_pointers == result_pointers

        # Assertion on padded dimensions for each heterogeneous tensor
        assert ground_truth_padded_dims == result_dims

    def test_concatenate_hetero_edge_indices(self):
        """
        Function for testing "concatenate_hetero_edge_indices" function
        in Data module

        """
        # Mock data preparation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Node features
        mock_feat = {
            "1": torch.tensor(
                [
                    [0.24, 0.56, 0.96, 0.54],
                    [0.78, 0.96, 0.12, 0.19],
                    [0.85, 0.91, 0.92, 0.13],
                    [1.91, 0.98, 0.54, 0.21],
                ],
                device=device,
            ),
            "2": torch.tensor([[0.97, 0.23], [0.21, 0.24], [0.29, 0.37]], device=device),
        }
        # Edge indexes ## IMPORTANT: these lines have been recently modified!!
        mock_edge_index = {
            ("1", "a", "2"): torch.tensor([[0, 2, 3], [2, 3, 1]], device=device, dtype=torch.int),
            ("2", "b", "1"): torch.tensor([[3, 1, 2], [1, 2, 0]], device=device, dtype=torch.int),
        }
        # Node pointers
        mock_node_pointers = [0, 4]

        # Ground truth outputs
        # Edge index
        ground_truth_edge_index = torch.tensor(
            [[0, 2, 3, 7, 5, 6], [6, 7, 5, 1, 2, 0]], device=device, dtype=torch.int
        )
        # Edge types
        ground_truth_edge_types = torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=torch.int)
        # Edge pointers
        ground_truth_edge_pointers = [0, 3]

        # Initialize data class
        data_class = Data(mock_feat, mock_edge_index)

        # Obtain result from current coding pipeline
        (
            result_edge_index,
            result_edge_types,
            result_edge_pointers,
        ) = data_class.concatenate_hetero_edge_indices(mock_node_pointers)

        # Run assertions
        # Assertions on the concatenated edge indexes
        assert torch.equal(ground_truth_edge_index.int(), result_edge_index.int())

        # Assertions on generated edge types
        assert torch.equal(ground_truth_edge_types.int(), result_edge_types.int())

        # Assertions on edge pointers
        assert ground_truth_edge_pointers == result_edge_pointers

    def test_pad_feat_tensors(self):
        """
        Function to test "pad_feat_tensors"

        """
        # Mock data preparation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Node features
        mock_feat = {
            "1": torch.tensor(
                [
                    [0.24, 0.56, 0.96, 0.54],
                    [0.78, 0.96, 0.12, 0.19],
                    [0.85, 0.91, 0.92, 0.13],
                    [1.91, 0.98, 0.54, 0.21],
                ],
                device=device,
            ),
            "2": torch.tensor([[0.97, 0.23], [0.21, 0.24], [0.29, 0.37]], device=device),
        }

        # Ground truth output
        # Padded tensors
        ground_truth_tensors = [
            torch.tensor(
                [
                    [0.24, 0.56, 0.96, 0.54],
                    [0.78, 0.96, 0.12, 0.19],
                    [0.85, 0.91, 0.92, 0.13],
                    [1.91, 0.98, 0.54, 0.21],
                ],
                device=device,
            ),
            torch.tensor(
                [[0.97, 0.23, 0.0, 0.0], [0.21, 0.24, 0.0, 0.0], [0.29, 0.37, 0.0, 0.0]],
                device=device,
            ),
        ]
        # Padded dimensions
        ground_truth_dims = [0, 2]
        # Pointers
        ground_truth_pointers = [0, 4]

        # Obtain results from coding pipeline
        padded_tensors, dims, pointers = pad_feat_tensors(list(mock_feat.values()))

        # Run assertions

        # Assertions on padded tensors
        for ground_truth_tensor, padded_tensor in zip(ground_truth_tensors, padded_tensors):
            assert torch.equal(ground_truth_tensor, padded_tensor)

        # Assertions on dimensions that have been padded
        assert ground_truth_dims == dims

        # Assertions on pointers for padding
        assert ground_truth_pointers == pointers


if __name__ == "__main__":
    TestData.test_preprocess_hetero_graph_hetero()
    TestData.test_preprocess_hetero_graph_homo()
    TestData.test_homo2hetero_nodes()
    TestData.test_homo2hetero_edges()
    TestData.test_hetero2homo_node_names()
    TestData.test_hetero2homo_edge_names()
    TestData.test_comp_graph(32)
    TestData.test_element_size_node()
    TestData.test_element_size_edge()
    TestData.test_build_edge_mask()
    TestData.test_perturb_node()
    TestData.test_perturb_edge()
    TestData.test_concat_features()
    TestData.test_config_val_dataframe()
    TestData.test_concatenate_hetero_features()
    TestData.test_concatenate_hetero_edge_indices()
    TestData.test_pad_feat_tensors()
