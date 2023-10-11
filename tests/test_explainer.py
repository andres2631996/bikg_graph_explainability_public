import torch
import pytest
import json
import numpy as np

from pathway_explanations.explainer import Explainer, set_seed
from .test_utils import load_GCN_model


class TestExplainer:
    def test_extract_index(self):
        """
        Function to test "extract_index" function
        in explainer.py script

        """
        # Create some mock data

        # Set up some mock data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create some heterogeneous mock data graph
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
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], dtype=torch.long, device=device
        )

        # Simulate some graph element names
        mock_names = ["1", "2", "3", "4", "5", "6", "7"]

        # Mock index for heterogeneous graph
        mock_index = None

        # Load parameter file
        with open("config/configs.json") as f:
            params = json.load(f)
            f.close()

        # Mock element from where to compute index
        mock_element = "3"

        # Define ground truth index
        ground_truth = 2

        # Run actual coding pipeline
        explainer_class = Explainer(
            mock_feat,
            mock_edge_index,
            None,
            params,
            mock_names,
            None,
            None,
            mock_index,
            problem="node",
        )
        result = explainer_class.extract_index(mock_element, mock_names)

        # Run assertions
        assert ground_truth == result

    def test_filter_hetero_names_node(self):
        """
        Function to test "filter_hetero_names" function
        in explainer.py script, for nodes

        """
        # Create some heterogeneous mock data graph
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Node features
        mock_feat = {
            "0": torch.tensor(
                [
                    [0.24, 0.56, 0.96, 0.54],
                    [0.78, 0.96, 0.12, 0.19],
                    [0.85, 0.91, 0.92, 0.13],
                    [1.91, 0.98, 0.54, 0.21],
                ],
                device=device,
            ),
            "1": torch.tensor(
                [[0.97, 0.23, 0.0, 0.0], [0.21, 0.24, 0.0, 0.0], [0.29, 0.37, 0.0, 0.0]],
                device=device,
            ),
        }

        # Simulate some node names
        mock_node_names = ["1", "2", "3", "4", "5", "6", "7"]

        # Edge index
        mock_edge_index = {
            ("0", "a", "1"): torch.tensor([[0, 2, 3], [5, 6, 4]], dtype=torch.long, device=device),
            ("1", "b", "0"): torch.tensor([[6, 4, 5], [1, 2, 0]], dtype=torch.long, device=device),
        }

        # Node types
        mock_node_types = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)

        # Node type names
        mock_node_type_names = ["0", "1"]

        # Mock element index
        mock_index = "1"

        # Load parameter file
        with open("config/configs.json") as f:
            params = json.load(f)
            f.close()

        # Build up ground-truth: take node names associated to node type "1"
        ground_truth = ["5", "6", "7"]

        # Run actual code
        explainer_class = Explainer(
            mock_feat, mock_edge_index, None, params, mock_node_names, None, None, mock_index
        )

        result = explainer_class.filter_hetero_names(
            mock_node_names, mock_node_types, None, mock_node_type_names, None
        )

        # Run assertions
        assert ground_truth == result

    def test_filter_hetero_names_edge(self):
        """
        Function to test "filter_hetero_names" function
        in explainer.py script, for edges

        """
        # Create some heterogeneous mock data graph
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Node features
        mock_feat = {
            "0": torch.tensor(
                [
                    [0.24, 0.56, 0.96, 0.54],
                    [0.78, 0.96, 0.12, 0.19],
                    [0.85, 0.91, 0.92, 0.13],
                    [1.91, 0.98, 0.54, 0.21],
                ],
                device=device,
            ),
            "1": torch.tensor(
                [[0.97, 0.23, 0.0, 0.0], [0.21, 0.24, 0.0, 0.0], [0.29, 0.37, 0.0, 0.0]],
                device=device,
            ),
        }

        # Edge index
        mock_edge_index = {
            ("0", "a", "1"): torch.tensor([[0, 2, 3], [5, 6, 4]], dtype=torch.long, device=device),
            ("1", "b", "0"): torch.tensor([[6, 4, 5], [1, 2, 0]], dtype=torch.long, device=device),
        }

        # Simulate some edge names
        mock_edge_names = ["1", "2", "3", "4", "5", "6"]

        # Edge types
        mock_edge_types = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long, device=device)

        # Edge type names
        mock_edge_type_names = [("0", "a", "1"), ("1", "b", "0")]

        # Mock element index: name of edge involved
        # (as it is a name of an edge category, it has to be a tuple:
        # source node type, relation name, target node type)
        mock_index = ("1", "b", "0")

        # Build up ground-truth: take node names associated to edge type "1"
        ground_truth = ["5", "6"]

        # Load parameter file
        with open("config/configs.json") as f:
            params = json.load(f)
            f.close()

        # Run actual code

        explainer_class = Explainer(
            mock_feat,
            mock_edge_index,
            None,
            params,
            mock_edge_names,
            None,
            None,
            mock_index,
            problem="edge",
        )
        result = explainer_class.filter_hetero_names(
            mock_edge_names, None, mock_edge_types, None, mock_edge_type_names
        )

        # Run assertions
        assert ground_truth == result

    def test_weight_stacking(self):
        """
        Function to test "weight_stacking" function
        in explainer.py script

        """

        # Create some heterogeneous mock data graph
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Node features
        mock_feat = {
            "0": torch.tensor(
                [
                    [0.24, 0.56, 0.96, 0.54],
                    [0.78, 0.96, 0.12, 0.19],
                    [0.85, 0.91, 0.92, 0.13],
                    [1.91, 0.98, 0.54, 0.21],
                ],
                device=device,
            ),
            "1": torch.tensor(
                [[0.97, 0.23, 0.0, 0.0], [0.21, 0.24, 0.0, 0.0], [0.29, 0.37, 0.0, 0.0]],
                device=device,
            ),
        }

        # Edge index
        mock_edge_index = {
            ("0", "a", "1"): torch.tensor([[0, 2, 3], [5, 6, 4]], dtype=torch.long, device=device),
            ("1", "b", "0"): torch.tensor([[6, 4, 5], [1, 2, 0]], dtype=torch.long, device=device),
        }

        # Simulate some edge names
        mock_edge_names = ["1", "2", "3", "4", "5", "6"]

        # Mock index for heterogeneous graph
        mock_index = ("1", "b", "0")

        # Load parameter file
        with open("config/configs.json") as f:
            params = json.load(f)
            f.close()

        # Mock weights to be stacked
        mock_weights = [
            torch.tensor([0.32, 0.34, 0.98, -0.12], device=device),
            torch.tensor([-0.14, 0.26, 0.12, 0.23], device=device),
            torch.tensor([0.21, 0.34, -0.94, 0.67], device=device),
        ]

        # Build up ground truth

        # Mean stacked weights
        ground_truth_mean = torch.tensor([0.13, 0.31, 0.05, 0.26], device=device)

        # Standard deviation of stacked weights
        ground_truth_std = torch.tensor([0.20, 0.04, 0.79, 0.32], device=device)

        # Run actual coding pipeline
        explainer_class = Explainer(
            mock_feat,
            mock_edge_index,
            None,
            params,
            mock_edge_names,
            None,
            None,
            mock_index,
            problem="edge",
        )
        result_mean, result_std = explainer_class.weight_stacking(mock_weights)

        # Run assertions

        # Obtain difference in means
        diff_mean = torch.abs(ground_truth_mean - result_mean).mean()

        # Obtain difference in standard deviations
        diff_std = torch.abs(ground_truth_std - result_std).mean()

        # Mean result assertion
        assert diff_mean.item() < 1e-2

        # Standard deviation result assertion
        assert diff_std.item() < 1e-2

    def test_run(self):
        """
        Test run function in Explainer class from explainer.py script

        """
        # Set up seed
        set_seed(0)

        # Build up some mock data from an artificial example
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mock features: generate random features, to be seeded
        # Set up 36 nodes with 84 features, as the black box model uses in the
        # example set up in here
        mock_feat = torch.randn((36, 84), device=device)

        # Mock edge index
        mock_edge_index = torch.tensor(
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
                    14,
                    16,
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
                    18,
                    19,
                    19,
                    19,
                    20,
                    20,
                    20,
                    21,
                    21,
                    21,
                    21,
                    22,
                    22,
                    22,
                    23,
                    23,
                    24,
                    24,
                    24,
                    24,
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
                    27,
                    28,
                    28,
                    28,
                    29,
                    29,
                    29,
                    30,
                    30,
                    30,
                    30,
                    30,
                    31,
                    31,
                    31,
                    32,
                    32,
                    32,
                    33,
                    33,
                    34,
                    34,
                    34,
                    35,
                    35,
                    34,
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
                    19,
                    28,
                    10,
                    16,
                    12,
                    18,
                    11,
                    18,
                    13,
                    12,
                    17,
                    14,
                    13,
                    15,
                    11,
                    18,
                    17,
                    15,
                    15,
                    13,
                    18,
                    16,
                    17,
                    12,
                    11,
                    16,
                    10,
                    25,
                    24,
                    25,
                    26,
                    21,
                    20,
                    26,
                    27,
                    22,
                    21,
                    27,
                    23,
                    22,
                    24,
                    23,
                    27,
                    19,
                    27,
                    24,
                    19,
                    20,
                    27,
                    21,
                    20,
                    25,
                    24,
                    26,
                    21,
                    22,
                    10,
                    30,
                    29,
                    28,
                    30,
                    35,
                    31,
                    32,
                    34,
                    35,
                    29,
                    30,
                    28,
                    32,
                    31,
                    33,
                    30,
                    32,
                    34,
                    33,
                    35,
                    30,
                    30,
                    34,
                    29,
                ],
            ],
            device=device,
        )

        # Define mock graph element names (from 1 to the number of node elements)
        # TODO: write function for link or graph prediction
        mock_names = [str(i) for i in range(mock_feat.shape[0])]

        # Mock pathway structure
        mock_pathways = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [10, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            [10, 28, 29, 30, 31, 32, 33, 34, 35],
        ]

        # Define pathway names
        mock_pathway_names = ["west", "north", "south", "east"]

        # Load parameter file
        with open("config/configs.json") as f:
            params = json.load(f)
            f.close()

        # Load GCN model: 1 hop model
        # (this will later give a computational graph of 5 nodes around the node of interest)
        # For the sake of testing, let's just load an untrained model
        # The original trained model
        arch = load_GCN_model(
            mock_feat.shape[-1], "../test_data/gcn_homo_1hop_lungCancer.pth.tar", True, True
        )

        # Set up some mock element to explain and
        # mock times to repeat execution and average
        mock_element = "10"
        mock_times = 3

        # Run actual coding pipeline
        explainer_class = Explainer(
            mock_feat,
            mock_edge_index,
            arch.to(device),
            params,
            mock_names,
            mock_pathways,
            mock_pathway_names,
            problem="node",
        )
        result_config_val, result_config_val_pathway = explainer_class.run(mock_element, mock_times)

        # Establish a series of assertions
        # to verify code validity

        # Check that the dataframe for configuration values consist of 3 columns
        # (element name + mean value + std value)
        # and the dataframe for the pathways has two columns (element name + mean value)
        assert (result_config_val.shape[-1] == 2) and (result_config_val_pathway.shape[-1] == 1)

        # Check that the column names of the configuration values
        # dataframe are the correct ones
        # Do the same for the pathway scores
        assert result_config_val.columns.tolist() == [
            "config_value_mean",
            "config_value_std",
        ]
        assert result_config_val_pathway.columns.tolist() == ["score"]

        # Check that the mean values of the configuration
        # value dataframe are sorted descendingly
        sorted_values = np.flip(np.sort(result_config_val["config_value_mean"].values))
        assert np.array_equal(result_config_val["config_value_mean"].values, sorted_values)

        # Check that the pathway values are also sorted
        # descendingly
        sorted_values = np.flip(np.sort(result_config_val_pathway["score"]))
        assert np.array_equal(result_config_val_pathway["score"].values, sorted_values)

        # Check that there are no NaN values in the dataframes
        config_val_array = result_config_val.values
        config_val_pathway_array = result_config_val_pathway.values
        assert not (np.isnan(config_val_array).any()) and not (
            np.isnan(config_val_pathway_array).any()
        )

        # Check that there are 15 nodes (2 hops) and
        # 4 pathways involved in dataframes,
        # according to the structure of the computational graph
        assert (result_config_val.shape[0] == 15) and (result_config_val_pathway.shape[0] == 4)


if __name__ == "__main__":
    TestExplainer().test_extract_index()
    TestExplainer().test_filter_hetero_names_node()
    TestExplainer().test_filter_hetero_names_edge()
    TestExplainer().test_weight_stacking()
    TestExplainer().test_run()
