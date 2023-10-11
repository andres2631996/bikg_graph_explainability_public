import torch
import pytest
from .test_utils import load_GCN_model

from pathway_explanations.model import Model


class TestModel:
    def test_get_hops(self):
        """
        Function to test get_hops function in Model module

        """
        # Load model in test_data/ folder
        model_file = "../test_data/gcn_homo_1hop_lungCancer.pth.tar"
        mock_features = 84  # Number of features used in training the model
        mock_arch = load_GCN_model(mock_features, model_file)

        # Set up ground truth number of hops
        ground_truth = 1  # Given model is 1 hop

        # Execute actual coding pipeline
        model_class = Model(mock_arch)
        result = model_class.get_hops()

        # Execute assertions
        assert ground_truth == result

    def test_infer(self):
        """
        Function to test infer function in Model module

        """
        # Create some homogeneous graph mock data
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
            [[0, 2, 3, 6, 4, 5], [5, 6, 4, 1, 2, 0]], device=device, dtype=torch.long
        )

        # Obtain model from checkpoint file in test_data/ folder
        # The model is untrained, so the output will be totally dependent
        # on the random initialization of the model
        model_file = "../test_data/gcn_homo_1hop_lungCancer.pth.tar"
        num_features = mock_feat.shape[-1]
        mock_arch = load_GCN_model(num_features, model_file, trained=False, eval=False)
        mock_arch.to(device)

        # Execute actual coding pipeline
        model_class = Model(mock_arch)
        result = model_class.infer(mock_feat, mock_edge_index)

        # Since the graph has been built quite at random, we do not really
        # know which outputs we will get from it, but we can check
        # for some other features

        # Assert that the outputs are located between 0 and 1
        # Assert as well that there as many outputs as nodes in the graph,
        # as the model is a node prediction model

        # TODO: when link prediction and graph prediction models are available, test
        # them too

        valid_ind = torch.where((result.flatten() >= 0.0) & (result.flatten() <= 1.0))[0]
        assert len(valid_ind) == mock_feat.shape[0]

    def test_hetero2homo_output(self):
        """
        Test function hetero2homo_output function
        in Model module

        """
        # Build some mock data on heterogeneous graph prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mock_output = {
            "1": torch.tensor([[0.5313], [0.5223], [0.5221], [0.5083]], device=device),
            "2": torch.tensor([[0.5080], [0.5313], [0.5282], [0.5313], [0.5223]], device=device),
        }

        # Build up ground-truth
        # Homogeneous graph output consists of concatenated
        # outputs
        ground_truth_output = torch.tensor(
            [
                [0.5313],
                [0.5223],
                [0.5221],
                [0.5083],
                [0.5080],
                [0.5313],
                [0.5282],
                [0.5313],
                [0.5223],
            ],
            device=device,
        )
        # Concatenated tensor types (4 elements of type 0, 5 elements of type 1)
        ground_truth_types = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long, device=device
        )

        # Run actual coding pipeline
        # No need to define an actual model
        model_class = Model(None)
        result_output, result_types = model_class.hetero2homo_output(mock_output)

        # Execute assertions

        # Assertions on prediction output
        assert torch.equal(ground_truth_output, result_output)

        # Asertions on prediction types
        assert torch.equal(ground_truth_types.int(), result_types.int())

    def test_extract_node_edge_output(self):
        """
        Function to test extract_result function in Model module

        """
        # Create some homogeneous graph mock data
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

        # Concatenated mock result:
        # simulate we have three concatenations of the same graph of seven nodes
        # And we have to take the same node in all concatenated results
        mock_result = torch.tensor(
            [
                [0.5313],
                [0.5223],
                [0.5221],
                [0.5083],
                [0.5080],
                [0.5313],
                [0.5282],
                [0.5313],
                [0.5223],
                [0.5221],
                [0.5083],
                [0.5080],
                [0.5313],
                [0.5282],
                [0.5313],
                [0.5223],
                [0.5221],
                [0.5083],
                [0.5080],
                [0.5313],
                [0.5282],
            ],
            device=device,
        )

        # Mock graph element index
        mock_index = 3

        # Build up ground-truth, result from node in mock_index for all concatenations
        ground_truth = torch.tensor([[0.5083], [0.5083], [0.5083]], device=device)

        # Run actual coding pipeline
        # Code is actually independent from Model module
        model_class = Model(None)
        num_elements = mock_feat.shape[0]
        result = model_class.extract_node_edge_output(mock_result, mock_index, num_elements)

        # Run assertions
        # The result should be the prediction for node 3 in the different concatenations
        assert torch.equal(ground_truth, result)

    def test_predict_hetero_output(self):
        """
        Function to test predict_hetero_output function in Model class

        """
        # Create some homogeneous graph mock data
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
            [[0, 2, 3, 6, 4, 5, 7, 9, 10, 13, 11, 12], [5, 6, 4, 1, 2, 0, 12, 13, 11, 8, 9, 7]],
            device=device,
            dtype=torch.long,
        )

        # Node types
        mock_node_types = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1], device=device, dtype=torch.long
        )
        mock_node_type_names = ["0", "1"]

        # Edge types
        mock_edge_types = torch.tensor(
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], device=device, dtype=torch.long
        )
        mock_edge_type_names = [("0", "a", "1"), ("1", "b", "0")]

        # Additional parameters
        mock_perturbs = 2  # Number of perturbations
        mock_num_nodes = 7  # Number of nodes in the original unperturbed heterogeneous graph
        mock_sub_ind = 1  # Index of node to be explained
        mock_padded_dims = [0, 2]  # Padded features in each case
        mock_problem = "node"  # Problem type

        # Mock model
        mock_features = 2
        model_file = "../test_data/gcn_hetero_1hop_lungCancer.pth.tar"
        mock_arch = load_GCN_model(mock_features, model_file, False, False, True)
        mock_arch.to(device)

        # Setup expected output

        # Output should be the GNN predictions for node of index 1 of the first type analysed
        ground_truth = torch.tensor([0.54, 0.97], device=device)

        # Define model class
        model_class = Model(mock_arch)

        # Run actual code function
        result = model_class.predict_hetero_output(
            mock_feat,
            mock_edge_index,
            mock_node_types,
            mock_edge_types,
            mock_node_type_names,
            mock_edge_type_names,
            mock_perturbs,
            mock_num_nodes,
            mock_sub_ind,
            mock_padded_dims,
            mock_problem,
        )

        # Run a series of assertions

        # Tensor length
        assert ground_truth.shape[0] == result.shape[0]

        # Tensor contents should be between 0 and 1
        ind_low = result >= 0
        ind_high = result <= 1

        assert (ind_low * ind_high).sum() == result.shape[0]


if __name__ == "__main__":
    TestModel().test_get_hops()
    TestModel().test_infer()
    TestModel().test_hetero2homo_output()
    TestModel().test_extract_node_edge_output()
    TestModel().test_predict_hetero_output()
