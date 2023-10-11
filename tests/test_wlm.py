import torch
import json
import numpy as np
import os, sys
import pytest
from torch.utils.data import DataLoader

from .test_utils import parameter_unpacker, load_GCN_model

from pathway_explanations.wlm import (
    LinearRegression,
    model_updates,
    regularizer,
    train_model,
    kernel_output,
    optimizer_scheduler,
    weighted_mse_loss,
)

from pathway_explanations.explainer import set_seed
from pathway_explanations.data import Data
from pathway_explanations.model import Model


class TestLinearRegression:
    def test_linear_regression(self):
        """
        Test Linear Regression function from wlm.py
        script

        """
        # Create some mock data
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

        # Call linear regression model from script
        num_features = mock_feat.shape[-1]
        model = LinearRegression(num_features).to(device)
        result = model(mock_feat)

        # Run assertions
        # Since the model is not trained, its output is stochastic,
        # so we will focus on asserting the datatype, shape and
        # bounds of the linear output

        # Assertion on datatype
        assert result.dtype == torch.float

        # Assertion on shape
        assert result.flatten().shape[0] == mock_feat.shape[0]

    def test_model_updates(self):
        """
        Test model_updates function from wlm.py script

        """
        # Create some mock data
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

        # Call linear regression model from script
        num_features = mock_feat.shape[-1]
        model = LinearRegression(num_features).to(device)
        result = model(mock_feat)

        # Set up ground truth values: model parameters,
        # and loss function value

        # Model parameters
        ground_truth_parameters = model.parameters()

        # Simulated loss function
        ground_truth_loss = torch.tensor([0.73], device=device)

        # Simulated best loss function value
        ground_truth_best_loss = torch.tensor([0.75], device=device)

        # Run coding pipeline, to get an update of the best
        # model parameters during training and the best value
        # for the loss function
        result_parameters, result_loss = model_updates(
            model, ground_truth_loss, ground_truth_best_loss
        )

        # Unpack layer names and parameters
        ground_truth = parameter_unpacker(ground_truth_parameters)
        result = parameter_unpacker(result_parameters)

        # Run assertions

        # Assertions on parameter names of each layer
        for g, r in zip(ground_truth, result):
            assert torch.equal(g, r)

        # Assertion on loss function value
        assert ground_truth_loss == result_loss

    def test_train_model(self):
        """
        Test train_model function from wlm.py script

        """
        # Set deterministic seed, to allow for reproducible testing
        # As the process is highly stochastic
        set_seed(0)

        # Load parameter file
        with open("config/configs.json") as f:
            params = json.load(f)
            f.close()

        # Prepare all mock data that are needed
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

        # Index of element to be explained: node number 1, for example
        mock_index = 1

        # Obtain total number of perturbations for the mock graph data
        # Number of perturbations per epoch X Number of epochs
        total_perturbations = int(params["interpret_samples"] * params["epochs"])

        # Mock mask, assume is a node prediction problem
        # TODO: expand tests to other kind of problems (link prediction, graph prediction, etc)
        num_elements = mock_feat.shape[0]  # Number of nodes to be explained
        mock_mask = torch.randint(
            low=0,
            high=2,
            size=(total_perturbations, num_elements),
            dtype=torch.bool,
            device=device,
        )

        # Build dataloader from mock mask, splitting it into as many
        # pieces as number of epochs for training linear regression model
        mock_loader = DataLoader(mock_mask, batch_size=params["interpret_samples"], num_workers=0)

        # Define mock model to be trained
        mock_model = LinearRegression(num_elements).to(device)

        # Define black box (randomly initialized)
        num_features = mock_feat.shape[-1]
        mock_black_box = load_GCN_model(num_features, None).to(device)

        # Run actual coding pipeline
        result_weights, result_losses, result_best_epoch = train_model(
            mock_loader,
            params,
            mock_feat,
            mock_edge_index,
            mock_model,
            mock_black_box,
            "node",
            element_index=mock_index,
        )

        # Set up some assertions that should be fulfilled for the model to work properly

        # Assert that there are as many weights as number of elements to explain
        assert result_weights[0].flatten().shape[0] == num_elements

        # Assert that there are no weights as NaN
        assert torch.isnan(result_weights[0]).sum() == 0

        # Assert that the model has been trained for the given number of epochs
        # Or at most for one more epoch, if the mask loader had to be partitioned
        # in one more split
        assert (len(result_losses) == params["epochs"]) or (
            len(result_losses) == params["epochs"] + 1
        )

        # Assert that the epoch with the minimal training loss is the one
        # given by the pipeline
        assert result_best_epoch == np.argmin(np.array(result_losses))

    def test_kernel_output(self):
        """
        Test kernel_output function from wlm.py script

        """
        # Set deterministic seed, to allow for reproducible testing
        # As the process is highly stochastic
        set_seed(0)

        # Prepare all mock data that are needed
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

        # Set up a mock mask with four perturbations
        mock_mask = torch.tensor(
            [
                [True, False, True, False, True, False, True, False, False],
                [False, True, False, True, False, True, False, True, True],
                [True, True, False, False, False, True, True, True, False],
                [True, False, True, True, True, False, False, False, True],
            ],
            device=device,
            dtype=torch.bool,
        )

        # Index of element to be explained: node number 1, for example
        mock_index = 1

        # Define black box (randomly initialized)
        num_features = mock_feat.shape[-1]
        mock_black_box = load_GCN_model(num_features, None).to(device)

        # Define Data class
        data_class = Data(mock_feat, mock_edge_index)

        # Define Model class
        model_class = Model(mock_black_box)

        # Set up ground truth

        # Ground truth kernel
        # Apply kernel Shap formula
        ground_truth_kernel = torch.tensor([1 / 305, 1 / 305, 1 / 305, 1 / 305], device=device)

        # Run actual coding pipeline
        result_kernel, result_output = kernel_output(
            mock_mask, data_class, model_class, "node", mock_index
        )

        # Set up some assertions that should be fulfilled
        # Kernel assertion
        # The assertion is approximate
        mae_kernel = torch.abs(ground_truth_kernel - result_kernel).mean()
        assert mae_kernel.item() < 1e-3

        # Assertion on output values, they should be between 0 and 1 (node classification problem)
        assert (result_output.min() >= 0) and (result_output.max() <= 1)

        # Assertion on output prediction shape
        num_elements = mock_mask.shape[0]
        assert result_output.flatten().shape[0] == num_elements

    def test_regularizer(self):
        """
        Test regularizer function from wlm.py script

        """

        # Create some mock data
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

        # Call linear regression model from script
        num_features = mock_feat.shape[-1]
        mock_model = LinearRegression(num_features).to(device)

        # Build up ground-truth for regularizer
        # Unpack parameters from mock model
        params = parameter_unpacker(mock_model.parameters())

        # Ground-truth regularization term
        ground_truth = (torch.abs(params[0].view(-1)).sum()) / (len(params[0].view(-1)))

        # Run actual coding pipeline
        # Assume a regularization factor of 1
        result = regularizer(mock_model, 1)

        # Assertions on regularizer values
        assert torch.equal(ground_truth, result)

    def test_optimizer_scheduler(self):
        """
        Test optimizer_scheduler function from wlm.py
        script

        """
        # Load parameter file
        with open("config/configs.json") as f:
            params = json.load(f)
            f.close()

        # Create some mock data
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

        # Call linear regression model from script
        num_features = mock_feat.shape[-1]
        mock_model = LinearRegression(num_features).to(device)

        # Run actual coding pipeline
        result_optim, result_sch = optimizer_scheduler(params, mock_model)

        # Execute assertions
        # Assertions on optimizer: it is usually Adam
        assert isinstance(result_optim, torch.optim.Adam)
        # Assertions on learning rate scheduler: it is usually ReduceLROnPlateau
        assert isinstance(result_sch, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_weighted_mse_loss(self):
        """
        Test weighted_mse_loss function from
        wlm.py script

        """

        # Create some mock data on predictions and reference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prediction
        mock_pred = torch.tensor([0.98, 0.23, -0.12, -0.24], device=device)

        # Reference
        mock_ref = torch.tensor([0.93, 0.29, -0.19, -0.31], device=device)

        # Kernels
        mock_kernel = torch.tensor([0.5, 0.85, 0.34, 0.78], device=device)

        # Ground truth computation
        ground_truth = (mock_kernel * (mock_pred - mock_ref) ** 2).mean() / (mock_kernel.sum())

        # Execution of actual coding pipeline
        result = weighted_mse_loss(mock_pred, mock_ref, mock_kernel)

        # Assertions
        assert torch.equal(ground_truth, result)


if __name__ == "__main__":
    TestLinearRegression().test_linear_regression()
    TestLinearRegression().test_model_updates()
    TestLinearRegression().test_train_model()
    TestLinearRegression().test_kernel_output()
    TestLinearRegression().test_regularizer()
    TestLinearRegression().test_optimizer_scheduler()
    TestLinearRegression().test_weighted_mse_loss()
