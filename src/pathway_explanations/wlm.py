import torch
import numpy as np
from torch import nn
import gc

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import networkx as nx

from pathway_explanations.data import Data
from pathway_explanations.kernels import Kernel
from pathway_explanations.model import Model


class LinearRegression(nn.Module):
    """
    Model for LIME/SHAP weighted linear regression

    i.e.
    linear_output = config_val*perturbation mask
    with
    loss = kernel(linear_output-GNN perturbed output)**2

    Params
    ------
    num_elements : int
        Number of elements (nodes or features to be explained)
    params : dict
        Hyperparameters

    Returns
    -------
    out : torch.tensor
        Predictions of weighted linear regression model

    """

    def __init__(self, num_elements):
        assert isinstance(num_elements, int)
        num_elements = int(num_elements)
        super(LinearRegression, self).__init__()

        self.layer = nn.Linear(num_elements, 1, bias=False)

    def forward(self, X):
        """
        Forward pass of weighted linear regression model

        Params
        ------
        X : torch.tensor
            Mask used in the training process

        Returns
        -------
        Forward pass of WLRM

        """
        return self.layer(X)


def model_updates(linear_model, loss, best_loss):
    """
    Check if the model improves on the loss for the training
    set, and in that case, extract model parameters

    i.e.
    if metric from current epoch is better,
    update metric

    Params
    ------
    linear_model : torch nn module
        Weighted linear regression model for
        approximation of configuration values
    loss : torch tensor
        Loss function tensor
    best_loss : float
        Best training loss registered so far in the
        training process

    Returns
    -------
    best_parameters : torch.tensor
        Parameters found for the best training iteration
        so far
    best_loss : float
         Updated best loss term

    """

    best_parameters = linear_model.parameters()
    # print("Best loss improved from {} to {}".format(best_loss,loss.item()))
    best_loss = loss.item()

    return best_parameters, best_loss


def regularizer(net, factor):
    """
    Compute L1 regularizer for weighted linear regression
    model

    i.e.
    regularizer = |coeffs|/sum(coeffs)

    Params:
    ------
    net : torch.nn.Module
        Network
    factor : float
        L1 factor

    Returns:
    -------
    loss : torch.tensor
        Loss function from L1 regularizer

    """

    to_regularise = []
    for param in net.parameters():
        to_regularise.append(param.view(-1))

    loss = torch.abs(torch.cat(to_regularise))

    return factor * (loss.sum() / loss.shape[0])


def train_model(
    mask_loader,
    params,
    feat,
    edge_index,
    linear_model,
    arch,
    problem,
    element_index=None,
    node_type=None,
    edge_type=None,
    node_type_names=None,
    edge_type_names=None,
    padded_dims=None,
):
    """
    Train weighted linear regression model

    Params
    ------
    mask_loader : torch.dataloader
        Dataloader
    params : dict
        Hyperparameters
    feat : torch.tensor of float
        Feature matrix
    edge_index : torch.tensor of float
        Edge indices
    linear_model : torch nn module
        Weighted linear regression model for
        approximation of configuration values
    arch : PyG model
        Relational Machine Learning model to be explained
    problem : str
        Graph element analysed
    element_index : int
        Index of element to be explained, for node and edge
        prediction cases (default: None)
    node_type : torch.tensor
        Node type for heterogeneous graph (default: None)
    edge_type : torch.tensor
        Edge type for heterogeneous graph (default: None)
    node_type_names : list of str
        Names of node types in heterogeneous graph (default: None)
    edge_type_names : list of tuple
        Names of edge types in heterogeneous graph (default: None)
    padded_dims : list of ints
        If node features have different sizes for each type,
        they need to be padded to be set into a homogeneous
        graph. How much each node type has been padded
        (default: None)

    Returns
    -------
    weights : list of float
        Weights of linear layer after training
    losses : list of float
        All loss function values
    best_epoch : int
        Epoch when minimal training loss is recorded

    """
    # Define data class
    data_class = Data(feat, edge_index)

    # Load optimizer and learning rate scheduler
    opt, sch = optimizer_scheduler(params, linear_model)

    best_loss = np.inf  # Loss value of reference for when to save model parameters
    best_epoch = 0  # Epoch of reference for when to save model parameters

    # Set black box model to evaluation mode
    arch.eval()
    model_class = Model(arch)

    # Store all losses
    losses = []

    for epoch, mask in enumerate(mask_loader):

        # Release un-needed memory in the GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load a set of rows with permutations for each iteration
        kernel, output = kernel_output(
            mask,
            data_class,
            model_class,
            problem,
            element_index,
            node_type,
            edge_type,
            node_type_names,
            edge_type_names,
            padded_dims,
        )

        # zero the parameter gradients
        opt.zero_grad()

        pred = linear_model(mask.float())

        del mask

        # Loss computation

        # L1 regularizer
        reg = regularizer(linear_model, params["l1_lambda"])

        loss = weighted_mse_loss(pred, output, kernel)
        loss += reg

        del pred, output, kernel, reg

        # Backprop (≈delta rule)
        loss.backward()
        opt.step()
        # sch.step(loss.item())

        # Loss storage
        losses.append(loss.item())

        # Keep model with lowest training loss
        if loss.item() < best_loss:
            # Update best training parameters, loss function value found and best epoch
            best_parameters, best_loss = model_updates(linear_model, loss, best_loss)
            best_epoch = epoch

    # Loop through parameters to find out the linear model weights
    weights = []
    try:
        for i, parameter in enumerate(best_parameters):
            if i == 0:
                weights.append(parameter[0])
                continue
    except:
        # The model has never learnt properly, so there are not "optimized parameters".
        # Use the final model parameters instead
        parameters = linear_model.parameters()

        for i, parameter in enumerate(parameters):
            if i == 0:
                weights.append(parameter[0])
                continue

    return weights, losses, best_epoch


# Helper functions


def kernel_output(
    mask,
    data_class,
    model_class,
    problem,
    element_index=None,
    node_type=None,
    edge_type=None,
    node_type_names=None,
    edge_type_names=None,
    padded_dims=None,
):
    """
    Compute perturbations, kernel, and perturbed
    outputs for each epoch of weighted linear
    regression model

    i.e.
    get mask --> obtain number of active elements in each permutation
    get kernel --> with number of active elements and number of total
                    elements

    (check:https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)


    Params
    ------
    mask : torch.tensor
        Binary mask with perturbation information
    data_class : Data object
        Class with data information on feature
        matrix and edge index
    model_class : Model object
        Class with model information
    problem : str
        Type of problem to be explained
    element_index : int
        Index of element to be explained, for node and edge
        prediction cases (default: None)
    node_type : torch.tensor
        Node type for heterogeneous graph (default: None)
    edge_type : torch.tensor
        Edge type for heterogeneous graph (default: None)
    node_type_names : list of str
        Names of node types in heterogeneous graph (default: None)
    edge_type_names : list of tuple
        Names of edge types in heterogeneous graph (default: None)
    padded_dims : list of ints
        If node features have different sizes for each type,
        they need to be padded to be set into a homogeneous
        graph. How much each node type has been padded
        (default: None)

    Returns
    -------
    kernel : torch.tensor
        Kernel for configuration value approximation
    output : torch.tensor
        Parametric relational model output

    """
    # Release un-needed memory in the GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Perturb graph topology
    concat_feat, concat_node_type, perturb_edge_index, perturbed_edge_type = data_class.perturbator(
        mask, problem, node_type, edge_type
    )

    # Release un-needed memory in the GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Determine number of node types involved
    # For heterogeneous graph
    node_types = 1
    if node_type_names is not None:
        unique_node_type = torch.unique(node_type)
        node_types = len(unique_node_type)

    # Compute perturbed output
    # Set edge index in perturbation to long datatype
    perturb_edge_index = perturb_edge_index.long()

    if (
        (node_type is not None)
        and (edge_type is not None)
        and (node_type_names is not None)
        and (edge_type_names is not None)
    ):
        if node_types < 2:  # Only for heterogeneous graphs with an only node type

            # Heterogeneous graph
            # Convert homogeneous graph with node types
            # and edge types back into heterogeneous

            # Heterogeneous node features
            concat_feat = data_class.homo2hetero(
                concat_feat, concat_node_type, node_type_names, padded_dims
            )

            # Heterogeneous edge indices
            perturb_edge_index = data_class.homo2hetero(
                perturb_edge_index, perturbed_edge_type, edge_type_names
            )

    # Release un-needed memory in the GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Black-box inference

    if node_types < 2:
        # Predict output for homogeneous graph
        # or heterogeneous graph with an only node type
        output = model_class.infer(
            concat_feat, perturb_edge_index, concat_node_type, perturbed_edge_type
        )
    else:
        # Predict output for heterogeneous graph with more than one node type

        output = model_class.predict_hetero_output(
            concat_feat,
            perturb_edge_index,
            concat_node_type,
            perturbed_edge_type,
            node_type_names,
            edge_type_names,
            mask.shape[0],
            mask.shape[1],
            element_index,
            padded_dims,
            problem,
        )

    # Compute kernel
    kernel_class = Kernel(mask)
    kernel = kernel_class.compute()

    # Release un-needed memory in the GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if (node_type is not None) and (edge_type is not None) and (isinstance(output, dict)):
        # Convert output from heterogeneous graph
        # back into homogeneous graph
        output, output_types = model_class.hetero2homo_output(output)

    # For node and edge prediction, extract just the predictions of the
    # relevant node or edge
    if element_index is not None:
        output = model_class.extract_node_edge_output(output, element_index, mask.shape[1])

    return kernel, output


def optimizer_scheduler(params, arch):
    """
    Load optimizer and learning rate scheduler

    i.e.
    read parameter file, and get optimizer and
    scheduler types specified, if available

    Parameters
    ----------
    params : dict
        Pipeline hyperparameters
    arch : torch model
        Weighted linear regression model to be trained

    Returns
    -------
    optimizer : PyTorch optimizer
        Loaded optimizer
    sch : PyTorch learning rate scheduler
        Loaded learning rate scheduler, to follow training loss

    """
    # Load hyperparameters
    opt = params["optimizer"]  # Usually Adam
    lr = params["lr"]  # Usually 0.01
    patience = params["lr_patience"]  # Usually 10 epochs

    assert isinstance(opt, str), "Optimizer is not string"
    assert isinstance(lr, float) or isinstance(lr, int), "Learning rate given is not numeric"
    assert isinstance(patience, float) or isinstance(
        patience, int
    ), "Patience for scheduler is not string"

    # Load optimizer
    # TODO: extend to more potential optimizers
    if opt.strip().lower() == "adam":
        optimizer = torch.optim.Adam(arch.parameters(), lr=abs(lr), weight_decay=1e-2)
    else:
        print("Optimizer choice not available. Please choose between 'adam'")

    # Load scheduler
    # TODO: extend to more potential learning rates
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=abs(int(patience)), verbose=True
    )

    return optimizer, sch


def weighted_mse_loss(input, target, weight):
    """
    Loss function for weighted linear regression,
    including L1 regularization

    i.e.
    loss = kernel(linear_output - GNN_output)**2

    Params
    ------
    input : torch tensor
        Output from linear regression model
    target : torch tensor
        Expected output from black-box model
    weight : torch tensor
        SHAP or LIME kernel
    phi : torch tensor
        Parameters from model to be trained

    Returns
    -------
    loss : torch.tensor
        Loss function value

    """

    diff = (input.flatten() - target) ** 2
    loss = torch.mean(weight * diff) / (weight.sum())

    return loss
