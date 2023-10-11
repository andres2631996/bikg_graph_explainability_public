import torch
from math import comb
import pytest

from pathway_explanations.kernels import Kernel


class TestKernel:
    def test_approximate_shap_kernel(self):
        """
        Function to test "approximate_shap_kernel" function in Kernel
        code module

        """
        # Build some mock index of simulated number of graph
        # elements active in a perturbation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mock_active = torch.tensor([1500], dtype=torch.long, device=device)
        mock_total = torch.tensor([2000], dtype=torch.long, device=device).item()

        # Define ground truth kernel SHAP with
        # weight between 1 and 1,000 elements
        # Define ground truth choose term as the num_total choose num_terms
        # weighted by the relative number of elements over 1,000
        ground_truth_choose = comb(1000, 750) * (mock_total / 1000)
        # Computation of ground-truth kernel
        ground_truth = 1999 / (ground_truth_choose * 1500 * 500)

        # Run actual coding pipeline
        kernel_class = Kernel(None)
        result = kernel_class.approximate_shap_kernel(mock_active, mock_total, device)

        # Perform assertions on the kernel values
        # Allow for a confidence interval, since results will not be exact due to
        # noise and floating point precision
        diff = ground_truth - result.item()
        assert (diff > -0.01) and (diff < 0.01)

    def test_original_shap_kernel(self):
        """
        Function to test "original_shap_kernel" function in Kernel
        code module

        """
        # Build mock data mask to be used in
        # obtaining kernel values for each perturbation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mask
        mock_mask = torch.tensor(
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

        # Number of active elements per perturbation element
        mock_active_elements = torch.tensor(
            [0, 4, 4, 2, 2, 7, 4, 6, 1], dtype=torch.long, device=device
        )

        # Ground-truth kernel computation
        # Combinatorial results for all rows of the mock_mask
        # Number of columns CHOOSE number of active elements
        combinatorial = torch.tensor([1, 126, 126, 1248480, 1248480, 36, 126, 84, 9], device=device)
        # Numerator term
        num = mock_mask.shape[-1] - 1
        # Denominator term
        den = combinatorial * mock_active_elements * (mock_mask.shape[-1] - mock_active_elements)
        ground_truth = (num) / (den)

        # Execute actual coding pipeline
        kernel_class = Kernel(mock_mask)
        result = kernel_class.original_shap_kernel(
            mock_active_elements, mock_mask.shape[-1] - 1, device
        )

        # Execute assertions
        # Set any infinites to 0
        ground_truth = torch.nan_to_num(ground_truth, posinf=0, neginf=0)
        result = torch.nan_to_num(result, posinf=0, neginf=0)

        # Run assertions approximately
        diff = torch.mean(ground_truth - result).item()
        assert (diff > -0.01) and (diff < 0.01)

    # No tests needed for compute function, as it is barely a function
    # that either calls approximate_shap_kernel or original_shap_kernel
    # function, without doing really much else


if __name__ == "__main__":
    TestKernel.test_approximate_shap_kernel()
    TestKernel.test_original_shap_kernel()
