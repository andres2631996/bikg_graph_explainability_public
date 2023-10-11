import torch
import os, sys
from scipy.special import binom


class Kernel:
    """
    Class for computing and manipulating kernel data to
    enforce locality in the approximation of configuration
    values

    Params
    ------
    mask : torch.tensor
        Binary mask with perturbation information

    """

    def __init__(self, mask):
        self.mask = mask

    @staticmethod
    def approximate_shap_kernel(num_active, num_total, device, ref=1000):
        """
        Approximate SHAP kernel when the M choose K term
        in the kernel SHAP equation gets very difficult
        to be computed for large samples (usually > 1,000)

        i.e.
        In the kernel computation, there is a "choose" term
        (number of elements "choose" number of active elements), if we
        have more than 1,000 elements to explain, this term
        goes to inf. Instead:
        - compute 1000 choose (0 to 1000) --> scaling vector
        - get ratio of number of active elements over number of elements (from 0 to 1)
        - get corresponding "choose" index from the scaling vector, according
        to the ratio
        - Use that "choose" value from the scaling vector to
        compute the kernel value

        The larger the value is from 1000, the more deviation
        there will be for a real KernelSHAP


        Params
        ------
        num_active : torch.tensor
            Number of active elements in each perturbation
        num_total : int
            Number of total elements to be explained
        device : torch.device
            Computational device
        ref : int
            Reference number of elements to be set to approx. Shap kernels

        Returns
        -------
        kernel : torch.tensor
            Approximated kernelSHAP

        """

        # Approximated choose term
        choose = ((binom(ref, torch.arange(ref, device="cpu")) + 1e-10) * num_total / (1000)).to(
            device
        )

        # Obtain index of coalition in a scale from 0 to 1000
        # choose[index] is the approximation of M choose K
        # if (num_active < num_total) and (num_active > 0):
        index = (num_active * 1000 / num_total).long()

        # Avoid getting indexes of the same length as the array
        index = torch.clip(index, min=0, max=len(choose) - 1)
        # Compute SHAP kernel
        kernel = (float(num_total)) / (
            choose[index] * num_active.double() * (num_total - num_active).double()
        )

        return kernel

    @staticmethod
    def original_shap_kernel(num_active, num_total, device):
        """
        If the number of elements to explain <= 1,000,
        compute the original SHAP kernel

        i.e.
        if the number of elements to explain <= 1000:
        get kernel as:
        kernel = num_total/((num_total choose num_active)*num_active*(num_total-num_active))

        Params
        ------
        num_active : torch.tensor
            Number of active elements in each perturbation
        num_total : int
            Number of total elements to be explained
        device : torch.device
            Computational device

        Returns
        -------
        kernel : torch.tensor
            Direct kernelSHAP

        """
        # Original kernel SHAP computation
        choose = binom(num_total + 1, (num_active).to("cpu")).to(device)

        kernel = (num_total) / (choose * (num_total + 1 - num_active) * num_active)

        return kernel

    def compute(self):
        """
        Compute kernelSHAP, if the number of elements
        to explain is > 1,000,approximate it to a
        kernelSHAP of 1,000 samples to avoid overflow

        i.e.
        if the number of elements to explain <= 1000:
        get kernel as:
        kernel = num_total/((num_total choose num_active)*num_active*(num_total-num_active))

        if the number of elements to explain < 1000:
        approximate choose term, and then apply same equation
        approximation will be worse as the number of elements grows further
        and further from 1000

        Returns
        -------
        kernel : torch.tensor
            Approximated kernel SHAP weights

        """

        # Obtain combinations of M choose K for M=1,000
        # (largest number that can be computed without
        # setting stuff to infinity and beyond)
        device = self.mask.device

        # Active elements in each perturbation
        num_active = torch.sum(self.mask, dim=1)
        # Total elements
        num_total = self.mask.shape[1] - 1

        if num_total > 1000:
            # Too many elements to explain, probable overflow
            # Approximate kernel to 1,000 samples
            # Reference number of elements to be considered to approximate Shap kernel
            ref = 1000
            kernel = torch.zeros(tuple([num_total]), device=device)  # Kernel initialization
            while (kernel.sum() == 0) and (ref > 0):
                kernel = self.approximate_shap_kernel(num_active, num_total, device, ref)
                if (kernel.sum() > 0) and (ref > 0):
                    # Kernel values are larger than 0
                    break
                else:
                    # Kernel values are all zero, we need less
                    # elements of reference in the choose term
                    ref = int(0.9 * ref)

        else:
            # Original kernelSHAP for less than 1,000 samples
            kernel = self.original_shap_kernel(num_active, num_total, device)

        # Compute SHAP kernel for 1 samples active to ensure
        # a symmetric kernel, in case that all samples are active
        # If all samples are active or zero, we will get posinf or neginf,
        # substituting posinf or neginf by zero
        kernel = torch.nan_to_num(kernel, posinf=0, neginf=0)

        return kernel
