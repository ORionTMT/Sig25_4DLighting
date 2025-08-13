import numpy as np
import torch


import torch

class TonemapHDRTensor(object):
    """
    Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
    Then, we calculate I_out = alpha * I_in ^ (1/gamma)
    input : torch tensor of images : [3, H, W]
    output : torch tensor of images : [3, H, W]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(tensor_img) * percentile) to

    def __call__(self, tensor_img, clip=True, alpha=None, gamma=True):
        # Assume tensor_img is of shape [3, H, W] (i.e., image[0])
        # Step 1: Apply gamma correction if gamma is True
        if gamma:
            power_tensor_img = torch.pow(tensor_img, 1 / self.gamma)
        else:
            power_tensor_img = tensor_img
        
        # Step 2: Calculate the non-zero mask
        non_zero = power_tensor_img > 0
        
        # Step 3: Compute the percentile
        if non_zero.any():
            r_percentile = torch.quantile(power_tensor_img[non_zero], self.percentile / 100.0)
        else:
            r_percentile = torch.quantile(power_tensor_img, self.percentile / 100.0)
        
        # Step 4: Compute alpha if not provided
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        
        # Step 5: Perform tone mapping
        tonemapped_img = alpha * power_tensor_img
        
        # Step 6: Clip the values between 0 and 1 if clip is True
        if clip:
            tonemapped_img_clip = torch.clamp(tonemapped_img, 0, 1)
        
        return tonemapped_img_clip.float(), alpha, tonemapped_img

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha, tonemapped_img