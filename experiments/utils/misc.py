import sys
import numpy as np
import torch
import cv2
import json


def save_df_as_json(df_data, save_path, file_name):
    """
    Save DataFrame as JSON file.

    Args:
        df_data (DataFrame): DataFrame to be converted to JSON.
        save_path (str): Path to save the JSON file.
        file_name (str): Name of the JSON file.

    Returns:
        None
    """
    to_dict = {}
    for index, row in list(df_data.iterrows()):
        to_dict[index] = dict(row)
    with open(r'{}{}.json'.format(save_path, file_name), 'w') as json_file:
        json.dump(to_dict, json_file, indent=3)


class ColorPrint:
    """
    Class to print colored messages to console.

    Methods:
        print_fail(message, end='\n'):
            Print a failure message in red.
        print_pass(message, end='\n'):
            Print a success message in green.
        print_warn(message, end='\n'):
            Print a warning message in yellow.
        print_info(message, end='\n'):
            Print an informational message in blue.
        print_bold(message, end='\n'):
            Print a bold message in white.
    """

    @staticmethod
    def print_fail(message, end='\n'):
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_pass(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_info(message, end='\n'):
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_bold(message, end='\n'):
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)


def pandas_col_to_numpy(df_col):
    """
    Convert pandas DataFrame column to numpy array.

    Args:
        df_col (pd.Series): DataFrame column containing string representations of arrays.

    Returns:
        np.ndarray: Numpy array converted from DataFrame column.
    """
    df_col = df_col.apply(
        lambda x: np.fromstring(x.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "), sep=", "))
    df_col = np.stack(df_col)
    return df_col


def pandas_string_to_numpy(arr_str):
    """
    Convert string representation of an array to numpy array.

    Args:
        arr_str (str): String representation of the array.

    Returns:
        np.ndarray: Numpy array converted from string representation.
    """
    arr_npy = np.fromstring(arr_str.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "), sep=", ")
    return arr_npy


def medfilter(x, W=20):
    """
    Apply median filter to 1D array.

    Args:
        x (np.ndarray): Input array to filter.
        W (int, optional): Window size for median filter. Defaults to 20.

    Returns:
        np.ndarray: Filtered array.
    """
    w = int(W / 2)
    x_new = np.copy(x)
    for i in range(0, x.shape[0]):
        if i < w:
            x_new[i] = np.mean(x[:i + w])
        elif i > x.shape[0] - w:
            x_new[i] = np.mean(x[i - w:])
        else:
            x_new[i] = np.mean(x[i - w:i + w])
    return x_new


def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Inverse normalization of a tensor.

    Args:
        tensor (torch.Tensor): Tensor to be normalized.
        mean (tuple): Mean values used for normalization. Defaults to (0.485, 0.456, 0.406).
        std (tuple): Standard deviation values used for normalization. Defaults to (0.229, 0.224, 0.225).

    Returns:
        torch.Tensor: Inverse normalized tensor.
    """
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def normalize(data, mean, std, eps=1e-8):
    """
    Normalize data using mean and standard deviation.

    Args:
        data (np.ndarray): Data to be normalized.
        mean (np.ndarray): Mean values for normalization.
        std (np.ndarray): Standard deviation values for normalization.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        np.ndarray: Normalized data.
    """
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    """
    Unnormalize data using mean and standard deviation.

    Args:
        data (np.ndarray): Data to be unnormalized.
        mean (np.ndarray): Mean values used for normalization.
        std (np.ndarray): Standard deviation values used for normalization.

    Returns:
        np.ndarray: Unnormalized data.
    """
    return data * std + mean


def normalize_max_min(data, dmax, dmin, eps=1e-8):
    """
    Normalize data between maximum and minimum values.

    Args:
        data (np.ndarray): Data to be normalized.
        dmax (float or np.ndarray): Maximum value or array of maximum values.
        dmin (float or np.ndarray): Minimum value or array of minimum values.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        np.ndarray: Normalized data.
    """
    return (data - dmin) / (dmax - dmin + eps)


def unnormalize_max_min(data, dmax, dmin):
    """
    Unnormalize data between maximum and minimum values.

    Args:
        data (np.ndarray): Data to be unnormalized.
        dmax (float or np.ndarray): Maximum value or array of maximum values.
        dmin (float or np.ndarray): Minimum value or array of minimum values.

    Returns:
        np.ndarray: Unnormalized data.
    """
    dmax = np.array(dmax)
    dmin = np.array(dmin)
    return data * (dmax - dmin) + dmin


def rolling_window(a, window):
    """
    Apply a rolling window to an array.

    Args:
        a (np.ndarray): Input array.
        window (int): Size of the rolling window.

    Returns:
        np.ndarray: Rolled array.
    """
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window - 1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad, mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def mean_and_plot(x, y, ax, ylabel, window=100):
    """
    Compute mean and plot data.

    Args:
        x (np.ndarray): X-axis data.
        y (np.ndarray): Y-axis data.
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
        ylabel (str): Label for the Y-axis.
        window (int, optional): Window size for mean calculation. Defaults to 100.

    Returns:
        None
    """
    mean = np.mean(rolling_window(y, window), axis=-1)
    std = np.std(rolling_window(y, window * 2), axis=-1)

    ax.plot(x, y, 'ko', markersize=1, alpha=0.3)
    ax.plot(x, mean, 'bo', markersize=1, alpha=0.5)
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, edgecolor='none')
    ax.set_xlabel('Force [N]')
    ax.set_ylabel(ylabel)
    ax.set_xlim([min(x), max(x)])


import torchvision.transforms.functional as TF

class ToGrayscale(object):
    """
    Convert image to grayscale version of image.

    Args:
        num_output_channels (int): Number of output channels (1 for grayscale, 3 for RGB).

    Methods:
        __call__(self, img):
            Convert input image to grayscale.

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Convert input image to grayscale.

        Args:
            img (PIL.Image or torch.Tensor): Input image to convert.

        Returns:
            PIL.Image or torch.Tensor: Grayscale version of input image.
        """
        return TF.to_grayscale(img, self.num_output_channels)


class AdjustGamma(object):
    """
    Perform gamma correction on an image.

    Args:
        gamma (float): Gamma value for correction.
        gain (float): Multiplicative factor. Default is 1.

    Methods:
        __call__(self, img):
            Perform gamma correction on the input image.
    """

    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img):
        """
        Perform gamma correction on the input image.

        Args:
            img (PIL.Image or torch.Tensor): Input image to perform gamma correction.

        Returns:
            PIL.Image or torch.Tensor: Gamma corrected image.
        """
        return TF.adjust_gamma(img, self.gamma, self.gain)

from PIL import ImageFilter
import random

class GaussianBlur(object):
    """
    Gaussian blur augmentation for images.

    Args:
        sigma (list or tuple): Range for sigma value.

    Methods:
        __call__(self, x):
            Apply Gaussian blur to the input image.
    """
    #Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        """
        Apply Gaussian blur to the input image.

        Args:
            x (PIL.Image): Input image to apply blur.

        Returns:
            PIL.Image: Blurred image.
        """
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoCropsTransform:
    """
    Take two random crops of one image as the query and key.

    Args:
        base_transform (callable): Transformation function for image preprocessing.

    Methods:
        __call__(self, x):
            Apply the transformation on the input image and return query and key crops.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        """
        Apply the transformation on the input image and return query and key crops.

        Args:
            x (torch.Tensor or PIL.Image): Input image to transform.

        Returns:
            torch.Tensor: Concatenation of query and key crops.
        """
        q = self.base_transform(x)
        k = self.base_transform(x)
        return torch.cat([q, k], dim=0)