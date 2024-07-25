import sys
import numpy as np
import torch
import json


def save_df_as_json(df_data, save_path, file_name):
    """
    Save DataFrame data to a JSON file.

    Args:
        df_data (pandas.DataFrame): DataFrame to save.
        save_path (str): Path to save the JSON file.
        file_name (str): Name of the JSON file.
    """
    to_dict = {}
    for index, row in list(df_data.iterrows()):
        to_dict[index] = dict(row)
    with open(r'{}{}.json'.format(save_path, file_name), 'w') as json_file:
        json.dump(to_dict, json_file, indent=3)


def find_strings_with_substring(string_list, substring):
    """
    Save DataFrame data to a JSON file.

    Args:
        df_data (pandas.DataFrame): DataFrame to save.
        save_path (str): Path to save the JSON file.
        file_name (str): Name of the JSON file.
    """
    result = []
    for string in string_list:
        if substring in string and 'finetune' not in string:
            result.append(string)
    return result


class ColorPrint:
    """
    Utility class for colored terminal output.
    """

    @staticmethod
    def print_fail(message, end='\n'):
        """
        Print a failure message in red.

        Args:
            message (str): Message to print.
            end (str, optional): Ending character(s). Defaults to '\n'.
        """
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_pass(message, end='\n'):
        """
        Print a success message in green.

        Args:
            message (str): Message to print.
            end (str, optional): Ending character(s). Defaults to '\n'.
        """
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        """
        Print a warning message in yellow.

        Args:
            message (str): Message to print.
            end (str, optional): Ending character(s). Defaults to '\n'.
        """
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_info(message, end='\n'):
        """
        Print an informational message in blue.

        Args:
            message (str): Message to print.
            end (str, optional): Ending character(s). Defaults to '\n'.
        """
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_bold(message, end='\n'):
        """
        Print a bold message in white.

        Args:
            message (str): Message to print.
            end (str, optional): Ending character(s). Defaults to '\n'.
        """
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)


def pandas_col_to_numpy(df_col):
    """
    Convert a pandas DataFrame column containing string representation of arrays to numpy arrays.

    Args:
        df_col (pandas.Series): DataFrame column containing string representation of arrays.

    Returns:
        numpy.ndarray: Numpy array representation of the column data.
    """
    df_col = df_col.apply(
        lambda x: np.fromstring(x.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "), sep=", "))
    df_col = np.stack(df_col)
    return df_col


def pandas_string_to_numpy(arr_str):
    """
    Convert a string representation of an array to a numpy array.

    Args:
        arr_str (str): String representation of the array.

    Returns:
        numpy.ndarray: Numpy array representation of the input string.
    """
    arr_npy = np.fromstring(arr_str.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "), sep=", ")
    return arr_npy


def medfilter(x, W=20):
    """
    Apply median filtering to a 1-dimensional numpy array.

    Args:
        x (numpy.ndarray): Input array to be filtered.
        W (int): Window size for the median filter. Default is 20.

    Returns:
        numpy.ndarray: Median filtered array.
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
        tensor (torch.Tensor): Input tensor to be normalized.
        mean (tuple or list or torch.Tensor): Mean values for normalization.
        std (tuple or list or torch.Tensor): Standard deviation values for normalization.

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
        data (numpy.ndarray or torch.Tensor): Input data to be normalized.
        mean (float or numpy.ndarray or torch.Tensor): Mean value or array of mean values.
        std (float or numpy.ndarray or torch.Tensor): Standard deviation value or array of std values.
        eps (float, optional): Small value to prevent division by zero. Default is 1e-8.

    Returns:
        numpy.ndarray or torch.Tensor: Normalized data.
    """
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    """
    Unnormalize data using mean and standard deviation.

    Args:
        data (numpy.ndarray or torch.Tensor): Input data to be unnormalized.
        mean (float or numpy.ndarray or torch.Tensor): Mean value or array of mean values.
        std (float or numpy.ndarray or torch.Tensor): Standard deviation value or array of std values.

    Returns:
        numpy.ndarray or torch.Tensor: Unnormalized data.
    """
    return data * std + mean


def normalize_max_min(data, dmax, dmin, eps=1e-8):
    """
    Normalize data using maximum and minimum values.

    Args:
        data (numpy.ndarray): Input data to be normalized.
        dmax (float or numpy.ndarray): Maximum value or array of maximum values.
        dmin (float or numpy.ndarray): Minimum value or array of minimum values.
        eps (float, optional): Small value to prevent division by zero. Default is 1e-8.

    Returns:
        numpy.ndarray: Normalized data.
    """
    return (data - dmin) / (dmax - dmin + eps)


def unnormalize_max_min(data, dmax, dmin):
    """
    Unnormalize data using maximum and minimum values.

    Args:
        data (numpy.ndarray): Input data to be unnormalized.
        dmax (float or numpy.ndarray): Maximum value or array of maximum values.
        dmin (float or numpy.ndarray): Minimum value or array of minimum values.

    Returns:
        numpy.ndarray: Unnormalized data.
    """
    dmax = np.array(dmax)
    dmin = np.array(dmin)
    return data * (dmax - dmin) + dmin


def rolling_window(a, window):
    """
    Create a rolling window view of the input array.

    Args:
        a (numpy.ndarray): Input array.
        window (int): Size of the rolling window.

    Returns:
        numpy.ndarray: Rolling window view of the input array.
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
    Calculate mean and plot with error bars.

    Args:
        x (numpy.ndarray): X-axis values.
        y (numpy.ndarray): Y-axis values.
        ax (matplotlib.axes.Axes): Axes object for plotting.
        ylabel (str): Label for the Y-axis.
        window (int, optional): Window size for calculating mean. Defaults to 100.
    """
    mean = np.mean(rolling_window(y, window), axis=-1)
    std = np.std(rolling_window(y, window * 2), axis=-1)

    ax.plot(x, y, 'ko', markersize=1, alpha=0.3)
    ax.plot(x, mean, 'bo', markersize=1, alpha=0.5)
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, edgecolor='none')
    ax.set_xlabel('Force [N]')
    ax.set_ylabel(ylabel)
    ax.set_xlim([min(x), max(x)])


