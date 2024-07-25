import numpy as np
import math


def T_inv(T_in):
    """
    Calculate the inverse transformation matrix `T_inv` of a given homogeneous transformation matrix `T_in`.

    Args:
        T_in (numpy.ndarray): 4x4 homogeneous transformation matrix.

    Returns:
        numpy.ndarray: Inverse of the input transformation matrix `T_in`.
    """
    R_in = T_in[:3, :3]
    t_in = T_in[:3, [-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out, t_in)
    return np.vstack((np.hstack((R_out, t_out)), np.array([0, 0, 0, 1])))


def convert_quat_xyzw_to_wxyz(q):
    """
    Convert a quaternion from XYZW convention to WXYZ convention.

    Args:
        q (list or numpy.ndarray): Quaternion in XYZW convention.

    Returns:
        list or numpy.ndarray: Quaternion converted to WXYZ convention.
    """
    q[0], q[1], q[2], q[3] = q[3], q[0], q[1], q[2]
    return q


def convert_quat_wxyz_to_xyzw(q):
    """
    Convert a quaternion from WXYZ convention to XYZW convention.

    Args:
        q (list or numpy.ndarray): Quaternion in WXYZ convention.

    Returns:
        list or numpy.ndarray: Quaternion converted to XYZW convention.
    """

    q[3], q[0], q[1], q[2] = q[0], q[1], q[2], q[3]
    return q


def unit_vector(data, axis=None, out=None):
    """
    Normalize a vector or array of vectors along a specified axis.

    Args:
        data (array_like): Input vector or array of vectors to be normalized.
        axis (int, optional): Axis along which to compute vector norms.
        out (numpy.ndarray, optional): Output array for storing results.

    Returns:
        numpy.ndarray: Normalized vector or array of vectors.
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def concatenate_matrices(*matrices):
    """
    Concatenate multiple homogeneous transformation matrices into a single matrix.

    Args:
        *matrices (numpy.ndarray): Homogeneous transformation matrices to concatenate.

    Returns:
        numpy.ndarray: Resulting concatenated homogeneous transformation matrix.
    """
    M = np.identity(4)
    for i in matrices:
        M = np.dot(M, i)
    return M


def rotation_matrix(angle, direction, point=None):
    """
    Generate a 4x4 homogeneous rotation matrix for rotation around a specified axis by a given angle.

    Args:
        angle (float): Angle of rotation in radians.
        direction (array_like): Unit vector specifying the axis of rotation.
        point (array_like, optional): Point around which to perform the rotation (default: origin).

    Returns:
        numpy.ndarray: Homogeneous rotation matrix representing the rotation around the specified axis.
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0, 0.0),
                  (0.0, cosa, 0.0),
                  (0.0, 0.0, cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(((0.0, -direction[2], direction[1]),
                   (direction[2], 0.0, -direction[0]),
                   (-direction[1], direction[0], 0.0)),
                  dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M
