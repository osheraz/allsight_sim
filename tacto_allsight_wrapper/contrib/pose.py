# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
SE(3) pose utilities

This module provides utilities for generating SE(3) poses from vertices and normals,
including shear transformations and delta rotations.
"""


import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple






def pose_from_vertex_normal(
    vertices: np.ndarray, normals: np.ndarray, shear_mag: float, delta: np.ndarray
) -> np.ndarray:
    """
    Generate SE(3) pose given vertices and normals.

    Args:
        vertices (np.ndarray): Vertices of the mesh (N, 3).
        normals (np.ndarray): Normals of the mesh (N, 3).
        shear_mag (float): Shear magnitude for pose generation.
        delta (np.ndarray): Delta rotations for pose generation (N, 1).

    Returns:
        np.ndarray: Generated SE(3) poses (N, 4, 4).
    """
    vertices = np.atleast_2d(vertices)
    normals = np.atleast_2d(normals)

    num_samples = vertices.shape[0]
    T = np.zeros((num_samples, 4, 4))  # transform from point coord to world coord
    T[:, 3, 3] = 1
    T[:, :3, 3] = vertices  # t

    # resolve ambiguous DoF
    """Find rotation of shear_vector so its orientation matches normal: np.dot(Rot, shear_vector) = normal
    https://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another """

    cos_shear_mag = np.random.uniform(
        low=np.cos(shear_mag), high=1.0, size=(num_samples,)
    )  # Base of shear cone
    shear_phi = np.random.uniform(
        low=0.0, high=2 * np.pi, size=(num_samples,)
    )  # Circle of shear cone

    # Axis v = (shear_vector \cross normal)/(||shear_vector \cross normal||)
    shear_vector = np.array(
        [
            np.sqrt(1 - cos_shear_mag**2) * np.cos(shear_phi),
            np.sqrt(1 - cos_shear_mag**2) * np.sin(shear_phi),
            cos_shear_mag,
        ]
    ).T
    shear_vector_skew = skew_matrix(shear_vector)
    v = np.einsum("ijk,jk->ik", shear_vector_skew, normals.T).T
    v = v / np.linalg.norm(v, axis=1).reshape(-1, 1)

    # find corner cases
    check = np.einsum("ij,ij->i", normals, np.array([[0, 0, 1]]))
    zero_idx_up = check > 0.9  # pointing up
    zero_idx_down = check < -0.9  # pointing down

    v_skew, sampledNormals_skew = skew_matrix(v), skew_matrix(normals)

    # Angle theta = \arccos(z_axis \dot normal)
    # elementwise: theta = np.arccos(np.dot(shear_vector,normal)/(np.linalg.norm(shear_vector)*np.linalg.norm(normal)))
    theta = np.arccos(
        np.einsum("ij,ij->i", shear_vector, normals)
        / (np.linalg.norm(shear_vector, axis=1) * np.linalg.norm(normals, axis=1))
    )

    identity_3d = np.zeros(v_skew.shape)
    np.einsum("iij->ij", identity_3d)[:] = 1
    # elementwise: Rot = np.identity(3) + v_skew*np.sin(theta) + np.linalg.matrix_power(v_skew,2) * (1-np.cos(theta)) # rodrigues
    Rot = (
        identity_3d
        + v_skew * np.sin(theta)
        + np.einsum("ijn,jkn->ikn", v_skew, v_skew) * (1 - np.cos(theta))
    )  # rodrigues

    if np.any(zero_idx_up):
        Rot[:3, :3, zero_idx_up] = np.dstack([np.identity(3)] * np.sum(zero_idx_up))
    if np.any(zero_idx_down):
        Rot[:3, :3, zero_idx_down] = np.dstack(
            [np.array([[1, 0, 0], [0, -1, -0], [0, 0, -1]])] * np.sum(zero_idx_down)
        )

    # Rotation about Z axis is still ambiguous, generating random rotation b/w [0, 2pi] about normal axis
    # elementwise: RotDelta = np.identity(3) + normal_skew*np.sin(delta[i]) + np.linalg.matrix_power(normal_skew,2) * (1-np.cos(delta[i])) # rodrigues
    RotDelta = (
        identity_3d
        + sampledNormals_skew * np.sin(delta)
        + np.einsum("ijn,jkn->ikn", sampledNormals_skew, sampledNormals_skew)
        * (1 - np.cos(delta))
    )  # rodrigues

    # elementwise:  RotDelta @ Rot
    tfs = np.einsum("ijn,jkn->ikn", RotDelta, Rot)
    T[:, :3, :3] = np.rollaxis(tfs, 2)
    return T





def skew_matrix(v: np.ndarray) -> np.ndarray:
    """
    Generate SE(3) pose given vertices and normals.

    Args:
        vertices (np.ndarray): Vertices of the mesh (N, 3).
        normals (np.ndarray): Normals of the mesh (N, 3).
        shear_mag (float): Shear magnitude for pose generation.
        delta (np.ndarray): Delta rotations for pose generation (N, 1).

    Returns:
        np.ndarray: Generated SE(3) poses (N, 4, 4).
    """
    v = np.atleast_2d(v)
    # vector to its skew matrix
    mat = np.zeros((3, 3, v.shape[0]))
    mat[0, 1, :] = -1 * v[:, 2]
    mat[0, 2, :] = v[:, 1]

    mat[1, 0, :] = v[:, 2]
    mat[1, 2, :] = -1 * v[:, 0]

    mat[2, 0, :] = -1 * v[:, 1]
    mat[2, 1, :] = v[:, 0]
    return mat



