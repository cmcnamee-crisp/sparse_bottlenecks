"""Used for t3 & t4, is_causal.

The inlp implementation is from: https://github.com/yanaiela/amnesic_probing/blob/120de7e63b6d2332da8859ff90d4340dbfda4ab6/amnesic_probing/tasks/utils.py#L132
"""

from typing import List

import numpy as np
import torch as th
from scipy import linalg


def get_projection(W: th.tensor) -> th.tensor:
    """
    Constructs the nullspace projection (while maintaining the original dimensions of the matrix.)

    Note: Unlike the original INLP, we here drop the I (iterative) and run a single iteration.
    In our original implementation, we executed this iteratively and investigated the behavior.

    Parameters
    ----------
        W : `th.tensor`
            The coef of a linear model.
    """
    device = W.device
    return th.tensor(_get_projection(W.detach().numpy()), device=device)


def _get_projection(W: np.array):
    # https://github.com/yanaiela/amnesic_probing/blob/120de7e63b6d2332da8859ff90d4340dbfda4ab6/amnesic_probing/debias/debias.py#L68
    input_dim = W.shape[1]
    null_space_projection = _get_projection_to_intersection_of_nullspaces(
        [_get_rowspace_projection(W)], input_dim
    )  # projection to W's rowspace
    return null_space_projection


def _get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = linalg.orth(W.T)  # orthogonal basis

    # w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace
    return P_W


def _get_projection_to_intersection_of_nullspaces(
    rowspace_projection_matrices: List[np.ndarray], input_dim: int
):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param input_dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - _get_rowspace_projection(Q)
    return P
