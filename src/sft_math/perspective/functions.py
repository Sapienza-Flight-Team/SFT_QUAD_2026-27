import numpy as np
from numpy import typing as np_typing
from sft_math.typing import Array, Float
from typing import Tuple, TypedDict


class Isography3DResult(TypedDict):
    percentage: float
    normals: Tuple[Array, Array]
    isometry_matrix: Array
    homography_matrix: Array


def isography3D(
    homography: np_typing.ArrayLike,
) -> Tuple[Isography3DResult, Isography3DResult | None]:
    """This function computes descriptors of a 3D isography

    Args:
        homography (np_typing.ArrayLike): A 3x3 homography matrix.

    Returns:
        Isography3DResult: A dictionary containing the following keys:
            - percentage: The change of distance from the observer
            - normals: A tuple containing the normals of the two planes
            - isometry_matrix: The isometry matrix
            - homography_matrix: The homography matrix

    Raises:
        - (ValueError) : if the homography is not a 3x3 matrix
        - (ValueError) : if the homography is not invertible
        - (ValueError) : if the homography is not a valid isometry with determinant 1
    """
    H: np_typing.NDArray[Float] = np.array(homography, dtype=Float)
    if H.shape != (
        3,
        3,
    ):
        raise ValueError("The homography has to be a 3x3 matrix")
    det = float(np.linalg.det(H).item())
    U: np_typing.NDArray[Float]
    S: np_typing.NDArray[Float]
    Vt: np_typing.NDArray[Float]
    U, S, Vt = np.linalg.svd(H)
    if S[1] == 0:
        raise ValueError("The homography has to be invertible")
    L: np_typing.NDArray[Float] = H / S[1]  # candidate
    S = S / S[1]  # normalisation

    # Report the change of distance from the observer
    percentage = float(np.abs(S[0] * S[2]).item())

    # Check if the isometry has a determinant of 1
    Q1t: np_typing.NDArray[Float] = np.zeros((3, 3), dtype=Float)
    Q1sign = np.sign(det)
    if Q1sign == 0:
        raise ValueError("The homography has to be invertible")
    Q1t = (
        np.array(
            [
                [
                    0,
                    S[0] * (1 - S[2] ** 2) ** 0.5,
                    Q1sign * S[2] * (S[0] ** 2 - 1) ** 0.5,
                ],
                [(S[0] ** 2 - S[2] ** 2) ** 0.5, 0, 0],
                [
                    0,
                    -S[2] * (S[0] ** 2 - 1) ** 0.5,
                    Q1sign * S[0] * (1 - S[2] ** 2) ** 0.5,
                ],
            ],
            dtype=Float,
        )
        / (S[0] ** 2 - S[2] ** 2) ** 0.5
    )
    Q2t: np_typing.NDArray[Float] = (
        np.array(
            [
                [0, (1 - S[2] ** 2) ** 0.5, (S[0] ** 2 - 1) ** 0.5],
                [(S[0] ** 2 - S[2] ** 2) ** 0.5, 0, 0],
                [0, -((S[0] ** 2 - 1) ** 0.5), (1 - S[2] ** 2) ** 0.5],
            ],
            dtype=Float,
        )
        / (S[0] ** 2 - S[2] ** 2) ** 0.5
    )

    # Check the signs
    caseApp = float(Vt.T[2, 0] * Q2t[0, 2] + Vt.T[2, 2] * Q2t[2, 2])
    caseApm = float(Vt.T[2, 0] * Q2t[0, 2] - Vt.T[2, 2] * Q2t[2, 2])
    caseBpp = float(U[2, 0] * Q1t[0, 2] + U[2, 2] * Q1t[2, 2])
    caseBpm = float(U[2, 0] * Q1t[0, 2] - U[2, 2] * Q1t[2, 2])

    # then this conditional statement, pbool will be:
    #  if pbool is True we choose the plane described by planeAp
    #  if pbool is False we choose the plane described by planeAm
    pbool: bool
    planeAp: float | None = None
    planeAm: float | None = None
    if Q1sign > 0:  # the two cases are concordants
        if caseApp * caseBpp <= 0 and caseApm * caseBpm <= 0:
            raise ValueError("The homography is not a valid isometry")
        if caseApp * caseBpp > 0:
            planeAp = float(np.abs(caseApp).item())
        if caseApm * caseBpm > 0:
            planeAm = float(np.abs(caseApm).item())
        # choose best normal
        pbool = planeAp is not None  # if True planeAp exists
        if planeAp is not None and planeAm is not None:  # if planeAp and planeAm exist
            pbool = (
                planeAp > planeAm
            )  # if True we choose the plane described by planeAp (ex. True -> planeAp better than planeAm)
    else:  # the two cases are discordants
        if caseApp * caseBpp >= 0 and caseApm * caseBpm >= 0:
            raise ValueError("The homography is not a valid isometry")
        if caseApp * caseBpp < 0:
            planeAp = float(np.abs(caseApp).item())
        if caseApm * caseBpm < 0:
            planeAm = float(np.abs(caseApm).item())
        # choose best normal
        pbool = planeAp is not None  # if True planeAp exists
        if planeAp is not None and planeAm is not None:  # if planeAp and planeAm exist
            pbool = (
                planeAp > planeAm
            )  # if True we choose the plane described by planeAp (ex. True -> planeAp better than planeAm)

    # first output: best normal

    # PN: the case of mm produces same results of the case of pp (similarly about pm and mp)
    Vtilde: np_typing.NDArray[Float]
    Utilde: np_typing.NDArray[Float]
    if pbool:  # we choose the planeAp (so the case pp)
        # PN: if caseApp < 0 then the normal of the planeA is inverted, so we choose the case mm to the case pp
        Vtilde = np.array(
            [[np.sign(caseApp), 0, 0], [0, 1, 0], [0, 0, np.sign(caseApp)]], dtype=Float
        )
        Utilde = Vtilde  # ? PN: just a formality
    else:  # we choose the planeAm (so the case pm)
        # PN: if caseApp < 0 then the normal of the planeA is inverted, so we choose the case mp to the case pm
        # the canonical form is planeA with positive normal
        Vtilde = np.array(
            [[np.sign(caseApp), 0, 0], [0, 1, 0], [0, 0, -np.sign(caseApp)]],
            dtype=Float,
        )
        Utilde = Vtilde  # ? PN: just a formality

    # Compute the orthogonal matrices QA and QB in canonical form (M=Id)
    QA = Vt.T @ Vtilde @ Q2t
    QB = U @ Utilde @ Q1t

    # Compute the planes' normals
    hatA = QA[:, 2]
    hatB = QB[:, 2]

    # Compute the orthogonal matrix of isometry
    Q = QB @ QA.T

    # Compute the translation of the isometry using canonical form (1+etaA=1, i.e. the distance of planeA from the observer is 1)
    t = (
        (S[0] - Q1sign * S[2])
        / (S[0] ** 2 - S[2] ** 2) ** 0.5
        * U
        @ Utilde
        @ np.array(
            [(S[0] ** 2 - 1) ** 0.5, 0, -Q1sign * (1 - S[2] ** 2) ** 0.5], dtype=Float
        )
    )
    hatA /= np.linalg.norm(hatA)
    hatB /= np.linalg.norm(hatB)

    first_out = Isography3DResult(
        percentage=percentage,
        normals=(hatA, hatB),
        isometry_matrix=np.concatenate([Q, t[..., None]], axis=1),
        homography_matrix=L,
    )

    # second output
    second_out: Isography3DResult | None = None
    if pbool and planeAm is None:
        second_out = None
    elif not pbool and planeAp is None:
        second_out = None
    else:
        if not pbool:
            Vtilde = np.array(
                [[np.sign(caseApp), 0, 0], [0, 1, 0], [0, 0, np.sign(caseApp)]],
                dtype=Float,
            )
            Utilde = Vtilde
        else:
            Vtilde = np.array(
                [[np.sign(caseApp), 0, 0], [0, 1, 0], [0, 0, -np.sign(caseApp)]],
                dtype=Float,
            )
            Utilde = Vtilde
        QA = Vt.T @ Vtilde @ Q2t
        QB = U @ Utilde @ Q1t
        t = (
            (S[0] - Q1sign * S[2])
            / (S[0] ** 2 - S[2] ** 2) ** 0.5
            * U
            @ Utilde
            @ np.array(
                [(S[0] ** 2 - 1) ** 0.5, 0, -Q1sign * (1 - S[2] ** 2) ** 0.5],
                dtype=Float,
            )
        )
        hatA /= np.linalg.norm(hatA)
        hatB /= np.linalg.norm(hatB)
        second_out = Isography3DResult(
            percentage=percentage,
            normals=(hatA, hatB),
            isometry_matrix=np.concatenate([Q, t[..., None]], axis=1),
            homography_matrix=L,
        )
    return first_out, second_out
