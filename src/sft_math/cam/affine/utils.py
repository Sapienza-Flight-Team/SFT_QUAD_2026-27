from sft_math.cam.affine import Affine
import numpy as np
from typing import Tuple


def broadcast_to(affine: Affine, shape: Tuple[int, ...]) -> Affine:
    try:
        shape = np.broadcast_shapes(affine.shape, shape)
    except ValueError:
        raise ValueError("The shapes of affine and shape must be broadcastable")
    else:
        return Affine(
            np.broadcast_to(affine.vector, shape + (affine.dimension,)),
            np.broadcast_to(
                affine.orthCoos2globalCoos_matrix,
                shape + (affine.dimension, affine.rank),
            ),
        ).squeeze()
