from sft_math.class_templates import ClassBase
from sft_math.typing import Array, Float
import numpy as np
from numpy import _typing as np_typing
from typing import Tuple, TypeAlias
import warnings


class Affine(ClassBase):

    eps: float = 1e-10
    rank: int
    dimension: int
    vector: Array
    orthCoos2globalCoos_matrix: Array  # shape: (..., dimension, rank)
    generatorCoos2orthCoos_matrix: Array  # shape: (..., num_of_generator, rank)
    shape: Tuple[int, ...]

    __constants__ = [
        (lambda x: x, "rank", ""),
        (lambda x: x, "dimension", ""),
        (lambda x: x, "vector", ""),
        (lambda x: x, "shape", ""),
    ]

    complement: Array  # shape: (..., dimension, dimension-ranking)

    def __init__(self, vector: np_typing.ArrayLike, generator: np_typing.ArrayLike):
        """Constructs an affine space from a generator and a vector.

        Args:
            generator (np_typing.ArrayLike): generator of the affine space
            vector (np_typing.ArrayLike): vector of the affine space

        Details:
            generator must have shape (..., d, r) and vector must have shape (..., d)
            with d dimension of the space and r number of generators of the affine space.
        """
        vector = np.array(
            np.array(vector, dtype=Float), copy=False, subok=True, ndmin=2
        )
        generator = np.array(
            np.array(generator, dtype=Float), copy=False, subok=True, ndmin=3
        )
        self.dimension = vector.shape[-1]
        if generator.shape[-2] != self.dimension:
            raise ValueError("Last dimension of generator and vector must be equal")
        try:
            self.shape = np.broadcast_shapes(generator.shape[:-2], vector.shape[:-1])
            vector = np.broadcast_to(vector, self.shape + (self.dimension,))
            generator = np.broadcast_to(
                generator, self.shape + (self.dimension, generator.shape[-1])
            )
        except ValueError:
            raise ValueError("Shapes of generator and vector must be broadcastable")
        Q, R = np.linalg.qr(generator, mode="complete")
        # Q has shape (..., d, d)
        # R has shape (..., d, num_of_generator)
        # compute rank
        if (
            R.shape[-1] == 0
        ):  # non ci sono generatori quindi l'affine è un punto il rank è 0
            ranks = 0
        else:
            ranks = np.linalg.matrix_rank(R, tol=self.eps)
        if np.any(ranks < np.max(ranks)):
            warnings.warn("Exists an affine space with lower rank")
        self.rank = int(np.max(ranks))  # rank <= min(d,r)
        self.orthCoos2globalCoos_matrix = Q[..., :, : self.rank]
        self.generatorCoos2orthCoos_matrix = R[..., : self.rank, :]
        self.vector = vector
        self.complement = Q[..., :, self.rank :]

    def __call__(self, coordinates: np_typing.ArrayLike) -> Array:
        """Get a global coordinate of a vector in the affine space from its local coordinates.

        Args:
            coordinates (np_typing.ArrayLike): local coordinates of the vector

        Returns:
            Array: global coordinates of the vector
        """
        coordinates = np.array(
            np.array(coordinates, dtype=Float), copy=False, subok=True, ndmin=2
        )
        if coordinates.shape[-1] != self.generatorCoos2orthCoos_matrix.shape[-1]:
            raise ValueError(
                "Last dimension of coordinates must be equal to the numbr of generators"
            )
        orthCoos = np.einsum(
            "...ik, ...k -> ...i", self.generatorCoos2orthCoos_matrix, coordinates
        )
        localCoos = np.einsum(
            "...ik, ...k -> ...i", self.orthCoos2globalCoos_matrix, orthCoos
        )
        return self.vector + localCoos

    def squeeze(self) -> "Affine":
        """Squeeze generator matrix with orthogonal coordinates to the affine space."""
        return Affine(self.vector, self.orthCoos2globalCoos_matrix)

    def orthogonal(self) -> "Affine":
        """Get the orthogonal complement of the affine space."""
        return Affine(self.vector, self.complement)

    def __pos__(self) -> "Affine":
        return Affine(self.vector, self.orthCoos2globalCoos_matrix).squeeze()

    def __neg__(self) -> "Affine":
        return Affine(-self.vector, self.orthCoos2globalCoos_matrix).squeeze()

    def __add__(self, other: "Affine") -> "Affine":
        if self.dimension != other.dimension:
            raise ValueError("Dimensions of the two affine spaces must be equal")
        try:
            shape = np.broadcast_shapes(self.shape, other.shape)
        except ValueError:
            raise ValueError("Shapes of the two affine spaces must be broadcastable")
        else:
            self_generator = np.broadcast_to(
                self.orthCoos2globalCoos_matrix,
                shape
                + (
                    self.dimension,
                    self.rank,
                ),
            )
            other_generator = np.broadcast_to(
                other.orthCoos2globalCoos_matrix,
                shape
                + (
                    other.dimension,
                    other.rank,
                ),
            )
            generator = np.concatenate([self_generator, other_generator], axis=-1)
            vector = self.vector + other.vector
            return Affine(vector, generator).squeeze()

    def __sub__(self, other: "Affine") -> "Affine":
        return self + (-other)

    def __mul__(self, other: np_typing.ArrayLike) -> "Affine":
        other = np.array(np.array(other, dtype=Float), copy=False, subok=True, ndmin=1)
        other = np.reshape(other, other.shape + (1,))
        try:
            np.broadcast_shapes(self.vector.shape, other.shape)
        except ValueError:
            raise ValueError(
                "Shapes of the affine space and the scalar must be broadcastable"
            )
        return Affine(self.vector * other, self.orthCoos2globalCoos_matrix).squeeze()

    def __rmul__(self, other: np_typing.ArrayLike) -> "Affine":
        return self * other

    def __truediv__(self, other: np_typing.ArrayLike) -> "Affine":
        other = np.array(other, dtype=Float)
        return self * (1 / other)

    def __iadd__(self, other: "Affine") -> "Affine":
        res = self + other
        for attr in vars(res):
            setattr(self, attr, getattr(res, attr))
        return self

    def __isub__(self, other: "Affine") -> "Affine":
        res = self - other
        for attr in vars(res):
            setattr(self, attr, getattr(res, attr))
        return self

    def __imul__(self, other: np_typing.ArrayLike) -> "Affine":
        res = self * other
        for attr in vars(res):
            setattr(self, attr, getattr(res, attr))
        return self

    def __and__(self, other: "Affine") -> "Affine":
        if self.dimension != other.dimension:
            raise ValueError("The dimensions of the two affine must be equal")
        try:
            shape = np.broadcast_shapes(self.shape, other.shape)
        except ValueError:
            raise ValueError("The shapes of the two affine must be broadcastable")
        else:
            M = np.concatenate(
                [
                    np.broadcast_to(
                        self.orthCoos2globalCoos_matrix,
                        shape + (self.dimension, self.rank),
                    ),
                    -np.broadcast_to(
                        other.orthCoos2globalCoos_matrix,
                        shape + (other.dimension, other.rank),
                    ),
                ],
                axis=-1,
            )  # (..., dimension, rank1 + rank2)
            rhs = other.vector - self.vector
            U, S, Vt = np.linalg.svd(M)
            rhs = np.array(np.einsum("...ki, ...k -> ...i", U, rhs), dtype=Float)
            null_rankarr = np.count_nonzero(S > self.eps, axis=-1)
            null_rank = np.min(null_rankarr)
            if null_rank < rhs.shape[-1]:
                if np.any(np.abs(rhs[..., null_rank:]) > self.eps):
                    raise RuntimeError("The affine spaces do not intersect")
            if np.any(null_rankarr != null_rank):
                warnings.warn("Incompatible rank detected")
            rhs = (
                rhs[..., :null_rank] / S[..., :null_rank]
            )  # has shape (..., null_rank)
            # aggiungo zeri, lo shape va da (..., null_rank) a (..., rank1 + rank2) con padding
            rhs = np.pad(
                rhs,
                tuple((0, 0) for _ in range(len(shape)))
                + ((0, self.rank + other.rank - null_rank),),
            )
            rhs = np.einsum(
                "...ki, ...k -> ...i", Vt, rhs
            )  # has shape (..., rank1 + rank2)
            vector = np.einsum(
                "...ik, ...k -> ...i",
                self.orthCoos2globalCoos_matrix,
                rhs[..., : self.rank],
            )
            B = Vt[..., null_rank:, : self.rank].copy()
            generator = np.einsum(
                "...rk, ...ck -> ...rc", self.orthCoos2globalCoos_matrix, B
            )
            return Affine(vector, generator).squeeze()

    def __invert__(self) -> "Affine":
        return Affine(self.vector, self.complement)

    def __or__(self, other: "Affine") -> "Affine":
        return ~((~self) & (~other))

    def __xor__(self, other: "Affine") -> "Affine":
        return (self & (~other)) | ((~self) & other)

    def __le__(self, other: "Affine") -> bool:
        return (self & other).rank == self.rank

    def __lt__(self, other: "Affine") -> bool:
        return (self <= other) and (self.rank < other.rank)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Affine):
            raise NotImplementedError
        return (self <= other) and (self.rank == other.rank)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Affine):
            raise NotImplementedError
        return not self == other

    def __gt__(self, other: "Affine") -> bool:
        return other < self

    def __ge__(self, other: "Affine") -> bool:
        return other <= self

    def is_parallel(self, other: "Affine") -> bool:
        if self.dimension != other.dimension:
            raise ValueError("Dimensions of the two affine spaces must be equal")
        try:
            np.broadcast_shapes(self.shape, other.shape)
        except ValueError:
            raise ValueError("Shapes of the two affine spaces must be broadcastable")
        if self.rank > other.rank:
            return False
        rhs = np.einsum(
            "...kr, ...kc -> ...rc", other.complement, self.orthCoos2globalCoos_matrix
        )
        return np.allclose(rhs, np.zeros_like(rhs), atol=self.eps)

    def is_orthogonal(self, other: "Affine") -> bool:
        return not self.orthogonal().is_parallel(other)

    def change_gen(self, matrix: np_typing.ArrayLike) -> "Affine":
        """Changes the inner coordinates of the affine space

        Args:
            matrix (np_typing.ArrayLike): transformation matrix with shape (..., curr_ngenerators, new_ngenerators)

        Returns:
            Affine: affine space with new outer coordinates
        """
        matrix = np.array(
            np.array(matrix, dtype=Float), copy=False, subok=True, ndmin=3
        )
        if self.generatorCoos2orthCoos_matrix[-1] != matrix.shape[-2]:
            raise ValueError(
                "The transformation matrix is not compatible with the generator"
            )
        try:
            np.broadcast_shapes(matrix.shape[:-2], self.shape)
        except:
            raise ValueError("matrix must be broadcastable")
        generator = np.einsum(
            "...rk, ...kc -> ...rc", self.generatorCoos2orthCoos_matrix, matrix
        )
        generator = np.einsum(
            "...rk, ...kc, -> ...rc", self.orthCoos2globalCoos_matrix, generator
        )
        return Affine(self.vector, generator)

    def change_vec(self, matrix: np_typing.ArrayLike) -> "Affine":
        """Changes the outer coordinates of the affine space

        Args:
            matrix (np_typing.ArrayLike): transformation matrix with shape (..., new_dimension, curr_dimension)

        Returns:
            Affine: affine space with new outer coordinates
        """
        matrix = np.array(
            np.array(matrix, dtype=Float), copy=False, subok=True, ndmin=3
        )
        if matrix.shape[-1] != self.dimension:
            raise ValueError(
                "The transformation matrix is not compatible with the dimension"
            )
        vector = np.einsum("...ik, ...k -> ...i", matrix, self.vector)
        generator = np.einsum(
            "...rk, ...kc -> ...rc", matrix, self.orthCoos2globalCoos_matrix
        )
        generator = np.einsum(
            "...rk, ...kc -> ...rc", generator, self.generatorCoos2orthCoos_matrix
        )
        return Affine(vector, generator)

    def __getitem__(self, key: Tuple) -> "Affine":
        return Affine(self.vector[key], self.orthCoos2globalCoos_matrix[key])

    def reshape(self, shape: Tuple[int, ...]) -> "Affine":
        return Affine(
            self.vector.reshape(shape + (self.dimension,)),
            self.orthCoos2globalCoos_matrix.reshape(
                shape
                + (
                    self.dimension,
                    self.rank,
                )
            ),
        )

    def broadcast_to(self, shape: Tuple[int, ...]) -> "Affine":
        return Affine(
            np.broadcast_to(self.vector, shape + (self.dimension,)),
            np.broadcast_to(
                self.orthCoos2globalCoos_matrix,
                shape
                + (
                    self.dimension,
                    self.rank,
                ),
            ),
        )


Point: TypeAlias = Affine
Straight: TypeAlias = Affine
