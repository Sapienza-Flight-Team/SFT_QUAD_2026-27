from abc import ABC, abstractmethod
from sft_math.class_templates import ClassBase
from sft_math.typing import Array
import numpy.typing as np_typing
from typing import TypeVar, Generic, Any

Geo_move = TypeVar("Geo_move", bound="GPSBase[Any, Any]")
Geo_base = TypeVar("Geo_base", bound="GPSBase[Any, Any]")
Geo_self = TypeVar("Geo_self", bound="GPSBase")


class GPSBase(ClassBase, ABC, Generic[Geo_base, Geo_move]):

    class Movement(ClassBase, ABC, Generic[Geo_self]):

        @abstractmethod
        def __neg__(self) -> "GPSBase.Movement[Geo_self]":
            """Negates the movement"""
            raise NotImplementedError

        @abstractmethod
        def to_vector(self) -> "GPSBase.Vector[Geo_self]":
            """Converts the movement to a vector"""
            raise NotImplementedError

    class Vector(ClassBase, ABC, Generic[Geo_self]):

        @abstractmethod
        def __neg__(self) -> "GPSBase.Vector[Geo_self]":
            raise NotImplementedError

        @abstractmethod
        def __pos__(self) -> "GPSBase.Vector[Geo_self]":
            raise NotImplementedError

        @abstractmethod
        def __add__(
            self, other: "GPSBase.Vector[Geo_self]"
        ) -> "GPSBase.Vector[Geo_self]":
            raise NotImplementedError

        @abstractmethod
        def __sub__(
            self, other: "GPSBase.Vector[Geo_self]"
        ) -> "GPSBase.Vector[Geo_self]":
            raise NotImplementedError

        @abstractmethod
        def __mul__(self, scalar: np_typing.ArrayLike) -> "GPSBase.Vector[Geo_self]":
            raise NotImplementedError

        @abstractmethod
        def __rmul__(self, scalar: np_typing.ArrayLike) -> "GPSBase.Vector[Geo_self]":
            raise NotImplementedError

        @abstractmethod
        def __truediv__(
            self, scalar: np_typing.ArrayLike
        ) -> "GPSBase.Vector[Geo_self]":
            raise NotImplementedError

        @abstractmethod
        def norm(self) -> Array:
            """Returns the norm of the vector"""
            raise NotImplementedError

        @abstractmethod
        def angle(self) -> Array:
            """Returns the angle of the vector"""
            raise NotImplementedError

        @abstractmethod
        def to_movement(self) -> "GPSBase.Movement[Geo_self]":
            """Converts the vector to a movement"""
            raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: Geo_base) -> Movement[Geo_move]:
        raise NotImplementedError

    @abstractmethod
    def __add__(self, movement: Movement[Geo_move]) -> Geo_base:
        raise NotImplementedError

    def map(self, gps: Geo_base) -> Vector[Geo_move]:
        """Maps the GPS coordinate to the manifold

        Args:
            gps (Geo_A): GPS coordinate

        Returns:
            Vector[Geo_A]: Vector on the manifold
        """
        return (-(self - gps)).to_vector()

    def unmap(self, vector: Vector[Geo_move]) -> Geo_base:
        """Unmaps the GPS coordinate from the manifold

        Args:
            vector (Vector[Geo_A]): Vector on the manifold

        Returns:
            Geo_A: GPS coordinate
        """
        return self + vector.to_movement()
