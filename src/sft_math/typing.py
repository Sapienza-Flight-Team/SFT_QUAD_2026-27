from typing import TypeAlias
import numpy as np
from numpy import typing as np_typing

Bool: TypeAlias = np.bool_
Int: TypeAlias = np.int64
Float: TypeAlias = np.float64
Bools: TypeAlias = np_typing.NDArray[Bool]
Index: TypeAlias = np_typing.NDArray[Int]
Array: TypeAlias = np_typing.NDArray[Float]
