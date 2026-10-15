from abc import ABC
from typing import Callable, Tuple


class ClassBase(ABC):
    __constants__: list[Tuple[Callable, str, str]]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__module__}.{self.__class__.__qualname__}(\n"
            + ",\n".join(
                [
                    "\t" + f"{k[1]}={k[0](getattr(self, k[1]))}{k[2]}"
                    for k in self.__constants__
                    if hasattr(self, k[1])
                ]
            )
            + "\n)"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{k[0](getattr(self, k[1]))}{k[2]}' for k in self.__constants__ if hasattr(self, k[1])])})"
