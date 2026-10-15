from stages.decision_system.default.class_templates import ClassHandlerTemplate
from typing import Dict, Optional, Any, List
from logging import Logger
import numpy as np
from numpy import typing as np_typing


class ClassHandler(ClassHandlerTemplate):

    n_classes: int
    classes: List[int]
    conf_M: np_typing.NDArray
    use_identity_mapping: bool

    def __init__(self, logger: Logger, configs: Dict[str, Optional[Any]]) -> None:
        self.logger = logger
        self.configs = configs

        if "classes" not in configs:
            raise ClassHandler.Failure("classes is required in configs")
        if "conf_matrix" in configs and not type(configs["conf_matrix"]) is str:
            raise ClassHandler.Failure(
                f"conf_matrx must be a str, but got {type(configs['conf_matrix'])}"
            )
        if not type(configs["classes"]) is list:
            raise ClassHandler.Failure(
                f"classes must be a list, but got {type(configs['classes'])}"
            )
        if not all([type(x) == int for x in configs["classes"]]):
            raise ClassHandler.Failure(
                f"classes must be a list of int, but got {configs['classes']}"
            )
        if not len(set(configs["classes"])) == len(configs["classes"]):
            raise ClassHandler.Failure(
                f"classes must be a list without repetition, but got {configs['classes']}"
            )
        relax_value = configs.get("relax_value", 0.0)
        if not type(relax_value) is float:
            raise ClassHandler.Failure(
                f"relax_value must be a float, but got {type(relax_value)}"
            )

        self.use_identity_mapping = "conf_matrix" not in configs

        # load confusion matrix
        if "conf_matrix" in configs:
            conf_M = np.array(np.load(configs["conf_matrix"]), dtype=np.float64)
        else:
            conf_M = np.eye(max(configs["classes"]) + 1, dtype=np.float64)
        if conf_M.ndim != 2:
            raise ClassHandler.Failure(
                f"conf_matrix must be a 2D array, but got {conf_M.ndim}D"
            )
        if conf_M.shape[0] != conf_M.shape[1]:
            raise ClassHandler.Failure(
                f"conf_matrix must be a square matrix, but got {conf_M.shape}"
            )
        if not np.allclose(np.sum(conf_M, axis=1), 1.0):
            raise ClassHandler.Failure(
                f"conf_matrix must be a stochastic matrix, but got {conf_M}"
            )
        self.n_classes = conf_M.shape[0]
        # load classes
        self.classes = configs["classes"]
        if not all([0 <= x < self.n_classes for x in self.classes]):
            raise ClassHandler.Failure(
                f"classes must be in [0, {self.n_classes}), but got {configs['classes']}"
            )
        if self.use_identity_mapping:
            self.conf_M = conf_M
        else:
            # make the final confusion matrix
            self.conf_M = conf_M[:, self.classes] + relax_value
            self.conf_M /= np.sum(self.conf_M, 1, keepdims=True)

    def __call__(self, classes: np_typing.ArrayLike) -> np_typing.NDArray[np.int64]:

        self.logger.info("Processing classes %s...", classes)

        np_classes = np.array(classes, dtype=np.int64)
        if np_classes.ndim != 1:
            raise ClassHandler.Failure(
                f"classes must be a 1D-matrix, but got {np_classes.shape}"
            )
        if self.use_identity_mapping:
            if np.any(0 > np_classes):
                raise ClassHandler.Failure(
                    f"classes must be non-negative, got {np_classes}"
                )
            return np_classes

        if np.any(0 > np_classes) or np.any(np_classes >= self.n_classes):
            raise ClassHandler.Failure(
                f"classes must be in [0, {self.n_classes}), but got {np_classes}"
            )
        probs = self.conf_M[np_classes]

        self.logger.info("Classes processed successfully")

        selected_classes = np.array(self.classes, dtype=np.int64)
        return np.array(
            [np.random.choice(selected_classes, p=row) for row in probs],
            dtype=np.int64,
        )
