from typing import Any, Iterator
from abc import ABCMeta, abstractmethod

class Orchestrator(Process) :



# role: assembling the pipeline and handling restarting of the pipeline


pipeline = Pipeline(config)
pipeline.start()