from core.stage_catalog import build_stage_catalog, read_profile

import yaml
from pathlib import Path
from importlib import import_module

def import_classes(catalog):
    """Import the single class defined in each selected stage module."""
    for stage in catalog.values():
        path = stage["path"].relative_to(Path(__file__).resolve().parent / "src")
        print(f"path: {path}")
        module = import_module(stage["module"])
        stage["class"] = next(
            cls for cls in vars(module).values()
            if isinstance(cls, type) and cls.__module__ == module.__name__
        )
    return catalog

def build_dependency_graph(catalog):
    # returns the graph of which stages produces and consumes what
    graph = {}

    for stage in catalog.values():
        with open(stage["contract"], encoding="utf-8") as file:
            contract = yaml.safe_load(file)

        for direction in ("input", "output"):
            for data_type in contract[direction]:
                data_stages = graph.setdefault(data_type, {"input": [], "output": []})
                data_stages[direction].append(stage["class"])

    return graph

catalog = build_stage_catalog()
#print(catalog)

catalog = read_profile(catalog)
#print(catalog)

for stage, contents in catalog.items():
    with open(contents["contract"], encoding="utf-8") as file:
        contract = yaml.safe_load(file)
        print(f"{stage}: {contract}")
