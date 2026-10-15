from pathlib import Path

STAGES_ROOT = Path(__file__).resolve().parents[1] / "stages"


def build_stage_catalog(root: Path = STAGES_ROOT) -> dict:
    """Discover stage implementations without importing their Python modules."""
    root = Path(root)
    catalog = {}
    for stage_file in sorted(root.glob("*/*/*_stage.py")):
        implementation = stage_file.parent
        catalog.setdefault(implementation.parent.name, {})[implementation.name] = {
            "path": stage_file,
            "config": next(implementation.glob("*_config.py")),
            "contract": next(implementation.glob("*_contract.yaml")),
            "module": ".".join(stage_file.relative_to(root.parent).with_suffix("").parts),
        }
    return catalog


def read_profile(catalog: dict, profile: list[dict]) -> dict:
    """Select implementations and flatten their settings from mission_config.yaml."""
    selected = {}
    for entry in profile:
        stage, implementations = next(iter(entry.items()))
        implementation, settings = next(iter(implementations[0].items()))
        selected[stage] = {
            **catalog[stage][implementation],
            "settings": {key: value for option in settings or [] for key, value in option.items()},
        }
    return selected
