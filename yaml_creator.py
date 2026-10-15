import yaml
from copy import deepcopy
from core.options import Options
from tools.config_file_generation.sft_config_dict import STFCONFIGDICT

EMULATABLE: Options = Options(("mavlink", "camera", None), default=2)


def get_config(to_emulate):
    base_config = deepcopy(STFCONFIGDICT)
    base_config = get_config_r(base_config, to_emulate, "", "", 0)

    if (
        "mavlink" in to_emulate
        and base_config.get("mavlink", {}).get("mock_type") == "simulated"
    ):
        base_config["mavlink"]["connection"] = "udpin:0.0.0.0:14540"

    return base_config

def should_use_default_without_prompt(
    config, top_level_key, key, prev_level_key, to_emulate
):
    """Return True when camera emulation means this option can use its default."""
    if key == "mock_metadata_csv":
        if "camera" not in to_emulate and "mavlink" not in to_emulate:
            return True
        return False

    if key == "emulation_use_random_data":
        if "camera" not in to_emulate:
            return True
        return False

    if top_level_key == "mavlink":
        if "mavlink" not in to_emulate:

            if key in [
                "mock_type",
                "unreal_host_ip",
                "unreal_video_stream_port",
                "unreal_trigger_port",
            ]:
                return True
        else:

            if key in [
                "unreal_host_ip",
                "unreal_video_stream_port",
                "unreal_trigger_port",
            ]:
                if config.get("mock_type") != "simulated":
                    return True

    if "camera" not in to_emulate:
        return False

    # In mock camera mode, avoid prompting for low-level camera profile tuning.
    if top_level_key == "camera" and prev_level_key == "profile":
        return True

    # In mock mode, use the default airdrop and maneuver bounds instead of prompting.
    if top_level_key == "airdrop_bound":
        return True

    return False


def get_config_r(config, to_emulate, top_level_key, prev_level_key, level):
    for key in config:
        # I want to keep all the keys at level 1 when visiting their children
        if level == 0:
            top_level_key = key

        if not isinstance(config[key], dict):
            # if it is a dictionary it is assumed to be a dictionary of options
            # manually out in the inf SFTCONFIGDICT

            if isinstance(config[key], Options):
                if isinstance(config[key].items, dict):
                    config[key] = set_emulation(config[key], top_level_key, to_emulate)
                elif isinstance(config[key].items, range | tuple):
                    # Passing config as first argument
                    if should_use_default_without_prompt(
                        config, top_level_key, key, prev_level_key, to_emulate
                    ):
                        config[key] = config[key].get_default()
                    else:
                        print(
                            f"In {top_level_key}{', ' + prev_level_key if prev_level_key != top_level_key else ''}"
                        )
                        choose_option(config, key)
                elif isinstance(config[key].items, str):
                    # Passing config as first argument
                    if should_use_default_without_prompt(
                        config, top_level_key, key, prev_level_key, to_emulate
                    ):
                        config[key] = config[key].get_default()
                    else:
                        print(
                            f"In {top_level_key}{', ' + prev_level_key if prev_level_key != top_level_key else ''}"
                        )
                        choose_string_option(config, key)
                else:
                    raise TypeError(f"Unsupported type for {key}: {type(config[key])}")
            else:
                continue
        else:
            # if it is a dictionary, call the function recursively
            config[key] = get_config_r(
                config[key], to_emulate, top_level_key, key, level + 1
            )

    return config


def choose_string_option(config: dict, key: str) -> None:
    options: Options = config[key]
    while True:
        print(
            f"Enter a value for '{key}' press enter for default ({options.display_default()}):"
        )
        choice = input("-> ")

        if not choice:
            config[key] = options.get_default()
            break

        config[key] = choice
        break


def set_emulation(dictionary: dict, component: str, to_emulate: list[str]) -> str:
    # to emulate stores what components to emulate, they have to match
    for emu in to_emulate:
        if emu == component:
            return dictionary["mock"]
    return dictionary["no_mock"]


def choose_option(config: dict, key: str) -> None:
    options: Options = config[key]
    items = options.items
    while True:
        print(
            f"Choose an option for '{key}' press enter for default ({options.display_default()}):"
        )

        if isinstance(items, range):
            print(f"Range: {items.start} - {items.stop} ")
        else:
            for i, option in enumerate(options):
                print(f"{i+1}: {options.alias[i]} ")

        choice = input("-> ")

        # go with the default
        if not choice:
            config[key] = options.get_default()
            if not check_constraints():
                config[key] = options
                continue
            break
        try:

            choice = int(choice)

            if isinstance(items, range):
                if choice >= items.start and choice <= items.stop:
                    config[key] = choice

                else:
                    raise ValueError
            else:
                config[key] = items[
                    choice - 1
                ]  # -1 to allow choosing last available non-range option

            # check if constraints are met
            if not check_constraints():
                config[key] = options
                continue

            break

        except (ValueError, KeyError, IndexError):
            print("Invalid input, try again")
            continue


def check_constraints() -> bool:
    if isinstance(STFCONFIGDICT["camera"]["profile"]["max_exposure_time"], int):
        if (
            STFCONFIGDICT["camera"]["profile"]["max_exposure_time"]
            < STFCONFIGDICT["camera"]["profile"]["min_exposure_time"]
        ):
            print("Constraint error, max exposure is less than min exposure")
            return False
    if isinstance(STFCONFIGDICT["camera"]["profile"]["max_auto_gain"], int):
        if (
            STFCONFIGDICT["camera"]["profile"]["max_auto_gain"]
            < STFCONFIGDICT["camera"]["profile"]["min_auto_gain"]
        ):
            print("Constraint error, max gain is less than min gain")
            return False

    return True


def save_yaml(config, filename="sft_config.yaml"):
    with open(filename, "w") as file:
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)


def main():

    while True:
        print("Choose an option, press enter for default")
        print("What do you want to emulate? (None is default)")
        print("to emulate more at the same time enter and add space")
        for i, emulated in enumerate(EMULATABLE):
            print(f"{i+1}: {emulated} ")
        mode = input("-> ")

        if not mode:
            to_emulate = [None]
            break
        try:
            indexes = list(map(int, mode.split()))
            if min(indexes) < 1 or max(indexes) > len(EMULATABLE):
                raise ValueError

            to_emulate = [EMULATABLE[idx - 1] for idx in indexes]
            break
        except ValueError:
            print("INVALID INPUT, expecting space separated integers")

    config = get_config(to_emulate)
    save_yaml(config)
    print(f"Configuration saved to sft_config.yaml")


if __name__ == "__main__":
    main()
    # check integrity of the config
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from common.config import SFTConfig

    config = SFTConfig("sft_config.yaml")
    print(config.config_dict)
