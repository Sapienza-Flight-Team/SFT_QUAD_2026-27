from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
from sft_math.cam.utils import get_mat
from typing import Dict
from common.config import SFTConfig


def select_csv(root_dir: str, input_: str):

    def merger(grouped: dict, root: str, name: str):
        merged_df = pd.DataFrame()
        last = 0
        for file in grouped[root]:
            if "metadata" not in file:
                continue
            df = metadata_checker(pd.read_csv(root + "/" + file).copy())
            df["teensy_timestamp"] = [
                (last + int(i * 1000 / default_fps)) for i in range(len(df))
            ]
            df["mission_state"] = [1 for i in range(len(df))]
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            last = int(merged_df.iloc[-1]["teensy_timestamp"])
        merged_df.to_csv(root + "/" + name, index=False)

    def metadata_checker(serie):
        # controllo se roll, pitch e yaw si chiamano  phi, theta, psi
        expected_cols = {"roll", "pitch", "yaw"}
        alt_cols = {"phi", "theta", "psi"}
        serie_cols = set(serie.columns)
        for exp, alt in zip(expected_cols, alt_cols):
            if exp not in serie_cols:
                if alt not in serie_cols:
                    raise Exception(f"Metadata missing both {exp} and {alt} columns")
                else:
                    serie.rename(columns={alt: exp}, inplace=True)
        return serie

    default_fps = 10
    grouped: Dict[str, List[str]] = {}
    acceptables: Dict[str, list[str]] = {}
    for path in Path(root_dir).rglob("*.csv"):
        folder = str(path.parent)
        file = path.name
        grouped.setdefault(folder, []).append(file)
    for i, (folder, files) in enumerate(grouped.items(), start=1):
        for j, file in enumerate(files, start=1):
            acceptables[str(i)] = [f"{i}.{j}"] + acceptables.get(str(i), [])
        acceptables[str(i)] += [str(i)]
    if len(grouped.keys()) == 0:
        raise Exception("Man ts pmo, sybau and create a dataset")
    while True:
        print(
            f"choose sub-root to select a single metadata, main root to merge all metadatas inside:"
        )
        for i, (folder, files) in enumerate(grouped.items(), start=1):
            print(f"{i}) {folder}")
            for j, file in enumerate(files, start=1):
                print(f"    {i}.{j}) {file}")
        input_ = input_ if len(input_) != 0 else input("Enter the chosen number: ")
        choice = input_.strip().split(".")
        print(f"Chosen: {input_}")
        if input_ not in [s for lst in acceptables.values() for s in lst]:
            raise Exception(
                "Tf you doin now? You tripping? Choose an actual dataset now fr"
            )
        root = int(choice[0]) - 1
        chosen = list(grouped.keys())[root]
        to_add = ""
        if len(choice) == 2:
            to_add_idx = int(choice[1]) - 1
            if (
                to_add_idx >= 0
                and to_add_idx < len(grouped[chosen])
                and root >= 0
                and root < len(grouped.keys())
            ):
                to_add = grouped[chosen][to_add_idx]
            else:
                print("Invalid input. Please try again.")
                continue
        elif len(choice) == 1:
            sub_len = len(grouped[list(grouped.keys())[int(choice[0]) - 1]])
            # I want to merge only if more than one metadata is present and it doesnt contain merged or corrected
            if sub_len == 1:
                print("Only one metadata present, no merge needed.")
                to_add = grouped[chosen][0]
            elif sub_len > 1:
                chosen = list(grouped.keys())[root]
                to_add = "merged_"
                for elem in grouped[list(grouped.keys())[root]]:
                    if (
                        "metadata" in elem
                        and "merged" not in elem
                        and "corrected" not in elem
                    ):
                        to_add += elem.split("_")[1][:-4] + "-"
                to_add = to_add[:-1] + ".csv"
                # creo un metadata misto tra i due se chosen + "/" + to_add non esiste
                if not Path(chosen + "/" + to_add).exists():
                    print("Merged metadata doesnt exists, creating it...")
                    merger(grouped, chosen, to_add)
                else:
                    print("Merged metadata already exists, selection phase skipped.")
            else:
                print(
                    "Something has gone bad, I dont know how you reached this point but I havent thought of a solution yet."
                )
                continue
        print(f"Selected metadata: {chosen + '/' + to_add}")
        return chosen + "/" + to_add


def teensy_fixer(metadata_path: str, fps: int) -> str:
    # if teensy_timestamp column is missing add it based on fps, if exists check for missing values and or 0 values and fill them, then write a corrected version of the metadata file
    metadata = pd.read_csv(metadata_path)
    print("teensy_timestamp column found, checking for missing or 0 values...")
    last = 0
    corrected = 0
    for i in range(len(metadata)):
        if (
            pd.isna(metadata.at[i, "teensy_timestamp"])
            or metadata.at[i, "teensy_timestamp"] == 0
        ):
            metadata.at[i, "teensy_timestamp"] = last + int(1000 / fps)
            corrected += 1
        last = metadata.at[i, "teensy_timestamp"]
    if corrected == 0:
        print(
            "No missing or 0 values found in teensy_timestamp column, no correction needed."
        )
        return metadata_path
    metadata["filename"] = metadata["filename"].str.replace(
        "metadata", "corrected_metadata"
    )
    # i also have to update metadata["filename"] to match the new path

    # save corrected metadata to a new file
    corrected_path = f"{metadata_path}".replace("metadata", "corrected_metadata")
    print(
        f"Corrected metadata saved to {corrected_path}, corrected {corrected} entries out of {len(metadata)}"
    )
    metadata.to_csv(corrected_path, index=False)
    return corrected_path
