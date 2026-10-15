from stages.decision_system.default import DecisionSystem, Mock
from common.multi_event import MultiEvent
from common.config import SFTConfig
import multiprocessing as mp
import time
from tqdm import tqdm
import argparse
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
from sft_math.cam.utils import get_mat
from typing import Dict, List
from common.structs import PhotoExif, WayPoint
from common.mavlink_structs import PixData
from core.logger import SFTLogger


def select_csv(root_dir: str, input_: str):

    def filter(serie):
        if (
            serie["h"] < 20
            or serie["h"] > 31
            or np.abs(
                np.rad2deg(
                    np.arccos(
                        (
                            get_mat(
                                serie["phi"], serie["theta"], 0, is_deg=True
                            ).trace()
                            - 1
                        )
                        / 2
                    )
                )
            )
            > 25
        ):
            return False
        return True

    def merger(grouped: dict, root: str, name: str):
        merged_df = pd.DataFrame()
        last = 0
        for file in grouped[root]:
            if "metadata" not in file:
                continue
            df = pd.read_csv(root + "/" + file).copy()
            df = df[df.apply(filter, axis=1)]
            df["teensy_timestamp"] = [
                (last + int(i * 1000 / config_dict["stage"]["fps"]))
                for i in range(len(df))
            ]
            df["mission_state"] = [1 for i in range(len(df))]
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            last = int(merged_df.iloc[-1]["teensy_timestamp"])
        merged_df.to_csv(root + "/" + name, index=False)

    grouped: Dict[str, List[str]] = {}
    acceptables = set()
    for path in Path(root_dir).rglob("*.csv"):
        folder = str(path.parent)
        file = path.name
        grouped.setdefault(folder, []).append(file)
    for i, (folder, files) in enumerate(grouped.items(), start=1):
        for j, file in enumerate(files, start=1):
            acceptables.add(f"{i}.{j}")
        acceptables.add(str(i))
    if len(grouped.keys()) == 0:
        raise Exception("No CSV files found in the specified directory.")
    while True:
        print(
            f"choose sub-root to select a single metadata, main root to merge all metadatas inside"
        )
        for i, (folder, files) in enumerate(grouped.items(), start=1):
            print(f"{i}) {folder}")
            for j, file in enumerate(files, start=1):
                print(f"    {i}.{j}) {file}")
        input_ = input_ if len(input_) != 0 else str(input("Enter the chosen number: "))
        choice = input_.strip().split(".")
        print(f"Chosen: {input_}")
        if input_ not in acceptables:
            raise Exception(
                "Man tf you doin now? You tripping? Choose an actual dataset now fr"
            )
        if len(choice) == 2:
            root, csv = int(choice[0]) - 1, int(choice[1]) - 1
            chosen = list(grouped.keys())[root]
            if (
                csv >= 0
                and csv < len(grouped[chosen])
                and root >= 0
                and root < len(grouped.keys())
            ):
                return chosen + "/" + grouped[chosen][int(csv)]
            else:
                print("Invalid input. Please try again.")
                continue
        elif len(choice) == 1:
            root, strg = int(choice[0]) - 1, "merged_"
            chosen = list(grouped.keys())[root]
            for elem in range(len(grouped[list(grouped.keys())[root]])):
                if "metadata" in grouped[list(grouped.keys())[root]][elem]:
                    strg += str(elem) + "-"
            strg = strg[:-1] + ".csv"
            # creo un metadata misto tra i due
            merger(grouped, chosen, strg)
            return chosen + "/" + strg


def csv_loader(csv_path: str):
    # Offline replay: read CSV rows and adapt them to the runtime queue format.
    pd_metadata = pd.read_csv(csv_path)
    inputs: List[PhotoExif] = []
    photo_idx: List[int] = np.array(
        pd_metadata["photo"].unique(), dtype=np.int64
    ).tolist()

    for photo in photo_idx:
        # seleziono il dataframe con photo = photo_idx
        indices = pd_metadata["photo"] == photo
        sub_df = pd_metadata.iloc[np.where(indices)[0]]
        if len(sub_df) != 0:
            photo_info = sub_df.iloc[0][
                [
                    "photo",
                    "lat",
                    "lon",
                    "teensy_timestamp",
                    "phi",
                    "theta",
                    "psi",
                    "h",
                ]
            ]
            targets_info = sub_df[
                [
                    "x",
                    "y",
                    "classification",
                ]
            ]

            photo = PhotoExif(
                filename=str(photo_info["photo"]),
                timestamp=int(photo_info["teensy_timestamp"]),
                exposure_time=0.0,
                gain=0.0,
                black_level=0,
                bsl_color_space=" ",
                gamma=0,
                light_source_preset=" ",
                pixel_format=" ",
                pix_data=PixData(
                    lat=np.array(photo_info["lat"], dtype=np.float64),
                    lon=np.array(photo_info["lon"], dtype=np.float64),
                    alt_rel=np.array(photo_info["h"], dtype=np.float32),
                    roll=np.array(photo_info["phi"], dtype=np.float32),
                    pitch=np.array(photo_info["theta"], dtype=np.float32),
                    yaw=np.array(photo_info["psi"], dtype=np.float32),
                    teensy_timestamp=np.array(
                        photo_info["teensy_timestamp"], dtype=np.uint32
                    ),
                ),
            )

            photo.bboxes = [
                {
                    "bbox": [float(x), float(y), float(x), float(y)],
                    "class": int(cls),
                }
                for x, y, cls in zip(
                    targets_info["x"].to_numpy(dtype=np.float64),
                    targets_info["y"].to_numpy(dtype=np.float64),
                    targets_info["classification"].to_numpy(dtype=np.int64),
                )
            ]
            inputs.append(photo)
    return inputs


# read the arguments
parser = argparse.ArgumentParser(
    description="This script is used to test the class stage over some dataset",
)

parser.add_argument("-rt", "--runtime", action="store_true", help="Realtime simulation")
parser.add_argument("--mock", action="store_true", help="Use mock as stage class")

args = parser.parse_args()

# load the configuration
configs = SFTConfig(r"sft_config.yaml")
logger = SFTLogger(configs)
logger.info("Starting pipeline")
config_dict = configs.config_dict

# main parameters
runtime: bool = args.runtime
metadata_csv: str = select_csv("data", "")

inputs = csv_loader(metadata_csv)

in_mission = MultiEvent(1)
to_update = None
in_q: mp.Queue = mp.Queue()
out_q: List[mp.Queue] = [mp.Queue(), mp.Queue()]
idx = 0

if args.mock:
    stage = Mock(
        configs,
        logger,
        in_mission,
        to_update,
        in_q,
        out_q,
        idx,
    )
else:
    stage = DecisionSystem(
        configs,
        logger,
        in_mission,
        to_update,
        in_q,
        out_q,
        idx,
    )

obj: List[List[WayPoint]]

if runtime:
    logger.info(f"Starting {stage.name}")
    stage.start()
    time.sleep(1)  # test di simulazione a coda vuota

    with (
        tqdm(
            total=len(inputs), desc=" Sending", position=0, unit="photo", leave=False
        ) as send_bar,
        tqdm(
            total=len(inputs), desc="Received", position=1, unit="photo", leave=False
        ) as recv_bar,
    ):
        # attivo la missione per 4 volte
        L = [int(len(inputs) / 4 * k + 0.5) for k in range(4 + 1)]
        k = 0
        time.sleep(1)
        for i in range(len(inputs)):
            # protocollo principale: attivare la missione se richiesto
            if i == L[k + 1] - 1:
                k += 1
                in_mission.events[0].set()
            if not stage.is_alive():
                raise Exception("process is not alive")
            send_bar.update(1)
            data = inputs[i]
            in_q.put(data)
            while out_q[1].qsize() > 0:
                out_q[1].get()
                recv_bar.update(1)
            # disattivo la missione
            if out_q[0].qsize() == k:
                in_mission.events[0].clear()
            # leggo da input il timestamp
            if i < len(inputs) - 1:
                time.sleep((inputs[i + 1].timestamp - inputs[i].timestamp) / 1000)
        if in_mission.events[0].is_set():
            in_mission.events[0].clear()
        # metto un timeout sull'attività di stage
        timeout = 1.0
        time_offset = time.time()
        while time.time() - time_offset < timeout:
            while out_q[1].qsize() > 0:
                out_q[1].get()
                recv_bar.update(1)
                time_offset = time.time()
    while not out_q[0].empty():
        obj = out_q[0].get()
        # mostro il risultato
        for i in range(len(obj)):
            logger.info(f"Mission {i}: lat={obj[i][0].lat}, lon={obj[i][0].lon}")
    # chiudo le code di output
    time.sleep(1)
    while not out_q[0].empty():
        out_q[0].get()
    while not out_q[1].empty():
        out_q[1].get()
    # stoppo lo stage
    time.sleep(1)
    logger.info(f"Stopping {stage.name}")
    stage.stop()
    # chiudo lo stage
    time.sleep(1)
    logger.info(f"Closing {stage.name}")
    stage.join()
    # chiudo la coda di input
    time.sleep(1)
    logger.info(f"Emptying {stage.name} input queue")
    while not in_q.empty():
        in_q.get()
    in_q.close()

else:
    logger.info(f"Starting {stage.name}")
    stage.start()
    bar = tqdm(
        total=len(inputs), desc="Received", position=0, unit="photo", leave=False
    )
    L = [int(len(inputs) / 4 * k + 0.5) for k in range(4 + 1)]
    for k in range(4):
        for i in range(L[k], L[k + 1]):
            data = inputs[i]
            in_q.put(data)
        while out_q[1].qsize() != len(range(L[k], L[k + 1])) + L[k]:
            # controllo se il processo stage è ancora attivo
            if not stage.is_alive():
                raise Exception("stage process is not alive")
            # setto la barra ad out_q[1].qsize
            if bar.n != out_q[1].qsize():
                bar.n = out_q[1].qsize()
                bar.last_print_n = out_q[1].qsize()
                bar.refresh()
        # attivo la missione
        in_mission.events[0].set()
        # attendo la ricezione della missione
        while out_q[0].qsize() != k + 1:
            if not stage.is_alive():
                raise Exception("stage process is not alive")
        # disattivo la missione se attiva
        in_mission.events[0].clear()

    bar.close()
    while not out_q[0].empty():
        obj = out_q[0].get()
        # mostro il risultato
        for i in range(len(obj)):
            logger.info(f"Mission {i}: lat={obj[i][0].lat}, lon={obj[i][0].lon}")
    while not out_q[1].empty():
        out_q[1].get()
    time.sleep(1)
    logger.info(f"Stopping {stage.name}")
    stage.stop()
    logger.info(f"Closing {stage.name}")
    stage.join()
    logger.info(f"Emptying {stage.name} input queue")
    while not in_q.empty():
        in_q.get()
    in_q.close()
