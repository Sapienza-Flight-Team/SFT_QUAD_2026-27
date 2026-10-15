import cv2
import numpy as np
from sft_math.cam.utils import get_rpy
import pandas as pd
from typing import List
from numpy import typing as np_typing
from common.metadata_QOL import select_csv
from common.metadata_QOL import teensy_fixer

if __name__ == "__main__":

    # descrittori ideali:
    desc = ["N", "E", "S", "W", "NE", "SE", "SW", "NW"]
    coos = {
        "N": (1, 0),
        "E": (0, 1),
        "S": (-1, 0),
        "W": (0, -1),
        "NE": (1, 1),
        "SE": (-1, 1),
        "SW": (-1, -1),
        "NW": (1, -1),
    }
    keypts = [coos[direction] for direction in desc]
    IDs = {direction: i for i, direction in enumerate(desc)}

    # inizio il test
    _input = input(
        "Welcome! In this test, the camera attitude will be extracted for each image,\n"
        "The software uses arUco to find 4 cardinal points in the image plus 4 auxiliary ones\n"
        "You only need to tell me if the experiment is dont with the sheet in horizontal or  vertical position:\n"
        "1) horizontal\n"
        "2) vertical\n"
        "Enter the chosen number: "
    )
    angle = ["yaw", "roll"][int(_input) - 1]
    # creo i marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # salvo i marker
    """path = "/home/stefano/Desktop/esperimento"
    for direction in desc:
        # creo l'immagine
        img = np.zeros((200, 200), dtype=np.uint8)
        # disegno il marker
        img = aruco.generateImageMarker(aruco_dict, IDs[direction], 200, img, 1)
        # salvo l'immagine
        cv2.imwrite(os.path.join(path, f"{direction}_{IDs[direction]}.png"), img)
    raise ValueError("Fatto")"""

    ############################################################################################
    default = ""
    # i want to read only the first 200 rows
    metadata = pd.read_csv(teensy_fixer(select_csv("data", default), 10))[:100]
    # le path delle immagini da esaminare sono nel campo filename
    filenames = metadata["filename"].values
    # elenco assetti
    positions = []
    # per ogni immagine
    for img_path in filenames:

        # apro l'immagine
        img = cv2.cvtColor(np.load(img_path), cv2.COLOR_BayerBG2BGR)
        # se l'immagine non esiste rimuovo la riga dal metadata e continuo
        if img is None:
            metadata = metadata[metadata["filename"] != img_path]
            print(f"{img_path} not found, skipping...")
            continue

        # individuo i marker
        detectorParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)
        corners, ids, rejectedImgPoints = detector.detectMarkers(img)

        if ids is None:
            metadata = metadata[metadata["filename"] != img_path]
            print(f"{img_path}: found no markers at all, skipping...")
            continue

        # rimuovo i marker sconosciuti
        corners = [corners[i] for i in range(len(ids)) if ids[i] in IDs.values()]
        ids = np.array([ids[i] for i in range(len(ids)) if ids[i] in IDs.values()])

        # creo il vettore dei descrittori e dei keypoints
        img_desc: List[str] = []
        list_img_kpts: List[np_typing.NDArray[np.float32]] = []
        for i in range(len(corners)):
            for j in range(len(corners[i])):
                img_desc.append(desc[int(ids[i].item())])
                list_img_kpts.append(
                    np.array(np.mean(corners[i][j], axis=0)).astype(np.float32)
                )  # media delle 4 coordinate
        img_kpts = np.array(list_img_kpts).astype(np.float32)

        # centro le coordinate in 0,0
        optical_center = np.array([img.shape[1] / 2, img.shape[0] / 2]).astype(
            np.float32
        )
        # centro i marker
        img_kpts = (img_kpts - optical_center).astype(np.float32)
        # converto le coordinate pixel in coordinate NE se la foto è correttamente allineata
        # se aumento la x vado verso S
        # se aumento la y vado verso E
        img_kpts = np.array([-img_kpts[:, 1], img_kpts[:, 0]]).T

        # creo i match dall'immagine a quella ideale
        matches = []
        for i in range(len(img_desc)):
            matches.append((tuple(img_kpts[i].tolist()), coos[img_desc[i]]))

        # se il numero di match è minore di 4 non posso fare nulla
        if len(matches) < 4:
            """# mostro su schermo l'immagine con i marker trovati
            tmp_img = cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
            # l'immagine è molto grande voglio una finestra più piccola
            tmp_img = cv2.resize(tmp_img, (0, 0), fx=0.2, fy=0.2)
            cv2.imshow("image with marker", tmp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
            # scarto l'immagine dai metadata
            metadata = metadata[metadata["filename"] != img_path]
            # report dell'eliminazione
            print(f"{img_path}: not enough matches found, skipping... ({len(matches)})")
            continue

        # individuo l'omografia in base ai match
        H = cv2.findHomography(
            np.array([match[0] for match in matches]),
            np.array([match[1] for match in matches]),
            method=cv2.RANSAC,
            ransacReprojThreshold=10,  # soglia di errore: sotto questa soglia il match è considerato buono, altrimenti è scartato
            # 10 è un valore molto alto, il che va bene poiché confidiamo che i match sino affidabili
            confidence=0.99,
        )[0]
        if H is None:
            # scarto l'immagine dai metadata
            metadata = metadata[metadata["filename"] != img_path]
            # report dell'eliminazione
            print(f"{img_path}: homography not found, skipping...")
            continue
        # inverto H per ottenere la trasformazione con dominio l'immagine ideale
        H = np.linalg.inv(H)
        # controllo che H sia una isometria
        N1 = np.linalg.norm(H[:, 0])
        N2 = np.linalg.norm(H[:, 1])
        if np.abs(np.log(N1 / N2)) > 0.15:
            """# mostro su schermo l'immagine con i marker trovati
            tmp_img = cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
            # l'immagine è molto grande voglio una finestra più piccola
            tmp_img = cv2.resize(tmp_img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("image with marker", tmp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
            # scarto l'immagine dai metadata
            metadata = metadata[metadata["filename"] != img_path]
            # report dell'eliminazione
            print(f"{img_path}: norms not found, skipping... ({N1}, {N2})")
            continue
        H[:, 0] /= N1
        H[:, 1] /= N2
        c = H[:, 0] @ H[:, 1]
        if np.abs(c).item() > 0.1:
            """# mostro su schermo l'immagine con i marker trovati
            tmp_img = cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
            # l'immagine è molto grande voglio una finestra più piccola
            tmp_img = cv2.resize(tmp_img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("image with marker", tmp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
            # scarto l'immagine dai metadata
            metadata = metadata[metadata["filename"] != img_path]
            # report dell'eliminazione
            print(f"{img_path}: wrong angle, skipping... ({c})")
            continue
        H[:, 2] = np.cross(H[:, 0], H[:, 1])
        # ortogonalizzo la matrice
        H[:, 0] = np.cross(H[:, 1], H[:, 2])
        H[:, 0] /= np.linalg.norm(H[:, 0])
        H[:, 1] /= np.linalg.norm(H[:, 1])
        H[:, 2] /= np.linalg.norm(H[:, 2])
        # H è l'isometria (a meno di traslazione) che giustifica la foto
        # calcolo roll, pitch, angle_photo da H.T (così da avere l'assetto dal punto di vista del foglio)
        r, p, y = get_rpy(H.T)
        positions.append((img_path, r, p, y))
    # recupero i dati roll dal metadata
    angle_ahrs = metadata[angle].values
    # mostro angle_ahrs in gradi
    angle_ahrs = np.rad2deg(np.array(angle_ahrs))

    # ricavo i teensy_timestamp
    teensy_timestamp = metadata["teensy_timestamp"].values.astype(np.float32) / 100000
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    angle_ahrs = angle_ahrs.astype(np.float32)
    angle_photo = np.array([position[3] for position in positions], dtype=np.float32)
    # creo il grafico degli angoli rispetto a teensy_timestamp
    plt.figure(figsize=(10, 6))
    start_idx = 0
    for i in range(1, len(angle_ahrs)):
        if (
            teensy_timestamp[i] - teensy_timestamp[i - 1] > 10.0
        ):  # se il gap tra i tempi è maggiore di un certo numero, allora è un' "interruzione"
            # Disegniamo la linea solo nel segmento continuo
            plt.plot(
                teensy_timestamp[start_idx:i],
                angle_ahrs[start_idx:i],
                color="blue",
            )
            plt.plot(
                teensy_timestamp[start_idx:i],
                angle_photo[start_idx:i],
                color="red",
            )
            start_idx = i  # aggiornare il punto di inizio per il prossimo segmento
    # va stampato l'ultimo segmento
    plt.plot(
        teensy_timestamp[start_idx:], angle_ahrs[start_idx:], color="blue", label="ahrs"
    )
    plt.plot(
        teensy_timestamp[start_idx:],
        angle_photo[start_idx:],
        color="red",
        label="photo",
    )
    plt.title("Comparison of angles")
    plt.grid()
    plt.legend()
    # mostro il grafico
    plt.show()

    print(f"Number of data shown: {len(positions)}")
    print(f"Number of data discarded: {len(filenames) - len(positions)}")

    # plot the biases
    diffs = angle_ahrs[1:] - angle_ahrs[:-1]
    diffs = (diffs + 180) % 360 - 180
    angle_ahrs[1:] = angle_ahrs[0] + np.cumsum(diffs)
    diffs = angle_photo[1:] - angle_photo[:-1]
    diffs = (diffs + 180) % 360 - 180
    angle_photo[1:] = angle_photo[0] + np.cumsum(diffs)
    biases = angle_ahrs - angle_photo
    print(f"mean bias: {np.mean(biases)}")
    print(f"bias variance: {np.var(biases)}")

    print("End")
