import cv2
import os
import glob
import argparse
import json
import re


def draw_yolo_boxes(image_path, label_path):
    """Legge un'immagine e il suo file YOLO, restituisce l'immagine con le bounding box."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    img_height, img_width, _ = image.shape

    # Controlla se esiste il file delle annotazioni
    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center = float(parts[1]), float(parts[2])
                norm_width, norm_height = float(parts[3]), float(parts[4])

                # Conversione da coordinate YOLO a pixel assoluti
                abs_width = norm_width * img_width
                abs_height = norm_height * img_height
                abs_x_center = x_center * img_width
                abs_y_center = y_center * img_height

                x_min = int(abs_x_center - (abs_width / 2))
                y_min = int(abs_y_center - (abs_height / 2))
                x_max = int(abs_x_center + (abs_width / 2))
                y_max = int(abs_y_center + (abs_height / 2))

                # Disegna il rettangolo e l'etichetta
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"Class {class_id}",
                    (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
    return image


def draw_json_boxes(image_path, bbox_path):
    """Legge un'immagine e il suo file JSON, restituisce l'immagine con le bounding box."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    img_height, img_width, _ = image.shape

    if not os.path.exists(bbox_path):
        return image

    with open(bbox_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    bbox_format = data.get("bbox_format", "xyxy")
    if bbox_format != "xyxy":
        raise ValueError(f"Formato bbox non supportato in {bbox_path}: {bbox_format}")

    for box_data in data.get("boxes", []):
        bbox = box_data.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x_min, y_min, x_max, y_max = [int(round(float(coord))) for coord in bbox]
        x_min = max(0, min(img_width - 1, x_min))
        y_min = max(0, min(img_height - 1, y_min))
        x_max = max(0, min(img_width - 1, x_max))
        y_max = max(0, min(img_height - 1, y_max))

        if x_max <= x_min or y_max <= y_min:
            continue

        class_id = box_data.get("class", "?")
        confidence = box_data.get("conf")
        label = f"Class {class_id}"
        if confidence is not None:
            label += f" {float(confidence):.2f}"

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        label_y = max(20, y_min - 8)
        cv2.putText(
            image,
            label,
            (x_min, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    return image


def image_paths_from_dir(images_dir):
    image_paths = []
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(images_dir, extension)))
    return sorted(image_paths, key=natural_sort_key)


def natural_sort_key(path):
    filename = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", filename)
    ]


def visualizer(images_dir, labels_dir, label_format, display_scale):

    # Cerca le immagini nella cartella 'images_dir' invece che in una cartella unica
    image_paths = image_paths_from_dir(images_dir)

    if not image_paths:
        print(f"Errore: Nessuna immagine trovata in '{images_dir}'.")
        return

    idx = 0
    total_images = len(image_paths)

    print("\n=== COMANDI DEL VISUALIZZATORE ===")
    print("Premi 'd' -> Immagine SUCCESSIVA")
    print("Premi 'a' -> Immagine PRECEDENTE")
    print("Premi 'q' o 'ESC' -> ESCI dal programma")
    print("==================================\n")

    while True:
        img_path = image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Disegna le bounding box
        if label_format == "json":
            label_path = os.path.join(labels_dir, base_name + ".bboxes.json")
            img_to_show = draw_json_boxes(img_path, label_path)
        else:
            label_path = os.path.join(labels_dir, base_name + ".txt")
            img_to_show = draw_yolo_boxes(img_path, label_path)

        if img_to_show is not None:
            # Aggiungi un contatore testuale in alto a sinistra (es. "3/100 - image03.jpg")
            info_text = f"{idx + 1}/{total_images} - {os.path.basename(img_path)}"
            cv2.putText(
                img_to_show,
                info_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            # Mostra la finestra
            cv2.imshow(
                "Bounding Box Visualizer",
                cv2.resize(img_to_show, (0, 0), fx=display_scale, fy=display_scale),
            )

        # Attende la pressione di un tasto
        key = cv2.waitKey(0) & 0xFF

        if key == ord("d"):
            # Va avanti (e ricomincia da zero se arriva alla fine)
            idx = (idx + 1) % total_images
        elif key == ord("a"):
            # Torna indietro (se è alla prima, va all'ultima)
            idx = (idx - 1) % total_images
        elif key == ord("q") or key == 27:  # 27 è il codice del tasto ESC
            # Esce dal programma
            break

    # Chiude tutte le finestre prima di terminare
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualizzatore per immagini e bounding box separate"
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        help="Percorso della cartella principale YOLO (che contiene 'images' e 'labels')",
    )
    parser.add_argument(
        "--images-dir",
        default="data/outs_converted",
        help="Cartella immagini da visualizzare",
    )
    parser.add_argument(
        "--labels-dir",
        default="data/outs",
        help="Cartella annotazioni",
    )
    parser.add_argument(
        "--format",
        choices=("json", "yolo"),
        default="json",
        help="Formato annotazioni: JSON .bboxes.json o YOLO .txt",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Scala di visualizzazione OpenCV",
    )
    args = parser.parse_args()

    if args.dataset_path and args.format == "yolo":
        cartella_immagini = os.path.join(args.dataset_path, "images")
        cartella_labels = os.path.join(args.dataset_path, "labels")
    else:
        cartella_immagini = args.images_dir
        cartella_labels = args.labels_dir

    # Avvia il visualizzatore
    visualizer(cartella_immagini, cartella_labels, args.format, args.scale)
