import argparse
import os
import numpy as np
from PIL import Image
from multiprocessing import Pool
import cv2
import pandas as pd


def convert_bayer_npy_to_rgb_png(file_path, output_folder, keep_old=True):
    try:
        # Load the .npy file
        data = np.load(file_path)

        # Ensure the data is in the correct format
        if data.ndim == 2:  # Bayer
            data = cv2.cvtColor(data, cv2.COLOR_BayerBG2RGB)
            image = Image.fromarray(data)
        elif data.ndim == 3 and data.shape[2] in [3, 4]:  # RGB or RGBA image
            image = Image.fromarray(data)
        else:
            raise ValueError(f"Unsupported array shape: {data.shape}")

        # Create the output file path
        base_name = os.path.basename(file_path)
        output_file_path = os.path.join(
            output_folder, base_name.replace(".npy", ".png")
        )

        # Save the image as a .png file
        image.save(output_file_path)
        # print(f"Converted {file_path} to {output_file_path}")

        if not keep_old:
            os.remove(file_path)
            # print(f"Deleted {file_path}")

    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")


def convert_rgb_png_to_bayer_npy(file_path, output_folder, keep_old=True):

    try:
        data = Image.open(file_path)

        if data.mode == "RGBA":
            data = data.convert("RGB")

        if data.mode != "RGB":
            raise ValueError(
                f"Unsupported PNG mode: {data.mode}, expected RGB or RGBA."
            )

        w, h = data.size
        data = np.asarray(data)

        # Initialize the BayerRG8 array
        bayer_rg = np.empty((h, w), dtype=np.uint8)

        # Separate R, G, B channels
        R = data[:, :, 0]
        G = data[:, :, 1]
        B = data[:, :, 2]

        # R channel - Even index row, Even index column (0::2, 0::2)
        bayer_rg[0::2, 0::2] = R[0::2, 0::2]

        # G channel - Even index row, Odd index column (0::2, 1::2), Odd index row, Even index column (1::2, 0::2)
        bayer_rg[0::2, 1::2] = G[0::2, 1::2]
        bayer_rg[1::2, 0::2] = G[1::2, 0::2]

        # B channel - Odd index row, Odd index column (1::2, 1::2)
        bayer_rg[1::2, 1::2] = B[1::2, 1::2]

        # Resulting image will be:
        # R G
        # G B

        # Create the output file path
        base_name = os.path.basename(file_path)
        output_file_path = os.path.join(
            output_folder, base_name.replace(".png", ".npy")
        )

        # Save the numpy array to a .npy file.
        np.save(output_file_path, bayer_rg)
        # print(f"Converted {file_path} to {output_file_path}")

        if not keep_old:
            os.remove(file_path)
            # print(f"Deleted {file_path}")

    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")


def convert_image_file(file_path, output_folder, keep_old=True):
    if file_path.endswith(".npy"):
        convert_bayer_npy_to_rgb_png(file_path, output_folder, keep_old)
    elif file_path.endswith(".png"):
        convert_rgb_png_to_bayer_npy(file_path, output_folder, keep_old)
    else:
        print(f"Skipping unsupported file: {file_path}")


def convert_image_files(file_paths, output_folder, keep_old=True, workers=None):
    os.makedirs(output_folder, exist_ok=True)

    files = [
        file_path
        for file_path in file_paths
        if file_path.endswith(".npy") or file_path.endswith(".png")
    ]
    if not files:
        print("No .npy or .png files to convert.")
        return

    with Pool(processes=workers) as pool:
        pool.starmap(
            convert_image_file,
            [(file_path, output_folder, keep_old) for file_path in files],
        )

    print("Conversion completed.")


def convert_dataset_from_metadata(
    metadata_file, input_folder, output_folder, keep_old=True
):

    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        print(f"Output folder {output_folder} does not exist. Creating it.")
        os.makedirs(output_folder)

    # load metadata file
    metadata = pd.read_csv(metadata_file)
    # i want to save a copy of the metadata file in the output folder but i want to update the filenames to point to the new files
    # create a copy of the metadata
    metadata_copy = metadata.copy()
    _from, _to = (
        (".npy", ".png")
        if metadata_copy["filename"].values[0].endswith(".npy")
        else (".png", ".npy")
    )
    metadata_copy["filename"] = (
        metadata_copy["filename"]
        .str.replace(f"{_from}$", _to, regex=True)
        .replace(f"{input_folder}/", f"{output_folder}/", regex=True)
    )
    # save the copy in the output folder
    metadata_copy.to_csv(
        os.path.join(output_folder, os.path.basename(metadata_file)), index=False
    )

    # Get a list of all file paths from the metadata["filename"]
    files = [filename for filename in metadata["filename"]]

    convert_image_files(files, output_folder, keep_old)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert .npy Bayer images to .png RGB images, or .png RGB images "
            "to .npy Bayer images."
        )
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Files to convert. Defaults to all .npy and .png files in data/outs.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default="data/outs_converted",
        help="Folder where converted files will be written.",
    )
    parser.add_argument(
        "--keep-old",
        action="store_true",
        help=(
            "Keep source files after conversion. By default source files are "
            "deleted."
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to the CPU count.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    files = args.files
    if not files:
        files = [
            os.path.join("data/outs", filename)
            for filename in os.listdir("data/outs")
            if filename.endswith(".npy") or filename.endswith(".png")
        ]

    convert_image_files(
        files,
        args.output_folder,
        keep_old=args.keep_old,
        workers=args.workers,
    )
