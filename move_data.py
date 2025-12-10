import os
import shutil
from pathlib import Path


def main():
    data_dir = Path("/mnt/project/data/")

    real_filepaths = []

    # ----------
    # DFC18
    # ----------
    num_real = 0

    train_list = data_dir / "real" / "DFC18" / "DFC18" / "train.txt"

    with open(train_list, "r") as f:
        image_list = [ln.strip() for ln in f if ln.strip()]

    for image_name in image_list:
        img_filepath = data_dir / "real/DFC18/DFC18" / "opt" / f"{image_name}.tif"
        real_filepaths.append(img_filepath)

    # ----------
    # DFC19
    # ----------
    d = data_dir / "real" / "DFC19" / "opt"
    filenames = os.listdir(d)

    for filename in filenames:
        img_filepath = d / filename
        real_filepaths.append(img_filepath)

    # ----------
    # GeoNRW
    # ----------
    d = data_dir / "real" / "geonrw" / "data"
    d_dirs = [d_dir for d_dir in os.listdir(d) if os.path.isdir(d / d_dir)]

    for d_dir in d_dirs:
        filenames = [f for f in os.listdir(d / d_dir) if f.endswith(".jp2")]
        for filename in filenames:
            img_filepath = d / d_dir / filename
            real_filepaths.append(img_filepath)

    # ----------
    # Copy to output dir
    # ----------
    output_dir = Path("/mnt/project/data/real/all/")
    for filepath in real_filepaths:
        filename = Path(filepath).name
        shutil.copy(filepath, output_dir / filename)


if __name__ == "__main__":
    main()