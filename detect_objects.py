"""Script to use a pretrained model to detect objects in a video or image file"""

import argparse
import gc
import shutil
import os
from pathlib import Path
import torch

from ultralytics import YOLO


MODEL_PATH = os.path.join(
    os.getcwd(), "Pretrained YOLOv8N two classes", "model_10_epochs", "weights"
)

SUPPORTED_TYPES = [
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
    "webm",
]


def get_arguments() -> dict:
    """
    Get all the arguments to be used in this script. We then return a
    dictionary with all these arguments.
    """
    parser = argparse.ArgumentParser(
        description="List of arguments for the detect objects"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=False,
        help="String containing the path to the file",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="best",
        choices=["best", "last"],
        help=(
            "String specifying the type of model, it can be either best "
            "(default) or last"
        ),
    )
    arguments = parser.parse_args().__dict__
    path = Path(arguments["path"])
    if not path.is_file():
        raise ValueError(f"No file found: {str(path)}")
    check_path_types(path)
    return arguments


def check_path_types(path: Path) -> None:
    """
    Check if the parsed path is one of the supported video/image format
    for the ultralytics YOLO model.
    """
    file_extension = path.suffix.replace(".", "")
    if file_extension not in SUPPORTED_TYPES:
        raise ValueError(f"{file_extension} is not a supported file type")


def load_model(model_type: str = "best") -> YOLO:
    """Load model if file exists"""
    # We empty the GPU memory before loading the model
    gc.collect()
    torch.cuda.empty_cache()
    # We load the model
    model_path = Path(MODEL_PATH) / f"{model_type}.pt"
    if not model_path.is_file():
        raise ValueError(f"Model file not found: {str(model_path)}")
    return YOLO(str(model_path))


def detect_objects(file_path: str, model: YOLO) -> None:
    """
    Detect objects using the pretrained YOLO model from ultralytics.
    We also save the results in the same folder of the original image.
    """
    # We delete any previous default prediction folder
    default_dir = Path(os.getcwd()) / "runs" / "detect" / "predict"
    if default_dir.exists():
        shutil.rmtree(default_dir)

    # We predict using the pretrained model here
    model(
        source=file_path,
        conf=0.25,
        iou=0.25,
        save=True,
    )

    # We copy the file from the prediction folder to the folder where
    # the original image was saved initially.
    file_path = Path(file_path)
    file_name = file_path.name
    labelled_name = file_path.stem + "_labelled" + file_path.suffix
    shutil.copy(default_dir / file_name, file_path.parent / labelled_name)

    # We delete the runs folder containing the predict folder
    shutil.rmtree(Path(os.getcwd()) / "runs")


def main() -> None:
    """
    Main function where we execute all actions.
    """
    arguments = get_arguments()
    model = load_model(arguments["model_type"])
    detect_objects(arguments["path"], model)


if __name__ == "__main__":
    main()
