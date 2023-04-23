"Here we prepare a yaml file to download datasets using fifityone API"

from pathlib import Path
from typing import Optional

import shutil
from requests.exceptions import ConnectionError as ReqConnectionError
import yaml
import fiftyone as fo
import fiftyone.zoo as foz

from utils import convert_coco_json


def get_yolo_data(
    directory: str,
    classes: Optional[list] = None,
    sample_size: int = 10000,
    skip_creation: bool = False,
) -> str:
    """
    This method allows us to download the MSCOCO dataset for specific classes
    We use the fiftyone API to download specific classes for traininig,
    validation and testing
    """
    if classes is None:
        raise ValueError(f"Expected a list of classes, got {classes}")

    # We create a yaml file to be used by the ultralytics model
    data_file = write_coco_yaml_ultralytics_file(directory, classes)
    if skip_creation:
        return data_file

    # Here we download the image and annotations for the MSCOCO dataset for the
    # classes provided. We may find some connection error due to multiple
    # requests to the same URL. To guarantee we're going to download the file,
    # we capture exception and wait until all images are downloaded.
    while True:
        try:
            foz.load_zoo_dataset(
                "coco-2017",
                splits=["train", "validation"],
                label_types=["detections", "segmentations"],
                classes=classes,
                max_samples=sample_size,
            )
            break
        except ReqConnectionError:
            continue

    # We collect the path of the COCO dataset downloaded by fiftyone API
    coco_dir = Path(fo.config.dataset_zoo_dir) / "coco-2017"
    data_dir = Path(directory) / "coco-2017"
    if (data_dir / "labels").exists():
        shutil.rmtree(data_dir / "labels")
    for dir_type in ["train", "validation"]:
        coco_cur_dir = coco_dir / dir_type
        convert_coco_json(coco_cur_dir, data_dir, classes, use_segments=False)
    return data_file


def write_coco_yaml_ultralytics_file(directory: str, classes: list) -> str:
    """
    We create the necessary yaml file for the ultralytics to read where
    MSCOCO files are and what classes we have.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    yaml_data = {
        "path": str(Path(directory) / "coco-2017"),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/validation",
        "names": dict(enumerate(classes)),
    }
    yaml_file_path = Path(directory).joinpath("coco_two_classes.yaml")
    # pylint: disable=unspecified-encoding
    with open(str(yaml_file_path), "w") as stream:
        yaml.dump(yaml_data, stream)
    return str(yaml_file_path)
