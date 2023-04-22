"Here we prepare a yaml file to download datasets using the ultralytics API"

from pathlib import Path
from typing import Optional


def get_yolo_data(
    directory: str, classes: Optional[list] = None, sample: bool = False
) -> str:
    """
    This method allows us to download the yolo dataset using either MSCOCO
    or MSCOCO128 for testing purposes. We use the base yaml files for both
    coco.yaml and coco128.yaml and modify them accordingly to reduce the number
    of classes and saving the dataset in the current directory.
    """
    # If no classes are passed, we use the default coco dataset with 80 classes
    if classes is None:
        if sample:
            return "coco128.yaml"
        return "coco.yaml"

    # Other, we open the default files in the ultralytics repo and change the
    # names object to contain only the classes we need
