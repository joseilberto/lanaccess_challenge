"Here we prepare a yaml file to download datasets using the ultralytics API"

from pathlib import Path
from typing import Optional

import os
import ultralytics
import yaml


def get_yolo_data(
    directory: str, classes: Optional[list] = None, sample: bool = False
) -> str:
    """
    This method allows us to download the yolo dataset using either MSCOCO
    or MSCOCO128 for testing purposes. We use the base yaml files for both
    coco.yaml and coco128.yaml and modify them accordingly to reduce the number
    of classes and saving the dataset in the current directory.
    """
    # We define the root file we would like to use
    coco_type = "coco128" if sample else "coco"
    coco_file = f"{coco_type}.yaml"

    # If no classes are passed, we use the default coco dataset with 80 classes
    if classes is None:
        return coco_file

    # Otherwise, we open the default files in the ultralytics repo and change
    # the names object to contain only the classes we need
    data_path = Path(directory)
    data_path.mkdir(parents=True, exist_ok=True)
    coco_output_file = data_path.joinpath(coco_file)

    coco_yaml = (
        Path(ultralytics.__path__[0]).joinpath("datasets").joinpath(coco_file)
    )
    if coco_yaml.is_file():
        # pylint: disable=unspecified-encoding
        with open(str(coco_yaml), "r") as stream:
            yaml_data = yaml.safe_load(stream)

        # Modifying path related variables
        yaml_data["path"] = str(data_path)
        yaml_data["train"] = os.path.join(coco_type, yaml_data["train"])
        yaml_data["val"] = os.path.join(coco_type, yaml_data["val"])

        # Validating the classes passed by the user
        validate_coco_classes(classes, yaml_data["names"])

        # Setting the classes in the names entry of yaml_data
        set_new_names(classes, yaml_data)

        # We save the file and return the string
        with open(str(coco_output_file), "w") as stream:
            yaml.dump(
                yaml_data, stream, default_flow_style=False, allow_unicode=True
            )
        if coco_output_file.is_file():
            return str(coco_output_file)
        raise ValueError(f"File {str(coco_output_file)} was not saved")
    raise ValueError(f"File not found: {str(coco_yaml)}")


def set_new_names(classes: list, yaml_data: dict) -> None:
    """Set the list of classes into the 'names' key of the yaml_data"""
    cls_count = 0
    new_names = {}
    for cls_name in classes:
        new_names[int(cls_count)] = cls_name
        cls_count += 1
    yaml_data["names"] = new_names


def validate_coco_classes(classes: list, names: dict) -> None:
    """
    Check if all the new classes are supported by COCO dataset. Otherwise
    raises an exception.
    """
    yaml_classes = list(names.values())
    not_present = [
        cls_name for cls_name in classes if cls_name not in yaml_classes
    ]
    if not_present:
        raise ValueError(
            f"The classes {not_present} are not present in the COCO dataset"
        )
