"Main file where we aim to create and train the model"

import os

from ultralytics import YOLO
from datasets import get_yolo_data


def main() -> None:
    """
    This function executes the model selection and training for this project
    """

    # Last start using a pretrained YOLOv8n model
    # We choose YOLOv8n because it is small and I assume our system specs
    # are fairly limited
    model = YOLO("yolov8n.yaml")

    # Here we use MSCOCO dataset with person and motorcycle categories only
    # so we fetch the dataset first
    data_dir = os.path.join(os.getcwd(), "data")
    data_file = get_yolo_data(
        data_dir,
        classes=["person", "motorcycle"],
        sample_size=10,
        skip_creation=True,
    )

    # Here we train the model
    model.train(
        data=data_file,
        epochs=5,
        imgsz=640,
        batch=16,
        patience=5,
        save=True,
        project="Pretrained YOLOv8N two classes",
        name="test1",
        exist_ok=False,
        pretrained=True,
        optimizer="Adam",
        verbose=True,
        cos_lr=True,
        lr0=0.01,
        lrf=0.0001,
        dropout=0.2,
    )


if __name__ == "__main__":
    main()
