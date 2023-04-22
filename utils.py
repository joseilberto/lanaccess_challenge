"Utils for training and prediction"

import json
import shutil

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np


COCO_CLASSES = [
    "0",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "12",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "26",
    "backpack",
    "umbrella",
    "29",
    "30",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "45",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "66",
    "dining table",
    "68",
    "69",
    "toilet",
    "71",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "83",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def convert_coco_json(
    coco_json_dir: str,
    output_dir: str,
    use_segments: bool = False,
    cls91to80: bool = False,
) -> None:
    """
    Convert MSCoco json label format to txt labels (a label file per image).
    This method was extracted from:https://github.com/ultralytics/JSON2YOLO
    """
    # Get the output directory and the image directory
    save_dir = make_dirs(output_dir)
    source_images_dir = Path(coco_json_dir) / "data"
    coco80 = coco91_to_coco80_class()
    # Import json
    for json_file in sorted(Path(coco_json_dir).resolve().glob("*.json")):
        folder_name = Path(save_dir) / "labels" / json_file.parent.stem
        folder_name.mkdir(exist_ok=True)

        images_folder_name = Path(save_dir) / "images" / json_file.parent.stem
        images_folder_name.mkdir(exist_ok=True)

        # pylint: disable=unspecified-encoding
        with open(json_file) as stream:
            data = json.load(stream)

        # Create image dict
        images = {f"{image['id']}": image for image in data["images"]}

        # Create image-annotations dict
        image_to_annotations = defaultdict(list)
        for ann in data["annotations"]:
            image_to_annotations[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(
            image_to_annotations.items(), desc=f"Annotations {json_file}"
        ):
            img = images[f"{img_id}"]
            height, width, file_name = (
                img["height"],
                img["width"],
                img["file_name"],
            )
            shutil.copy(
                source_images_dir / file_name, images_folder_name / file_name
            )
            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= width  # normalize x
                box[[1, 3]] /= height  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = (
                    coco80[ann["category_id"] - 1]
                    if cls91to80
                    else ann["category_id"] - 1
                )  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        segment = merge_multi_segment(ann["segmentation"])
                        segment = (
                            (
                                np.concatenate(segment, axis=0)
                                / np.array([width, height])
                            )
                            .reshape(-1)
                            .tolist()
                        )
                    else:
                        segment = [
                            j for i in ann["segmentation"] for j in i
                        ]  # all segments concatenated
                        segment = (
                            (
                                np.array(segment).reshape(-1, 2)
                                / np.array([width, height])
                            )
                            .reshape(-1)
                            .tolist()
                        )
                    segment = [cls] + segment
                    if segment not in segments:
                        segments.append(segment)

            # Write labels file
            with open(
                (folder_name / file_name).with_suffix(".txt"), "a"
            ) as file:
                for idx, bbox in enumerate(bboxes):
                    # class, box or segments
                    line = segments[idx] if use_segments else bbox
                    if len(line) > 0:
                        line = " ".join([str(elem) for elem in line])
                        file.write(f"{line}\n")


def make_dirs(directory: str) -> Path:
    """
    Create directory structure for images
    This method was extracted from:https://github.com/ultralytics/JSON2YOLO
    """
    directory = Path(directory)
    for path in directory, directory / "labels", directory / "images":
        path.mkdir(parents=True, exist_ok=True)  # make dir
    return directory


def merge_multi_segment(segments: list) -> list:
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all
    segments into one.
    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.

    This method was extracted from:https://github.com/ultralytics/JSON2YOLO
    """
    segment_list = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    segment_list.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    segment_list.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    segment_list.append(segments[i][nidx:])
    return segment_list


def min_index(arr1: np.ndarray, arr2: np.ndarray) -> tuple:
    """Find a pair of indexes with the shortest distance.
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).

    This method was extracted from:https://github.com/ultralytics/JSON2YOLO
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def coco91_to_coco80_class() -> list:  #
    """
    Converts 80-index (val2014) to 91-index (paper)

    This method was extracted from:https://github.com/ultralytics/JSON2YOLO
    """
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]
