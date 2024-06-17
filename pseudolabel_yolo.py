import argparse
import glob
import json
import os
from torch.multiprocessing import Process, set_start_method
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

try:
    set_start_method("spawn")
except RuntimeError:
    pass

from functools import partial

import numpy as np
import torch

import cv2

from tqdm import tqdm

from folder_list import folder_list_imagenet, folder_list_all


class YoloPredictor:
    """
    End-to-end predictor for ViTDet given a cfg and checkpoint
    Create a simple end-to-end predictor with the given config and checkpoint that runs on
    single device for a single input image.


    Examples:
    ::
        pred = ViTDetPredictor(cfg, checkpoint, device)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg, checkpoint="yolov8n.pt", device="cuda"):
        self.cfg = cfg
        from ultralytics import YOLO

        # Load a model
        self.model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
        self.model.to(device)
        self.model.eval()
        self.input_format = "RGB"  # TODO: unhardcode this

        # self.aug = instantiate(self.cfg.dataloader.test.mapper.augmentations)[0]

        # self.input_format = self.cfg.dataloader.test.mapper.image_format
        # assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

            # Run batched infer nce on a list of images
            results = model(
                ["datasets/my_examples/img1.png", "datasets/my_examples/img2.png", "datasets/my_examples/img3.png"]
            )  # return a list of Results objects


def get_predictor(config_file, model_weights, device="cuda:0"):
    cfg = LazyConfig.load(config_file)
    predictor = ViTDetPredictor(cfg, model_weights, device=device)
    return predictor


# Modified from https://www.immersivelimit.com/create-coco-annotations-from-scratch
def create_sub_mask_polygon(sub_mask, height, width):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask.cpu().numpy(), 0.5, positive_orientation="low")

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col, row)

        # Make a polygon and simplify it
        try:
            poly = Polygon(contour)
        except:
            continue

        poly_simple = poly.simplify(1.0, preserve_topology=False)
        if isinstance(poly_simple, MultiPolygon):
            # On the rare case where this becomes a MultiPolygon, just use the non-simple one
            segmentation = np.array(poly.exterior.coords)
        else:
            segmentation = np.array(poly_simple.exterior.coords)

        if segmentation.ndim < 2:
            continue

        segmentation[:, 0] = segmentation[:, 0] / width
        segmentation[:, 1] = segmentation[:, 1] / height
        segmentation = segmentation.ravel().tolist()
        segmentations.append(segmentation)

    return segmentations


def create_object_bbox(bbox, height, width):
    # Format is (x1, y1, x2, y2)
    bbox[0::2] = bbox[0::2] / width
    bbox[1::2] = bbox[1::2] / height
    return bbox.cpu().tolist()


def annotate_image(inst, class_names, thresh=0.2, save_segmentation=True):
    inst = inst[inst.scores > thresh]
    num_instances, h, w = len(inst), inst._image_size[0], inst._image_size[1]

    annotations = {
        "num_instances": num_instances,
        "image_height": h,
        "image_width": w,
        "instances": [],
    }

    for bbox, score, cls, mask in zip(inst.pred_boxes, inst.scores, inst.pred_classes, inst.pred_masks):
        obj = {}
        obj["boxes"] = create_object_bbox(bbox, height=h, width=w)
        obj["score"] = score.item()
        obj["class_id"] = cls.item()
        obj["class_name"] = class_names[cls.item()]
        if save_segmentation:
            obj["segmentation"] = create_sub_mask_polygon(mask, height=h, width=w)
        annotations["instances"].append(obj)

    return annotations


def save_pseudo_detection(
    config_file, model_weights, classes_list, rgb_dir, save_dir, thresh=0.2, device="cuda:0", save_segmentation=True
):

    predictor = get_predictor(config_file, model_weights, device=device)

    if classes_list is None:
        classes_list = [name for name in os.listdir(rgb_dir) if os.path.isdir(os.path.join(rgb_dir, name))]

    for class_idx, class_id in enumerate(classes_list):
        print(f"{class_id}: {class_idx} / {len(classes_list)}")
        os.makedirs(os.path.join(save_dir, class_id), exist_ok=True)

        class_img_paths = list(glob.glob(os.path.join(rgb_dir, class_id, "*.JPEG")))
        class_img_paths.extend(list(glob.glob(os.path.join(rgb_dir, class_id, "*.jpg"))))
        class_img_paths.extend(list(glob.glob(os.path.join(rgb_dir, class_id, "*.png"))))

        for i, img_path in enumerate(tqdm(class_img_paths)):
            img_name = (
                img_path.split("/")[-1].replace(".JPEG", ".json").replace(".jpg", ".json").replace(".png", ".json")
            )
            save_path = os.path.join(save_dir, class_id, img_name)

            im = cv2.imread(img_path)
            outputs = predictor(im)
            annotations = annotate_image(
                outputs["instances"],
                predictor.metadata.thing_classes,
                thresh=thresh,
                save_segmentation=save_segmentation,
            )
            with open(save_path, "w") as f:
                json.dump(annotations, f)

        print(f"{class_id} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="/scratch/david/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py",
    )
    parser.add_argument(
        "--model_weights", type=str, default="/scratch/david/detectron2/checkpoints/coco_cascade_mask_rcnn_vitdet_h.pkl"
    )
    parser.add_argument("--rgb_dir", type=str, default="/datasets/imagenet_multitask/train/rgb")
    parser.add_argument("--save_dir", type=str, default="/datasets/imagenet_multitask/train/vitdet_coco")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_proc_per_gpu", type=int, default=1)
    parser.add_argument("--start_class", type=int, default=0)
    parser.add_argument("--end_class", type=int, default=None)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--folder_list", type=str, default="auto", choices=["imagenet", "all", "none", "auto"])
    parser.add_argument("--detection_threshold", type=float, default=0.2)
    parser.add_argument("--save_segmentation", action="store_true")
    parser.add_argument("--no_save_segmentation", action="store_false", dest="save_segmentation")
    parser.set_defaults(save_segmentation=True)
    args = parser.parse_args()

    if args.folder_list == "all":
        folder_list = folder_list_all

    elif args.folder_list == "imagenet":
        folder_list = folder_list_imagenet
    elif args.folder_list == "none":
        folder_list = None
    elif args.folder_list == "auto":
        folder_list = sorted([f.name for f in os.scandir(args.rgb_dir) if f.is_dir()])
    else:
        raise ValueError("Invalid folder list name")

    if folder_list is None:
        save_pseudo_detection(
            config_file=args.config_file,
            model_weights=args.model_weights,
            classes_list=None,
            rgb_dir=args.rgb_dir,
            save_dir=args.save_dir,
            device=f"cuda:{args.device_id}",
            thresh=args.detection_threshold,
            save_segmentation=args.save_segmentation,
        )

        exit(0)

    if args.end_class is None:
        args.end_class = len(folder_list)
    folder_list_trunc = folder_list[args.start_class : args.end_class]

    total_proc = args.n_gpus * args.n_proc_per_gpu
    assert len(folder_list_trunc) % total_proc == 0
    n_classes_per_proc = len(folder_list_trunc) // total_proc

    processes = []
    for gpu_idx in range(args.n_gpus):
        for proc_idx in range(args.n_proc_per_gpu):
            start_class = (gpu_idx * args.n_proc_per_gpu + proc_idx) * n_classes_per_proc
            end_class = start_class + n_classes_per_proc
            classes_list = folder_list_trunc[start_class:end_class]

            print(f"Starting {start_class + args.start_class} to {end_class + args.start_class}")

            func = Process(
                target=partial(
                    save_pseudo_detection,
                    config_file=args.config_file,
                    model_weights=args.model_weights,
                    classes_list=classes_list,
                    rgb_dir=args.rgb_dir,
                    save_dir=args.save_dir,
                    device=f"cuda:{gpu_idx + args.device_id}",
                    thresh=args.detection_threshold,
                    save_segmentation=args.save_segmentation,
                )
            )
            func.start()
            processes.append(func)

    for process in processes:
        process.join()
