# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path

import numpy as np
import torch


class Detector:
    def __init__(
        self,
        name="vitdet",
        device="cuda",
        model_size="b",
        test_score_thresh: float = 0.25,
        **kwargs,
    ):
        self.device = device

        if name == "vitdet":
            print(f"Loading ViTDet-{model_size.upper()} detector...")
            self.detector = load_detectron2_vitdet(
                model_size=model_size,
                test_score_thresh=test_score_thresh,
                **kwargs,
            )
            self.detector_func = run_detectron2_vitdet
            self.batch_func = run_detectron2_vitdet_batch

            self.detector = self.detector.to(self.device)
            self.detector.eval()
        else:
            raise NotImplementedError

    def run_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **kwargs)

    def run_detection_batch(self, images: list[np.ndarray], **kwargs) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.batch_func(self.detector, images, **kwargs)


VITDET_VARIANTS = {
    "b": {
        "config": "cascade_mask_rcnn_vitdet_b_100ep.py",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_b/f325358525/model_final_435fa9.pkl",
        "checkpoint_file": "model_final_435fa9.pkl",
    },
    "l": {
        "config": "cascade_mask_rcnn_vitdet_l_100ep.py",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_l/f328021305/model_final_1a9f28.pkl",
        "checkpoint_file": "model_final_1a9f28.pkl",
    },
    "h": {
        "config": "cascade_mask_rcnn_vitdet_h_75ep.py",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl",
        "checkpoint_file": "model_final_f05665.pkl",
    },
}


def load_detectron2_vitdet(
    path: str = "",
    model_size: str = "b",
    test_score_thresh: float = 0.25,
):
    """Load a ViTDet Cascade Mask R-CNN detector.

    Args:
        path: Directory containing the checkpoint file. If empty, downloads
              from the detectron2 model zoo.
        model_size: One of "b" (ViT-Base, fastest), "l" (ViT-Large),
                    "h" (ViT-Huge, most accurate).
        test_score_thresh: Internal NMS score threshold. Detections below this
            are discarded before they leave the model. Lower for ByteTrack
            (e.g. 0.1) to retain partial-occlusion detections.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate, LazyConfig

    if model_size not in VITDET_VARIANTS:
        raise ValueError(f"Unknown model_size '{model_size}', expected one of {list(VITDET_VARIANTS)}")
    variant = VITDET_VARIANTS[model_size]

    cfg_path = Path(__file__).parent / variant["config"]
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        variant["checkpoint_url"]
        if path == ""
        else os.path.join(path, variant["checkpoint_file"])
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = test_score_thresh
    detector = instantiate(detectron2_cfg.model)
    checkpointer = DetectionCheckpointer(detector)
    checkpointer.load(detectron2_cfg.train.init_checkpoint)

    detector.eval()
    return detector


def run_detectron2_vitdet(
    detector,
    img,
    det_cat_id: int = 0,  # 0: person
    bbox_thr: float = 0.5,
    default_to_full_image: bool = True,
):
    import detectron2.data.transforms as T

    height, width = img.shape[:2]

    IMAGE_SIZE = 1024
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
    img_transformed = transforms(T.AugInput(img)).apply_image(img)
    img_transformed = torch.as_tensor(
        img_transformed.astype("float32").transpose(2, 0, 1)
    )
    inputs = {"image": img_transformed, "height": height, "width": width}

    with torch.no_grad():
        det_out = detector([inputs])

    det_instances = det_out[0]["instances"]
    valid_idx = (det_instances.pred_classes == det_cat_id) & (
        det_instances.scores > bbox_thr
    )
    if valid_idx.sum() == 0 and default_to_full_image:
        boxes = np.array([0, 0, width, height]).reshape(1, 4)
        scores = np.array([1.0])
    else:
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        scores = det_instances.scores[valid_idx].cpu().numpy()

    sorted_indices = np.argsort(scores)[::-1]
    return boxes[sorted_indices].reshape(-1, 4), scores[sorted_indices]


def run_detectron2_vitdet_batch(
    detector,
    images: list[np.ndarray],
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    default_to_full_image: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run ViTDet detection on a batch of images in a single forward pass.

    Memory scales linearly with batch size (~2-3 GB per image for ViTDet-B
    at 1024px). Batch sizes of 4-8 are practical on a 24 GB GPU.
    """
    import detectron2.data.transforms as T

    IMAGE_SIZE = 768
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)

    inputs = []
    for img in images:
        height, width = img.shape[:2]
        img_transformed = transforms(T.AugInput(img)).apply_image(img)
        img_transformed = torch.as_tensor(
            img_transformed.astype("float32").transpose(2, 0, 1)
        )
        inputs.append({"image": img_transformed, "height": height, "width": width})

    with torch.no_grad():
        det_outputs = detector(inputs)

    results = []
    for img, det_out in zip(images, det_outputs):
        height, width = img.shape[:2]
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == det_cat_id) & (
            det_instances.scores > bbox_thr
        )
        if valid_idx.sum() == 0 and default_to_full_image:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            scores = np.array([1.0])
        else:
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            scores = det_instances.scores[valid_idx].cpu().numpy()

        sorted_indices = np.argsort(scores)[::-1]
        results.append((boxes[sorted_indices].reshape(-1, 4), scores[sorted_indices]))

    return results
