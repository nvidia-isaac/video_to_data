# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Optional, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import NoCollate, prepare_batch
from sam_3d_body.utils import recursive_to
from torch.utils.data import default_collate
from torchvision.transforms import ToTensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()
        self.num_vertices = self.model.head_pose.keypoint_mapping.shape[1] - 127

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            # print("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            print("Found boxes:", boxes)
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            assert (
                bboxes is not None
            ), "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img, boxes)
        else:
            masks, masks_score = None, None

        #################### Construct batch data samples ####################
        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)

        #################### Run model inference on an image ####################
        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        if cam_int is not None:
            # print("Using provided camera intrinsics...")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image = batch["img_ori"][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()

        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        # out = recursive_to(out, "cpu")
        # out = recursive_to(out, "numpy")
        out = recursive_to(out, DEVICE)
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                    "mhr_model_params": out["mhr_model_params"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out

    @torch.no_grad()
    def process_batch(
        self,
        images: list,
        bboxes: Optional[list] = None,
        cam_ints: Optional[list] = None,
        inference_type: str = "body",
    ):
        """Run inference on a batch of B frames, each with a single person.

        Amortizes Python/PyTorch dispatcher overhead across B frames — this is
        the path that gets the wall-time win on dispatch-bound machines (e.g.
        cluster nodes with slower per-core CPUs).

        Args:
            images: list of B RGB numpy images. May have differing H,W; the
                top-down transform crops/resizes each to the model's input.
            bboxes: optional list of length B; each entry is a (4,) or (1, 4)
                xyxy bbox, or None (use full image).
            cam_ints: optional list of length B; each entry is a (3, 3) torch
                tensor or None (use default FOV from prepare_batch).
            inference_type: "body" only — full/hand variants would need their
                own batched implementations and aren't supported here.

        Returns:
            list[dict] of length B, one dict per frame, with the same keys as
            ``process_one_image`` (minus the lhand/rhand fields).
        """
        assert inference_type == "body", (
            "process_batch only supports inference_type='body'"
        )
        B = len(images)
        if B == 0:
            return []

        # Build per-frame transformed data dicts (one person per frame).
        data_list = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            if bboxes is not None and bboxes[i] is not None:
                box = np.asarray(bboxes[i]).reshape(-1, 4)[0]
            else:
                box = np.array([0, 0, w, h], dtype=np.float32)
            data_info = dict(
                img=img,
                bbox=box,
                bbox_format="xyxy",
                mask=np.zeros((h, w, 1), dtype=np.uint8),
                mask_score=np.array(0.0, dtype=np.float32),
            )
            data_list.append(self.transform(data_info))

        # Collate B per-frame dicts -> tensors with leading dim B.
        batch = default_collate(data_list)

        # Add num_person=1 dim so the model sees (B, 1, ...) — its
        # _initialize_batch dispatches on dim==5 to set _max_num_person.
        for key in [
            "img", "img_size", "ori_img_size", "bbox_center",
            "bbox_scale", "bbox", "affine_trans", "mask", "mask_score",
        ]:
            if key in batch:
                batch[key] = batch[key].unsqueeze(1).float()
        if "mask" in batch:
            batch["mask"] = batch["mask"].unsqueeze(2)
        batch["person_valid"] = torch.ones((B, 1))

        # Camera intrinsics: stack provided ones; fill defaults for any missing.
        ci_list = []
        for i, img in enumerate(images):
            ci = cam_ints[i] if cam_ints is not None else None
            if ci is not None:
                ci_list.append(torch.as_tensor(ci).reshape(3, 3).float())
            else:
                h, w = img.shape[:2]
                f = (h * h + w * w) ** 0.5
                ci_list.append(torch.tensor(
                    [[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]],
                    dtype=torch.float32,
                ))
        batch["cam_int"] = torch.stack(ci_list)

        # img_ori is a list of NoCollate-wrapped originals; not a tensor.
        img_ori = [NoCollate(img) for img in images]

        # Move tensor entries to GPU (skip the img_ori list entry).
        batch = recursive_to(batch, "cuda")
        batch["img_ori"] = img_ori
        self.model._initialize_batch(batch)
        batch["cam_int"] = batch["cam_int"].to(batch["img"])

        # Forward pass.
        pose_output = self.model.forward_step(batch, decoder_type="body")
        out = pose_output["mhr"]
        out = recursive_to(out, DEVICE)

        # Outputs are shape (B*P, ...) = (B, ...) since P=1.
        all_out = []
        for i in range(B):
            all_out.append({
                "bbox": batch["bbox"][i, 0].cpu().numpy(),
                "focal_length": out["focal_length"][i],
                "pred_keypoints_3d": out["pred_keypoints_3d"][i],
                "pred_keypoints_2d": out["pred_keypoints_2d"][i],
                "pred_vertices": out["pred_vertices"][i],
                "pred_cam_t": out["pred_cam_t"][i],
                "pred_pose_raw": out["pred_pose_raw"][i],
                "global_rot": out["global_rot"][i],
                "body_pose_params": out["body_pose"][i],
                "hand_pose_params": out["hand"][i],
                "scale_params": out["scale"][i],
                "shape_params": out["shape"][i],
                "expr_params": out["face"][i],
                "mask": None,
                "pred_joint_coords": out["pred_joint_coords"][i],
                "pred_global_rots": out["joint_global_rots"][i],
                "mhr_model_params": out["mhr_model_params"][i],
            })
        return all_out
