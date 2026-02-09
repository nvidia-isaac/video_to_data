"""
SAM2 video to masks processing function.
Can be called directly from command line or imported as a function.
"""
from modules.sam2.sam2_utils import build_sam2_video_predictor_low_mem
from modules.sam2.datatypes import Sam2Prompts
import os
import sys
import argparse
import json
from PIL import Image

# Singleton predictor instance
_predictor = None

def _get_predictor():
    global _predictor
    if _predictor is None:
        checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
        if checkpoint_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
        
        config_file = os.environ.get("CONFIG_FILE", "configs/sam2.1/sam2.1_hiera_l.yaml")
        checkpoint_file = os.environ.get("CHECKPOINT_FILE", "sam2.1_hiera_large.pt")
        
        os.environ["HYDRA_CONFIG_SEARCH_PATH"] = checkpoint_dir
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_file)
        _predictor = build_sam2_video_predictor_low_mem(
            config_file=config_file,
            ckpt_path=ckpt_path,
        )
    return _predictor

def video_to_masks(video_path: str, prompts_path: str, masks_dir: str):
    """Process a video with SAM2 prompts and save masks to files."""
    # Load prompts from file
    with open(prompts_path, "r") as f:
        prompts = Sam2Prompts.from_dict(json.load(f))
    
    # Get predictor
    predictor = _get_predictor()
    
    # Initialize inference state
    inference_state = predictor.init_state(video_path)
    
    # Add prompts
    for prompt in prompts.prompts:
        box = [prompt.box.x0, prompt.box.y0, prompt.box.x1, prompt.box.y1] if prompt.box else None
        points = [[p.x, p.y] for p in prompt.points] if prompt.points else None
        point_labels = prompt.point_labels if prompt.point_labels else None
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt.frame_index,
            obj_id=prompt.object_id,
            points=points,
            labels=point_labels,
            box=box,
        )
    
    # Store masks in memory
    masks_dict = {}  # {object_id: {frame_idx: mask_array}}
    
    # Run propagation in both directions (reverse then forward)
    for reverse in [True, False]:
        for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state, reverse=reverse):
            for i, obj_id in enumerate(object_ids):
                mask_data = (masks[i, 0] > 0.0).cpu().numpy().astype(bool)
                
                if obj_id not in masks_dict:
                    masks_dict[obj_id] = {}
                masks_dict[obj_id][frame_idx] = mask_data
    
    # Save masks to files
    os.makedirs(masks_dir, exist_ok=True)
    for obj_id, frame_masks in masks_dict.items():
        obj_mask_dir = os.path.join(masks_dir, str(obj_id))
        os.makedirs(obj_mask_dir, exist_ok=True)
        for frame_idx, mask_array in frame_masks.items():
            mask_img = Image.fromarray((mask_array * 255).astype('uint8'), mode='L')
            mask_path = os.path.join(obj_mask_dir, f"{frame_idx:06d}.png")
            mask_img.save(mask_path, format='PNG')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to masks using SAM2")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--masks_dir", type=str, required=True, help="Output directory for masks")
    
    args = parser.parse_args()
    video_to_masks(args.video_path, args.prompts_path, args.masks_dir)

