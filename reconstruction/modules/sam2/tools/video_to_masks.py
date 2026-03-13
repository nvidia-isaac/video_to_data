"""
SAM2 video to masks processing function.
Can be called directly from command line or imported as a function.
"""
from modules.sam2._impl.sam2_utils import build_sam2_video_predictor_low_mem
from modules.common.datatypes import Sam2Prompts
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
        data_dir = os.environ.get("DATA_DIR", "/data")
        checkpoint_dir = os.environ.get("CHECKPOINT_DIR", os.path.join(data_dir, "sam2/checkpoints/sam2.1-hiera-large"))
        
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
    
    # Create output directories for each object
    os.makedirs(masks_dir, exist_ok=True)
    obj_dirs = {}  # Track which object directories we've created
    
    # Run propagation in both directions (reverse then forward)
    # Write masks immediately as they're generated to avoid keeping them in memory
    for reverse in [True, False]:
        for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state, reverse=reverse):
            for i, obj_id in enumerate(object_ids):
                # Convert mask to numpy and write immediately
                mask_data = (masks[i, 0] > 0.0).cpu().numpy().astype(bool)
                
                # Create object directory if needed
                if obj_id not in obj_dirs:
                    obj_mask_dir = os.path.join(masks_dir, str(obj_id))
                    os.makedirs(obj_mask_dir, exist_ok=True)
                    obj_dirs[obj_id] = obj_mask_dir
                else:
                    obj_mask_dir = obj_dirs[obj_id]
                
                # Write mask to file immediately (overwrites if frame was processed before)
                mask_img = Image.fromarray((mask_data * 255).astype('uint8'), mode='L')
                mask_path = os.path.join(obj_mask_dir, f"{frame_idx:06d}.png")
                mask_img.save(mask_path, format='PNG')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to masks using SAM2")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--masks_dir", type=str, required=True, help="Output directory for masks")
    
    args = parser.parse_args()
    video_to_masks(args.video_path, args.prompts_path, args.masks_dir)

