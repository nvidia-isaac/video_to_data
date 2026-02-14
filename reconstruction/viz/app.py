import argparse
import os
import h5py
import json
import numpy as np
import cv2
from flask import Flask, send_from_directory, jsonify, request, send_file
from PIL import Image
from modules.common.datatypes import DepthImage, CameraIntrinsics, Mask

app = Flask(__name__, static_folder='static')

# Global variable to store the job directory
JOB_DIR = ""

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/data/<path:filename>')
def serve_job_data(filename):
    """Serve files directly from the job directory (mesh, video, etc.)"""
    file_path = os.path.join(JOB_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {filename}"}), 404
    
    # Disable caching completely - remove ETag and Last-Modified headers
    response = send_file(file_path, conditional=False)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    # Remove ETag and Last-Modified to prevent 304 responses
    response.headers.pop('ETag', None)
    response.headers.pop('Last-Modified', None)
    return response

@app.route('/api/metadata')
def get_metadata():
    """Return basic info about the job results."""
    metadata = {
        "has_object": os.path.exists(os.path.join(JOB_DIR, "mesh_input.glb")),
        "has_human": os.path.exists(os.path.join(JOB_DIR, "smpl_results.h5")),
        "has_video": os.path.exists(os.path.join(JOB_DIR, "video.mp4")),
    }
    # Check for transform file (contains scale information)
    transform_path = os.path.join(JOB_DIR, "transform.json")
    if os.path.exists(transform_path):
        with open(transform_path, 'r') as f:
            transform = json.load(f)
            metadata["object_scale"] = transform.get("scale", [1.0, 1.0, 1.0])
    else:
        metadata["object_scale"] = [1.0, 1.0, 1.0]
    return jsonify(metadata)

@app.route('/api/human/params')
def get_human_params():
    """Get basic human parameters (metadata only, no vertex data)."""
    h5_path = os.path.join(JOB_DIR, "smpl_results.h5")
    if not os.path.exists(h5_path):
        return jsonify({"error": "No human data found"}), 404
    
    with h5py.File(h5_path, 'r') as f:
        # Only return metadata, not all vertex data
        data = {
            "gender": f['gender'][()].decode('utf-8') if isinstance(f['gender'][()], bytes) else f['gender'][()],
            "model_type": f['model_type'][()].decode('utf-8') if isinstance(f['model_type'][()], bytes) else f['model_type'][()],
            "num_frames": len(f['poses']) if 'poses' in f else 0,
            "has_vertices": 'vertices' in f,
            "has_faces": 'faces' in f
        }
        # Include faces if available (they're the same for all frames)
        if 'faces' in f:
            data['faces'] = f['faces'][:].tolist()
    return jsonify(data)

@app.route('/api/human/frame/<int:frame_idx>')
def get_human_frame(frame_idx):
    """Get human parameters for a specific frame (frame-by-frame loading)."""
    h5_path = os.path.join(JOB_DIR, "smpl_results.h5")
    if not os.path.exists(h5_path):
        return jsonify({"error": "No human data found"}), 404
    
    with h5py.File(h5_path, 'r') as f:
        num_frames = len(f['poses']) if 'poses' in f else 0
        if frame_idx < 0 or frame_idx >= num_frames:
            return jsonify({"error": f"Frame index {frame_idx} out of range [0, {num_frames-1}]"}), 404
        
        data = {
            "frame_idx": frame_idx,
            "pose": f['poses'][frame_idx].tolist(),
            "beta": f['betas'][frame_idx].tolist() if 'betas' in f else f['betas'][0].tolist(),  # Betas are usually constant
            "transl": f['transls'][frame_idx].tolist(),
        }
        # Include vertices for this frame if available
        if 'vertices' in f:
            data['vertices'] = f['vertices'][frame_idx].tolist()
    return jsonify(data)

@app.route('/api/object/poses')
def get_object_poses():
    """Load all pose JSON files from the poses directory."""
    poses_dir = os.path.join(JOB_DIR, "poses")
    if not os.path.exists(poses_dir):
        return jsonify({"error": "No object poses found"}), 404
    
    pose_files = sorted([f for f in os.listdir(poses_dir) if f.endswith('.json')])
    all_poses = []
    for pf in pose_files:
        with open(os.path.join(poses_dir, pf), 'r') as f:
            all_poses.append(json.load(f))
    
    return jsonify(all_poses)

@app.route('/api/objects')
def get_objects():
    """List all available object IDs from masks directory."""
    masks_dir = os.path.join(JOB_DIR, "masks")
    if not os.path.exists(masks_dir):
        return jsonify({"objects": []})
    
    objects = []
    for item in os.listdir(masks_dir):
        obj_path = os.path.join(masks_dir, item)
        if os.path.isdir(obj_path):
            try:
                obj_id = int(item)
                objects.append(obj_id)
            except ValueError:
                continue
    
    return jsonify({"objects": sorted(objects)})

@app.route('/api/frame/<int:frame_idx>/depth')
def get_depth_image(frame_idx):
    """Get depth image for a specific frame."""
    depth_path = os.path.join(JOB_DIR, "depth", f"{frame_idx:06d}.png")
    if not os.path.exists(depth_path):
        return jsonify({"error": "Depth image not found"}), 404
    
    return send_file(depth_path, mimetype='image/png')

@app.route('/api/frame/<int:frame_idx>/rgb')
def get_rgb_image(frame_idx):
    """Get RGB image for a specific frame by extracting from video."""
    video_path = os.path.join(JOB_DIR, "video.mp4")
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Frame not found"}), 404
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Save to BytesIO and send
    from io import BytesIO
    img_io = BytesIO()
    pil_image.save(img_io, format='PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/api/frame/<int:frame_idx>/mask/<int:obj_id>')
def get_mask_image(frame_idx, obj_id):
    """Get mask image for a specific frame and object."""
    mask_path = os.path.join(JOB_DIR, "masks", str(obj_id), f"{frame_idx:06d}.png")
    if not os.path.exists(mask_path):
        return jsonify({"error": "Mask image not found"}), 404
    
    return send_file(mask_path, mimetype='image/png')

@app.route('/api/frame/<int:frame_idx>/intrinsics')
def get_intrinsics(frame_idx):
    """Get camera intrinsics for a specific frame."""
    intrinsics_path = os.path.join(JOB_DIR, "intrinsics", f"{frame_idx:06d}.json")
    if not os.path.exists(intrinsics_path):
        # Try to get from first frame if available
        intrinsics_path = os.path.join(JOB_DIR, "intrinsics", "000000.json")
        if not os.path.exists(intrinsics_path):
            return jsonify({"error": "Intrinsics not found"}), 404
    
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    
    return jsonify(intrinsics)

@app.route('/api/frame/<int:frame_idx>/depth/raw')
def get_depth_raw(frame_idx):
    """Get depth image as raw float array (more precise than PNG)."""
    depth_path = os.path.join(JOB_DIR, "depth", f"{frame_idx:06d}.png")
    if not os.path.exists(depth_path):
        return jsonify({"error": "Depth image not found"}), 404
    
    try:
        depth_img = DepthImage.load(depth_path)
        # Convert to list - this can be slow for large images, so we'll skip it if it takes too long
        # For now, just return the numpy array shape info and let frontend use PNG
        # Actually, let's downsample or use a more efficient format
        depth_array = depth_img.depth
        
        # Downsample if too large to avoid JSON serialization issues
        if depth_array.size > 1000000:  # If more than 1M pixels
            # Downsample by factor of 2
            import cv2
            depth_downsampled = cv2.resize(depth_array, (depth_array.shape[1]//2, depth_array.shape[0]//2), interpolation=cv2.INTER_NEAREST)
            return jsonify({
                "depth": depth_downsampled.flatten().tolist(),
                "width": depth_downsampled.shape[1],
                "height": depth_downsampled.shape[0],
                "downsampled": True
            })
        else:
            return jsonify({
                "depth": depth_array.flatten().tolist(),
                "width": depth_img.width(),
                "height": depth_img.height(),
                "downsampled": False
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ephemeral 3D Viewer for Reconstruction Jobs")
    parser.add_argument("--dir", required=True, help="Absolute path to the job directory")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    args = parser.parse_args()

    JOB_DIR = os.path.abspath(args.dir)
    print(f"Starting viewer for job directory: {JOB_DIR}")
    
    if not os.path.exists(JOB_DIR):
        print(f"Error: Directory {JOB_DIR} does not exist.")
        exit(1)

    app.run(host='0.0.0.0', port=args.port, debug=False)

