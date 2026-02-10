import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
# Patch numpy for chumpy compatibility
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'complex'): np.complex = complex
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'unicode'): np.unicode = str
if not hasattr(np, 'str'): np.str = str

import os
import json
from flask import Flask, request, send_file, jsonify
from modules.common.server_utils import create_job_directory, save_uploaded_file, zip_directory
from modules.nlf.tasks import video_to_smpl_task

app = Flask(__name__)
PORT = 8005

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "module": "nlf"}), 200

@app.route('/process/video_to_smpl', methods=['POST'])
def process_video_to_smpl():
    if 'video' not in request.files or 'masks' not in request.files:
        return jsonify({"error": "Missing required files: 'video' and 'masks' (zip)"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded files
        video_file = request.files['video']
        masks_zip = request.files['masks']
        
        video_path = os.path.join(input_dir, "video.mp4")
        masks_zip_path = os.path.join(input_dir, "masks.zip")
        masks_dir = os.path.join(input_dir, "masks")
        intrinsics_path = os.path.join(input_dir, "intrinsics.json")
        
        save_uploaded_file(video_file, video_path)
        save_uploaded_file(masks_zip, masks_zip_path)
        
        # Unzip masks
        import zipfile
        with zipfile.ZipFile(masks_zip_path, 'r') as zip_ref:
            zip_ref.extractall(masks_dir)
        
        # Get parameters
        gender = request.form.get('gender', 'neutral')
        model_type = request.form.get('model_type', 'smplh')
        render_debug = request.form.get('render_debug', 'false').lower() == 'true'
        
        intrinsics_json = request.form.get('intrinsics')
        if not intrinsics_json:
            return jsonify({"error": "Missing 'intrinsics' JSON"}), 400
        
        with open(intrinsics_path, 'w') as f:
            f.write(intrinsics_json)
        
        # Prepare output path
        output_h5_path = os.path.join(output_dir, "smpl_params.h5")
        debug_render_dir = os.path.join(output_dir, "debug_render") if render_debug else None
        
        # Submit Celery task
        task = video_to_smpl_task.delay(
            video_path=video_path,
            masks_dir=masks_dir,
            intrinsics_path=intrinsics_path,
            gender=gender,
            model_type=model_type,
            output_path=output_h5_path,
            render_debug=render_debug,
            debug_dir=debug_render_dir
        )
        
        # Wait for task completion (timeout 1 hour)
        result = task.get(timeout=3600)
        
        # Create zip file of results
        zip_path = os.path.join(output_dir, "results.zip")
        zip_directory(output_dir, zip_path)
        
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"nlf_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)

