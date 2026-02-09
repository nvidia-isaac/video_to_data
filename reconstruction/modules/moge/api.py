"""
HTTP API server for MoGe module.
Handles file uploads, processes tasks via Celery, and returns zipped results.
"""
from flask import Flask, request, send_file, jsonify
import os
from modules.common.server_utils import create_job_directory, save_uploaded_file, zip_directory
from modules.moge.worker import celery_app
from modules.moge.tasks import image_to_depth, video_to_depth

app = Flask(__name__)
PORT = 8002

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "module": "moge"}), 200

@app.route('/process/image_to_depth', methods=['POST'])
def process_image_to_depth():
    """Process image to depth endpoint."""
    if 'image' not in request.files:
        return jsonify({"error": "Missing required file: 'image'"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded file
        image_file = request.files['image']
        image_path = os.path.join(input_dir, "image.jpg")
        save_uploaded_file(image_file, image_path)
        
        # Prepare output paths
        depth_path = os.path.join(output_dir, "depth.png")
        intrinsics_path = os.path.join(output_dir, "intrinsics.json")
        
        # Submit Celery task
        task = image_to_depth.delay(image_path, depth_path, intrinsics_path)
        
        # Wait for task completion
        result = task.get(timeout=3600)  # 1 hour timeout
        
        # Create zip file
        zip_path = os.path.join(output_dir, "results.zip")
        zip_directory(output_dir, zip_path)
        
        # Return zip file
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"moge_image_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process/video_to_depth', methods=['POST'])
def process_video_to_depth():
    """Process video to depth endpoint."""
    if 'video' not in request.files:
        return jsonify({"error": "Missing required file: 'video'"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded file
        video_file = request.files['video']
        video_path = os.path.join(input_dir, "video.mp4")
        save_uploaded_file(video_file, video_path)
        
        # Get optional batch_size parameter
        batch_size = int(request.form.get('batch_size', 8))
        
        # Prepare output paths
        depth_folder = os.path.join(output_dir, "depth")
        intrinsics_folder = os.path.join(output_dir, "intrinsics")
        
        # Submit Celery task
        task = video_to_depth.delay(video_path, depth_folder, intrinsics_folder, batch_size)
        
        # Wait for task completion
        result = task.get(timeout=3600)  # 1 hour timeout
        
        # Create zip file
        zip_path = os.path.join(output_dir, "results.zip")
        zip_directory(output_dir, zip_path)
        
        # Return zip file
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"moge_video_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)

