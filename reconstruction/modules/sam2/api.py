"""
HTTP API server for SAM2 module.
Handles file uploads, processes tasks via Celery, and returns zipped results.
"""
from flask import Flask, request, send_file, jsonify
import os
import time
from modules.common.server_utils import create_job_directory, save_uploaded_file, zip_directory
from modules.sam2.worker import celery_app
from modules.sam2.tasks import video_to_masks

app = Flask(__name__)
PORT = 8001

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "module": "sam2"}), 200

@app.route('/process/video_to_masks', methods=['POST'])
def process_video_to_masks():
    """Process video to masks endpoint."""
    if 'video' not in request.files or 'prompts' not in request.files:
        return jsonify({"error": "Missing required files: 'video' and 'prompts'"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded files
        video_file = request.files['video']
        prompts_file = request.files['prompts']
        
        video_path = os.path.join(input_dir, "video.mp4")
        prompts_path = os.path.join(input_dir, "prompts.json")
        
        save_uploaded_file(video_file, video_path)
        save_uploaded_file(prompts_file, prompts_path)
        
        # Prepare output paths
        masks_dir = os.path.join(output_dir, "masks")
        
        # Submit Celery task
        task = video_to_masks.delay(video_path, prompts_path, masks_dir)
        
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
            download_name=f"sam2_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)

