"""
HTTP API server for SAM2 module.
Handles file uploads, processes tasks via Celery, and returns zipped results.
"""
from flask import Flask, request, send_file, jsonify, render_template
import os
import time
import json
from modules.common.server_utils import create_job_directory, save_uploaded_file, zip_directory
from modules.sam2.worker import celery_app
from modules.sam2.tasks import video_to_masks

app = Flask(__name__, template_folder='templates', static_folder='static')
PORT = 8001

@app.route('/')
def index():
    """Render the annotation UI."""
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload video and return job_id."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file"}), 400
    
    job_id, input_dir, output_dir = create_job_directory()
    video_file = request.files['video']
    video_path = os.path.join(input_dir, "video.mp4")
    save_uploaded_file(video_file, video_path)
    
    return jsonify({"job_id": job_id})

@app.route('/video/<job_id>')
def get_video(job_id):
    """Serve the uploaded video."""
    video_path = f"/data/jobs/{job_id}/input/video.mp4"
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    return f"Not found: {video_path}", 404

@app.route('/run_sam2', methods=['POST'])
def run_sam2():
    """Execute SAM2 with provided prompts."""
    job_id = request.form.get('job_id')
    prompts_json = request.form.get('prompts')
    
    if not job_id or not prompts_json:
        return jsonify({"error": "Missing job_id or prompts"}), 400
    
    input_dir = f"/data/jobs/{job_id}/input"
    output_dir = f"/data/jobs/{job_id}/output"
    
    video_path = os.path.join(input_dir, "video.mp4")
    prompts_path = os.path.join(input_dir, "prompts.json")
    
    with open(prompts_path, 'w') as f:
        f.write(prompts_json)
    
    # Prepare output paths
    masks_dir = os.path.join(output_dir, "masks")
    
    try:
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

