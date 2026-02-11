import os
import json
import uuid
from flask import Flask, request, send_file, jsonify, render_template
from modules.common.server_utils import create_job_directory, save_uploaded_file, zip_directory
from modules.orchestrator.tasks import orchestrate_reconstruction, celery_app

app = Flask(__name__, template_folder='templates', static_folder='static')
PORT = 8000

@app.route('/')
def index():
    """Render the unified pipeline UI."""
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

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    """Execute the full reconstruction pipeline."""
    job_id = request.form.get('job_id')
    prompts_json = request.form.get('prompts')
    
    if not job_id or not prompts_json:
        return jsonify({"error": "Missing job_id or prompts"}), 400
    
    input_dir = f"/data/jobs/{job_id}/input"
    output_dir = f"/data/jobs/{job_id}/output"
    
    prompts_path = os.path.join(input_dir, "prompts.json")
    with open(prompts_path, 'w') as f:
        f.write(prompts_json)
    
    try:
        # Submit the orchestration task
        task = orchestrate_reconstruction.delay(job_id)
        return jsonify({"status": "started", "task_id": task.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get the status of a pipeline task."""
    from celery.result import AsyncResult
    task = AsyncResult(task_id, app=celery_app)
    
    result = task.result
    if isinstance(result, Exception):
        result = str(result)
    
    response = {
        "status": task.status,
        "result": result if task.ready() else None,
        "info": task.info if not task.ready() else None
    }
    return jsonify(response)

@app.route('/download/<job_id>')
def download_results(job_id):
    """Download zipped results."""
    zip_path = f"/data/jobs/{job_id}/output/results.zip"
    if os.path.exists(zip_path):
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"reconstruction_results_{job_id}.zip"
        )
    return "Results not ready", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)

