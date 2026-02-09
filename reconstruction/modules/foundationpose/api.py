"""
HTTP API server for FoundationPose module.
Handles file uploads, processes tasks via Celery, and returns zipped results.
"""
from flask import Flask, request, send_file, jsonify
import os
from modules.common.server_utils import create_job_directory, save_uploaded_file, zip_directory
from modules.foundationpose.worker import celery_app
from modules.foundationpose.tasks import video_to_poses, align_mesh_scale, transform_mesh, simplify_mesh

app = Flask(__name__)
PORT = 8005

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "module": "foundationpose"}), 200

@app.route('/process/video_to_poses', methods=['POST'])
def process_video_to_poses():
    """Process video to poses endpoint."""
    required_files = ['video', 'mesh']
    if not all(f in request.files for f in required_files):
        return jsonify({"error": f"Missing required files: {required_files}"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded files
        video_file = request.files['video']
        mesh_file = request.files['mesh']
        
        video_path = os.path.join(input_dir, "video.mp4")
        mesh_path = os.path.join(input_dir, "mesh.glb")
        
        save_uploaded_file(video_file, video_path)
        save_uploaded_file(mesh_file, mesh_path)
        
        # Optional files
        depth_folder = None
        masks_folder = None
        camera_intrinsics_path = None
        
        if 'depth_folder' in request.files:
            # Handle zip file for depth folder
            depth_zip = request.files['depth_folder']
            depth_folder = os.path.join(input_dir, "depth")
            os.makedirs(depth_folder, exist_ok=True)
            # For now, assume depth is provided as individual files or we'll handle it differently
            # This is a simplified version - you may need to extract zip files
        
        if 'masks_folder' in request.files:
            masks_zip = request.files['masks_folder']
            masks_folder = os.path.join(input_dir, "masks")
            os.makedirs(masks_folder, exist_ok=True)
        
        if 'camera_intrinsics' in request.files:
            intrinsics_file = request.files['camera_intrinsics']
            camera_intrinsics_path = os.path.join(input_dir, "intrinsics.json")
            save_uploaded_file(intrinsics_file, camera_intrinsics_path)
        
        # Prepare output paths
        poses_dir = os.path.join(output_dir, "poses")
        
        # Get optional parameters
        reference_frame = int(request.form.get('reference_frame', 0))
        target_width = int(request.form.get('target_width')) if request.form.get('target_width') else None
        target_height = int(request.form.get('target_height')) if request.form.get('target_height') else None
        
        # Submit Celery task
        task = video_to_poses.delay(
            video_path, depth_folder, masks_folder, camera_intrinsics_path,
            mesh_path, poses_dir, reference_frame, target_width, target_height
        )
        
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
            download_name=f"foundationpose_poses_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process/align_mesh', methods=['POST'])
def process_align_mesh():
    """Process align mesh scale endpoint."""
    required_files = ['mesh', 'depth', 'mask', 'intrinsics', 'transform']
    if not all(f in request.files for f in required_files):
        return jsonify({"error": f"Missing required files: {required_files}"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded files
        mesh_file = request.files['mesh']
        depth_file = request.files['depth']
        mask_file = request.files['mask']
        intrinsics_file = request.files['intrinsics']
        transform_file = request.files['transform']
        
        mesh_path = os.path.join(input_dir, "mesh.glb")
        depth_path = os.path.join(input_dir, "depth.png")
        mask_path = os.path.join(input_dir, "mask.png")
        intrinsics_path = os.path.join(input_dir, "intrinsics.json")
        transform_path = os.path.join(input_dir, "transform.json")
        
        save_uploaded_file(mesh_file, mesh_path)
        save_uploaded_file(depth_file, depth_path)
        save_uploaded_file(mask_file, mask_path)
        save_uploaded_file(intrinsics_file, intrinsics_path)
        save_uploaded_file(transform_file, transform_path)
        
        # Prepare output path
        output_transform_path = os.path.join(output_dir, "refined_transform.json")
        
        # Submit Celery task
        task = align_mesh_scale.delay(
            depth_path, mask_path, intrinsics_path, mesh_path,
            output_transform_path, transform_path
        )
        
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
            download_name=f"foundationpose_align_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process/transform_mesh', methods=['POST'])
def process_transform_mesh():
    """Process transform mesh endpoint."""
    required_files = ['input_mesh', 'transform']
    if not all(f in request.files for f in required_files):
        return jsonify({"error": f"Missing required files: {required_files}"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded files
        input_mesh_file = request.files['input_mesh']
        transform_file = request.files['transform']
        
        input_mesh_path = os.path.join(input_dir, "input_mesh.glb")
        transform_path = os.path.join(input_dir, "transform.json")
        
        save_uploaded_file(input_mesh_file, input_mesh_path)
        save_uploaded_file(transform_file, transform_path)
        
        # Prepare output path
        output_mesh_path = os.path.join(output_dir, "scaled_mesh.glb")
        
        # Submit Celery task
        task = transform_mesh.delay(input_mesh_path, output_mesh_path, transform_path)
        
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
            download_name=f"foundationpose_transform_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process/simplify_mesh', methods=['POST'])
def process_simplify_mesh():
    """Process simplify mesh endpoint."""
    if 'input_mesh' not in request.files:
        return jsonify({"error": "Missing required file: 'input_mesh'"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded file
        input_mesh_file = request.files['input_mesh']
        input_mesh_path = os.path.join(input_dir, "input_mesh.glb")
        save_uploaded_file(input_mesh_file, input_mesh_path)
        
        # Prepare output path
        output_mesh_path = os.path.join(output_dir, "simplified_mesh.glb")
        
        # Get optional parameters
        face_count = int(request.form.get('faces')) if request.form.get('faces') else None
        factor = float(request.form.get('factor')) if request.form.get('factor') else None
        
        # Submit Celery task
        task = simplify_mesh.delay(input_mesh_path, output_mesh_path, face_count, factor)
        
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
            download_name=f"foundationpose_simplify_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)

