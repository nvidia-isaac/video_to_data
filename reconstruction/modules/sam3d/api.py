"""
HTTP API server for SAM3D module.
Handles file uploads, processes tasks via Celery, and returns zipped results.
"""
from flask import Flask, request, send_file, jsonify
import os
from modules.common.server_utils import create_job_directory, save_uploaded_file, zip_directory
from modules.sam3d.worker import celery_app
from modules.sam3d.tasks import image_to_mesh, render_debug_image

app = Flask(__name__)
PORT = 8004

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "module": "sam3d"}), 200

@app.route('/process/image_to_mesh', methods=['POST'])
def process_image_to_mesh():
    """Process image to mesh endpoint."""
    if 'image' not in request.files or 'mask' not in request.files:
        return jsonify({"error": "Missing required files: 'image' and 'mask'"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded files
        image_file = request.files['image']
        mask_file = request.files['mask']
        
        image_path = os.path.join(input_dir, "image.jpg")
        mask_path = os.path.join(input_dir, "mask.png")
        
        save_uploaded_file(image_file, image_path)
        save_uploaded_file(mask_file, mask_path)
        
        # Prepare output paths
        mesh_path = os.path.join(output_dir, "mesh.glb")
        transform_path = os.path.join(output_dir, "transform.json")
        intrinsics_path = os.path.join(output_dir, "intrinsics.json")
        
        # Get optional parameters
        seed = int(request.form.get('seed')) if request.form.get('seed') else None
        stage1_only = request.form.get('stage1_only', 'false').lower() == 'true'
        with_mesh_postprocess = request.form.get('with_mesh_postprocess', 'false').lower() == 'true'
        with_texture_baking = request.form.get('with_texture_baking', 'false').lower() == 'true'
        with_layout_postprocess = request.form.get('with_layout_postprocess', 'false').lower() == 'true'
        use_vertex_color = request.form.get('use_vertex_color', 'true').lower() == 'true'
        stage1_inference_steps = int(request.form.get('stage1_inference_steps')) if request.form.get('stage1_inference_steps') else None
        
        # Submit Celery task
        task = image_to_mesh.delay(
            image_path, mask_path, mesh_path, transform_path, intrinsics_path,
            seed=seed,
            stage1_only=stage1_only,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=with_texture_baking,
            with_layout_postprocess=with_layout_postprocess,
            use_vertex_color=use_vertex_color,
            stage1_inference_steps=stage1_inference_steps
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
            download_name=f"sam3d_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process/render_debug', methods=['POST'])
def process_render_debug():
    """Process render debug image endpoint."""
    if 'image' not in request.files or 'mesh' not in request.files or 'transform' not in request.files or 'intrinsics' not in request.files:
        return jsonify({"error": "Missing required files: 'image', 'mesh', 'transform', 'intrinsics'"}), 400
    
    try:
        # Create job directory
        job_id, input_dir, output_dir = create_job_directory()
        
        # Save uploaded files
        image_file = request.files['image']
        mesh_file = request.files['mesh']
        transform_file = request.files['transform']
        intrinsics_file = request.files['intrinsics']
        
        image_path = os.path.join(input_dir, "image.jpg")
        mesh_path = os.path.join(input_dir, "mesh.glb")
        transform_path = os.path.join(input_dir, "transform.json")
        intrinsics_path = os.path.join(input_dir, "intrinsics.json")
        
        save_uploaded_file(image_file, image_path)
        save_uploaded_file(mesh_file, mesh_path)
        save_uploaded_file(transform_file, transform_path)
        save_uploaded_file(intrinsics_file, intrinsics_path)
        
        # Prepare output path
        output_image_path = os.path.join(output_dir, "debug.jpg")
        num_vertices_to_use = int(request.form.get('num_vertices_to_use', 5000))
        
        # Submit Celery task
        task = render_debug_image.delay(
            image_path, mesh_path, transform_path, intrinsics_path,
            output_image_path, num_vertices_to_use
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
            download_name=f"sam3d_debug_results_{job_id}.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)

