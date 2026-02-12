import argparse
import os
import h5py
import json
import numpy as np
from flask import Flask, send_from_directory, jsonify, request

app = Flask(__name__, static_folder='static')

# Global variable to store the job directory
JOB_DIR = ""

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/data/<path:filename>')
def serve_job_data(filename):
    """Serve files directly from the job directory (mesh, video, etc.)"""
    return send_from_directory(JOB_DIR, filename)

@app.route('/api/metadata')
def get_metadata():
    """Return basic info about the job results."""
    metadata = {
        "has_object": os.path.exists(os.path.join(JOB_DIR, "mesh_input.glb")),
        "has_human": os.path.exists(os.path.join(JOB_DIR, "smpl_results.h5")),
        "has_video": os.path.exists(os.path.join(JOB_DIR, "video.mp4")),
    }
    return jsonify(metadata)

@app.route('/api/human/params')
def get_human_params():
    """Parse the SMPL .h5 file and return parameters as JSON."""
    h5_path = os.path.join(JOB_DIR, "smpl_results.h5")
    if not os.path.exists(h5_path):
        return jsonify({"error": "No human data found"}), 404
    
    with h5py.File(h5_path, 'r') as f:
        # Convert numpy arrays to lists for JSON serialization
        data = {
            "poses": f['poses'][:].tolist(),
            "betas": f['betas'][:].tolist(),
            "transls": f['transls'][:].tolist(),
            "gender": f['gender'][()].decode('utf-8') if isinstance(f['gender'][()], bytes) else f['gender'][()],
            "model_type": f['model_type'][()].decode('utf-8') if isinstance(f['model_type'][()], bytes) else f['model_type'][()]
        }
        # Include vertices and faces if available
        if 'vertices' in f:
            data['vertices'] = f['vertices'][:].tolist()
        if 'faces' in f:
            data['faces'] = f['faces'][:].tolist()
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

