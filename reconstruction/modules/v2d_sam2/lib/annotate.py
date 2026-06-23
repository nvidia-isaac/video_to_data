# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Web-based annotation tool for SAM2 prompts.
Allows interactive annotation of objects using points and boxes.
"""
import argparse
import os
import json
import cv2
from flask import Flask, send_from_directory, jsonify, request, send_file
from v2d.common.datatypes import Point, BoundingBox
from v2d.sam2.lib.datatypes import Sam2Prompt, Sam2Prompts

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(SCRIPT_DIR, 'static'))

VIDEO_PATH = ""
PROMPTS_PATH = ""
PROMPTS = Sam2Prompts(prompts=[])

def load_prompts():
    """Load prompts from file if it exists."""
    global PROMPTS
    if os.path.exists(PROMPTS_PATH):
        with open(PROMPTS_PATH, 'r') as f:
            PROMPTS = Sam2Prompts.from_dict(json.load(f))
    else:
        PROMPTS = Sam2Prompts(prompts=[])

def save_prompts():
    """Save prompts to file."""
    os.makedirs(os.path.dirname(PROMPTS_PATH), exist_ok=True)
    with open(PROMPTS_PATH, 'w') as f:
        json.dump(PROMPTS.to_dict(), f, indent=2)

@app.route('/')
def index():
    return send_from_directory('static', 'annotate.html')

@app.route('/api/video/info')
def get_video_info():
    """Get video information (frame count, fps, etc.)"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 500

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return jsonify({
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height
    })

@app.route('/api/frame/<int:frame_idx>')
def get_frame(frame_idx):
    """Get a specific video frame as JPEG."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 500

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Could not read frame"}), 404

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    from PIL import Image
    import io
    img = Image.fromarray(frame_rgb)
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG', quality=95)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

@app.route('/api/prompts')
def get_prompts():
    """Get all prompts."""
    return jsonify(PROMPTS.to_dict())

@app.route('/api/prompts', methods=['POST'])
def add_prompt():
    """Add a new prompt."""
    data = request.json

    prompt = Sam2Prompt(
        frame_index=data['frame_index'],
        object_id=data['object_id'],
        points=[Point(x=p['x'], y=p['y']) for p in data.get('points', [])] if data.get('points') else None,
        point_labels=data.get('point_labels'),
        box=BoundingBox(**data['box']) if data.get('box') else None
    )

    PROMPTS.prompts.append(prompt)
    save_prompts()

    return jsonify({"success": True, "prompt": prompt.to_dict()})

@app.route('/api/prompts/<int:prompt_idx>', methods=['DELETE'])
def delete_prompt(prompt_idx):
    """Delete a prompt by index."""
    if 0 <= prompt_idx < len(PROMPTS.prompts):
        PROMPTS.prompts.pop(prompt_idx)
        save_prompts()
        return jsonify({"success": True})
    return jsonify({"error": "Invalid prompt index"}), 404

@app.route('/api/prompts/clear', methods=['POST'])
def clear_prompts():
    """Clear all prompts."""
    PROMPTS.prompts = []
    save_prompts()
    return jsonify({"success": True})

@app.route('/api/prompts/frame/<int:frame_idx>')
def get_prompts_for_frame(frame_idx):
    """Get all prompts for a specific frame."""
    frame_prompts = [p for p in PROMPTS.prompts if p.frame_index == frame_idx]
    return jsonify({
        "prompts": [p.to_dict() for p in frame_prompts]
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Web-based SAM2 annotation tool")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to save prompts JSON")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")

    args = parser.parse_args()

    VIDEO_PATH = args.video_path
    PROMPTS_PATH = args.prompts_path

    load_prompts()

    print(f"Starting annotation server on http://{args.host}:{args.port}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Prompts will be saved to: {PROMPTS_PATH}")

    app.run(host=args.host, port=args.port, debug=False)
