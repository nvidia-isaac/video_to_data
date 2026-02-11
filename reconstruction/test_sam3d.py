from celery import Celery
import time

# Initialize Celery with the same Redis URL used in docker-compose
# Note: Use 'localhost' if running from the host machine
app = Celery('sam3d', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Define the task signature by name (must match the name in tasks.py)
task_name = 'sam3d.image_to_mesh'

# Arguments matching the sam3d-image-to-mesh-test service in docker-compose.yaml
# Note: These paths are relative to the INSIDE of the worker container
kwargs = {
    "image_path": "/workspace/modules/sam3d/tests/test_data/test_image_2.jpg",
    "mask_path": "/workspace/modules/sam3d/tests/test_data/test_mask_2.png",
    "mesh_path": "/workspace/modules/sam3d/tests/test_data/output/test_mesh.glb",
    "transform_path": "/workspace/modules/sam3d/tests/test_data/output/test_intrinsics.json",
    "intrinsics_path": "/workspace/modules/sam3d/tests/test_data/output/test_transform.json"
}


print(f"Sending task {task_name}...")
# Send the task to the 'sam3d.image_to_mesh' queue as configured in common/celery_utils.py
result = app.send_task(task_name, kwargs=kwargs, queue='sam3d.image_to_mesh')

print(f"Task ID: {result.id}")
print("Waiting for result...")

try:
    # Wait for completion (timeout 1 hour as per API)
    response = result.get(timeout=3600)
    print("Task completed successfully!")
    print(f"Result: {response}")
except Exception as e:
    print(f"Task failed: {e}")