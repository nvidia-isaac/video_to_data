import subprocess
import os

IMAGE_NAME = "v2d_unidepth"

current_dir = os.path.dirname(os.path.abspath(__file__))

module_dir = os.path.join(current_dir, "..")
root_dir = os.path.join(module_dir, "..")
lib_dir = os.path.join(module_dir, "lib")

dockerfile_path = os.path.join(current_dir, "Dockerfile")

print(os.path.abspath(module_dir))
print(os.path.abspath(lib_dir))
print(os.path.abspath(dockerfile_path))
print(os.path.abspath(root_dir))

def build_docker_image() -> None:
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, "-f", dockerfile_path, root_dir], check=True)

if __name__ == "__main__":
    build_docker_image()