import subprocess
import os

IMAGE_NAME = "v2d_moge"

def run_shell() -> None:
    subprocess.run([
        "docker",
        "run",
        "-it",
        "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        IMAGE_NAME, "bash"], check=True)

if __name__ == "__main__":
    run_shell()
