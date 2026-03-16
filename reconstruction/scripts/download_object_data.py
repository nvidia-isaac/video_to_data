import os
import subprocess
import argparse

def download_session_data(session_id: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(os.path.join(output_dir, session_id), exist_ok=True)
    cmd = [
        "wget",
        f"https://pdx.s8k.io/v1/AUTH_team-isaac/recordings/v2d/ego/{session_id}/{session_id}_color.mp4",
        "-O", os.path.join(output_dir, session_id, f"{session_id}_color.mp4"),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_list_txt_path", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    with open(args.session_list_txt_path, "r") as f:
        session_ids = f.readlines()
    session_ids = [session_id.strip() for session_id in session_ids]
    for session_id in session_ids:
        print(f"Downloading session data for {session_id}")
        download_session_data(session_id, args.output_dir)
        print(f"Session data for {session_id} downloaded")