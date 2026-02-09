"""
Common utilities for HTTP API servers across all modules.
Handles job directory management, file operations, and result zipping.
"""
import os
import uuid
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Tuple


def create_job_directory(base_data_dir: str = "/data") -> Tuple[str, str, str]:
    """
    Create a unique job directory structure.
    
    Args:
        base_data_dir: Base data directory (default: /data)
    
    Returns:
        Tuple of (job_id, input_dir, output_dir)
    """
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(base_data_dir, "jobs", job_id)
    input_dir = os.path.join(job_dir, "input")
    output_dir = os.path.join(job_dir, "output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return job_id, input_dir, output_dir


def save_uploaded_file(file, destination_path: str) -> str:
    """
    Save an uploaded file to the destination path.
    
    Args:
        file: Flask file object from request.files
        destination_path: Full path where file should be saved
    
    Returns:
        Full path to saved file
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    file.save(destination_path)
    return destination_path


def zip_directory(directory_path: str, output_zip_path: str) -> str:
    """
    Create a zip file from a directory.
    
    Args:
        directory_path: Directory to zip
        output_zip_path: Path for output zip file
    
    Returns:
        Path to created zip file
    """
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)
    
    return output_zip_path


def cleanup_job_directory(job_dir: str, keep_output: bool = False):
    """
    Clean up job directory, optionally keeping output.
    
    Args:
        job_dir: Full path to job directory
        keep_output: If True, only remove input directory
    """
    if not os.path.exists(job_dir):
        return
    
    if keep_output:
        input_dir = os.path.join(job_dir, "input")
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
    else:
        shutil.rmtree(job_dir)

