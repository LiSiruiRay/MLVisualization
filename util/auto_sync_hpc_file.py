# Author: ray
# Date: 3/29/24
# Description:
import subprocess

from dotenv import load_dotenv
import os

from util.common import get_proje_root_path

# Load the environment variables from .env file
load_dotenv()

# Read the USERNAME environment variable
hpc_user_name = os.getenv('HPC_USERNAME')


def sync_hpc_files(source_path: str = "/scratch/sl9625/FEDformer_meta_script/scripts/",
                   dest_path: str = "testing_resync"):
    # /scratch/sl9625/FEDformer_meta_script/scripts/
    # Define the rsync command
    rsync_command = [
        "rsync",
        "-avz",
        "--partial",
        "--progress",  # TODO: change here
        "-e", "ssh -i ~/.ssh/id_ed25519",  # Specify the path to your SSH private key
        f"{hpc_user_name}:{source_path}",
        # Replace with your HPC username, hostname, and source directory
        f"{dest_path}"
        # Replace with your local destination directory
    ]
    print(f"command: {rsync_command}")
    # Execute the rsync command
    result = subprocess.run(rsync_command, capture_output=True, text=True)

    # Check if the rsync command was successful
    if result.returncode == 0:
        print("Synchronization successful!")
        print(result.stdout)
    else:
        print("An error occurred during synchronization.")
        print(result.stderr)


if __name__ == "__main__":
    proje_root = get_proje_root_path()
    dest_path = os.path.join(proje_root, "hpc_sync_files")
    source_folder = "/scratch/sl9625/FEDformer_meta_script"
    to_sync_folders = ["checkpoints", "meta_info", "results", "test_results"]
    source_paths = [os.path.join(source_folder, to_sync_folder) for to_sync_folder in to_sync_folders]
    [sync_hpc_files(source_path=source_path, dest_path=dest_path) for source_path in source_paths]
