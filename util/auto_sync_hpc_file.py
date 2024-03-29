# Author: ray
# Date: 3/29/24
# Description:
import subprocess

from dotenv import load_dotenv
import os

# Load the environment variables from .env file
load_dotenv()

# Read the USERNAME environment variable
hpc_user_name = os.getenv('HPC_USERNAME')


def sync_hpc_files():
    # Define the rsync command
    rsync_command = [
        "rsync",
        "-avz",
        "--partial",
        "--progress", # TODO: change here
        "-e", "ssh -i ~/.ssh/id_ed25519",  # Specify the path to your SSH private key
        f"{hpc_user_name}:/scratch/sl9625/",  # Replace with your HPC username, hostname, and source directory
        "/Users/ray/rayfile/self-project/research_ml_visualization"  # Replace with your local destination directory
    ]

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
    sync_hpc_files()
