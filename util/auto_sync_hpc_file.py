# Author: ray
# Date: 3/29/24
# Description:
import subprocess


def sync_hpc_files():
    # Define the rsync command
    rsync_command = [
        "rsync",
        "-avz",
        "--partial",
        "--progress",
        "-e", "ssh -i /path/to/your/private/key",  # Specify the path to your SSH private key
        "user@hostname:/scratch/user/directory/",  # Replace with your HPC username, hostname, and source directory
        "/local/destination/directory/"  # Replace with your local destination directory
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
