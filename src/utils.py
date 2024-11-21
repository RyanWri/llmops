import os
from datetime import datetime


def generate_run_id():
    # Get the current date and time
    now = datetime.now()
    # Format the datetime as YYYYMMDD_HHMM
    run_id = now.strftime("%Y%m%d_%H%M")
    return run_id


def create_run_directory(logs_dir, run_id):
    """
    Create a directory in logs_dir with the specified run_id.

    Args:
        logs_dir (str): The base directory where logs are stored.
        run_id (str): The unique identifier for this run.

    Returns:
        str: The full path of the created run directory.
    """
    # Build the path for the run directory
    run_directory = os.path.join(logs_dir, run_id)

    # Create the directory if it doesn't exist
    os.makedirs(run_directory, exist_ok=True)

    return run_directory
