import os


def get_project_root(target_folder="breast_cancer_diagnosis_project"):
    """
    Finds the project root folder by name, starting from the current working directory
    and going up the folder hierarchy.

    Args:
        target_folder (str): Name of the project root folder.

    Returns:
        str: Absolute path to the project root.

    Raises:
        FileNotFoundError: If the project root folder is not found.
    """
    current = os.getcwd()
    while True:
        if os.path.basename(current) == target_folder:
            return current
        new_path = os.path.dirname(current)
        if new_path == current:
            raise FileNotFoundError(f"Folder '{target_folder}' not found")
        current = new_path


# Usage example
project_root = get_project_root()
os.chdir(project_root)
print("Project root:", project_root)
