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


def choose_top_level_folder(project_path):
    """
    Lists all top-level folders in the project root and allows the user to choose one.

    Args:
        project_path (str): Path to the project root.

    Returns:
        str: Path to the chosen folder.
    """
    folders = [
        f
        for f in os.listdir(project_path)
        if os.path.isdir(os.path.join(project_path, f))
    ]
    print("Top-level folders in the project:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    choice = int(input("Select a folder by number: ")) - 1
    if 0 <= choice < len(folders):
        return os.path.join(project_path, folders[choice])
    else:
        raise ValueError("Invalid selection")


# Usage example
project_root = get_project_root()
os.chdir(project_root)
print("Project root:", project_root)

selected_folder = choose_top_level_folder(project_root)
print("You selected:", selected_folder)
