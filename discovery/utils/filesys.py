from typing import Optional
import os

PROJECT_DIR_NAME = "discovery"


def find_root_directory(path: str) -> str:
    """Finds the subpath to the root dir of the project in `path`."""
    where = path.find(PROJECT_DIR_NAME)
    if where == -1:
        raise ValueError(f"Cannot find the root directory of the project in '{path}'.")
    return path[: where + len(PROJECT_DIR_NAME)]


def set_directory_in_project(
    rel_path: Optional[str] = None, create_dirs: bool = False
) -> str:
    """Changes the working dir to rel_path from the root of the project."""
    path = find_root_directory(os.getcwd())
    if rel_path is not None:
        path = os.path.join(path, rel_path)
    if create_dirs:
        os.makedirs(path, exist_ok=True)
    os.chdir(path)
    new_dir = os.getcwd()
    print("Changed working directory to", new_dir)
    return new_dir
