import os
from typing import List  # using List as a type


##
# @brief Get a list of all the subfolders in a given folder.
# @param folder_path The path of the folder
# @return List of subfolders in the given folder.
def get_subfolders(folder_path: str) -> List[str]:
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    return subfolders


def make_path(folder, *argv):

    for item in argv:
        folder = os.path.join(folder, item)

    return folder


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def file_exists(path):
    return os.path.exists(path)
