from pathlib import Path
import sys

def add_path(path):
    print('Initialising path to directory: ', src_path)
    if path not in sys.path:
        sys.path.insert(0, path)

file_path = Path(__file__)
repo_path = file_path.parents[1]
repo_name = file_path.parents[1].stem
src_path = repo_path / repo_name

add_path(str(src_path))