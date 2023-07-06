"""
input/output ops
"""

import json
import os


def load_file(file_path):
    """
    :param file_path: .json, path of file
    :return: dict/list, file
    """
    # load file
    file_path = os.path.join(file_path)
    print(f"Parsing {file_path}")
    with open(file_path, 'r') as f:
        file_json = json.load(f)
    return file_json
