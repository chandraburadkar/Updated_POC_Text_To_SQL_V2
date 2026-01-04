import os

EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    ".git",
    ".idea",
    ".pytest_cache",
    ".DS_Store",
    ".ipynb_checkpoints",
    "pyvenv.cfg","static","share",
    ".venv_mcp"
}

EXCLUDE_FILES = {
    ".DS_Store"
}

def print_tree(root_dir, prefix=""):
    items = sorted(os.listdir(root_dir))
    items = [i for i in items if i not in EXCLUDE_DIRS and i not in EXCLUDE_FILES]

    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last = idx == len(items) - 1

        connector = "└── " if is_last else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(".")
    print("\nPROJECT STRUCTURE:\n")
    print(PROJECT_ROOT)
    print_tree(PROJECT_ROOT)