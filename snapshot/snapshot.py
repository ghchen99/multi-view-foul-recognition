import os
import fnmatch
from pathlib import Path
from typing import List, Set, Tuple

DEFAULT_IGNORE_PATTERNS = {
    '.git/',
    '.git/**',
    '.DS_Store',
    '__pycache__/',
    '**/__pycache__/',
    '*.pyc',
    '.env',
    'node_modules/',
    '.idea/',
    '.vscode/',
    '.pytest_cache/',
    '*.swp',
    '.gitignore'
}

def parse_gitignore() -> Set[str]:
    """Parse .gitignore file and return a set of patterns."""
    ignore_patterns = DEFAULT_IGNORE_PATTERNS.copy()
    try:
        with open('.gitignore', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle directory patterns
                    if line.endswith('/'):
                        ignore_patterns.add(line)
                        ignore_patterns.add(f"{line}**")
                    # Handle wildcards
                    if '*' in line:
                        ignore_patterns.add(line)
                    else:
                        ignore_patterns.add(line)
                        ignore_patterns.add(f"**/{line}")
    except FileNotFoundError:
        pass
    return ignore_patterns

def should_ignore(path: str, ignore_patterns: Set[str]) -> bool:
    """
    Check if a path should be ignored based on patterns.
    Handles both file and directory patterns.
    """
    # Convert path to use forward slashes for consistency
    path = path.replace(os.sep, '/')
    
    # Always ignore hidden files and directories (starting with .)
    if os.path.basename(path).startswith('.'):
        return True

    for pattern in ignore_patterns:
        # Handle directory-specific patterns
        if pattern.endswith('/') and os.path.isdir(path):
            if fnmatch.fnmatch(f"{path}/", pattern):
                return True
        # Handle patterns with wildcards
        if '**' in pattern:
            # Replace ** with a reasonable pattern that matches any number of directories
            pattern_regex = pattern.replace('**', '*')
            if fnmatch.fnmatch(path, pattern_regex):
                return True
        # Direct pattern matching
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
        # Check if the path or any of its parent directories match the pattern
        path_parts = path.split('/')
        for i in range(len(path_parts)):
            subpath = '/'.join(path_parts[:i+1])
            if fnmatch.fnmatch(subpath, pattern):
                return True
    return False

def is_binary(file_path: str) -> bool:
    """Check if a file is binary by reading its first few thousand bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(4096)
            if not chunk:  # empty file
                return False
            
            textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
            return bool(chunk.translate(None, textchars))
    except (IOError, OSError):
        return True

def read_file_content(file_path: str) -> str:
    """Attempt to read file content with various encodings."""
    if is_binary(file_path):
        return "[Binary file]"

    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except (IOError, OSError):
            return "[Error: Unable to read file]"
    
    return "[Error: Unable to decode file with available encodings]"

def generate_tree(path: str = ".", level: int = 0, ignore_patterns: Set[str] = None) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Generate a tree structure of the directory and collect file contents.
    Returns a tuple of (tree_lines, file_contents).
    """
    if ignore_patterns is None:
        ignore_patterns = parse_gitignore()

    tree_lines = []
    file_contents = []
    
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        return [], []

    # Count items that aren't ignored for proper tree formatting
    valid_items = [item for item in items if not should_ignore(os.path.join(path, item), ignore_patterns)]
    last_idx = len(valid_items) - 1

    for idx, item in enumerate(items):
        full_path = os.path.join(path, item)
        rel_path = os.path.relpath(full_path)

        if should_ignore(rel_path, ignore_patterns):
            continue

        # Determine if this is the last item for proper tree formatting
        is_last = idx >= last_idx
        # Create tree line with proper formatting
        if level == 0:
            prefix = ""
        else:
            prefix = "└── " if is_last else "├── "
            prefix = "    " * (level - 1) + prefix

        tree_lines.append(f"{prefix}{item}")

        if os.path.isdir(full_path):
            # Recursively process directory
            subtree_lines, subtree_contents = generate_tree(full_path, level + 1, ignore_patterns)
            tree_lines.extend(subtree_lines)
            file_contents.extend(subtree_contents)
        else:
            # Read and store file content
            content = read_file_content(full_path)
            file_contents.append((rel_path, content))

    return tree_lines, file_contents

def create_snapshot(output_file: str = "snapshot.txt"):
    """Create a snapshot of the current directory and save it to a file."""
    tree_lines, file_contents = generate_tree()

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write directory tree
        f.write("Directory Structure:\n")
        f.write("===================\n\n")
        for line in tree_lines:
            f.write(line + "\n")
        
        # Write file contents
        f.write("\nFile Contents:\n")
        f.write("=============\n\n")
        for filepath, content in file_contents:
            f.write(f"--- {filepath} ---\n")
            f.write(content)
            f.write("\n\n")

if __name__ == "__main__":
    create_snapshot()
    print("Snapshot has been created in 'snapshot.txt'")