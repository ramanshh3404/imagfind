import os
from PIL import Image
from typing import Dict, Tuple, Any

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def get_image_info(filepath: str) -> Tuple[int, int, int]:
    """Returns (width, height, size_bytes) of the image.
    
    If the image cannot be read, width and height default to 0.
    """
    size_bytes = os.path.getsize(filepath)
    try:
        with Image.open(filepath) as img:
            width, height = img.size
    except Exception:
        width, height = 0, 0
    return width, height, size_bytes

def scan_directory(directory: str, recursive: bool = False) -> Dict[str, Dict[str, Any]]:
    """Scans the directory for image files.
    
    Returns a dict mapping relative path (normalized with forward slashes) 
    to its metadata: abs_path, filename, mtime, and size_bytes.
    
    It ignores hidden files/directories (starting with .), internal directories (starting with _),
    and common virtual env/database directories.
    """
    directory = os.path.abspath(directory)
    images = {}
    
    ignore_folders = {'node_modules', 'venv', '.venv', '.git', '.imagfind_db', '__pycache__'}
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            # Prune directories in-place to prevent os.walk from entering them
            dirs[:] = [
                d for d in dirs 
                if not d.startswith('.') 
                and not d.startswith('_') 
                and d not in ignore_folders
            ]
            
            for file in files:
                if file.startswith('.'):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, start=directory).replace('\\', '/')
                    try:
                        mtime = os.path.getmtime(abs_path)
                        size = os.path.getsize(abs_path)
                        images[rel_path] = {
                            "abs_path": abs_path,
                            "filename": file,
                            "mtime": mtime,
                            "size_bytes": size
                        }
                    except Exception:
                        # Skip if file was deleted/inaccessible during scan
                        continue
    else:
        # Non-recursive top-level scan
        try:
            for item in os.listdir(directory):
                if item.startswith('.') or item.startswith('_') or item in ignore_folders:
                    continue
                abs_path = os.path.join(directory, item)
                if os.path.isfile(abs_path):
                    ext = os.path.splitext(item)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        rel_path = item  # at root level, relative path is just the filename
                        try:
                            mtime = os.path.getmtime(abs_path)
                            size = os.path.getsize(abs_path)
                            images[rel_path] = {
                                "abs_path": abs_path,
                                "filename": item,
                                "mtime": mtime,
                                "size_bytes": size
                            }
                        except Exception:
                            continue
        except Exception:
            pass
            
    return images

def format_size(bytes_size: int) -> str:
    """Formats bytes to a human-readable string (e.g. 1.2 MB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def get_dir_size(dirpath: str) -> int:
    """Returns the total size in bytes of all files in the directory."""
    total_size = 0
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        for root, _, files in os.walk(dirpath):
            for file in files:
                fp = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(fp)
                except Exception:
                    pass
    return total_size
