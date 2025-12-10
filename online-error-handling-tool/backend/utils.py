"""
Shared utility functions for routes
"""
import os
import numpy as np
from PIL import Image
import tifffile
from flask import jsonify


def get_image_dimensions(file, use_temp_file=False):
    """
    Get dimensions of an uploaded image file.
    
    Args:
        file: Flask file object
        use_temp_file: If True, save file temporarily and load via load_image_or_stack
        
    Returns:
        dict: {"shape": [height, width, ...]} or {"error": str}
    """
    if not file:
        return {"error": "No file provided"}
    
    try:
        if use_temp_file:
            # Save temporarily for volume loading
            temp_path = f"temp_{file.filename}"
            file.save(temp_path)
            try:
                from backend.volume_manager import load_image_or_stack
                volume = load_image_or_stack(temp_path)
                shape = volume.shape
                return {"shape": list(shape)}
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            # Direct read for simple cases
            fname = file.filename.lower() if file.filename else ""
            if fname.endswith((".tif", ".tiff")):
                arr = tifffile.imread(file)
                shape = arr.shape
            else:
                img = Image.open(file)
                # PIL returns (width, height), convert to (height, width, channels)
                if hasattr(img, 'size'):
                    w, h = img.size
                    # Try to get channel info
                    if hasattr(img, 'mode'):
                        mode = img.mode
                        if mode in ('RGB', 'RGBA'):
                            channels = len(mode)
                        elif mode == 'L':
                            channels = 1
                        else:
                            channels = 1
                        shape = [h, w] if channels == 1 else [h, w, channels]
                    else:
                        shape = [h, w]
                else:
                    shape = img.size[::-1]  # (width, height) -> (height, width)
            
            return {"shape": list(shape)}
    except Exception as e:
        return {"error": str(e)}


def jsonify_dimensions(file, use_temp_file=False):
    """
    Get image dimensions and return as JSON response.
    
    Args:
        file: Flask file object
        use_temp_file: If True, save file temporarily and load via load_image_or_stack
        
    Returns:
        Flask JSON response
    """
    result = get_image_dimensions(file, use_temp_file)
    if "error" in result:
        status_code = 400 if result.get("error") == "No file provided" else 500
        return jsonify(result), status_code
    return jsonify(result)

