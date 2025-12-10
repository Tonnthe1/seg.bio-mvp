"""
Error Handling Tool - Detection Workflow Routes
---------------------------------------------
Handles the error detection workflow (original functionality).
"""

import os
from flask import Blueprint, render_template, request, redirect, url_for, current_app, jsonify
from backend.data_manager import DataManager
from backend.volume_manager import list_images_for_path, build_mask_stack_from_pairs, build_mask_path_mapping
from backend.lazy_stack import LazyMaskLoader
from backend.session_manager import SessionManager
from backend.utils import jsonify_dimensions

bp = Blueprint("detection_workflow", __name__, url_prefix="")


def _uploads_root() -> str:
    env_dir = os.environ.get("PROOFREADING_UPLOAD_DIR")
    if env_dir:
        base_dir = os.path.abspath(env_dir)
    else:
        base_dir = os.path.join(os.path.expanduser("~"), "proofreading_uploads")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def register_detection_workflow_routes(app):
    app.register_blueprint(bp)

@bp.route("/detection/load")
def detection_load():
    """Load dataset for error detection workflow."""
    # Check if there's existing data
    volume = current_app.config.get("DETECTION_VOLUME")
    mask = current_app.config.get("DETECTION_MASK")
    image_path = current_app.config.get("DETECTION_IMAGE_PATH")
    mask_path = current_app.config.get("DETECTION_MASK_PATH")
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    layers = session_state.get("layers", [])
    
    has_existing_data = volume is not None and image_path is not None and len(layers) > 0
    
    if has_existing_data:
        return render_template("detection_load.html",
                             has_existing_data=True,
                             existing_image_path=os.path.basename(image_path),
                             existing_mask_path=os.path.basename(mask_path) if mask_path else None,
                             existing_shape=" × ".join(map(str, volume.shape)),
                             existing_mode3d=volume.ndim == 3,
                             existing_layers_count=len(layers))
    else:
        return render_template("detection_load.html")

@bp.route("/detection/clear", methods=["POST"])
def detection_clear():
    """Clear previously loaded data for error detection."""
    try:
        session_manager = current_app.session_manager
        
        # Clear session data
        session_manager.reset_session()
        
        # Clear app config
        current_app.config.pop("DETECTION_VOLUME", None)
        current_app.config.pop("DETECTION_MASK", None)
        current_app.config.pop("DETECTION_IMAGE_PATH", None)
        current_app.config.pop("DETECTION_MASK_PATH", None)
        current_app.config.pop("DETECTION_DATA_MANAGER", None)
        
        return jsonify({"success": True, "message": "Data cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/detection/load", methods=["POST"])
def detection_load_post():
    """Handle dataset loading for error detection."""
    try:
        session_manager = current_app.session_manager
        
        # Clear previously loaded data before loading new data
        session_manager.reset_session()
        current_app.config.pop("DETECTION_VOLUME", None)
        current_app.config.pop("DETECTION_MASK", None)
        current_app.config.pop("DETECTION_IMAGE_PATH", None)
        current_app.config.pop("DETECTION_MASK_PATH", None)
        current_app.config.pop("DETECTION_DATA_MANAGER", None)
        
        # Get form data
        load_mode = request.form.get("load_mode", "path")
        image_path = request.form.get("image_path", "").strip()
        mask_path = request.form.get("mask_path", "").strip()
        
        if load_mode == "path":
            if not image_path:
                return render_template("detection_load.html", 
                                    error="Image path is required",
                                    image_path=image_path,
                                    mask_path=mask_path)
            
            if not os.path.exists(image_path):
                return render_template("detection_load.html", 
                                    error="Image file not found",
                                    image_path=image_path,
                                    mask_path=mask_path)
            
            if mask_path and not os.path.exists(mask_path):
                return render_template("detection_load.html", 
                                    error="Mask file not found",
                                    image_path=image_path,
                                    mask_path=mask_path)
        
        else:  # upload mode
            # Allow multiple image files to form a stack
            image_files = request.files.getlist("image_file")
            mask_file = request.files.get("mask_file")

            # Filter out empty entries
            image_files = [f for f in image_files if f and f.filename]
            if not image_files:
                return render_template("detection_load.html", 
                                    error="At least one image file is required")

            # Save uploaded files temporarily
            upload_dir = _uploads_root()

            saved_image_paths = []
            for f in image_files:
                dst = os.path.join(upload_dir, f.filename)
                f.save(dst)
                saved_image_paths.append(dst)

            # If a single file, keep a string path; if multiple, keep list
            image_path = saved_image_paths[0] if len(saved_image_paths) == 1 else saved_image_paths

            if mask_file and mask_file.filename:
                mask_path = os.path.join(upload_dir, mask_file.filename)
                mask_file.save(mask_path)
            else:
                mask_path = ""
        
        # Load data
        data_manager = DataManager()
        
        # Decide source list before loading image, to avoid stacking masks as images
        prepared_image_source = image_path
        is_dir_or_glob_or_list = isinstance(image_path, list) or (isinstance(image_path, str) and (os.path.isdir(image_path) or any(ch in image_path for ch in ['*','?','['])))
        mask_dir_for_pairing = None
        if is_dir_or_glob_or_list:
            # Determine mask_dir early for same-folder pairing logic
            if mask_path and os.path.isdir(mask_path):
                mask_dir_for_pairing = mask_path
            elif isinstance(image_path, str) and os.path.isdir(image_path):
                # If mask path is empty or not a dir, pairing may still occur in same image folder
                mask_dir_for_pairing = image_path

            try:
                # If same folder for image and mask, and *_0000 exist, drive by those only
                if isinstance(image_path, str) and os.path.isdir(image_path) and mask_dir_for_pairing and os.path.isdir(mask_dir_for_pairing) \
                   and os.path.abspath(image_path) == os.path.abspath(mask_dir_for_pairing):
                    all_images = list_images_for_path(image_path)
                    driver_images = [fp for fp in all_images if os.path.splitext(os.path.basename(fp))[0].endswith('_0000')]
                    if driver_images:
                        prepared_image_source = driver_images
            except Exception:
                pass

        # Load image (supports path, directory/glob, or list-of-paths); use prepared source
        # Cache ordered image file list for later lookups
        if is_dir_or_glob_or_list:
            if isinstance(prepared_image_source, list):
                cached_image_files = list(prepared_image_source)
            else:
                cached_image_files = list_images_for_path(prepared_image_source)
        else:
            if isinstance(prepared_image_source, list):
                cached_image_files = list(prepared_image_source)
            else:
                cached_image_files = [prepared_image_source]
        current_app.config["DETECTION_IMAGE_FILES"] = cached_image_files

        volume, volume_info = data_manager.load_image(prepared_image_source)
        
        # Load mask if provided, or pair per-file for folders/globs/lists
        mask = None
        mask_info = {}
        lazy_volume = bool(volume_info.get("lazy"))
        if is_dir_or_glob_or_list:
            mask_dir = None
            if mask_path and os.path.isdir(mask_path):
                mask_dir = mask_path
            elif not mask_path:
                mask_dir = None
            else:
                # mask provided as single file; use standard loader
                mask, mask_info = data_manager.load_mask(mask_path)

            if mask is None:
                # Pair using the same driver list we used to load the volume
                if isinstance(prepared_image_source, list):
                    image_files = list(prepared_image_source)
                else:
                    image_files = list_images_for_path(prepared_image_source)

                mask_paths_for_cache = build_mask_path_mapping(image_files, mask_dir)
                current_app.config["DETECTION_MASK_FILES"] = list(mask_paths_for_cache)

                if lazy_volume:
                    lazy_mask = LazyMaskLoader(mask_paths_for_cache, data_manager.current_volume.slice_shape)
                    mask = lazy_mask
                    mask_info = {
                        "shape": lazy_mask.shape,
                        "dtype": "uint8",
                        "ndim": lazy_mask.ndim,
                        "is_3d": True,
                        "num_slices": lazy_mask.shape[0],
                        "lazy": True
                    }
                    data_manager.current_mask = lazy_mask
                    data_manager.mask_info = mask_info
                else:
                    mask = build_mask_stack_from_pairs(image_files, mask_dir)
                    if mask is not None:
                        current_app.config["DETECTION_MASK_FILES"] = list(build_mask_path_mapping(image_files, mask_dir))
                        data_manager.current_mask = mask
                        data_manager.mask_info = {
                            "shape": mask.shape,
                            "dtype": str(mask.dtype),
                            "ndim": mask.ndim,
                            "is_3d": mask.ndim == 3,
                            "num_slices": mask.shape[0] if mask.ndim == 3 else 1
                        }
        elif mask_path:
            mask, mask_info = data_manager.load_mask(mask_path)
            if isinstance(mask_path, list):
                current_app.config["DETECTION_MASK_FILES"] = list(mask_path)
            elif os.path.isfile(mask_path):
                current_app.config["DETECTION_MASK_FILES"] = [mask_path]
        
        # Validate compatibility
        if not data_manager.validate_data_compatibility():
            return render_template("detection_load.html", 
                                error="Image and mask dimensions are incompatible",
                                image_path=image_path if load_mode == "path" else "",
                                mask_path=mask_path if load_mode == "path" else "")
        
        # Update session
        # Store representative name for uploads
        image_name_val = None
        if load_mode == "upload":
            if isinstance(image_path, list):
                image_name_val = f"{len(image_path)}_files_stack"
            else:
                image_name_val = os.path.basename(image_path)
        else:
            image_name_val = os.path.basename(image_path) if isinstance(image_path, str) else "stack"

        session_manager.update(
            mode3d=volume_info["is_3d"],
            image_path=image_path if isinstance(image_path, str) else (image_path[0] if image_path else None),
            mask_path=mask_path,
            load_mode=load_mode,
            image_name=image_name_val
        )
        session_manager.set_image_info(image_path, load_mode)
        
        # Add lightweight placeholders for layers (defer overlay generation)
        total_slices = volume_info.get("num_slices", 1)
        for z in range(total_slices):
            session_manager.add_layer({
                "z": z
            })
        
        # Store data in app config for API access
        current_app.config["DETECTION_VOLUME"] = volume
        current_app.config["DETECTION_MASK"] = mask
        current_app.config["DETECTION_IMAGE_PATH"] = image_path
        current_app.config["DETECTION_MASK_PATH"] = mask_path
        current_app.config["DETECTION_DATA_MANAGER"] = data_manager
        
        # Redirect to detection page
        return redirect(url_for("detection.detection"))
        
    except Exception as e:
        return render_template("detection_load.html", 
                            error=f"Error loading dataset: {str(e)}")

@bp.route("/api/dims", methods=["POST"])
def api_dims():
    """Get dimensions of uploaded file."""
    return jsonify_dimensions(request.files.get("file"), use_temp_file=True)
