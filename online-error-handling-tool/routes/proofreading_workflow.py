"""
Error Handling Tool - Standalone Proofreading Workflow Routes
-----------------------------------------------------------
Handles the standalone proofreading workflow (similar to PFTool).
"""

import os
import io
import math
import base64
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
from flask import Blueprint, render_template, request, current_app, jsonify, send_file, redirect, url_for
from PIL import Image
from backend.volume_manager import load_image_or_stack, load_mask_like, save_mask, list_images_for_path, build_mask_stack_from_pairs, build_mask_path_mapping, stack_2d_images
from backend.data_manager import _prepare_mask_for_display, _normalize_image_slice_to_rgb, _mask_slice_to_rgba


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _uploads_root() -> str:
    env_dir = os.environ.get("PROOFREADING_UPLOAD_DIR")
    if env_dir:
        base_dir = os.path.abspath(env_dir)
    else:
        base_dir = os.path.join(os.path.expanduser("~"), "proofreading_uploads")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _saved_masks_root() -> str:
    path = os.path.join(PROJECT_ROOT, "_saved_masks")
    os.makedirs(path, exist_ok=True)
    return path


def _saved_masks_dir_for(image_path) -> str:
    candidate = None
    if isinstance(image_path, list) and image_path:
        candidate = image_path[0]
    elif isinstance(image_path, str):
        candidate = image_path

    base_name = "dataset"
    if isinstance(candidate, str) and candidate:
        norm = os.path.abspath(candidate)
        if os.path.isdir(norm):
            base_name = os.path.basename(norm)
        else:
            base_name = os.path.splitext(os.path.basename(norm))[0]

    target = os.path.join(_saved_masks_root(), base_name)
    os.makedirs(target, exist_ok=True)
    return target
def _store_file_lists(image_files=None, mask_files=None):
    if image_files is not None:
        current_app.config["PROOFREADING_IMAGE_FILES"] = list(image_files)
    if mask_files is not None:
        current_app.config["PROOFREADING_MASK_FILES"] = list(mask_files)


def _ensure_cached_image_files():
    files = current_app.config.get("PROOFREADING_IMAGE_FILES")
    if files:
        return files
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    image_path = session_state.get("image_path", "")
    files = list_images_for_path(image_path)
    current_app.config["PROOFREADING_IMAGE_FILES"] = files
    return files


def _resolve_mask_for_index(img_fp, idx, default_mask_path=None):
    mask_files = current_app.config.get("PROOFREADING_MASK_FILES")
    if mask_files and idx < len(mask_files):
        mask_fp = mask_files[idx]
        if mask_fp:
            return mask_fp

    def build_extensions(base_ext):
        exts = [base_ext.lower()]
        for e in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            if e not in exts:
                exts.append(e)
        return exts

    def try_dir(dir_path, base_name, extensions):
        if not dir_path or not os.path.isdir(dir_path):
            return None
        # Try _mask suffix
        for e in extensions:
            cand = os.path.join(dir_path, f"{base_name}_mask{e}")
            if os.path.exists(cand):
                return cand
        # Try _pred_skeleton suffix (for files like Image95_00001_pred_skeleton.tif)
        for e in extensions:
            cand = os.path.join(dir_path, f"{base_name}_pred_skeleton{e}")
            if os.path.exists(cand):
                return cand
        # Try _pred suffix (for files like Image94_00001_pred.tif)
        for e in extensions:
            cand = os.path.join(dir_path, f"{base_name}_pred{e}")
            if os.path.exists(cand):
                return cand
        # Try _prediction suffix
        for e in extensions:
            cand = os.path.join(dir_path, f"{base_name}_prediction{e}")
            if os.path.exists(cand):
                return cand
        # Try nnUNet-style (remove _0000 suffix)
        if base_name.endswith("_0000"):
            trimmed = base_name[:-5]
            for e in extensions:
                cand = os.path.join(dir_path, f"{trimmed}{e}")
                if os.path.exists(cand):
                    return cand
        return None

    if not img_fp:
        return default_mask_path

    base, ext = os.path.splitext(os.path.basename(img_fp))
    extensions = build_extensions(ext)
    session_state = current_app.session_manager.snapshot()
    mask_path = session_state.get("mask_path", default_mask_path)
    search_dirs = []
    if mask_path and os.path.isdir(mask_path):
        search_dirs.append(mask_path)
    img_dir = os.path.dirname(img_fp)
    if img_dir:
        search_dirs.append(img_dir)
        # Check if there's a directory with _pred or _predictions suffix in the parent directory
        parent_dir = os.path.dirname(img_dir)
        dir_name = os.path.basename(img_dir)
        pred_dir_candidates = [
            os.path.join(parent_dir, f"{dir_name}_predictions"),
            os.path.join(parent_dir, f"{dir_name}_pred")
        ]
        for pred_dir in pred_dir_candidates:
            if os.path.isdir(pred_dir) and pred_dir not in search_dirs:
                search_dirs.append(pred_dir)

    for d in search_dirs:
        result = try_dir(d, base, extensions)
        if result:
            return result

    if mask_path and os.path.isfile(mask_path):
        return mask_path
    return None
from backend.lazy_stack import LazySliceLoader, LazyMaskLoader
from backend.utils import jsonify_dimensions
from backend.data_manager import DataManager

bp = Blueprint("proofreading_workflow", __name__, url_prefix="/standalone_proofreading")

def register_proofreading_workflow_routes(app):
    app.register_blueprint(bp)


def _infer_num_slices(volume):
    """Best-effort slice count for numpy arrays or lazy loaders."""
    if volume is None:
        return 0
    shape = getattr(volume, "shape", None)
    if isinstance(shape, tuple) and shape:
        if len(shape) <= 2:
            return 1
        return max(1, int(shape[0]))
    try:
        return max(1, len(volume))
    except (TypeError, AttributeError):
        return 1


def _ensure_proofreading_data_manager(volume, mask):
    """Attach or refresh a DataManager for standalone proofreading previews."""
    data_manager = current_app.config.get("PROOFREADING_DATA_MANAGER")
    if data_manager is None:
        data_manager = DataManager()
        current_app.config["PROOFREADING_DATA_MANAGER"] = data_manager

    total_slices = _infer_num_slices(volume)
    ndim_guess = getattr(volume, "ndim", 3 if total_slices > 1 else 2)

    data_manager.current_volume = volume
    data_manager.volume_info = {
        "shape": getattr(volume, "shape", None),
        "dtype": str(getattr(volume, "dtype", "uint8")),
        "ndim": ndim_guess,
        "is_3d": total_slices > 1,
        "num_slices": total_slices,
        "lazy": hasattr(volume, "get_slice")
    }

    data_manager.current_mask = mask
    if mask is not None:
        mask_shape = getattr(mask, "shape", None)
        mask_ndim = getattr(mask, "ndim", 3 if mask_shape and len(mask_shape) >= 3 else 2)
        data_manager.mask_info = {
            "shape": mask_shape,
            "dtype": str(getattr(mask, "dtype", "uint8")),
            "ndim": mask_ndim,
            "is_3d": mask_ndim == 3,
            "num_slices": mask_shape[0] if mask_shape and mask_ndim == 3 else 1,
            "lazy": hasattr(mask, "get_slice")
        }

    return data_manager, total_slices


def _apply_pending_mask_edits(mask_array: Optional[np.ndarray], mask_edits: Dict[int, np.ndarray]):
    """
    Merge pending edits (stored in PROOFREADING_MASK_EDITS) into the in-memory mask array.
    Returns (mask_array, edits_applied_flag).
    """
    if mask_array is None or not isinstance(mask_array, np.ndarray) or not mask_edits:
        return mask_array, False

    def _resize_patch(patch: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if patch.shape == target_shape:
            return patch
        return cv2.resize(patch, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    applied = False
    if mask_array.ndim == 2:
        patch = None
        if 0 in mask_edits:
            patch = mask_edits[0]
        elif mask_edits:
            patch = next(iter(mask_edits.values()))
        if patch is not None:
            patch = np.asarray(patch, dtype=np.uint8)
            patch = _resize_patch(patch, mask_array.shape)
            mask_array[:, :] = patch
            applied = True
    else:
        depth = mask_array.shape[0]
        for idx, sl in mask_edits.items():
            if sl is None or idx is None or idx < 0 or idx >= depth:
                continue
            patch = np.asarray(sl, dtype=np.uint8)
            patch = _resize_patch(patch, mask_array[idx].shape)
            mask_array[idx] = patch
            applied = True

    return mask_array, applied


def _ensure_mask_matches_volume(mask_array: Optional[np.ndarray], volume) -> Optional[np.ndarray]:
    if isinstance(mask_array, np.ndarray):
        vol_ndim = getattr(volume, "ndim", None)
        if mask_array.ndim == 3 and mask_array.shape[0] == 1 and vol_ndim == 2:
            return np.squeeze(mask_array, axis=0)
    return mask_array

@bp.route("/load")
def proofreading_load():
    """Load dataset for standalone proofreading workflow."""
    # Check if there's existing data
    volume = current_app.config.get("PROOFREADING_VOLUME")
    mask = current_app.config.get("PROOFREADING_MASK")
    image_path = current_app.config.get("PROOFREADING_IMAGE_PATH")
    mask_path = current_app.config.get("PROOFREADING_MASK_PATH")
    
    has_existing_data = volume is not None and image_path is not None
    
    if has_existing_data:
        return render_template("proofreading_load.html",
                             has_existing_data=True,
                             existing_image_path=os.path.basename(image_path),
                             existing_mask_path=os.path.basename(mask_path) if mask_path else None,
                             existing_shape=" × ".join(map(str, volume.shape)),
                             existing_mode3d=volume.ndim == 3)
    else:
        return render_template("proofreading_load.html")

@bp.route("/clear", methods=["POST"])
def clear_data():
    """Clear existing data."""
    try:
        # Clear app config
        current_app.config.pop("PROOFREADING_VOLUME", None)
        current_app.config.pop("PROOFREADING_MASK", None)
        current_app.config.pop("PROOFREADING_IMAGE_PATH", None)
        current_app.config.pop("PROOFREADING_MASK_PATH", None)
        current_app.config.pop("PROOFREADING_EDITED_SLICES", None)
        current_app.config.pop("PROOFREADING_MASK_EDITS", None)
        current_app.config.pop("PROOFREADING_NUM_SLICES", None)
        current_app.config.pop("PROOFREADING_DATA_MANAGER", None)
        
        # Clear session
        session_manager = current_app.session_manager
        session_manager.reset_session()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/load", methods=["POST"])
def proofreading_load_post():
    """Handle dataset loading for standalone proofreading."""
    try:
        # Clear previously loaded data before loading new data
        session_manager = current_app.session_manager
        session_manager.reset_session()
        current_app.config.pop("PROOFREADING_VOLUME", None)
        current_app.config.pop("PROOFREADING_MASK", None)
        current_app.config.pop("PROOFREADING_IMAGE_PATH", None)
        current_app.config.pop("PROOFREADING_MASK_PATH", None)
        current_app.config.pop("PROOFREADING_NUM_SLICES", None)
        current_app.config.pop("PROOFREADING_DATA_MANAGER", None)
        
        # Get form data
        load_mode = request.form.get("load_mode", "path")
        image_path = request.form.get("image_path", "").strip()
        mask_path = request.form.get("mask_path", "").strip()
        
        if load_mode == "path":
            if not image_path:
                return render_template("proofreading_load.html", 
                                    error="Image path is required",
                                    image_path=image_path,
                                    mask_path=mask_path)
            
            if not os.path.exists(image_path):
                return render_template("proofreading_load.html", 
                                    error="Image file not found",
                                    image_path=image_path,
                                    mask_path=mask_path)
            
            if mask_path and not os.path.exists(mask_path):
                return render_template("proofreading_load.html", 
                                    error="Mask file not found",
                                    image_path=image_path,
                                    mask_path=mask_path)
        
        else:  # upload mode
            # Allow multiple image files to form a stack
            image_files = request.files.getlist("image_file")
            mask_file = request.files.get("mask_file")

            image_files = [f for f in image_files if f and f.filename]
            if not image_files:
                return render_template("proofreading_load.html", 
                                    error="At least one image file is required")

            # Save uploaded files temporarily outside repo
            upload_dir = _uploads_root()

            saved_image_paths = []
            for f in image_files:
                dst = os.path.join(upload_dir, f.filename)
                f.save(dst)
                saved_image_paths.append(dst)

            image_path = saved_image_paths[0] if len(saved_image_paths) == 1 else saved_image_paths

            if mask_file and mask_file.filename:
                mask_path = os.path.join(upload_dir, mask_file.filename)
                mask_file.save(mask_path)
            else:
                mask_path = ""
        
        # Store paths in session manager
        session_manager = current_app.session_manager
        
        # Determine original filename based on load mode
        if load_mode == "upload":
            if isinstance(image_path, list):
                original_filename = f"{len(image_path)}_files_stack"
            else:
                original_filename = os.path.basename(image_path)
        else:
            if isinstance(image_path, list) or any(ch in str(image_path) for ch in ['*','?','[']) or os.path.isdir(image_path):
                original_filename = "stack"
            else:
                original_filename = os.path.basename(image_path)
        
        session_manager.update(
            image_path=image_path,
            mask_path=mask_path,
            load_mode=load_mode,
            image_name=original_filename,  # Store original filename
            mode3d=False  # Will be updated after loading
        )
        
        # Prepare driver images when working from dir/glob/list and masks live in same folder
        prepared_image_source = image_path
        is_dir_or_glob_or_list = isinstance(image_path, list) or (isinstance(image_path, str) and (os.path.isdir(image_path) or any(ch in image_path for ch in ['*','?','['])))
        if is_dir_or_glob_or_list:
            try:
                # Same-folder pairing if mask_path is same dir as images (or mask not provided)
                img_dir = image_path if (isinstance(image_path, str) and os.path.isdir(image_path)) else None
                mask_dir_candidate = mask_path if (mask_path and os.path.isdir(mask_path)) else img_dir
                if img_dir and mask_dir_candidate and os.path.abspath(img_dir) == os.path.abspath(mask_dir_candidate):
                    all_images = list_images_for_path(image_path)
                    drivers = [fp for fp in all_images if os.path.splitext(os.path.basename(fp))[0].endswith('_0000')]
                    if drivers:
                        prepared_image_source = drivers
            except Exception:
                pass

        # Load volume using prepared source (lazy for folders/globs/lists)
        if is_dir_or_glob_or_list:
            if isinstance(prepared_image_source, list):
                image_files = list(prepared_image_source)
            else:
                image_files = list_images_for_path(prepared_image_source)
            _store_file_lists(image_files=image_files)
            volume = LazySliceLoader(image_files)
        else:
            if isinstance(prepared_image_source, list):
                image_files = list(prepared_image_source)
                volume = stack_2d_images(prepared_image_source)
            else:
                image_files = [prepared_image_source]
                volume = load_image_or_stack(prepared_image_source)
            _store_file_lists(image_files=image_files)

        # Optional: build mask per-file pairing when working with folders/globs/lists
        mask = None
        if is_dir_or_glob_or_list:
            mask_dir = None
            if mask_path and os.path.isdir(mask_path):
                mask_dir = mask_path
            elif not mask_path:
                mask_dir = None
            else:
                mask = load_mask_like(mask_path, volume)

            if mask is None:
                mask_paths = build_mask_path_mapping(image_files, mask_dir)
                _store_file_lists(mask_files=mask_paths)
                mask = LazyMaskLoader(mask_paths, volume.slice_shape)
        else:
            mask = load_mask_like(mask_path, volume) if mask_path else None
            if mask_path:
                if isinstance(mask_path, list):
                    _store_file_lists(mask_files=mask_path)
                elif os.path.isfile(mask_path):
                    _store_file_lists(mask_files=[mask_path])
        
        print(f"DEBUG: Loaded volume shape: {getattr(volume, 'shape', None)}")
        print(f"DEBUG: Loaded mask shape: {getattr(mask, 'shape', None)}")
        
        # Store in app config for the session
        current_app.config["PROOFREADING_VOLUME"] = volume
        current_app.config["PROOFREADING_MASK"] = mask
        current_app.config["PROOFREADING_IMAGE_PATH"] = image_path
        current_app.config["PROOFREADING_MASK_PATH"] = mask_path
        current_app.config["PROOFREADING_EDITED_SLICES"] = set()
        current_app.config["PROOFREADING_MASK_EDITS"] = {}
        data_manager, num_slices = _ensure_proofreading_data_manager(volume, mask)
        current_app.config["PROOFREADING_NUM_SLICES"] = num_slices
        
        # Update session with 3D info
        mode3d = getattr(volume, 'ndim', 2) == 3
        session_manager.update(mode3d=mode3d)
        
        print(f"DEBUG: Stored in config - volume: {current_app.config.get('PROOFREADING_VOLUME') is not None}, mask: {current_app.config.get('PROOFREADING_MASK') is not None}")
        
        # Redirect to proofreading editor
        return redirect(url_for("proofreading_workflow.proofreading_editor"))
        
    except Exception as e:
        return render_template("proofreading_load.html", 
                            error=f"Error loading dataset: {str(e)}")

@bp.route("/editor")
def proofreading_editor():
    """Standalone proofreading editor."""
    volume = current_app.config.get("PROOFREADING_VOLUME")
    mask = current_app.config.get("PROOFREADING_MASK")
    
    if volume is None:
        return redirect(url_for("proofreading_workflow.proofreading_load"))
    
    # Ensure mask exists (create empty mask if none provided)
    if mask is None:
        if hasattr(volume, 'get_slice'):
            mask = LazyMaskLoader([None] * volume.shape[0], volume.slice_shape)
        elif volume.ndim == 2:
            mask = np.zeros_like(volume, dtype=np.uint8)
        elif volume.ndim == 3:
            mask = np.zeros_like(volume, dtype=np.uint8)
        current_app.config["PROOFREADING_MASK"] = mask
        print(f"DEBUG: Created empty mask with shape {getattr(mask, 'shape', None)}")
    
    mode3d = getattr(volume, 'ndim', 2) == 3
    num_slices = volume.shape[0] if mode3d else 1
    
    return render_template("proofreading_standalone.html",
                         mode3d=mode3d,
                         num_slices=num_slices,
                         volume_shape=volume.shape,
                         mask_shape=mask.shape if mask is not None else None,
                         z=0,  # Current slice index (always 0 for standalone)
                         slice_index=0)  # For template compatibility

@bp.route("/api/slice/<int:z>")
def api_slice(z):
    """Get image slice for standalone proofreading."""
    try:
        volume = current_app.config.get("PROOFREADING_VOLUME")
        print(f"DEBUG: api_slice called with z={z}, volume is None: {volume is None}")
        
        # If volume is not loaded, try to reload from session
        if volume is None:
            session_manager = current_app.session_manager
            session_state = session_manager.snapshot()
            image_path = session_state.get("image_path")
            
            if image_path:
                print(f"DEBUG: Reloading volume from {image_path}")
                if isinstance(image_path, list):
                    volume = stack_2d_images(image_path)
                else:
                    volume = load_image_or_stack(image_path)
                current_app.config["PROOFREADING_VOLUME"] = volume
                
                # Also reload mask if available
                mask_path = session_state.get("mask_path")
                if mask_path and os.path.exists(mask_path):
                    mask = load_mask_like(mask_path, volume)
                    current_app.config["PROOFREADING_MASK"] = mask
                    current_app.config["PROOFREADING_MASK_PATH"] = mask_path
            else:
                print("DEBUG: No volume loaded in api_slice")
                return jsonify(error="No volume loaded"), 404
        
        if hasattr(volume, 'get_slice'):
            total = volume.shape[0]
            z = int(np.clip(z, 0, total - 1))
            sl = volume.get_slice(z)
        elif volume.ndim == 2:
            sl = volume
        else:
            z = int(np.clip(z, 0, volume.shape[0] - 1))
            sl = volume[z]
        
        print(f"DEBUG: Processing slice shape: {sl.shape}")
        
        # Convert to RGB with consistent normalization (using shared helper)
        rgb = _normalize_image_slice_to_rgb(sl)
        
        print(f"DEBUG: RGB shape: {rgb.shape}")
        
        bio = io.BytesIO()
        Image.fromarray(rgb).save(bio, format="PNG")
        bio.seek(0)
        return send_file(bio, mimetype="image/png")
        
    except Exception as e:
        print(f"DEBUG: Error in api_slice: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@bp.route("/api/mask/<int:z>")
def api_mask(z):
    """Get mask slice for standalone proofreading."""
    try:
        mask = current_app.config.get("PROOFREADING_MASK")
        volume = current_app.config.get("PROOFREADING_VOLUME")
        print(f"DEBUG: api_mask called with z={z}, volume is None: {volume is None}, mask is None: {mask is None}")
        
        # If volume is not loaded, try to reload from session
        if volume is None:
            session_manager = current_app.session_manager
            session_state = session_manager.snapshot()
            image_path = session_state.get("image_path")
            
            if image_path:
                print(f"DEBUG: Reloading volume from {image_path}")
                if isinstance(image_path, list):
                    volume = stack_2d_images(image_path)
                else:
                    volume = load_image_or_stack(image_path)
                current_app.config["PROOFREADING_VOLUME"] = volume
                
                # Also reload mask if available
                mask_path = session_state.get("mask_path")
                if mask_path and os.path.exists(mask_path):
                    mask = load_mask_like(mask_path, volume)
                    current_app.config["PROOFREADING_MASK"] = mask
                    current_app.config["PROOFREADING_MASK_PATH"] = mask_path

        # if no mask loaded but an image exists, create a blank one
        if mask is None and volume is not None:
            if volume.ndim == 2:
                mask = np.zeros_like(volume, dtype=np.uint8)
            elif volume.ndim == 3:
                mask = np.zeros_like(volume, dtype=np.uint8)
            current_app.config["PROOFREADING_MASK"] = mask

        if mask is None:
            return jsonify(error="No mask loaded"), 404

        mask_edits = current_app.config.get("PROOFREADING_MASK_EDITS", {})
        if z in mask_edits:
            sl = mask_edits[z]
        elif hasattr(mask, 'get_slice'):
            z = int(np.clip(z, 0, mask.shape[0] - 1))
            print(f"DEBUG: Calling mask.get_slice({z}) on LazyMaskLoader")
            sl = mask.get_slice(z)
            print(f"DEBUG: Got slice from LazyMaskLoader - shape: {sl.shape}, dtype: {sl.dtype}, min: {sl.min()}, max: {sl.max()}, non-zero: {np.count_nonzero(sl)}")
        elif mask.ndim == 2:
            sl = mask
        else:
            z = int(np.clip(z, 0, mask.shape[0] - 1))
            sl = mask[z]
        
        print(f"DEBUG: Processing mask slice shape: {sl.shape}")
        print(f"DEBUG: Mask slice dtype: {sl.dtype}, min: {sl.min()}, max: {sl.max()}, non-zero: {np.count_nonzero(sl)}")
        
        # Convert mask to RGBA using shared helper
        rgba = _mask_slice_to_rgba(sl, opacity=230)
        
        non_zero_count = np.count_nonzero(rgba[:,:,3] > 0)
        print(f"DEBUG: Final mask RGBA - shape: {rgba.shape}, non-zero pixels: {non_zero_count}, alpha range: 0-{rgba[:,:,3].max()}")
        
        if non_zero_count == 0:
            print(f"ERROR: Mask is completely empty! This means the mask file has no data or is not loading correctly.")
            print(f"  Check server logs for LazyMaskLoader debug messages to see what's happening.")
        
        im = Image.fromarray(rgba, mode='RGBA')
        bio = io.BytesIO()
        im.save(bio, format="PNG")
        bio.seek(0)
        return send_file(bio, mimetype="image/png")
        
    except Exception as e:
        print(f"DEBUG: Error in api_mask: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@bp.route("/api/names/<int:z>")
def api_names(z):
    """Return source image filename and paired mask filename for current slice.
    When images and masks share the same folder and a pair is found, suppress mask filename display.
    Supports nnUNet-style pairing (image *_0000, mask without suffix).
    """
    try:
        session_manager = current_app.session_manager
        session_state = session_manager.snapshot()
        image_path = session_state.get("image_path", "")
        mask_path = session_state.get("mask_path", "")

        images = current_app.config.get("PROOFREADING_IMAGE_FILES")
        if not images:
            images = _ensure_cached_image_files()
        if not images:
            return jsonify(image=None, mask=None)
        idx = int(np.clip(z, 0, len(images) - 1))
        img_fp = images[idx]

        mask_fp = _resolve_mask_for_index(img_fp, idx, default_mask_path=mask_path)
        mask_name = os.path.basename(mask_fp) if mask_fp else None

        return jsonify(image=os.path.basename(img_fp), mask=mask_name)
    except Exception as e:
        return jsonify(error=str(e)), 500


@bp.route("/api/layers")
def api_layers():
    """Return paginated layer previews to avoid loading thousands of slices at once."""
    try:
        volume = current_app.config.get("PROOFREADING_VOLUME")
        mask = current_app.config.get("PROOFREADING_MASK")
        if volume is None:
            return jsonify(success=False, error="No dataset loaded"), 404

        per_page = request.args.get("per_page", 36, type=int)
        per_page = max(1, min(per_page, 72))
        page = request.args.get("page", 1, type=int)

        data_manager, total_slices = _ensure_proofreading_data_manager(volume, mask)
        if total_slices == 0:
            return jsonify(success=True, layers=[], pagination={
                "current_page": 1,
                "total_pages": 1,
                "per_page": per_page,
                "total_layers": 0,
                "start_idx": 0,
                "end_idx": 0
            })

        total_pages = max(1, math.ceil(total_slices / per_page))
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_slices)

        layers = data_manager.generate_layers_range(start_idx, end_idx)

        return jsonify(success=True, layers=layers, pagination={
            "current_page": page,
            "total_pages": total_pages,
            "per_page": per_page,
            "total_layers": total_slices,
            "start_idx": start_idx + 1,
            "end_idx": end_idx
        })
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@bp.route("/api/mask/update", methods=["POST"])
def api_mask_update():
    """Update mask for standalone proofreading."""
    data = request.get_json(force=True)
    mask = current_app.config.get("PROOFREADING_MASK")
    volume = current_app.config.get("PROOFREADING_VOLUME")
    mask_edits = current_app.config.setdefault("PROOFREADING_MASK_EDITS", {})

    if mask is None and volume is not None and not hasattr(volume, 'get_slice'):
        if volume.ndim == 2:
            mask = np.zeros_like(volume, dtype=np.uint8)
        elif volume.ndim == 3:
            mask = np.zeros_like(volume, dtype=np.uint8)
        current_app.config["PROOFREADING_MASK"] = mask
    elif mask is None:
        return jsonify(success=False, error="No mask or image loaded"), 404

    mask_edits = current_app.config.get("PROOFREADING_MASK_EDITS", {})
    mask_is_lazy = hasattr(mask, "get_slice")

    edited = set(current_app.config.get("PROOFREADING_EDITED_SLICES", set()))
    mask_is_lazy = hasattr(mask, 'get_slice')

    def store_edit(z_idx: int, arr: np.ndarray):
        if mask_is_lazy:
            mask_edits[z_idx] = arr.astype(np.uint8)
        else:
            if mask.ndim == 2:
                resized = cv2.resize(arr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask[:, :] = resized
            else:
                resized = cv2.resize(arr, (mask.shape[2], mask.shape[1]), interpolation=cv2.INTER_NEAREST)
                mask[z_idx] = resized
        edited.add(z_idx)

    # --- Batch updates ---
    if "full_batch" in data:
        for item in data["full_batch"]:
            z = int(item["z"])
            png_bytes = base64.b64decode(item["png"])
            img = Image.open(io.BytesIO(png_bytes)).convert("L")
            arr = (np.array(img) > 127).astype(np.uint8)
            store_edit(z, arr)
        if not mask_is_lazy:
            current_app.config["PROOFREADING_MASK"] = mask
        current_app.config["PROOFREADING_EDITED_SLICES"] = edited
        print(f"✅ Batch updated {len(data['full_batch'])} slice(s)")
        return jsonify(success=True)

    # --- Single slice update ---
    if "full_png" in data:
        z = int(data.get("z", 0))
        png_bytes = base64.b64decode(data["full_png"])
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        arr = (np.array(img) > 127).astype(np.uint8)
        store_edit(z, arr)
        if not mask_is_lazy:
            current_app.config["PROOFREADING_MASK"] = mask
        current_app.config["PROOFREADING_EDITED_SLICES"] = edited
        print(f"✅ Replaced full slice {z}")
        return jsonify(success=True)

    return jsonify(success=False, error="Invalid data"), 400

@bp.route("/api/save", methods=["POST"])
def api_save():
    """Save mask for standalone proofreading."""
    # Ensure we always return JSON, even if there's an error before we get here
    # Wrap entire function in try-except to ensure JSON responses
    try:
        mask = current_app.config.get("PROOFREADING_MASK")
        volume = current_app.config.get("PROOFREADING_VOLUME")

        if mask is None and volume is not None:
            if hasattr(volume, 'get_slice'):
                mask = LazyMaskLoader([None] * volume.shape[0], volume.slice_shape)
            elif volume.ndim == 2:
                mask = np.zeros_like(volume, dtype=np.uint8)
            elif volume.ndim == 3:
                mask = np.zeros_like(volume, dtype=np.uint8)
            current_app.config["PROOFREADING_MASK"] = mask
        elif mask is None:
            return jsonify(success=False, error="No mask or image loaded"), 404

        # Get mask_edits and mask_is_lazy
        mask_edits = current_app.config.get("PROOFREADING_MASK_EDITS", {})
        mask_is_lazy = hasattr(mask, "get_slice")

        # Get session data
        session_manager = current_app.session_manager
        session_state = session_manager.snapshot()
        img_path = session_state.get("image_path", "")
        mask_path = session_state.get("mask_path", "")
        load_mode = session_state.get("load_mode", "path")

        # Generate save destination metadata
        src_is_dir = bool(img_path and isinstance(img_path, str) and os.path.isdir(img_path))
        src_is_glob = bool(img_path and isinstance(img_path, str) and any(ch in img_path for ch in ['*','?','[']) and not os.path.exists(img_path))
        src_is_list = isinstance(img_path, list)
        cached_image_files = current_app.config.get("PROOFREADING_IMAGE_FILES") or []
        cached_has_multiple = len(cached_image_files) > 1
        dataset_has_multiple_sources = src_is_dir or src_is_glob or src_is_list or cached_has_multiple

        def _resolve_source_files():
            if cached_image_files:
                return cached_image_files
            if src_is_list and isinstance(img_path, list):
                return list(img_path)
            if src_is_dir and isinstance(img_path, str):
                selected = []
                for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                    selected = sorted([
                        os.path.join(img_path, f)
                        for f in os.listdir(img_path)
                        if f.lower().endswith(ext)
                    ])
                    if selected:
                        break
                return selected
            if src_is_glob and isinstance(img_path, str):
                import glob as _glob
                return sorted(_glob.glob(img_path))
            if isinstance(img_path, str) and img_path:
                return [img_path]
            return []

        def _mask_dir_anchor():
            if src_is_dir and isinstance(img_path, str):
                return img_path
            if cached_image_files:
                first = cached_image_files[0]
                parent = os.path.dirname(first)
                return parent if parent else first
            if src_is_list and isinstance(img_path, list) and img_path:
                first = img_path[0]
                parent = os.path.dirname(first)
                return parent if parent else first
            if isinstance(img_path, str) and os.path.isfile(img_path):
                parent = os.path.dirname(img_path)
                return parent if parent else img_path
            return img_path

        source_files_for_naming = _resolve_source_files()
        mask_files_cache = current_app.config.get("PROOFREADING_MASK_FILES", [])

        # Folder/glob/list datasets: always save per-slice files, only for edited slices
        if dataset_has_multiple_sources:
            edited = current_app.config.get("PROOFREADING_EDITED_SLICES", set())
            edited_list = sorted(edited)
            if not edited_list:
                return jsonify(success=True, message="No edited slices to save"), 200

            if mask_path and os.path.isdir(mask_path):
                mask_dir = mask_path
            else:
                mask_dir = _saved_masks_dir_for(_mask_dir_anchor())
                session_manager.update(mask_path=mask_dir)
                current_app.config["PROOFREADING_MASK_PATH"] = mask_dir
            os.makedirs(mask_dir, exist_ok=True)

            def _slice_output_path(z_idx, base_name, ext_hint):
                allowed_exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
                ext_use = ext_hint if ext_hint in allowed_exts else ".tif"
                out_fp = None
                if mask_files_cache and z_idx < len(mask_files_cache) and mask_files_cache[z_idx]:
                    out_fp = mask_files_cache[z_idx]
                else:
                    out_fp = os.path.join(mask_dir, f"{base_name}_mask{ext_use}")
                return out_fp

            saved_paths = []
            for z in edited_list:
                if z < 0:
                    continue
                if source_files_for_naming:
                    src_idx = min(max(z, 0), len(source_files_for_naming) - 1)
                    src_fp = source_files_for_naming[src_idx]
                    base = os.path.splitext(os.path.basename(src_fp))[0]
                    ext_out = os.path.splitext(src_fp)[-1].lower()
                else:
                    base = f"slice_{z:04d}"
                    ext_out = ".tif"

                out_fp = _slice_output_path(z, base, ext_out)
                os.makedirs(os.path.dirname(out_fp), exist_ok=True)

                sl = mask_edits.pop(z, None)
                if sl is None:
                    if mask_is_lazy:
                        try:
                            sl = mask.get_slice(z)
                        except Exception:
                            sl = None
                    elif isinstance(mask, np.ndarray):
                        if mask.ndim == 3 and z < mask.shape[0]:
                            sl = mask[z]
                        elif mask.ndim == 2:
                            sl = mask.copy()
                if sl is None:
                    continue

                try:
                    save_mask(sl, out_fp, preserve_format_from=out_fp if os.path.exists(out_fp) else None)
                    saved_paths.append(out_fp)
                except Exception as slice_save_err:
                    print(f"ERROR saving slice {z} to {out_fp}: {slice_save_err}")
                    import traceback
                    traceback.print_exc()
                    raise

            current_app.config["PROOFREADING_MASK_EDITS"] = mask_edits
            current_app.config["PROOFREADING_EDITED_SLICES"] = set()
            if saved_paths:
                return jsonify(success=True, message=f"Mask slice(s) saved to {os.path.dirname(saved_paths[0])}")
            return jsonify(success=True, message="No slices were saved (no data available)"), 200

        # Default behavior: single file path (TIFF stack or 2D image)
        # If no edits were recorded, skip saving to avoid overwriting the original mask
        edited_global = current_app.config.get("PROOFREADING_EDITED_SLICES", set())
        if not edited_global and not mask_edits:
            return jsonify(success=True, message="No edited slices to save"), 200

        # Determine save directory and filename
        # Only create _uploads folder for actual uploads, not when reading from file paths
        if load_mode == "upload":
            image_name = session_state.get("image_name", "image")
            if not image_name or image_name == "image":
                image_name = "image"
            base_dir = _saved_masks_dir_for(image_name)
            base_name = os.path.splitext(os.path.basename(image_name))[0]
        elif img_path and os.path.exists(img_path):
            base_dir = _saved_masks_dir_for(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
        else:
            base_dir = _saved_masks_root()
            base_name = "image"

        ext = ".tif"
        if isinstance(img_path, str) and img_path:
            _, src_ext = os.path.splitext(img_path.lower())
            if src_ext in [".png", ".jpg", ".jpeg"]:
                ext = src_ext

        # For file paths (not uploads), try to find original mask file first
        # Only create new mask file if no original exists
        if load_mode == "path" and img_path and os.path.exists(img_path):
            # If mask_path is already a file, use it
            if mask_path and os.path.isfile(mask_path):
                print(f"DEBUG: Using existing mask file from session: {mask_path}")
            else:
                # Try to find original mask file using naming patterns
                base, _ = os.path.splitext(os.path.basename(img_path))
                img_dir = os.path.dirname(img_path)
                found_original = False
                
                # Try common mask naming patterns in order of preference
                for suffix in ["_pred_skeleton", "_pred", "_prediction", "_mask"]:
                    candidate = os.path.join(img_dir, f"{base}{suffix}{ext}")
                    if os.path.exists(candidate):
                        mask_path = candidate
                        found_original = True
                        print(f"DEBUG: Found original mask file: {mask_path}")
                        break
                
                # If not found, also check parent directory for prediction folders
                if not found_original and img_dir:
                    parent_dir = os.path.dirname(img_dir)
                    dir_name = os.path.basename(img_dir)
                    for pred_dir_name in [f"{dir_name}_predictions", f"{dir_name}_pred"]:
                        pred_dir = os.path.join(parent_dir, pred_dir_name)
                        if os.path.isdir(pred_dir):
                            for suffix in ["_pred_skeleton", "_pred", "_prediction", "_mask"]:
                                candidate = os.path.join(pred_dir, f"{base}{suffix}{ext}")
                                if os.path.exists(candidate):
                                    mask_path = candidate
                                    found_original = True
                                    print(f"DEBUG: Found original mask file in prediction folder: {mask_path}")
                                    break
                            if found_original:
                                break
                
                # Only create new file if no original found
                if not found_original:
                    mask_dir = _saved_masks_dir_for(img_path)
                    mask_path = os.path.join(mask_dir, f"{base_name}_mask{ext}")
                    print(f"DEBUG: No original mask found, will create new file in repo folder: {mask_path}")
        elif not mask_path:
            # For uploads or fallback cases, create new mask file
            mask_path = os.path.join(base_dir, f"{base_name}_mask{ext}")
        
        session_manager.update(mask_path=mask_path)
        current_app.config["PROOFREADING_MASK_PATH"] = mask_path

        try:
            # Convert LazyMaskLoader to numpy array if needed
            if hasattr(mask, 'get_slice') and not isinstance(mask, np.ndarray):
                # It's a LazyMaskLoader - convert to numpy array
                try:
                    mask_shape = mask.shape
                    if len(mask_shape) == 2 or mask_shape[0] == 1:
                        # 2D or single slice
                        mask_array = mask.get_slice(0)
                    else:
                        # For 3D, we need to stack all slices
                        slices = [mask.get_slice(i) for i in range(mask_shape[0])]
                        mask_array = np.stack(slices, axis=0)
                    mask = mask_array
                except Exception as e:
                    print(f"ERROR: Failed to convert LazyMaskLoader to numpy array: {e}")
                    import traceback
                    traceback.print_exc()
                    return jsonify(success=False, error=f"Cannot save LazyMaskLoader: {str(e)}"), 400
            
            # Ensure mask is a numpy array before saving
            if not isinstance(mask, np.ndarray):
                return jsonify(success=False, error=f"Mask must be a numpy array, got {type(mask)}"), 400

            mask, edits_applied = _apply_pending_mask_edits(mask, mask_edits)
            mask = _ensure_mask_matches_volume(mask, volume)
            if edits_applied:
                current_app.config["PROOFREADING_MASK"] = mask
            
            print(f"DEBUG: Saving mask with shape {mask.shape if mask is not None else 'None'}")
            print(f"DEBUG: Saving to path: {mask_path}")
            print(f"DEBUG: Mask path exists: {os.path.exists(mask_path) if mask_path else 'No path'}")
            
            # Save preserving original format if file exists
            try:
                save_mask(mask, mask_path, preserve_format_from=mask_path if os.path.exists(mask_path) else None)
                print(f"DEBUG: Save completed successfully")
            except Exception as save_err:
                print(f"ERROR in save_mask: {save_err}")
                import traceback
                traceback.print_exc()
                return jsonify(success=False, error=f"Failed to save mask: {str(save_err)}"), 500
            
            current_app.config["PROOFREADING_MASK_EDITS"] = {}
            current_app.config["PROOFREADING_EDITED_SLICES"] = set()
            current_app.config["PROOFREADING_MASK"] = mask
            return jsonify(success=True, message=f"Mask saved to {mask_path}")
        except Exception as e:
            print(f"Save error: {e}")
            import traceback
            traceback.print_exc()
            # Ensure we always return JSON, not HTML
            try:
                error_msg = str(e) if e else "Unknown error"
                return jsonify(success=False, error=f"Failed to save mask: {error_msg}"), 500
            except Exception as json_err:
                # If even jsonify fails, return a simple JSON string
                from flask import Response
                error_msg = str(e) if e else "Unknown error"
                return Response(
                    f'{{"success": false, "error": "Failed to save mask: {error_msg}"}}',
                    mimetype='application/json',
                    status=500
                )
    except BaseException as e:
        # Catch even system exits and keyboard interrupts to return JSON
        print(f"Critical error in save: {e}")
        import traceback
        traceback.print_exc()
        from flask import Response
        error_msg = str(e) if e else "Unknown critical error"
        try:
            return Response(
                f'{{"success": false, "error": "Critical error: {error_msg}"}}',
                mimetype='application/json',
                status=500
            )
        except:
            # Last resort - return minimal JSON
            return '{"success": false, "error": "Critical error occurred"}', 500, {'Content-Type': 'application/json'}

@bp.route("/api/dims", methods=["POST"])
def api_dims():
    """Get dimensions of uploaded file for standalone proofreading."""
    return jsonify_dimensions(request.files.get("file"), use_temp_file=True)
