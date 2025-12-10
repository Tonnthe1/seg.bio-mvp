"""
Error Handling Tool - Proofreading Routes
----------------------------------------
Handles the integrated proofreading interface for correcting incorrect layers.
"""

import os
import io
import base64
from typing import Optional, Tuple

import numpy as np
from flask import Blueprint, render_template, request, current_app, jsonify, send_file, redirect, url_for
from PIL import Image
from backend.volume_manager import list_images_for_path, load_image_or_stack, load_mask_like, save_mask, stack_2d_images
from backend.data_manager import _normalize_image_slice_to_rgb, _mask_slice_to_rgba

try:
    from backend.ai.sam_utils import apply_sam_segmentation
except ImportError:  # pragma: no cover
    apply_sam_segmentation = None


def _uploads_root() -> str:
    env_dir = os.environ.get("PROOFREADING_UPLOAD_DIR")
    if env_dir:
        base_dir = os.path.abspath(env_dir)
    else:
        base_dir = os.path.join(os.path.expanduser("~"), "proofreading_uploads")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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


def _extract_mask_slice(mask: np.ndarray, slice_idx: int) -> np.ndarray:
    if mask is None:
        return None
    if isinstance(mask, np.ndarray) and mask.ndim == 3:
        idx = max(0, min(slice_idx, mask.shape[0] - 1))
        return mask[idx]
    return mask


def _ensure_mask_matches_volume(mask_array: Optional[np.ndarray], volume) -> Optional[np.ndarray]:
    if isinstance(mask_array, np.ndarray):
        vol_ndim = getattr(volume, "ndim", None)
        if mask_array.ndim == 3 and mask_array.shape[0] == 1 and vol_ndim == 2:
            return np.squeeze(mask_array, axis=0)
    return mask_array


def _ensure_mask_matches_volume(mask_array: Optional[np.ndarray], volume) -> Optional[np.ndarray]:
    if isinstance(mask_array, np.ndarray):
        vol_ndim = getattr(volume, "ndim", None)
        if mask_array.ndim == 3 and mask_array.shape[0] == 1 and vol_ndim == 2:
            return np.squeeze(mask_array, axis=0)
    return mask_array
def _cached_detection_image_files():
    files = current_app.config.get("DETECTION_IMAGE_FILES")
    if files:
        return files
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    image_path = session_state.get("image_path", "")
    files = list_images_for_path(image_path)
    current_app.config["DETECTION_IMAGE_FILES"] = files
    return files


def _resolve_detection_mask(img_fp, idx):
    mask_files = current_app.config.get("DETECTION_MASK_FILES")
    if mask_files and idx < len(mask_files):
        mask_fp = mask_files[idx]
        if mask_fp:
            return mask_fp

    def build_exts(ext):
        exts = [ext.lower()]
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
        if base_name.endswith("_0000"):
            trimmed = base_name[:-5]
            for e in extensions:
                cand = os.path.join(dir_path, f"{trimmed}{e}")
                if os.path.exists(cand):
                    return cand
        return None

    if not img_fp:
        return None

    base, ext = os.path.splitext(os.path.basename(img_fp))
    extensions = build_exts(ext)
    session_state = current_app.session_manager.snapshot()
    mask_path = session_state.get("mask_path", "")
    search_dirs = []
    if mask_path and os.path.isdir(mask_path):
        search_dirs.append(mask_path)
    img_dir = os.path.dirname(img_fp)
    if img_dir:
        search_dirs.append(img_dir)

    for d in search_dirs:
        result = try_dir(d, base, extensions)
        if result:
            return result

    if mask_path and os.path.isfile(mask_path):
        return mask_path
    return None
from backend.utils import jsonify_dimensions

bp = Blueprint("proofreading", __name__, url_prefix="")

def register_proofreading_routes(app):
    app.register_blueprint(bp)


def _prepare_layer_context(layer_id: str, include_layers: bool = True):
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()

    if not session_state.get("layers"):
        raise RuntimeError("No detection session is currently active.")

    incorrect_layers = session_manager.get_incorrect_layers() or []
    if not incorrect_layers:
        raise RuntimeError("No incorrect layers found for proofreading.")

    current_layer = None
    layer_index = -1
    for idx, layer in enumerate(incorrect_layers):
        if layer["id"] == layer_id:
            current_layer = layer
            layer_index = idx
            break

    if current_layer is None:
        raise ValueError("Requested layer is not in the proofreading queue.")

    slice_idx = current_layer.get("z", 0)
    data_manager = current_app.config.get("DETECTION_DATA_MANAGER")
    volume = current_app.config.get("DETECTION_VOLUME")
    mask = current_app.config.get("DETECTION_MASK")

    if data_manager is not None:
        image_slice, mask_slice = data_manager.get_slice(slice_idx)
    else:
        if volume is None:
            ipath = session_state.get("image_path", "")
            volume = stack_2d_images(ipath) if isinstance(ipath, list) else load_image_or_stack(ipath)
        if mask is None:
            mask = load_mask_like(session_state.get("mask_path"), volume)

        if getattr(volume, "ndim", 2) == 3:
            if slice_idx >= volume.shape[0]:
                raise ValueError(
                    f"Slice index {slice_idx} out of range for volume with {volume.shape[0]} slices"
                )
            image_slice = volume[slice_idx]
            if mask is not None and getattr(mask, "ndim", 0) == 3:
                mask_slice = mask[slice_idx]
            else:
                mask_slice = mask
        else:
            image_slice = volume
            mask_slice = mask

    current_app.config["INTEGRATED_VOLUME"] = image_slice
    current_app.config["INTEGRATED_MASK"] = mask_slice
    current_app.config["CURRENT_SLICE_INDEX"] = slice_idx
    current_app.config["CURRENT_LAYER_ID"] = layer_id

    context = {
        "layers": session_state.get("layers", []) if include_layers else [],
        "incorrect_layers": incorrect_layers,
        "current_layer": current_layer,
        "layer_index": layer_index,
        "total_incorrect": len(incorrect_layers),
        "progress": session_manager.get_progress_stats(),
        "mode3d": False,
        "image_path": session_state.get("image_path", ""),
        "mask_path": session_state.get("mask_path", ""),
        "num_slices": 1,
        "volume_shape": getattr(image_slice, "shape", None),
        "mask_shape": getattr(mask_slice, "shape", None) if mask_slice is not None else None,
        "slice_index": slice_idx,
    }
    return context

@bp.route("/proofreading")
def proofreading():
    """Layer selection page for incorrect layers."""
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    
    if not session_state.get("layers"):
        return redirect(url_for("landing.landing"))
    
    # Get incorrect layers for proofreading
    incorrect_layers = session_manager.get_incorrect_layers()
    
    if not incorrect_layers:
        all_layers = session_state.get("layers", [])
        return render_template(
            "proofreading_selection.html",
            layers=all_layers,  # Pass all layers for navigation
            incorrect_layers=[],  # No incorrect layers
            progress=session_manager.get_progress_stats(),
            mode3d=session_state.get("mode3d", False),
            image_path=session_state.get("image_path", ""),
            mask_path=session_state.get("mask_path", ""),
            warning="No incorrect layers found for proofreading."
        )
    
    # Pass all layers to ensure navigation is visible
    all_layers = session_state.get("layers", [])
    
    return render_template(
        "proofreading_selection.html",
        layers=all_layers,  # Pass all layers for navigation
        incorrect_layers=incorrect_layers,  # Pass incorrect layers for selection
        progress=session_manager.get_progress_stats(),
        mode3d=session_state.get("mode3d", False),
        image_path=session_state.get("image_path", ""),
        mask_path=session_state.get("mask_path", "")
    )

@bp.route("/proofreading/edit/<layer_id>")
def proofreading_edit(layer_id):
    """Proofreading editor for a specific incorrect layer."""
    try:
        context = _prepare_layer_context(layer_id, include_layers=True)
        return render_template("proofreading.html", **context)
    except Exception as e:
        session_manager = current_app.session_manager
        session_state = session_manager.snapshot()
        return render_template(
            "proofreading.html",
            layers=session_state.get("layers", []),
            incorrect_layers=session_manager.get_incorrect_layers(),
            current_layer=None,
            layer_index=-1,
            total_incorrect=0,
            progress=session_manager.get_progress_stats(),
            mode3d=False,
            image_path=session_state.get("image_path", ""),
            mask_path=session_state.get("mask_path", ""),
            num_slices=1,
            volume_shape=None,
            mask_shape=None,
            slice_index=0,
            warning=f"Error loading data for proofreading: {e}"
        )

@bp.route("/api/proofreading_layer/<layer_id>")
def api_proofreading_layer(layer_id):
    """Get layer data for proofreading interface."""
    try:
        session_manager = current_app.session_manager
        layers = session_manager.get("layers", [])
        
        # Find the layer
        layer = None
        for l in layers:
            if l["id"] == layer_id:
                layer = l
                break
        
        if not layer:
            return jsonify({"error": "Layer not found"}), 404
        
        # Get volume and mask
        volume = current_app.config.get("INTEGRATED_VOLUME")
        mask = current_app.config.get("INTEGRATED_MASK")
        
        if volume is None or mask is None:
            return jsonify({"error": "Volume or mask not loaded"}), 400
        
        # Get layer slice
        if volume.ndim == 3:
            slice_idx = layer.get("slice_index", 0)
            image_slice = volume[slice_idx]
            mask_slice = mask[slice_idx] if mask.ndim == 3 else mask
        else:
            image_slice = volume
            mask_slice = mask
        
        # Convert to base64 for display
        image_pil = Image.fromarray(image_slice)
        mask_pil = Image.fromarray(mask_slice * 255)
        
        # Create overlay
        overlay = Image.blend(image_pil.convert('RGB'), mask_pil.convert('RGB'), 0.3)
        
        # Convert to base64
        def image_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            "layer": layer,
            "image_base64": image_to_base64(image_pil),
            "mask_base64": image_to_base64(mask_pil),
            "overlay_base64": image_to_base64(overlay),
            "slice_index": layer.get("slice_index", 0),
            "total_slices": volume.shape[0] if volume.ndim == 3 else 1
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/proofreading/api/load_layer", methods=["POST"])
def api_load_proofreading_layer():
    """AJAX endpoint to switch proofreading layers without a full page refresh."""
    data = request.get_json(force=True) or {}
    layer_id = data.get("layer_id")
    if not layer_id:
        return jsonify({"success": False, "error": "Missing layer_id"}), 400
    try:
        context = _prepare_layer_context(layer_id, include_layers=False)
        response = {
            "success": True,
            "current_layer": context["current_layer"],
            "layer_index": context["layer_index"],
            "total_incorrect": context["total_incorrect"],
            "volume_shape": context["volume_shape"],
            "mask_shape": context["mask_shape"],
            "slice_index": context["slice_index"],
            "incorrect_layers": context["incorrect_layers"],
            "progress": context["progress"],
        }
        return jsonify(response)
    except Exception as exc:
        current_app.logger.exception("Failed to load proofreading layer via API")
        return jsonify({"success": False, "error": str(exc)}), 400

@bp.route("/api/save_proofreading", methods=["POST"])
def api_save_proofreading():
    """Save proofreading changes for a layer."""
    try:
        data = request.get_json()
        layer_id = data.get("layer_id")
        mask_data = data.get("mask_data")  # Base64 encoded mask
        
        if not layer_id or not mask_data:
            return jsonify({"success": False, "error": "Missing layer_id or mask_data"}), 400
        
        # Decode mask data
        mask_bytes = base64.b64decode(mask_data.split(',')[1])
        mask_pil = Image.open(io.BytesIO(mask_bytes))
        mask_array = np.array(mask_pil.convert('L')) > 128
        
        # Get current volume and mask
        volume = current_app.config.get("INTEGRATED_VOLUME")
        mask = current_app.config.get("INTEGRATED_MASK")
        
        if volume is None or mask is None:
            return jsonify({"success": False, "error": "Volume or mask not loaded"}), 400
        
        # Update mask for this layer
        session_manager = current_app.session_manager
        layers = session_manager.get("layers", [])
        
        for layer in layers:
            if layer["id"] == layer_id:
                slice_idx = layer.get("slice_index", 0)
                if volume.ndim == 3:
                    mask[slice_idx] = mask_array.astype(np.uint8)
                else:
                    mask[:] = mask_array.astype(np.uint8)
                break
        
        # Update mask in app config
        current_app.config["INTEGRATED_MASK"] = mask
        
        # Save mask to file
        session_state = session_manager.snapshot()
        mask_path = session_state.get("mask_path", "")
        if mask_path:
            save_mask(mask, mask_path)
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/mark_corrected", methods=["POST"])
def api_mark_corrected():
    """Mark a layer as corrected after proofreading."""
    try:
        data = request.get_json()
        layer_id = data.get("layer_id")
        
        if not layer_id:
            return jsonify({"success": False, "error": "Missing layer_id"}), 400
        
        # Update layer status to correct
        session_manager = current_app.session_manager
        session_manager.update_layer_status(layer_id, "correct", {"proofread": True})
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/slice/<int:slice_idx>")
def api_slice(slice_idx):
    """Get image slice for proofreading."""
    try:
        # For incorrect layer editing, we only have one slice
        volume = current_app.config.get("INTEGRATED_VOLUME")
        if volume is None:
            return jsonify({"error": "Volume not loaded"}), 400
        
        # Since we're editing a specific incorrect layer, we only have one slice
        if slice_idx != 0:
            return jsonify({"error": "Only slice 0 available for incorrect layer editing"}), 400
        
        # Convert to RGB with consistent normalization (using shared helper)
        rgb = _normalize_image_slice_to_rgb(volume)
        
        # Convert to PIL Image and return as PNG
        from PIL import Image
        bio = io.BytesIO()
        Image.fromarray(rgb).save(bio, format="PNG")
        bio.seek(0)
        return send_file(bio, mimetype="image/png")
        
    except Exception as e:
        print(f"DEBUG: Error in api_slice: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@bp.route("/api/mask/<int:slice_idx>")
def api_mask(slice_idx):
    """Get mask slice for proofreading."""
    try:
        # For incorrect layer editing, we only have one slice
        mask = current_app.config.get("INTEGRATED_MASK")
        if mask is None:
            return jsonify({"error": "Mask not loaded"}), 400
        
        # Since we're editing a specific incorrect layer, we only have one slice
        if slice_idx != 0:
            return jsonify({"error": "Only slice 0 available for incorrect layer editing"}), 400
        
        # Get mask slice (already a single slice for integrated)
        sl = np.asarray(mask)
        
        print(f"DEBUG: Processing mask slice shape: {sl.shape}")
        print(f"DEBUG: Mask slice dtype: {sl.dtype}, min: {sl.min()}, max: {sl.max()}, non-zero: {np.count_nonzero(sl)}")
        
        # Convert mask to RGBA using shared helper
        rgba = _mask_slice_to_rgba(sl, opacity=230)
        
        # Convert to PIL Image and return as PNG
        from PIL import Image
        bio = io.BytesIO()
        Image.fromarray(rgba).save(bio, format="PNG")
        bio.seek(0)
        return send_file(bio, mimetype="image/png")
        
    except Exception as e:
        print(f"DEBUG: Error in api_mask: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@bp.route("/api/names_current")
def api_names_current():
    """Return the current image filename and paired mask filename for the incorrect-layer proofreading view."""
    try:
        slice_idx = current_app.config.get("CURRENT_SLICE_INDEX", 0)

        images = _cached_detection_image_files()
        if not images:
            session_state = current_app.session_manager.snapshot()
            image_path = session_state.get("image_path", "")
            img_fp = image_path if isinstance(image_path, str) else None
        else:
            img_fp = images[slice_idx] if slice_idx < len(images) else images[-1]

        img_name = os.path.basename(img_fp) if img_fp else None
        mask_fp = _resolve_detection_mask(img_fp, slice_idx)
        mask_name = os.path.basename(mask_fp) if mask_fp else None

        return jsonify(image=img_name, mask=mask_name)
    except Exception as e:
        return jsonify(error=str(e)), 500

@bp.route("/api/mask/update", methods=["POST"])
def api_mask_update():
    """Update mask with edited slices (matching standalone logic)."""
    try:
        data = request.get_json(force=True)
        mask = current_app.config.get("INTEGRATED_MASK")
        volume = current_app.config.get("INTEGRATED_VOLUME")
        
        if mask is None and volume is not None:
            if volume.ndim == 2:
                mask = np.zeros_like(volume, dtype=np.uint8)
            elif volume.ndim == 3:
                mask = np.zeros_like(volume, dtype=np.uint8)
            current_app.config["INTEGRATED_MASK"] = mask
        elif mask is None:
            return jsonify(success=False, error="No mask or image loaded"), 404
        
        # For integrated proofreading, we only work with slice 0
        def store_edit(arr: np.ndarray):
            # Resize if needed to match mask dimensions
            if arr.shape != mask.shape:
                import cv2
                resized = cv2.resize(arr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask[:] = resized
            else:
                mask[:] = arr.astype(np.uint8)
            current_app.config["INTEGRATED_MASK"] = mask
        
        # --- Batch updates ---
        if "full_batch" in data:
            for item in data["full_batch"]:
                z = int(item.get("z", 0))
                if z != 0:
                    continue  # Only allow slice 0 for integrated proofreading
                png_bytes = base64.b64decode(item["png"])
                img = Image.open(io.BytesIO(png_bytes)).convert("L")
                arr = (np.array(img) > 127).astype(np.uint8)
                store_edit(arr)
            print(f"✅ Batch updated slice(s)")
            return jsonify(success=True)
        
        # --- Single slice update ---
        if "full_png" in data:
            z = int(data.get("z", 0))
            if z != 0:
                return jsonify(success=False, error="Only slice 0 available for incorrect layer editing"), 400
            png_bytes = base64.b64decode(data["full_png"])
            img = Image.open(io.BytesIO(png_bytes)).convert("L")
            arr = (np.array(img) > 127).astype(np.uint8)
            store_edit(arr)
            print(f"✅ Replaced full slice {z}")
            return jsonify(success=True)
        
        return jsonify(success=False, error="Invalid data"), 400
        
    except Exception as e:
        print(f"DEBUG: Error in api_mask_update: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/save", methods=["POST"])
def api_save():
    """Save mask to file."""
    try:
        mask = current_app.config.get("INTEGRATED_MASK")
        volume = current_app.config.get("INTEGRATED_VOLUME")
        slice_idx = current_app.config.get("CURRENT_SLICE_INDEX", 0)
        
        if mask is None and volume is not None:
            if volume.ndim == 2:
                mask = np.zeros_like(volume, dtype=np.uint8)
            elif volume.ndim == 3:
                mask = np.zeros_like(volume, dtype=np.uint8)
            current_app.config["INTEGRATED_MASK"] = mask
        elif mask is None:
            return jsonify({"success": False, "error": "No mask or image loaded"}), 400

        if isinstance(mask, np.ndarray):
            mask = _ensure_mask_matches_volume(mask, volume)
            current_app.config["INTEGRATED_MASK"] = mask
        
        # Get session data
        session_manager = current_app.session_manager
        session_state = session_manager.snapshot()
        mask_path = session_state.get("mask_path", "")
        image_path = session_state.get("image_path", "")
        load_mode = session_state.get("load_mode", "path")

        # If working from a folder/glob/multi-file
        src_is_dir = bool(image_path and isinstance(image_path, str) and os.path.isdir(image_path))
        src_is_glob = bool(image_path and isinstance(image_path, str) and any(ch in image_path for ch in ['*','?','[']) and not os.path.exists(image_path))
        src_is_list = isinstance(image_path, list)
        cached_image_files = current_app.config.get("DETECTION_IMAGE_FILES") or []
        cached_has_multiple = len(cached_image_files) > 1
        dataset_has_multiple_sources = src_is_dir or src_is_glob or src_is_list or cached_has_multiple

        def _resolve_source_files():
            if cached_image_files:
                return cached_image_files
            if src_is_list and isinstance(image_path, list):
                return list(image_path)
            if src_is_dir and isinstance(image_path, str):
                selected = []
                for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                    selected = sorted([
                        os.path.join(image_path, f)
                        for f in os.listdir(image_path)
                        if f.lower().endswith(ext)
                    ])
                    if selected:
                        break
                return selected
            if src_is_glob and isinstance(image_path, str):
                import glob as _glob
                return sorted(_glob.glob(image_path))
            if isinstance(image_path, str) and image_path:
                return [image_path]
            return []

        def _mask_dir_anchor():
            if src_is_dir and isinstance(image_path, str):
                return image_path
            if cached_image_files:
                first = cached_image_files[0]
                parent = os.path.dirname(first)
                return parent if parent else first
            if src_is_list and isinstance(image_path, list) and image_path:
                first = image_path[0]
                parent = os.path.dirname(first)
                return parent if parent else first
            if isinstance(image_path, str) and os.path.isfile(image_path):
                parent = os.path.dirname(image_path)
                return parent if parent else image_path
            return image_path

        source_files_for_naming = _resolve_source_files()
        mask_files = current_app.config.get("PROOFREADING_MASK_FILES") or []

        if dataset_has_multiple_sources:
            src_fp = None
            try:
                if source_files_for_naming:
                    src_fp = source_files_for_naming[slice_idx] if slice_idx < len(source_files_for_naming) else source_files_for_naming[-1]
            except Exception:
                src_fp = None

            if src_fp is None and source_files_for_naming:
                src_fp = source_files_for_naming[-1]

            if src_fp is None:
                base = f"slice_{slice_idx:04d}"
                ext_out = ".tif"
            else:
                base = os.path.splitext(os.path.basename(src_fp))[0]
                ext_out = os.path.splitext(src_fp)[-1].lower()
                if ext_out not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                    ext_out = ".tif"

            out_fp = None
            if mask_files and slice_idx < len(mask_files) and mask_files[slice_idx]:
                out_fp = mask_files[slice_idx]
            elif mask_path and os.path.isdir(mask_path):
                out_fp = os.path.join(mask_path, f"{base}_mask{ext_out}")
            else:
                mask_dir_source = _mask_dir_anchor()
                mask_dir = _saved_masks_dir_for(mask_dir_source)
                out_fp = os.path.join(mask_dir, f"{base}_mask{ext_out}")
                session_manager.update(mask_path=mask_dir)

            os.makedirs(os.path.dirname(out_fp), exist_ok=True)

            mask_slice = _extract_mask_slice(mask, slice_idx)
            save_mask(mask_slice, out_fp, preserve_format_from=out_fp if os.path.exists(out_fp) else None)
            return jsonify({"success": True, "message": f"Mask slice saved to {out_fp}"})

        # Generate mask path if not provided (file-based)
        # Only create shared uploads folder for actual uploads, not when reading from file paths
        if not mask_path:
            if dataset_has_multiple_sources:
                mask_path = _saved_masks_dir_for(_mask_dir_anchor())
            else:
                if load_mode == "upload":
                    image_name = session_state.get("image_name", "image")
                    if not image_name or image_name == "image":
                        image_name = "image"
                    base_dir = _saved_masks_dir_for(image_name)
                    original_base = os.path.splitext(os.path.basename(image_name))[0]
                elif image_path and os.path.exists(image_path):
                    base_dir = _saved_masks_dir_for(image_path)
                    original_base = os.path.splitext(os.path.basename(image_path))[0]
                else:
                    base_dir = _saved_masks_root()
                    original_base = "image"

                # Detect extension
                ext = ".tif"
                if isinstance(image_path, str) and image_path:
                    _, src_ext = os.path.splitext(image_path.lower())
                    if src_ext in [".png", ".jpg", ".jpeg"]:
                        ext = src_ext

                # For file paths (not uploads), try to find original mask file first
                # Only create new mask file if no original exists
                if load_mode == "path" and image_path and os.path.exists(image_path):
                    # If mask_path is already a file, use it
                    if mask_path and os.path.isfile(mask_path):
                        print(f"DEBUG: Using existing mask file from session: {mask_path}")
                    else:
                        # Try to find original mask file using naming patterns
                        base, _ = os.path.splitext(os.path.basename(image_path))
                        img_dir = os.path.dirname(image_path)
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
                            mask_dir = _saved_masks_dir_for(image_path)
                            mask_path = os.path.join(mask_dir, f"{original_base}_mask{ext}")
                            print(f"DEBUG: No original mask found, will create new file in repo folder: {mask_path}")
                elif not mask_path:
                    # For uploads or fallback cases, create new mask file
                    mask_path = os.path.join(base_dir, f"{original_base}_mask{ext}")
            
            session_manager.update(mask_path=mask_path)
        
        try:
            full_volume = None
            full_mask = None
            if mask_path and os.path.isfile(mask_path):
                full_volume = load_image_or_stack(image_path)
                full_mask = load_mask_like(mask_path, full_volume)
                full_mask = _ensure_mask_matches_volume(full_mask, full_volume)

            dataset_base = "image"
            if source_files_for_naming:
                src_for_base = source_files_for_naming[slice_idx] if slice_idx < len(source_files_for_naming) else source_files_for_naming[-1]
                dataset_base = os.path.splitext(os.path.basename(src_for_base))[0]
            elif isinstance(image_path, list) and image_path:
                idx = min(slice_idx, len(image_path) - 1)
                dataset_base = os.path.splitext(os.path.basename(image_path[idx]))[0]
            elif isinstance(image_path, str) and image_path:
                dataset_base = os.path.splitext(os.path.basename(image_path))[0]

            if full_mask is not None:
                if full_mask.ndim == 3 and slice_idx < full_mask.shape[0]:
                    full_mask[slice_idx] = mask
                elif full_mask.ndim == 2:
                    full_mask[:] = mask

                is_stack = full_mask.ndim >= 3 and getattr(full_volume, "ndim", 2) >= 3
                if is_stack:
                    slice_mask = _extract_mask_slice(full_mask, slice_idx)
                    dir_source = _mask_dir_anchor()
                    target_dir = _saved_masks_dir_for(dir_source)
                    out_name = f"{dataset_base}_z{slice_idx:04d}.tif"
                    out_path = os.path.join(target_dir, out_name)
                    save_mask(
                        slice_mask,
                        out_path,
                        preserve_format_from=out_path if os.path.exists(out_path) else None,
                    )
                    session_manager.update(mask_path=target_dir)
                    current_app.config["PROOFREADING_MASK"] = full_mask
                    current_app.config["PROOFREADING_MASK_EDITS"] = {}
                    current_app.config["PROOFREADING_EDITED_SLICES"] = set()

                    detection_mask = current_app.config.get("DETECTION_MASK")
                    if detection_mask is not None:
                        if detection_mask.ndim == 3 and slice_idx < detection_mask.shape[0]:
                            detection_mask[slice_idx] = slice_mask
                        elif detection_mask.ndim == 2:
                            detection_mask[:] = slice_mask
                        current_app.config["DETECTION_MASK"] = detection_mask

                    return jsonify({"success": True, "message": f"Mask slice {slice_idx} saved to {out_path}"})
                else:
                    save_mask(full_mask, mask_path, preserve_format_from=mask_path if mask_path and os.path.exists(mask_path) else None)
                    detection_mask = current_app.config.get("DETECTION_MASK")
                    if detection_mask is not None:
                        replacement = _extract_mask_slice(full_mask, slice_idx) if full_mask.ndim >= 3 else full_mask
                        if detection_mask.ndim == 3 and slice_idx < detection_mask.shape[0]:
                            detection_mask[slice_idx] = replacement
                        elif detection_mask.ndim == 2:
                            detection_mask[:] = replacement
                        current_app.config["DETECTION_MASK"] = detection_mask

                    current_app.config["PROOFREADING_MASK"] = full_mask
                    current_app.config["PROOFREADING_MASK_EDITS"] = {}
                    current_app.config["PROOFREADING_EDITED_SLICES"] = set()
                    return jsonify({"success": True, "message": f"Mask slice {slice_idx} saved to {mask_path}"})
            else:
                dir_source = _mask_dir_anchor()
                target_dir = _saved_masks_dir_for(dir_source)
                slice_mask = _extract_mask_slice(mask, slice_idx) if isinstance(mask, np.ndarray) and mask.ndim >= 3 else mask
                out_name = f"{dataset_base}_z{slice_idx:04d}.tif"
                out_path = os.path.join(target_dir, out_name)
                save_mask(slice_mask, out_path, preserve_format_from=out_path if os.path.exists(out_path) else None)
                session_manager.update(mask_path=target_dir)
                return jsonify({"success": True, "message": f"New mask saved to {out_path}"})

        except Exception as e:
            print(f"Save error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({"success": False, "error": f"Failed to save mask: {str(e)}"}), 500
        
    except Exception as e:
        print(f"Save error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Failed to save mask: {str(e)}"}), 500

@bp.route("/api/sam/segment_current", methods=["POST"])
def sam_segment_current():
    """SAM segmentation endpoint for integrated proofreading."""
    try:
        if apply_sam_segmentation is None:
            return jsonify({"error": "SAM integration not available in this build"}), 501

        data = request.get_json()
        points = data.get("points", [])
        point_labels = data.get("point_labels", [])

        if not points or not point_labels:
            return jsonify({"error": "Points and point_labels are required"}), 400

        volume = current_app.config.get("INTEGRATED_VOLUME")
        if volume is None:
            return jsonify({"error": "No volume loaded"}), 400

        image_slice = volume if volume.ndim == 2 else volume[0]

        try:
            from backend.ai.sam_init import initialize_sam_model
            model = initialize_sam_model()
        except ImportError as e:
            return jsonify({"error": f"SAM not available: {str(e)}"}), 500

        mask = current_app.config.get("INTEGRATED_MASK")
        current_mask_slice = None
        if mask is not None:
            current_mask_slice = mask if mask.ndim == 2 else mask[0]

        result_mask, mask_b64 = apply_sam_segmentation(
            model,
            image_slice,
            current_mask_slice,
            points,
            point_labels,
            slice_index=0,
        )

        if mask is None:
            mask = result_mask.copy()
        else:
            if mask.ndim == 2:
                mask = result_mask.copy()
            else:
                mask[0] = result_mask
        current_app.config["INTEGRATED_MASK"] = mask

        return jsonify({"success": True, "mask": mask_b64})
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@bp.route("/api/dims", methods=["POST"])
def api_dims():
    """Get dimensions of uploaded file."""
    return jsonify_dimensions(request.files.get("file"), use_temp_file=False)