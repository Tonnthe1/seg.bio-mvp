"""
Error Detection Tool - Landing Page Routes
------------------------------------------
Handles dataset loading and initial setup.
"""

import os
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, current_app, flash
from PIL import Image
from backend.data_manager import DataManager
from backend.session_manager import SessionManager
from backend.utils import get_image_dimensions
from datetime import datetime
import uuid

bp = Blueprint("landing", __name__, url_prefix="")


def _uploads_root() -> str:
    env_dir = os.environ.get("PROOFREADING_UPLOAD_DIR")
    if env_dir:
        base_dir = os.path.abspath(env_dir)
    else:
        base_dir = os.path.join(os.path.expanduser("~"), "proofreading_uploads")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def register_landing_routes(app):
    app.register_blueprint(bp)

@bp.route("/", methods=["GET", "POST"])
def landing():
    """Main landing page for dataset loading."""
    if request.method == "POST":
        return handle_dataset_load()
    
    # Get current session state
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    layers = session_state.get("layers", [])
    
    return render_template("landing.html", layers=layers)

def handle_dataset_load():
    """Handle dataset loading from form submission."""
    try:
        load_mode = request.form.get("load_mode", "path")
        
        if load_mode == "upload":
            return handle_upload()
        else:
            return handle_path_load()
            
    except Exception as e:
        flash(f"Error loading dataset: {str(e)}", "error")
        return redirect(url_for("landing.landing"))

def handle_upload():
    """Handle file upload."""
    image_file = request.files.get("image_file")
    mask_file = request.files.get("mask_file")
    
    if not image_file or image_file.filename == "":
        flash("Please select an image file", "error")
        return redirect(url_for("landing.landing"))
    
    upload_dir = _uploads_root()
    
    # Save uploaded files
    image_path = os.path.join(upload_dir, image_file.filename)
    image_file.save(image_path)
    
    mask_path = None
    if mask_file and mask_file.filename != "":
        mask_path = os.path.join(upload_dir, mask_file.filename)
        mask_file.save(mask_path)
    
    return load_dataset(image_path, mask_path, "upload")

def handle_path_load():
    """Handle path-based loading."""
    image_path = request.form.get("image_path", "").strip()
    mask_path = request.form.get("mask_path", "").strip()
    
    if not image_path:
        flash("Please provide an image path", "error")
        return redirect(url_for("landing.landing"))
    
    if not os.path.exists(image_path):
        flash(f"Image file not found: {image_path}", "error")
        return redirect(url_for("landing.landing"))
    
    if mask_path and not os.path.exists(mask_path):
        flash(f"Mask file not found: {mask_path}", "error")
        return redirect(url_for("landing.landing"))
    
    return load_dataset(image_path, mask_path if mask_path else None, "path")

def load_dataset(image_path: str, mask_path: str = None, load_mode: str = "path"):
    """Load dataset and redirect to detection interface."""
    try:
        # Initialize data manager
        data_manager = DataManager()
        
        # Load image
        volume, volume_info = data_manager.load_image(image_path)
        
        # Load mask if provided
        mask = None
        mask_info = {}
        if mask_path:
            mask, mask_info = data_manager.load_mask(mask_path)
        
        # Validate compatibility
        if not data_manager.validate_data_compatibility():
            flash("Image and mask dimensions are incompatible", "error")
            return redirect(url_for("landing.landing"))
        
        # Generate layers
        layers = data_manager.generate_all_layers()
        
        # Update session
        session_manager = current_app.session_manager
        session_manager.update(
            mode3d=volume_info["is_3d"],
            image_path=image_path,
            mask_path=mask_path,
            load_mode=load_mode,
            session_id=str(uuid.uuid4()),
            created_at=datetime.now().isoformat()
        )
        session_manager.set_image_info(image_path, load_mode)
        
        # Add layers to session
        for layer_data in layers:
            layer_id = session_manager.add_layer(layer_data)
        
        # Store data in app config for API access
        current_app.config["LANDING_VOLUME"] = volume
        current_app.config["LANDING_MASK"] = mask
        current_app.config["DATA_MANAGER"] = data_manager
        
        flash(f"Dataset loaded successfully! Found {len(layers)} layer(s)", "success")
        return redirect(url_for("detection.detection"))
        
    except Exception as e:
        flash(f"Error loading dataset: {str(e)}", "error")
        return redirect(url_for("landing.landing"))

@bp.route("/api/dims", methods=["POST"])
def api_dims():
    """API endpoint to get image dimensions."""
    result = get_image_dimensions(request.files.get("file"), use_temp_file=False)
    if "error" in result:
        status_code = 400 if result.get("error") == "No file provided" else 500
        return result, status_code
    return result

@bp.route("/reset")
def reset():
    """Reset session and return to landing page."""
    current_app.session_manager.reset_session()
    
    # Clear all workflow config variables
    # Landing page
    current_app.config.pop("LANDING_VOLUME", None)
    current_app.config.pop("LANDING_MASK", None)
    current_app.config.pop("DATA_MANAGER", None)
    
    # Error Detection workflow
    current_app.config.pop("DETECTION_VOLUME", None)
    current_app.config.pop("DETECTION_MASK", None)
    current_app.config.pop("DETECTION_IMAGE_PATH", None)
    current_app.config.pop("DETECTION_MASK_PATH", None)
    current_app.config.pop("DETECTION_DATA_MANAGER", None)
    
    # Standalone Proofreading workflow
    current_app.config.pop("PROOFREADING_VOLUME", None)
    current_app.config.pop("PROOFREADING_MASK", None)
    current_app.config.pop("PROOFREADING_IMAGE_PATH", None)
    current_app.config.pop("PROOFREADING_MASK_PATH", None)
    
    # Integrated Proofreading workflow
    current_app.config.pop("INTEGRATED_VOLUME", None)
    current_app.config.pop("INTEGRATED_MASK", None)
    current_app.config.pop("CURRENT_SLICE_INDEX", None)
    current_app.config.pop("CURRENT_LAYER_ID", None)
    
    flash("Session reset successfully", "info")
    return redirect(url_for("landing.landing"))
