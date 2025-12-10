"""
Error Detection Tool - Detection Routes
--------------------------------------
Handles the main error detection interface and layer annotation.
"""

import io
import base64
import numpy as np
from flask import Blueprint, render_template, request, current_app, jsonify, send_file, redirect, url_for
from PIL import Image
from backend.ai.error_detection import ErrorDetection

bp = Blueprint("detection", __name__, url_prefix="")

def register_detection_routes(app):
    app.register_blueprint(bp)

@bp.route("/detection")
def detection():
    """Main error detection interface."""
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    
    if not session_state.get("layers"):
        return redirect(url_for("landing.landing"))
    
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = 36
    
    # Get all layers and calculate pagination
    all_layers = session_state["layers"]
    total_layers = len(all_layers)
    total_pages = (total_layers + per_page - 1) // per_page  # Ceiling division
    
    # Ensure page is within valid range
    page = max(1, min(page, total_pages))
    
    # Get layers for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    layers = all_layers[start_idx:end_idx]

    # Lazy-generate overlays for current page only
    data_manager = current_app.config.get("DETECTION_DATA_MANAGER")
    if data_manager is not None:
        # Ensure each layer on this page has overlay and related fields
        for i, layer in enumerate(layers):
            if not layer.get("overlay"):
                z = int(layer.get("z", start_idx + i))
                try:
                    layer_data = data_manager.generate_layer_for_z(z)
                    # Update session layer with generated fields
                    session_manager.update_layer_fields(layer["id"], {
                        "z": z,
                        "image_slice": layer_data.get("image_slice"),
                        "mask_slice": layer_data.get("mask_slice"),
                        "overlay": layer_data.get("overlay"),
                        "has_mask": layer_data.get("has_mask"),
                        "mask_coverage": layer_data.get("mask_coverage"),
                    })
                    # Reflect updates in local variable used for render
                    layers[i].update({
                        "z": z,
                        "image_slice": layer_data.get("image_slice"),
                        "mask_slice": layer_data.get("mask_slice"),
                        "overlay": layer_data.get("overlay"),
                        "has_mask": layer_data.get("has_mask"),
                        "mask_coverage": layer_data.get("mask_coverage"),
                    })
                except Exception:
                    # If generation fails, continue without blocking UI
                    pass
    
    # Get progress statistics
    progress = session_manager.get_progress_stats()
    
    return render_template(
        "detection.html",
        layers=layers,
        progress=progress,
        mode3d=session_state.get("mode3d", False),
        image_path=session_state.get("image_path", ""),
        mask_path=session_state.get("mask_path", ""),
        pagination={
            'current_page': page,
            'total_pages': total_pages,
            'per_page': per_page,
            'total_layers': total_layers,
            'start_idx': start_idx + 1,  # 1-based indexing for display
            'end_idx': min(end_idx, total_layers)
        }
    )

@bp.route("/api/layer/<int:layer_index>")
def api_layer(layer_index):
    """Get specific layer data."""
    session_manager = current_app.session_manager
    layers = session_manager.get("layers", [])
    
    if layer_index >= len(layers):
        return jsonify({"error": "Layer index out of range"}), 404
    
    layer = layers[layer_index]
    return jsonify(layer)

@bp.route("/api/layer/<int:layer_index>/overlay")
def api_layer_overlay(layer_index):
    """Get overlay image for a specific layer."""
    session_manager = current_app.session_manager
    layers = session_manager.get("layers", [])
    
    if layer_index >= len(layers):
        return jsonify({"error": "Layer index out of range"}), 404
    
    layer = layers[layer_index]
    overlay_data = layer.get("overlay")
    
    if not overlay_data:
        return jsonify({"error": "No overlay data available"}), 404
    
    # Decode base64 and return as image
    image_data = base64.b64decode(overlay_data)
    return send_file(io.BytesIO(image_data), mimetype="image/png")

@bp.route("/api/annotate", methods=["POST"])
def api_annotate():
    """Annotate a layer with status and error information."""
    try:
        data = request.get_json()
        layer_id = data.get("layer_id")
        status = data.get("status")
        annotation = data.get("annotation", {})
        
        if not layer_id or not status:
            return jsonify({"success": False, "error": "Missing required fields"}), 400
        
        # Validate status
        valid_statuses = ["correct", "incorrect", "unsure", "unlabeled"]
        if status not in valid_statuses:
            return jsonify({"success": False, "error": f"Invalid status. Must be one of: {valid_statuses}"}), 400
        
        # Update session
        session_manager = current_app.session_manager
        session_manager.update_layer_status(layer_id, status, annotation)
        
        # Get updated progress
        progress = session_manager.get_progress_stats()
        
        
        return jsonify({
            "success": True,
            "progress": progress
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/batch_annotate", methods=["POST"])
def api_batch_annotate():
    """Annotate multiple layers at once."""
    try:
        data = request.get_json()
        annotations = data.get("annotations", [])
        
        if not annotations:
            return jsonify({"success": False, "error": "No annotations provided"}), 400
        
        session_manager = current_app.session_manager
        
        # Process each annotation
        for annotation_data in annotations:
            layer_id = annotation_data.get("layer_id")
            status = annotation_data.get("status")
            annotation = annotation_data.get("annotation", {})
            
            if layer_id and status:
                session_manager.update_layer_status(layer_id, status, annotation)
        
        # Get updated progress
        progress = session_manager.get_progress_stats()
        
        return jsonify({
            "success": True,
            "progress": progress
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/batch_update_all_pages", methods=["POST"])
def api_batch_update_all_pages():
    """Batch update all layers across all pages."""
    try:
        data = request.get_json()
        action = data.get("action")  # 'correct', 'incorrect', or 'clear'
        scope = data.get("scope")    # 'all'
        
        if not action:
            return jsonify({"success": False, "error": "Missing action"}), 400
        
        session_manager = current_app.session_manager
        session_state = session_manager.snapshot()
        all_layers = session_state.get("layers", [])
        
        # Filter to only unselected layers (skip already marked layers)
        unselected_layers = []
        for layer in all_layers:
            layer_id = layer.get("id")
            current_status = layer.get("status", "unlabeled")
            
            # Only apply to unselected layers (skip already marked layers)
            # For 'clear' action, process all layers
            # For other actions, only process unlabeled layers
            if action == "clear":
                # Clear action affects all layers
                unselected_layers.append(layer)
            elif current_status == "unlabeled":
                # Other actions only affect unlabeled layers
                unselected_layers.append(layer)
        
        # Update only unselected layers
        updated_count = 0
        for layer in unselected_layers:
            layer_id = layer.get("id")
            if layer_id:
                if action == "clear":
                    session_manager.update_layer_status(layer_id, "unlabeled")
                else:
                    session_manager.update_layer_status(layer_id, action)
                updated_count += 1
        
        # Get updated progress
        progress = session_manager.get_progress_stats()
        
        return jsonify({
            "success": True,
            "progress": progress,
            "updated_count": updated_count,
            "total_unselected": len(unselected_layers)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/progress")
def api_progress():
    """Get current progress statistics."""
    session_manager = current_app.session_manager
    progress = session_manager.get_progress_stats()
    return jsonify(progress)

@bp.route("/api/layers_by_status/<status>")
def api_layers_by_status(status):
    """Get layers filtered by status."""
    session_manager = current_app.session_manager
    layers = session_manager.get_layers_by_status(status)
    return jsonify(layers)

@bp.route("/api/error_analysis/<layer_id>")
def api_error_analysis(layer_id):
    """Get error analysis for a specific layer."""
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
        
        # Perform error analysis
        error_detector = ErrorDetection()
        analysis = error_detector.analyze_layer(layer)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/error_suggestions/<layer_id>")
def api_error_suggestions(layer_id):
    """Get error type suggestions for a layer."""
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
        
        # Get error suggestions
        error_detector = ErrorDetection()
        suggestions = error_detector.get_error_type_suggestions(layer)
        
        return jsonify({"suggestions": suggestions})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/validate_annotation", methods=["POST"])
def api_validate_annotation():
    """Validate an annotation."""
    try:
        data = request.get_json()
        annotation = data.get("annotation", {})
        
        error_detector = ErrorDetection()
        is_valid, errors = error_detector.validate_annotation(annotation)
        
        return jsonify({
            "valid": is_valid,
            "errors": errors
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
