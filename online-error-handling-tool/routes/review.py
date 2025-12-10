"""
Error Detection Tool - Review Routes
------------------------------------
Handles the review and categorization interface for annotated layers.
"""

from flask import Blueprint, render_template, request, current_app, jsonify, redirect, url_for

bp = Blueprint("review", __name__, url_prefix="")

def register_review_routes(app):
    app.register_blueprint(bp)

@bp.route("/review")
def review():
    """Review and categorize annotated layers."""
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    
    if not session_state.get("layers"):
        return redirect(url_for("landing.landing"))
    
    # Get all layers for review
    all_layers = session_state.get("layers", [])
    incorrect_layers = session_manager.get_incorrect_layers()
    unsure_layers = session_manager.get_unsure_layers()
    
    # Get progress statistics
    progress = session_manager.get_progress_stats()
    
    return render_template(
        "review.html",
        layers=all_layers,  # Show all layers by default
        incorrect_layers=incorrect_layers,
        unsure_layers=unsure_layers,
        progress=progress,
        mode3d=session_state.get("mode3d", False)
    )

@bp.route("/api/error_statistics")
def api_error_statistics():
    """Get detailed error statistics."""
    try:
        session_manager = current_app.session_manager
        progress = session_manager.get_progress_stats()
        
        return jsonify(progress)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/review_statistics")
def api_review_statistics():
    """Get review statistics for consistency check."""
    try:
        session_manager = current_app.session_manager
        progress = session_manager.get_progress_stats()
        
        return jsonify(progress)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/proofreading_queue")
def api_proofreading_queue():
    """Get proofreading queue for incorrect layers."""
    try:
        session_manager = current_app.session_manager
        incorrect_layers = session_manager.get_incorrect_layers()
        
        return jsonify(incorrect_layers)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/update_annotation", methods=["POST"])
def api_update_annotation():
    """Update annotation for a layer."""
    try:
        data = request.get_json()
        layer_id = data.get("layer_id")
        annotation = data.get("annotation", {})
        
        if not layer_id:
            return jsonify({"success": False, "error": "Missing layer_id"}), 400
        
        # Basic validation
        if not annotation.get("status") in ["correct", "incorrect", "unsure", "unlabeled"]:
            return jsonify({"success": False, "error": "Invalid status"}), 400
        
        # Update session
        session_manager = current_app.session_manager
        session_manager.update_layer_status(layer_id, annotation.get("status", "unlabeled"), annotation)
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/batch_update_annotations", methods=["POST"])
def api_batch_update_annotations():
    """Update multiple annotations at once."""
    try:
        data = request.get_json()
        updates = data.get("updates", [])
        
        if not updates:
            return jsonify({"success": False, "error": "No updates provided"}), 400
        
        session_manager = current_app.session_manager
        
        # Process each update
        for update in updates:
            layer_id = update.get("layer_id")
            annotation = update.get("annotation", {})
            
            if layer_id and annotation:
                # Basic validation
                if annotation.get("status") in ["correct", "incorrect", "unsure", "unlabeled"]:
                    session_manager.update_layer_status(
                        layer_id, 
                        annotation.get("status", "unlabeled"), 
                        annotation
                    )
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/export_annotations", methods=["POST"])
def api_export_annotations():
    """Export annotations for a specific status."""
    try:
        data = request.get_json()
        status = data.get("status", "incorrect")
        include_correct = data.get("include_correct", False)
        
        session_manager = current_app.session_manager
        layers = session_manager.get("layers", [])
        
        # Filter layers by status
        filtered_layers = []
        for layer in layers:
            if layer.get("status") == status or (include_correct and layer.get("status") == "correct"):
                filtered_layers.append(layer)
        
        return jsonify({"layers": filtered_layers, "count": len(filtered_layers)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/layer_details/<layer_id>")
def api_layer_details(layer_id):
    """Get detailed information about a specific layer."""
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
        
        # Return layer details
        return jsonify({
            "layer": layer,
            "analysis": {
                "status": layer.get("status", "unlabeled"),
                "annotation": layer.get("annotation", {}),
                "has_errors": layer.get("status") == "incorrect"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/filter_layers", methods=["POST"])
def api_filter_layers():
    """Filter layers based on criteria."""
    try:
        data = request.get_json()
        status_filter = data.get("status", "all")
        error_type_filter = data.get("error_type", "all")
        severity_filter = data.get("severity", "all")
        
        session_manager = current_app.session_manager
        layers = session_manager.get("layers", [])
        
        # Apply filters
        filtered_layers = []
        for layer in layers:
            # Status filter
            if status_filter != "all" and layer.get("status") != status_filter:
                continue
            
            # Error type filter
            if error_type_filter != "all":
                annotation = layer.get("annotation", {})
                if annotation.get("error_type") != error_type_filter:
                    continue
            
            # Severity filter
            if severity_filter != "all":
                annotation = layer.get("annotation", {})
                if annotation.get("severity") != severity_filter:
                    continue
            
            filtered_layers.append(layer)
        
        return jsonify(filtered_layers)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
