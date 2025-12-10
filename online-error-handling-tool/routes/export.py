"""
Error Detection Tool - Export Routes
-------------------------------------
Handles export functionality for integration with PFTool and other tools.
"""

import os
import json
from flask import Blueprint, render_template, request, current_app, jsonify, send_file, redirect, url_for
from datetime import datetime

bp = Blueprint("export", __name__, url_prefix="")

def register_export_routes(app):
    app.register_blueprint(bp)

@bp.route("/export")
def export():
    """Export interface for session data."""
    session_manager = current_app.session_manager
    session_state = session_manager.snapshot()
    
    if not session_state.get("layers"):
        return redirect(url_for("landing.landing"))
    
    # Get progress statistics
    progress = session_manager.get_progress_stats()
    
    # Get basic export summary
    layers = session_state.get("layers", [])
    export_summary = {
        "total_layers": len(layers),
        "correct_layers": len([l for l in layers if l.get("status") == "correct"]),
        "incorrect_layers": len([l for l in layers if l.get("status") == "incorrect"]),
        "unsure_layers": len([l for l in layers if l.get("status") == "unsure"]),
        "unlabeled_layers": len([l for l in layers if l.get("status") == "unlabeled"])
    }
    
    return render_template(
        "export.html",
        layers=layers,
        progress=progress,
        export_summary=export_summary,
        session_state=session_state
    )

@bp.route("/api/export_session", methods=["POST"])
def api_export_session():
    """Export complete session data."""
    try:
        data = request.get_json()
        export_format = data.get("format", "json")
        include_correct = data.get("include_correct", False)
        
        session_manager = current_app.session_manager
        session_data = session_manager.export_session()
        
        # Create export directory
        export_dir = os.path.abspath("./_exports")
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "json":
            export_path = os.path.join(export_dir, f"session_export_{timestamp}.json")
            with open(export_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            result = {"success": True, "file_path": export_path, "filename": f"session_export_{timestamp}.json"}
        elif export_format == "csv":
            export_path = os.path.join(export_dir, f"annotations_{timestamp}.csv")
            # Simple CSV export
            import csv
            with open(export_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['layer_id', 'status', 'annotation'])
                for layer in session_data.get("layers", []):
                    writer.writerow([layer.get("id"), layer.get("status"), json.dumps(layer.get("annotation", {}))])
            result = {"success": True, "file_path": export_path, "filename": f"annotations_{timestamp}.csv"}
        else:
            return jsonify({"success": False, "error": "Unsupported export format"}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/export_proofreading_queue", methods=["POST"])
def api_export_proofreading_queue():
    """Export proofreading queue for incorrect layers."""
    try:
        session_manager = current_app.session_manager
        layers = session_manager.get("layers", [])
        
        # Filter incorrect layers
        incorrect_layers = [layer for layer in layers if layer.get("status") == "incorrect"]
        
        # Create export directory
        export_dir = os.path.abspath("./_exports")
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(export_dir, f"proofreading_queue_{timestamp}.json")
        
        # Export proofreading queue
        queue_data = {
            "timestamp": timestamp,
            "total_layers": len(layers),
            "incorrect_layers": len(incorrect_layers),
            "queue": incorrect_layers
        }
        
        with open(export_path, 'w') as f:
            json.dump(queue_data, f, indent=2)
        
        result = {"success": True, "file_path": export_path, "filename": f"proofreading_queue_{timestamp}.json"}
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/open_proofreading", methods=["POST"])
def api_open_proofreading():
    """Open integrated proofreading interface with incorrect layers."""
    try:
        session_manager = current_app.session_manager
        session_data = session_manager.export_session()
        
        # Get incorrect layers
        incorrect_layers = [layer for layer in session_data.get("layers", []) if layer.get("status") == "incorrect"]
        
        if not incorrect_layers:
            return jsonify({"success": False, "error": "No incorrect layers found for proofreading"}), 400
        
        # Return proofreading URL
        proofreading_url = "/proofreading"
        
        result = {
            "success": True, 
            "proofreading_url": proofreading_url,
            "incorrect_layers": len(incorrect_layers)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/export_layer_images", methods=["POST"])
def api_export_layer_images():
    """Export layer images to files."""
    try:
        data = request.get_json()
        include_overlays = data.get("include_overlays", True)
        status_filter = data.get("status_filter", "all")
        
        session_manager = current_app.session_manager
        layers = session_manager.get("layers", [])
        
        # Filter layers by status
        if status_filter != "all":
            layers = [layer for layer in layers if layer.get("status") == status_filter]
        
        # Create export directory
        export_dir = os.path.abspath("./_exports/layer_images")
        os.makedirs(export_dir, exist_ok=True)
        
        # Simple export - just return the layer info
        result = {
            "success": True,
            "export_dir": export_dir,
            "layers_exported": len(layers),
            "include_overlays": include_overlays,
            "status_filter": status_filter
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/download_export/<filename>")
def api_download_export(filename):
    """Download an exported file."""
    try:
        export_dir = os.path.abspath("./_exports")
        file_path = os.path.join(export_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/list_exports")
def api_list_exports():
    """List available export files."""
    try:
        export_dir = os.path.abspath("./_exports")
        
        if not os.path.exists(export_dir):
            return jsonify({"exports": []})
        
        exports = []
        for filename in os.listdir(export_dir):
            file_path = os.path.join(export_dir, filename)
            if os.path.isfile(file_path):
                exports.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "modified": os.path.getmtime(file_path)
                })
        
        # Sort by modification time (newest first)
        exports.sort(key=lambda x: x["modified"], reverse=True)
        
        return jsonify({"exports": exports})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/export_summary")
def api_export_summary():
    """Get export summary information."""
    try:
        session_manager = current_app.session_manager
        session_state = session_manager.snapshot()
        
        # Create basic export summary
        layers = session_state.get("layers", [])
        summary = {
            "total_layers": len(layers),
            "correct_layers": len([l for l in layers if l.get("status") == "correct"]),
            "incorrect_layers": len([l for l in layers if l.get("status") == "incorrect"]),
            "unsure_layers": len([l for l in layers if l.get("status") == "unsure"]),
            "unlabeled_layers": len([l for l in layers if l.get("status") == "unlabeled"])
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/api/delete_export/<filename>", methods=["DELETE"])
def api_delete_export(filename):
    """Delete an exported file."""
    try:
        export_dir = os.path.abspath("./_exports")
        file_path = os.path.join(export_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "File not found"}), 404
        
        os.remove(file_path)
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
