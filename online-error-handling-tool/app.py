"""
Error Handling Tool (EHTool)
============================
Integrated application combining error detection and proofreading functionality.
Users can select layers as correct, incorrect, or unsure, with incorrect layers
sent to proofreading tool and unsure layers available for later review.
"""

import os
import atexit
import signal
import sys
from flask import Flask, jsonify, request
from routes.landing import register_landing_routes
from routes.detection import register_detection_routes
from routes.review import register_review_routes
from routes.proofreading import register_proofreading_routes
from routes.export import register_export_routes
from routes.detection_workflow import register_detection_workflow_routes
from routes.proofreading_workflow import register_proofreading_workflow_routes
from backend.session_manager import SessionManager

def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("EHTool_SECRET", "dev-secret")
    
    # Attach a global session manager to app
    app.session_manager = SessionManager()
    
    # Register global error handlers to ensure JSON responses for API routes
    # These must be registered BEFORE routes to ensure they catch all errors
    @app.errorhandler(404)
    def not_found(error):
        # Check if this is an API route
        if hasattr(request, 'path') and request.path.startswith('/api/'):
            return jsonify(success=False, error="Not found"), 404
        # For non-API routes, return default Flask 404
        return error
    
    @app.errorhandler(500)
    def internal_error(error):
        # Check if this is an API route
        if hasattr(request, 'path') and request.path.startswith('/api/'):
            import traceback
            error_msg = str(error) if error else "Unknown error"
            print(f"500 Error in API route {request.path if hasattr(request, 'path') else 'unknown'}: {error_msg}")
            traceback.print_exc()
            try:
                return jsonify(success=False, error=f"Internal server error: {error_msg}"), 500
            except:
                # If jsonify fails, return raw JSON
                from flask import Response
                return Response(
                    f'{{"success": false, "error": "Internal server error: {error_msg}"}}',
                    mimetype='application/json',
                    status=500
                )
        # For non-API routes, return default Flask 500
        return error
    
    # Add an after_request handler to catch HTML errors and convert to JSON for API routes
    @app.after_request
    def after_request(response):
        # If it's an API route and we got an error response that's HTML, convert to JSON
        if hasattr(request, 'path') and request.path.startswith('/api/'):
            if response.status_code >= 400:
                content_type = response.content_type or ''
                response_data = response.get_data()
                # Check if response is HTML (starts with <!doctype or has html content type)
                is_html = ('html' in content_type.lower() or 
                          (response_data and len(response_data) > 0 and 
                           (response_data[:20].startswith(b'<!') or b'<!doctype' in response_data[:100])))
                if is_html:
                    try:
                        from flask import jsonify
                        error_msg = "Internal server error"
                        # Try to extract error message from HTML if possible
                        if response_data:
                            try:
                                data_str = response_data.decode('utf-8', errors='ignore')
                                # Look for error messages in the HTML
                                import re
                                error_match = re.search(r'<title>(.*?)</title>', data_str, re.IGNORECASE | re.DOTALL)
                                if error_match:
                                    error_msg = error_match.group(1).strip()
                            except:
                                pass
                        return jsonify(success=False, error=error_msg), response.status_code
                    except Exception as e:
                        # If conversion fails, return minimal JSON
                        from flask import Response
                        return Response(
                            '{"success": false, "error": "Internal server error"}',
                            mimetype='application/json',
                            status=response.status_code
                        )
        return response
    
    # Register routes
    register_landing_routes(app)
    register_detection_routes(app)
    register_review_routes(app)
    register_proofreading_routes(app)
    register_export_routes(app)
    register_detection_workflow_routes(app)
    register_proofreading_workflow_routes(app)
    
    return app

def cleanup():
    """Cleanup function to handle resource cleanup."""
    try:
        # Suppress multiprocessing warnings
        import warnings
        warnings.filterwarnings("ignore", category=ResourceWarning)
    except:
        pass

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n🛑 Shutting down Error Handling Tool...")
    sys.exit(0)

if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    host = os.environ.get("EHTool_HOST", "0.0.0.0")
    port = int(os.environ.get("EHTool_PORT", "5004"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app = create_app()
    print(f"✅ Error Handling Tool running on http://{host}:{port}  (debug={debug})")
    app.run(host=host, port=port, debug=debug)
