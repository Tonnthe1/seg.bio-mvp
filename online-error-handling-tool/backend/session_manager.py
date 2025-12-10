import os
import threading
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

class SessionManager:
    """
    Session manager for Error Detection Tool.
    Tracks dataset info, user annotations, and progress.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.state: Dict[str, Any] = {
            "mode3d": False,                    # whether current dataset is 3D
            "image_path": None,                 # path to loaded image or stack
            "mask_path": None,                  # path to loaded mask (optional)
            "load_mode": "path",                # "upload" or "path"
            "image_name": None,                 # filename of image
            "image_ext": None,                  # extension of current image
            "layers": [],                       # list of layer data
            "annotations": {},                  # layer_id -> annotation data
            "current_layer": 0,                # currently viewing layer
            "session_id": None,                # unique session identifier
            "created_at": None,                 # session creation time
            "last_updated": None,               # last update time
        }

    def update(self, **kwargs):
        """Update one or more session fields atomically."""
        with self._lock:
            for k, v in kwargs.items():
                self.state[k] = v
            self.state["last_updated"] = datetime.now().isoformat()

    def get(self, key: str, default=None):
        """Get a session value safely."""
        with self._lock:
            return self.state.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Return a full copy of session state (thread-safe)."""
        with self._lock:
            return dict(self.state)

    def set_image_info(self, image_path: str, load_mode: str = "path"):
        """Extract filename and extension from the image path."""
        with self._lock:
            if image_path:
                base_name = os.path.basename(image_path)
                _, ext = os.path.splitext(base_name)
                self.state["image_name"] = base_name
                self.state["image_ext"] = ext.lower()
            else:
                self.state["image_name"] = None
                self.state["image_ext"] = None
            self.state["load_mode"] = load_mode
            self.state["last_updated"] = datetime.now().isoformat()

    def add_layer(self, layer_data: Dict[str, Any]) -> str:
        """Add a new layer to the session."""
        with self._lock:
            layer_id = f"layer_{len(self.state['layers'])}"
            layer_data["id"] = layer_id
            layer_data["status"] = "unlabeled"  # correct, incorrect, unsure, unlabeled
            layer_data["created_at"] = datetime.now().isoformat()
            self.state["layers"].append(layer_data)
            self.state["last_updated"] = datetime.now().isoformat()
            return layer_id

    def update_layer_status(self, layer_id: str, status: str, annotation: Optional[Dict] = None):
        """Update the status of a specific layer."""
        with self._lock:
            for layer in self.state["layers"]:
                if layer["id"] == layer_id:
                    layer["status"] = status
                    layer["updated_at"] = datetime.now().isoformat()
                    if annotation:
                        layer["annotation"] = annotation
                    break
            self.state["last_updated"] = datetime.now().isoformat()

    def update_layer_fields(self, layer_id: str, fields: Dict[str, Any]):
        """Update arbitrary fields of a specific layer (e.g., overlay, mask_coverage)."""
        with self._lock:
            for layer in self.state["layers"]:
                if layer["id"] == layer_id:
                    layer.update(fields)
                    layer["updated_at"] = datetime.now().isoformat()
                    break
            self.state["last_updated"] = datetime.now().isoformat()

    def get_layers_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all layers with a specific status."""
        with self._lock:
            return [layer for layer in self.state["layers"] if layer["status"] == status]

    def get_incorrect_layers(self) -> List[Dict[str, Any]]:
        """Get all layers marked as incorrect."""
        return self.get_layers_by_status("incorrect")

    def get_unsure_layers(self) -> List[Dict[str, Any]]:
        """Get all layers marked as unsure."""
        return self.get_layers_by_status("unsure")

    def get_correct_layers(self) -> List[Dict[str, Any]]:
        """Get all layers marked as correct."""
        return self.get_layers_by_status("correct")

    def get_progress_stats(self) -> Dict[str, int]:
        """Get progress statistics."""
        with self._lock:
            total = len(self.state["layers"])
            correct = len(self.get_correct_layers())
            incorrect = len(self.get_incorrect_layers())
            unsure = len(self.get_unsure_layers())
            unlabeled = total - correct - incorrect - unsure
            
            return {
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "unsure": unsure,
                "unlabeled": unlabeled,
                "completion_rate": (correct + incorrect + unsure) / total if total > 0 else 0
            }

    def export_session(self) -> Dict[str, Any]:
        """Export session data for integration with PFTool."""
        with self._lock:
            return {
                "session_info": {
                    "session_id": self.state["session_id"],
                    "created_at": self.state["created_at"],
                    "last_updated": self.state["last_updated"],
                    "image_path": self.state["image_path"],
                    "mask_path": self.state["mask_path"],
                    "mode3d": self.state["mode3d"]
                },
                "layers": self.state["layers"],
                "progress": self.get_progress_stats()
            }

    def reset_session(self):
        """Reset the session to initial state."""
        with self._lock:
            self.state = {
                "mode3d": False,
                "image_path": None,
                "mask_path": None,
                "load_mode": "path",
                "image_name": None,
                "image_ext": None,
                "layers": [],
                "annotations": {},
                "current_layer": 0,
                "session_id": None,
                "created_at": None,
                "last_updated": None,
            }
