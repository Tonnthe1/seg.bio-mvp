import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tifffile
from PIL import Image

from backend.lazy_stack import LazySliceLoader, LazyMaskLoader
from backend.volume_manager import list_images_for_path, build_mask_path_mapping


def _ensure_grayscale_2d(arr: np.ndarray, kind: str) -> np.ndarray:
    """
    Ensure an array is 2D grayscale.

    Args:
        arr: Input array that may be 2D or multi-channel.
        kind: Text description used in error messages.
    """
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        return np.mean(arr[:, :, :3], axis=2)

    raise ValueError(f"Unsupported {kind} dimensions: {arr.ndim}")


def _prepare_mask_for_display(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert mask data into uint8 [0, 255] for visualization purposes."""
    if mask is None:
        return None

    mask_arr = np.asarray(mask)

    if mask_arr.dtype != np.uint8:
        mask_arr = mask_arr.astype(np.float32)
        if mask_arr.size == 0:
            return np.zeros_like(mask_arr, dtype=np.uint8)

        max_val = float(mask_arr.max())
        min_val = float(mask_arr.min())

        if max_val <= 1.0 and min_val >= 0.0:
            mask_arr *= 255.0
        elif max_val > min_val:
            mask_arr = (mask_arr - min_val) / (max_val - min_val)
            mask_arr *= 255.0

        mask_arr = np.clip(mask_arr, 0, 255).astype(np.uint8)
    else:
        # Promote {0,1} masks to {0,255} for visibility
        unique_vals = np.unique(mask_arr)
        if unique_vals.size <= 2 and set(unique_vals.tolist()).issubset({0, 1}):
            mask_arr = (mask_arr * 255).astype(np.uint8)

    return mask_arr


def _mask_to_reference_scale(mask: Optional[np.ndarray], reference: np.ndarray) -> Optional[np.ndarray]:
    """Match mask dtype/range to the reference image for serialization."""
    if mask is None:
        return None

    mask_bool = (np.asarray(mask) > 0)
    ref_dtype = reference.dtype

    if np.issubdtype(ref_dtype, np.integer):
        info = np.iinfo(ref_dtype)
        high = info.max
        return (mask_bool.astype(ref_dtype)) * high
    elif np.issubdtype(ref_dtype, np.floating):
        return mask_bool.astype(ref_dtype)
    else:
        # Fallback to uint8 if dtype is unexpected
        return (mask_bool.astype(np.uint8)) * 255


def _enhance_contrast(arr: np.ndarray) -> np.ndarray:
    """
    Apply contrast enhancement to improve visibility of dark images.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    if arr.ndim != 2:
        return arr
    
    # Normalize to uint8 if needed
    if arr.max() > 255 or arr.dtype != np.uint8:
        if arr.max() > 0:
            arr = (arr / arr.max() * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr)


def _normalize_image_slice_to_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Normalize image slice to RGB format with CLAHE enhancement.
    Used by both integrated and standalone proofreading.
    
    Args:
        arr: Input image slice (2D or 3D with channels)
        
    Returns:
        RGB image as uint8 array of shape (H, W, 3)
    """
    arr = np.asarray(arr)
    
    # If already RGB, just ensure uint8
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.astype(np.uint8)
    
    # Normalize to 0-255 range
    if arr.max() > 0:
        arr = (arr / arr.max() * 255.0)
    else:
        arr = arr.astype(np.float64)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    # Apply CLAHE for contrast enhancement (only for 2D)
    if arr.ndim == 2:
        arr = _enhance_contrast(arr)
    
    # Convert to RGB by stacking channels
    rgb = np.stack([arr] * 3, axis=-1)
    
    # Ensure consistent data type and range
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _mask_slice_to_rgba(mask_slice: np.ndarray, opacity: int = 230) -> np.ndarray:
    """
    Convert mask slice to RGBA format with grayscale overlay.
    Used by both integrated and standalone proofreading.
    
    Args:
        mask_slice: Input mask slice (2D array)
        opacity: Alpha channel value (0-255), default 230 (90% opacity)
        
    Returns:
        RGBA image as uint8 array of shape (H, W, 4)
    """
    sl = np.asarray(mask_slice)
    
    # Convert mask to uint8 without normalization (preserve original values)
    if sl.dtype != np.uint8:
        if sl.max() <= 1.0 and sl.min() >= 0.0:
            mask_uint8 = (sl * 255).astype(np.uint8)
        elif sl.max() > 255:
            mask_uint8 = ((sl.astype(np.float32) / sl.max()) * 255).astype(np.uint8) if sl.max() > 0 else sl.astype(np.uint8)
        else:
            mask_uint8 = np.clip(sl, 0, 255).astype(np.uint8)
    else:
        mask_uint8 = sl.copy()
    
    # Create RGBA: grayscale mask with transparency
    h, w = mask_uint8.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Binary check: any non-zero value means mask pixel
    mask_binary = mask_uint8 > 0
    
    # Debug: check if we're losing mask data
    if np.count_nonzero(mask_binary) == 0 and np.count_nonzero(mask_uint8) > 0:
        # Try a more lenient threshold
        mask_binary = mask_uint8 >= 1
    
    # Set white color with transparency where mask exists
    rgba[mask_binary, 0] = 255  # R
    rgba[mask_binary, 1] = 255  # G
    rgba[mask_binary, 2] = 255  # B
    rgba[mask_binary, 3] = opacity  # A
    
    return rgba
    return clahe.apply(arr)

class DataManager:
    """
    Handles loading, processing, and managing image and mask data.
    Supports both 2D and 3D datasets.
    """

    def __init__(self):
        self.current_volume = None
        self.current_mask = None
        self.volume_info = {}
        self.mask_info = {}

    def _resolve_slice_index(self, z: int) -> int:
        total = self.volume_info.get("num_slices")
        if total is None and hasattr(self.current_volume, 'shape'):
            shape = getattr(self.current_volume, 'shape', None)
            if isinstance(shape, tuple) and shape:
                total = shape[0]
        if total is None and hasattr(self.current_volume, 'ndim') and getattr(self.current_volume, 'ndim', 0) == 3:
            total = self.current_volume.shape[0]
        if not total or total <= 0:
            total = 1
        return max(0, min(z, total - 1))

    def _get_mask_slice_for_index(self, z: int):
        if self.current_mask is None:
            return None
        if hasattr(self.current_mask, 'get_slice'):
            return self.current_mask.get_slice(z)
        if getattr(self.current_mask, 'ndim', 0) == 2:
            return self.current_mask
        return self.current_mask[z]

    def load_image(self, image_path: Union[str, List[str]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load image data from file path, directory/glob, or list of files.
        Returns: (image_array, metadata_dict)
        """
        # If list/dir/glob of files is provided, lazily stream them
        if isinstance(image_path, list) or any(ch in str(image_path) for ch in ['*', '?', '[']) or os.path.isdir(image_path):
            if isinstance(image_path, list):
                files = list(image_path)
            else:
                files = list_images_for_path(image_path)
            if not files:
                raise FileNotFoundError("No image files found for loading")
            loader = LazySliceLoader(files)
            info = {
                "shape": loader.shape,
                "dtype": str(loader.dtype),
                "ndim": loader.ndim,
                "is_3d": True,
                "num_slices": loader.shape[0],
                "lazy": True
            }
            self.current_volume = loader
            self.volume_info = info
            return loader, info


        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        ext = os.path.splitext(image_path.lower())[1]
        
        if ext in ['.tif', '.tiff']:
            # Load TIFF stack
            volume = tifffile.imread(image_path)
            info = {
                "shape": volume.shape,
                "dtype": str(volume.dtype),
                "ndim": volume.ndim,
                "is_3d": volume.ndim == 3,
                "num_slices": volume.shape[0] if volume.ndim == 3 else 1
            }
        else:
            # Load 2D image
            img = Image.open(image_path)
            volume = np.array(img)
            
            # Handle different image formats properly
            if volume.ndim == 3:
                # If it's a color image, convert to grayscale
                if volume.shape[2] == 3:  # RGB
                    volume = np.mean(volume, axis=2).astype(volume.dtype)
                elif volume.shape[2] == 4:  # RGBA
                    volume = np.mean(volume[:, :, :3], axis=2).astype(volume.dtype)
                else:
                    # Other multi-channel formats, take first channel
                    volume = volume[:, :, 0]
            elif volume.ndim == 2:
                # Already grayscale, keep as is
                pass
            else:
                raise ValueError(f"Unsupported image format with {volume.ndim} dimensions")
            
            info = {
                "shape": volume.shape,
                "dtype": str(volume.dtype),
                "ndim": volume.ndim,
                "is_3d": False,
                "num_slices": 1
            }

        self.current_volume = volume
        self.volume_info = info
        return volume, info

    def load_mask(self, mask_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load mask data from file path.
        Returns: (mask_array, metadata_dict)
        """
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        ext = os.path.splitext(mask_path.lower())[1]
        
        if ext in ['.tif', '.tiff']:
            # Load TIFF stack
            mask = tifffile.imread(mask_path)
            info = {
                "shape": mask.shape,
                "dtype": str(mask.dtype),
                "ndim": mask.ndim,
                "is_3d": mask.ndim == 3,
                "num_slices": mask.shape[0] if mask.ndim == 3 else 1
            }
        else:
            # Load 2D image
            img = Image.open(mask_path)
            mask = np.array(img)
            
            # Handle different image formats properly
            if mask.ndim == 3:
                # If it's a color image, convert to grayscale
                if mask.shape[2] == 3:  # RGB
                    mask = np.mean(mask, axis=2).astype(mask.dtype)
                elif mask.shape[2] == 4:  # RGBA
                    mask = np.mean(mask[:, :, :3], axis=2).astype(mask.dtype)
                else:
                    # Other multi-channel formats, take first channel
                    mask = mask[:, :, 0]
            elif mask.ndim == 2:
                # Already grayscale, keep as is
                pass
            else:
                raise ValueError(f"Unsupported mask format with {mask.ndim} dimensions")
            
            info = {
                "shape": mask.shape,
                "dtype": str(mask.dtype),
                "ndim": mask.ndim,
                "is_3d": False,
                "num_slices": 1
            }

        self.current_mask = mask
        self.mask_info = info
        return mask, info

    def get_slice(self, z: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get a specific slice from the volume and mask.
        Returns: (image_slice, mask_slice)
        """
        if self.current_volume is None:
            raise ValueError("No volume loaded")

        if hasattr(self.current_volume, 'get_slice'):
            z = self._resolve_slice_index(z)
            image_slice = self.current_volume.get_slice(z)
        elif getattr(self.current_volume, 'ndim', 0) == 2:
            image_slice = self.current_volume
            z = 0
        else:
            z = self._resolve_slice_index(z)
            image_slice = self.current_volume[z]

        mask_slice = self._get_mask_slice_for_index(z)
        return image_slice, mask_slice

    def create_overlay(self, image_slice: np.ndarray, mask_slice: Optional[np.ndarray], 
                      alpha: float = 0.4) -> np.ndarray:
        """
        Create an overlay of image and mask.
        Returns: RGB overlay image
        """
        # Ensure image_slice is 2D
        image_slice = _ensure_grayscale_2d(image_slice, "image slice")

        # Normalize image to 0-255
        if image_slice.max() > 1:
            img_norm = image_slice.astype(np.float32)
        else:
            img_norm = (image_slice * 255).astype(np.float32)
        
        img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
        
        # Apply contrast enhancement for better visibility of dark images
        img_norm = _enhance_contrast(img_norm)
        
        # Convert to RGB
        img_rgb = np.stack([img_norm] * 3, axis=-1)

        if mask_slice is not None:
            mask_slice = _ensure_grayscale_2d(mask_slice, "mask slice")
            mask_slice = _prepare_mask_for_display(mask_slice)

            if mask_slice.shape != image_slice.shape:
                mask_slice = cv2.resize(mask_slice, (image_slice.shape[1], image_slice.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            mask_binary = (mask_slice > 0).astype(np.uint8)
            if np.any(mask_binary):
                overlay_color = np.zeros_like(img_rgb, dtype=np.uint8)
                overlay_color[mask_binary > 0] = np.array([255, 0, 0], dtype=np.uint8)
                img_rgb = cv2.addWeighted(img_rgb, 1 - alpha, overlay_color, alpha, 0)

        return img_rgb

    def array_to_base64(self, arr: np.ndarray, format: str = "PNG") -> str:
        """Convert numpy array to base64 string."""
        if len(arr.shape) == 2:
            img = Image.fromarray(arr)
        else:
            img = Image.fromarray(arr)
        
        bio = io.BytesIO()
        img.save(bio, format=format)
        bio.seek(0)
        return base64.b64encode(bio.getvalue()).decode()

    def create_layer_data(self, z: int, image_slice: np.ndarray, 
                         mask_slice: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Create layer data for a specific slice.
        Returns: layer data dictionary
        """
        overlay = self.create_overlay(image_slice, mask_slice)
        
        # Prepare enhanced image slice for display
        enhanced_image = _ensure_grayscale_2d(image_slice, "image slice")
        if enhanced_image.max() > 1:
            enhanced_image = enhanced_image.astype(np.float32)
        else:
            enhanced_image = (enhanced_image * 255).astype(np.float32)
        enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
        enhanced_image = _enhance_contrast(enhanced_image)
        
        mask_serialized = None
        if mask_slice is not None:
            mask_for_serialization = _mask_to_reference_scale(mask_slice, image_slice)
            mask_serialized = self.array_to_base64(mask_for_serialization)

        return {
            "z": z,
            "image_slice": self.array_to_base64(enhanced_image),
            "mask_slice": mask_serialized,
            "overlay": self.array_to_base64(overlay),
            "has_mask": mask_slice is not None,
            "mask_coverage": float(np.sum(mask_slice > 0) / mask_slice.size) if mask_slice is not None else 0.0
        }

    def generate_layer_for_z(self, z: int) -> Dict[str, Any]:
        """
        Generate full layer data for a specific z index using current volume/mask.
        """
        if self.current_volume is None:
            raise ValueError("No volume loaded")
        image_slice, mask_slice = self.get_slice(z)
        return self.create_layer_data(z, image_slice, mask_slice)

    def generate_layers_range(self, start_z: int, end_z: int) -> List[Dict[str, Any]]:
        """
        Generate layer data for z in [start_z, end_z) (end exclusive).
        """
        if self.current_volume is None:
            raise ValueError("No volume loaded")
        if self.current_volume.ndim == 2:
            start_z, end_z = 0, 1
        end_z = min(end_z, self.current_volume.shape[0] if self.current_volume.ndim == 3 else 1)
        start_z = max(0, start_z)
        layers: List[Dict[str, Any]] = []
        for z in range(start_z, end_z):
            layers.append(self.generate_layer_for_z(z))
        return layers

    def generate_all_layers(self) -> List[Dict[str, Any]]:
        """
        Generate layer data for all slices.
        Returns: list of layer data dictionaries
        """
        if self.current_volume is None:
            raise ValueError("No volume loaded")

        layers = []
        
        if self.current_volume.ndim == 2:
            # 2D case
            image_slice, mask_slice = self.get_slice(0)
            layer_data = self.create_layer_data(0, image_slice, mask_slice)
            layers.append(layer_data)
        else:
            # 3D case
            for z in range(self.current_volume.shape[0]):
                image_slice, mask_slice = self.get_slice(z)
                layer_data = self.create_layer_data(z, image_slice, mask_slice)
                layers.append(layer_data)

        return layers

    def validate_data_compatibility(self) -> bool:
        """
        Validate that image and mask data are compatible.
        Returns: True if compatible, False otherwise
        """
        if self.current_volume is None or self.current_mask is None:
            return True  # No mask is valid

        try:
            image_slice, mask_slice = self.get_slice(0)
            if mask_slice is None:
                return True
            return image_slice.shape == mask_slice.shape
        except Exception:
            return False

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        summary = {
            "volume_loaded": self.current_volume is not None,
            "mask_loaded": self.current_mask is not None,
            "compatible": self.validate_data_compatibility()
        }
        
        if self.current_volume is not None:
            summary.update({
                "volume_shape": self.current_volume.shape,
                "volume_dtype": str(self.current_volume.dtype),
                "is_3d": self.current_volume.ndim == 3,
                "num_slices": self.current_volume.shape[0] if self.current_volume.ndim == 3 else 1
            })
        
        if self.current_mask is not None:
            summary.update({
                "mask_shape": self.current_mask.shape,
                "mask_dtype": str(self.current_mask.dtype),
                "mask_is_3d": self.current_mask.ndim == 3
            })

        return summary
