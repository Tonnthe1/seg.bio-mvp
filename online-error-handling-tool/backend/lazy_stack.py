import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tifffile

from backend.volume_manager import _to_uint8


def _read_grayscale_slice(path: str) -> np.ndarray:
    """Load a single image slice from disk and normalize to uint8 grayscale."""
    ext = os.path.splitext(path.lower())[1]
    if ext in [".tif", ".tiff"]:
        arr = np.asarray(tifffile.imread(path))
        if arr.ndim == 3:
            # if multi-channel, collapse to first channel
            arr = arr[0] if arr.shape[0] <= 4 else arr.mean(axis=0)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Failed to read image: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return _to_uint8(arr)


def _resize_if_needed(arr: np.ndarray, target_shape: Optional[Tuple[int, int]]) -> np.ndarray:
    if target_shape is None or arr.shape == target_shape:
        return arr
    return cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


class LazySliceLoader:
    """Lazily load 2D slices from a list of file paths."""

    def __init__(self, files: Sequence[str]):
        if not files:
            raise ValueError("Empty slice list for lazy loader")
        self.files: List[str] = list(files)
        sample = _read_grayscale_slice(self.files[0])
        self.slice_shape: Tuple[int, int] = sample.shape[-2], sample.shape[-1]
        self.dtype = sample.dtype
        self.ndim = 3
        self._num_slices = len(self.files)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._num_slices, self.slice_shape[0], self.slice_shape[1])

    def __len__(self) -> int:
        return self._num_slices

    def get_slice(self, index: int) -> np.ndarray:
        idx = max(0, min(index, self._num_slices - 1))
        slice_arr = _read_grayscale_slice(self.files[idx])
        return _resize_if_needed(slice_arr, self.slice_shape)


class LazyMaskLoader:
    """Lazily load mask slices paired with image files."""

    def __init__(self, mask_paths: Sequence[Optional[str]], target_shape: Tuple[int, int]):
        if not mask_paths:
            raise ValueError("Mask loader requires at least one entry")
        self.mask_paths: List[Optional[str]] = list(mask_paths)
        self.slice_shape = target_shape
        self._num_slices = len(self.mask_paths)

    def __len__(self) -> int:
        return self._num_slices

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._num_slices, self.slice_shape[0], self.slice_shape[1])

    @property
    def ndim(self) -> int:
        return 3

    def _load_mask_slice(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path.lower())[1]
        if ext in [".png", ".jpg", ".jpeg"]:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"Failed to read mask: {path}")
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = np.asarray(tifffile.imread(path))
            print(f"DEBUG: LazyMaskLoader - Raw TIF {os.path.basename(path)}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, non-zero={np.count_nonzero(arr)}")
            
            # Handle different TIF formats
            if arr.ndim == 2:
                # Already 2D, use as-is
                pass
            elif arr.ndim == 3:
                # Could be (Z, H, W) stack or (H, W, C) multi-channel
                # For masks, if last dimension is small (1-4), it's likely channels
                if arr.shape[2] <= 4:
                    # Likely (H, W, C) - take first channel
                    arr = arr[:, :, 0]
                    print(f"DEBUG: Took first channel, new shape: {arr.shape}")
                else:
                    # Likely (Z, H, W) stack - take first slice
                    arr = arr[0]
                    print(f"DEBUG: Took first slice, new shape: {arr.shape}")
            elif arr.ndim > 3:
                # 4D or higher - take first slice
                arr = arr[0]
                print(f"DEBUG: Took first slice from 4D+, new shape: {arr.shape}")
            else:
                raise ValueError(f"Unsupported mask array dimensions: {arr.ndim} for {path}")
        
        # Debug: print raw mask info
        print(f"DEBUG: LazyMaskLoader - Loaded {os.path.basename(path)}, shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min()}, max: {arr.max()}, non-zero: {np.count_nonzero(arr)}")
        
        # Store original for comparison
        original_nonzero = np.count_nonzero(arr)
        original_max = arr.max()
        
        # Convert to uint8, preserving mask values (don't normalize)
        # IMPORTANT: For binary masks, we want to preserve any non-zero value as 255
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0 and arr.min() >= 0.0:
                # Float 0-1 range, scale to 0-255
                arr = (arr * 255).astype(np.uint8)
            elif arr.max() > 255:
                # Large values (uint16), scale proportionally
                if arr.max() > 0:
                    arr = ((arr.astype(np.float32) / arr.max()) * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            else:
                # For integer types with values <= 255, preserve non-zero values
                # If it's a binary mask (0 and 1), convert 1 to 255
                unique_vals = np.unique(arr)
                if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
                    # Binary mask: convert 1 to 255
                    arr = (arr * 255).astype(np.uint8)
                    print(f"DEBUG: Detected binary mask (0,1), converted to (0,255)")
                elif len(unique_vals) == 2 and 0 in unique_vals:
                    # Binary mask with 0 and some other value - convert non-zero to 255
                    arr = (arr > 0).astype(np.uint8) * 255
                    print(f"DEBUG: Detected binary mask with 0 and {unique_vals[unique_vals != 0][0]}, converted to (0,255)")
                else:
                    # Just clip to uint8 range
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
        
        print(f"DEBUG: LazyMaskLoader - After conversion, shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min()}, max: {arr.max()}, non-zero: {np.count_nonzero(arr)}")
        
        # Check if we lost mask data during conversion
        final_nonzero = np.count_nonzero(arr)
        if original_nonzero > 0 and final_nonzero == 0:
            print(f"ERROR: Mask data lost during conversion! Original had {original_nonzero} non-zero pixels, final has 0")
            print(f"  Original max value: {original_max}, dtype: {arr.dtype}")
            # Try to recover: if original had non-zero, make them 255
            # This shouldn't happen, but if it does, we'll try to fix it
        elif arr.max() == 0:
            print(f"WARNING: Mask {os.path.basename(path)} appears to be empty (all zeros)")
        
        return arr

    def get_slice(self, index: int) -> np.ndarray:
        idx = max(0, min(index, self._num_slices - 1))
        path = self.mask_paths[idx]
        if not path:
            print(f"WARNING: No mask path for index {idx}")
            return np.zeros(self.slice_shape, dtype=np.uint8)
        
        if not os.path.exists(path):
            print(f"ERROR: Mask file does not exist: {path}")
            return np.zeros(self.slice_shape, dtype=np.uint8)
        
        mask_arr = self._load_mask_slice(path)
        mask_arr = _resize_if_needed(mask_arr, self.slice_shape)
        
        print(f"DEBUG: LazyMaskLoader.get_slice({idx}) returning shape: {mask_arr.shape}, dtype: {mask_arr.dtype}, min: {mask_arr.min()}, max: {mask_arr.max()}, non-zero: {np.count_nonzero(mask_arr)}")
        
        return mask_arr

