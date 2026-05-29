# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""TensorRT-based Foundation Stereo inference for stereo depth estimation.

Adapted from real2sim's tensorrt_inference_base.py, stereo_inference_base.py,
and foundation_stereo.py into a single self-contained module.
"""

import numpy as np
import cv2
import tensorrt as trt
from cuda.bindings import driver as cuda_driver
from cuda.bindings import runtime as cuda_runtime


# Foundation Stereo model constants
MODEL_INPUT_HEIGHT = 576
MODEL_INPUT_WIDTH = 960
BASE_MODEL_NAME = 'deployable_foundationstereo_small_576x960_v2.0'

# ImageNet normalization parameters used during training
IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STDDEV = [58.395, 57.12, 57.375]

# Stereo depth conversion constants
MIN_DISPARITY_THRESHOLD = 0.01
MAX_DEPTH_METERS = 655.35  # clip before inverse-depth encoding; beyond this quantization error grows large


class FoundationStereoInference:
    """TensorRT-based Foundation Stereo inference engine.

    Takes a stereo image pair (left + right BGR images) and returns a metric
    depth map (in meters) in the original image resolution.

    Usage:
        with FoundationStereoInference(engine_path) as inference:
            depth = inference.run(left_image, right_image, fx, baseline)
    """

    def __init__(self, engine_file_path: str, verbose: bool = False):
        self.engine_file_path = engine_file_path
        self.verbose = verbose
        self._initialized = False

        self._check_cuda()
        self._init_trt()
        self._load_engine()
        self._setup_bindings()
        self._initialized = True

        if verbose:
            print(f"FoundationStereoInference initialized: {engine_file_path}")
            print(f"TensorRT version: {trt.__version__}")
            print(f"Input shapes: {self.input_shapes}")
            print(f"Output shapes: {self.output_shapes}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if not self._initialized:
            return
        try:
            for mem_info in self._inputs.values():
                cuda_runtime.cudaFree(mem_info[0])
            for mem_info in self._outputs.values():
                cuda_runtime.cudaFree(mem_info[0])
            if hasattr(self, '_context'):
                del self._context
            if hasattr(self, '_engine'):
                del self._engine
            if hasattr(self, '_stream'):
                cuda_runtime.cudaStreamDestroy(self._stream)
        except Exception as e:
            print(f"Warning: error during CUDA cleanup: {e}")
        finally:
            self._initialized = False

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _check_cuda(self):
        err, = cuda_driver.cuInit(0)
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA init failed: {err}")
        err, count = cuda_driver.cuDeviceGetCount()
        if count == 0:
            raise RuntimeError("No CUDA devices found")

    def _init_trt(self):
        self._logger = trt.Logger(trt.Logger.VERBOSE if self.verbose else trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        trt.init_libnvinfer_plugins(self._logger, "")

    def _load_engine(self):
        try:
            with open(self.engine_file_path, 'rb') as f:
                self._engine = self._runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TensorRT engine from {self.engine_file_path}: {e}"
            ) from e
        if self._engine is None:
            raise RuntimeError(
                f"TensorRT engine is None after loading {self.engine_file_path}. "
                "Check TRT version compatibility."
            )
        self._context = self._engine.create_execution_context()
        # Set static input shapes
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = self._engine.get_tensor_shape(name)
                self._context.set_input_shape(name, shape)

    def _setup_bindings(self):
        INPUT_NAMES = ['left_image', 'right_image']
        OUTPUT_NAMES = ['disparity']

        self._inputs = {}
        self._outputs = {}
        self.input_shapes = {}
        self.output_shapes = {}

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # Replace any -1 dims with 1 for memory allocation
            alloc_shape = tuple(max(1, d) for d in shape)
            volume = int(np.prod(alloc_shape))
            size = volume * np.dtype(dtype).itemsize

            err, ptr = cuda_runtime.cudaMalloc(size)
            if err != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed for {name}: {err}")

            if is_input and name in INPUT_NAMES:
                self._inputs[name] = (ptr, size, alloc_shape, dtype)
                self.input_shapes[name] = shape
            elif not is_input and name in OUTPUT_NAMES:
                self._outputs[name] = (ptr, size, alloc_shape, dtype)
                self.output_shapes[name] = shape

        err, self._stream = cuda_runtime.cudaStreamCreate()
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_image(self, image: np.ndarray):
        """Preprocess a BGR image for Foundation Stereo.

        Args:
            image: BGR uint8 image (H×W×3)

        Returns:
            (nchw_array, metadata) where nchw_array is float32 (1×3×576×960)
            and metadata contains scale/padding info for coordinate transformation.
        """
        h, w = image.shape[:2]
        scale_w = MODEL_INPUT_WIDTH / w
        scale_h = MODEL_INPUT_HEIGHT / h
        scale = min(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

        pad_w = MODEL_INPUT_WIDTH - new_w
        pad_h = MODEL_INPUT_HEIGHT - new_h
        padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        for c in range(3):
            rgb[:, :, c] = (rgb[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STDDEV[c]

        nchw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])  # (1,3,H,W)

        metadata = {
            'scale': scale,
            'resized_w': new_w,
            'resized_h': new_h,
            'pad_w': pad_w,
            'pad_h': pad_h,
            'orig_w': w,
            'orig_h': h,
        }
        return nchw, metadata

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _copy_to_device(self, name: str, array: np.ndarray):
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        ptr, size, _, _ = self._inputs[name]
        err, = cuda_runtime.cudaMemcpyAsync(
            ptr, array.ctypes.data, size,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self._stream,
        )
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpyAsync H→D failed for {name}: {err}")

    def _copy_from_device(self, name: str) -> np.ndarray:
        ptr, size, _, dtype = self._outputs[name]
        runtime_shape = tuple(self._context.get_tensor_shape(name))
        result = np.empty(runtime_shape, dtype=dtype)
        err, = cuda_runtime.cudaMemcpyAsync(
            result.ctypes.data, ptr, size,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self._stream,
        )
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpyAsync D→H failed for {name}: {err}")
        return result

    def infer(self, left_image: np.ndarray, right_image: np.ndarray):
        """Run stereo inference.

        Args:
            left_image: BGR uint8 image (H×W×3)
            right_image: BGR uint8 image (H×W×3), same size as left_image

        Returns:
            (disparity, metadata) — float32 disparity in original image resolution
            and preprocessing metadata from the left image.
        """
        left_nchw, metadata = self.preprocess_image(left_image)
        right_nchw, _ = self.preprocess_image(right_image)

        # Copy inputs
        self._copy_to_device('left_image', left_nchw)
        self._copy_to_device('right_image', right_nchw)

        # Set dynamic shapes and tensor addresses
        for name, arr in [('left_image', left_nchw), ('right_image', right_nchw)]:
            self._context.set_input_shape(name, arr.shape)
            self._context.set_tensor_address(name, int(self._inputs[name][0]))
        for name in self._outputs:
            self._context.set_tensor_address(name, int(self._outputs[name][0]))

        # Execute
        self._context.execute_async_v3(stream_handle=self._stream)

        # Retrieve output
        raw_disparity = self._copy_from_device('disparity')

        # Synchronize
        err, = cuda_runtime.cudaStreamSynchronize(self._stream)
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamSynchronize failed: {err}")

        # Collapse batch/channel dims → (H, W)
        d = raw_disparity
        if d.ndim == 4:
            d = d[0]
        if d.ndim == 3:
            d = d[0] if d.shape[0] == 1 else d[:, :, 0] if d.shape[2] == 1 else d[0]

        # Transform back to original image coordinates
        disparity = self._transform_to_original(d, metadata)
        return disparity, metadata

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _transform_to_original(self, disparity: np.ndarray, metadata: dict) -> np.ndarray:
        """Crop padding, scale disparity values, resize to original dims."""
        cropped = disparity[:metadata['resized_h'], :metadata['resized_w']]
        scaled = cropped / metadata['scale']
        return cv2.resize(
            scaled,
            (metadata['orig_w'], metadata['orig_h']),
            interpolation=cv2.INTER_LINEAR,
        )


def disparity_to_depth(disparity_px: np.ndarray, fx_px: float, baseline_m: float) -> np.ndarray:
    """Convert a disparity map to metric depth in meters.

    Formula: depth_m = fx_px * baseline_m / disparity_px

    Args:
        disparity_px: float32 disparity map (H×W) in pixels, at original image resolution
        fx_px: focal length in pixels (x-axis)
        baseline_m: stereo baseline in meters

    Returns:
        float32 depth map in meters (H×W); invalid/zero-disparity pixels are 0.
    """
    depth_m = np.zeros_like(disparity_px, dtype=np.float32)
    valid = disparity_px > MIN_DISPARITY_THRESHOLD
    if np.any(valid):
        depth_m_vals = (fx_px * baseline_m) / disparity_px[valid]
        depth_m_vals = np.clip(depth_m_vals, 0.0, MAX_DEPTH_METERS)
        depth_m[valid] = depth_m_vals
    return depth_m
