"""Zero-copy TensorRT execution for FoundationPose CUDA tensors."""

from __future__ import annotations

from pathlib import Path


class TensorRTModule:
    """Execute a dynamic-batch TensorRT engine using PyTorch CUDA buffers."""

    def __init__(self, engine_path: str | Path):
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The nvidia_tensorrt backend requires both TensorRT and PyTorch"
            ) from exc

        self._trt = trt
        self._torch = torch
        self.engine_path = Path(engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as stream:
            self._engine = self._runtime.deserialize_cuda_engine(stream.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {self.engine_path}")
        # TensorRT synchronizes when handed CUDA's legacy default stream. A
        # dedicated torch stream keeps execution asynchronous while explicit
        # stream dependencies preserve the caller's normal torch semantics.
        self._stream = torch.cuda.Stream()
        self.input_names = self._tensor_names(trt.TensorIOMode.INPUT)
        self.output_names = self._tensor_names(trt.TensorIOMode.OUTPUT)

    def _tensor_names(self, mode) -> list[str]:
        return [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(index)) == mode
        ]

    def _torch_dtype(self, name: str):
        dtype = self._engine.get_tensor_dtype(name)
        mapping = {
            self._trt.float32: self._torch.float32,
            self._trt.float16: self._torch.float16,
            self._trt.int32: self._torch.int32,
            self._trt.int8: self._torch.int8,
            self._trt.bool: self._torch.bool,
        }
        try:
            return mapping[dtype]
        except KeyError as exc:
            raise TypeError(f"Unsupported TensorRT dtype for {name}: {dtype}") from exc

    def __call__(self, input_a, input_b) -> dict[str, object]:
        torch = self._torch
        inputs = {"inputA": input_a, "inputB": input_b}
        if set(self.input_names) != set(inputs):
            raise RuntimeError(
                f"Unexpected TensorRT inputs {self.input_names}; expected inputA,inputB"
            )

        batch = None
        device = None
        for name, tensor in inputs.items():
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            tensor = tensor.contiguous()
            if tensor.dtype != self._torch_dtype(name):
                tensor = tensor.to(self._torch_dtype(name))
            if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (6, 160, 160):
                raise ValueError(
                    f"{name} must have shape (N,6,160,160), got {tuple(tensor.shape)}"
                )
            if batch is None:
                batch = tensor.shape[0]
                device = tensor.device
            elif tensor.shape[0] != batch:
                raise ValueError("TensorRT input batches do not match")
            elif tensor.device != device:
                raise ValueError("TensorRT inputs must be on the same CUDA device")
            inputs[name] = tensor
            tensor.record_stream(self._stream)
            if not self._context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"Engine rejected shape {tuple(tensor.shape)} for {name}")
            self._context.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for name in self.output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Unresolved TensorRT output shape for {name}: {shape}")
            output = torch.empty(
                shape, device=inputs["inputA"].device, dtype=self._torch_dtype(name)
            )
            output.record_stream(self._stream)
            self._context.set_tensor_address(name, output.data_ptr())
            outputs[name] = output

        caller_stream = torch.cuda.current_stream(device=inputs["inputA"].device)
        self._stream.wait_stream(caller_stream)
        if not self._context.execute_async_v3(stream_handle=self._stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed: {self.engine_path}")
        caller_stream.wait_stream(self._stream)
        return outputs

    def close(self) -> None:
        self._context = None
        self._engine = None
        self._runtime = None
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
