from __future__ import annotations

import io
import pickle
import zlib
from typing import Any

import numpy as np


RECORD_CODEC_ATTR = "cari4d_record_codec"
RECORD_CODEC = "pickle5+zlib1"
PICKLE_PROTOCOL = 5
ZLIB_LEVEL = 1


def _restore_ndarray(data: bytes, dtype_descr: Any, shape: tuple[int, ...], order: str) -> np.ndarray:
    dtype = np.dtype(dtype_descr)
    return np.frombuffer(data, dtype=dtype).reshape(shape, order=order).copy(order=order)


def _restore_numpy_scalar(data: bytes, dtype_descr: Any) -> np.generic:
    return np.frombuffer(data, dtype=np.dtype(dtype_descr), count=1)[0]


class _PortableProtocol5Pickler(pickle.Pickler):
    def reducer_override(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if type(value) is not np.ndarray:
                raise TypeError(f"Unsupported NumPy array subclass in render H5 record: {type(value).__name__}")
            if value.dtype.hasobject:
                raise TypeError("Object-dtype arrays are not supported in render H5 records")
            order = "F" if value.flags.f_contiguous and not value.flags.c_contiguous else "C"
            dtype_descr = np.lib.format.dtype_to_descr(value.dtype)
            return _restore_ndarray, (value.tobytes(order=order), dtype_descr, value.shape, order)
        if isinstance(value, np.generic):
            if value.dtype.hasobject:
                raise TypeError("Object-dtype NumPy scalars are not supported in render H5 records")
            return _restore_numpy_scalar, (value.tobytes(), np.lib.format.dtype_to_descr(value.dtype))
        return NotImplemented


def _pickle_protocol5(value: Any) -> bytes:
    stream = io.BytesIO()
    _PortableProtocol5Pickler(stream, protocol=PICKLE_PROTOCOL).dump(value)
    return stream.getvalue()


def create_pickled_dataset(parent: Any, key: str, value: Any) -> Any:
    payload = _pickle_protocol5(value)
    node = parent.create_dataset(key, data=np.void(zlib.compress(payload, level=ZLIB_LEVEL)))
    node.attrs[RECORD_CODEC_ATTR] = RECORD_CODEC
    return node


def decode_pickled_payload(payload: bytes, codec: Any) -> Any:
    if isinstance(codec, bytes):
        codec = codec.decode("utf-8")
    if codec is None:
        return pickle.loads(payload)
    if codec != RECORD_CODEC:
        raise ValueError(f"Unsupported render H5 record codec: {codec!r}")
    return pickle.loads(zlib.decompress(payload))


def load_pickled_dataset(node: Any) -> Any:
    return decode_pickled_payload(bytes(node[()]), node.attrs.get(RECORD_CODEC_ATTR))
