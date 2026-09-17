# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal, gr00t-free ZMQ client for a GR00T ``run_gr00t_server`` inference server.

This mirrors ``Isaac-GR00T/gr00t/policy/server_client.py`` (``PolicyClient`` +
``MsgSerializer``) so the IsaacLab container can talk to a GR00T inference server
**without installing the ``gr00t`` package** (the container is Python 3.11; GR00T needs
3.10). It is deliberately NOT a verbatim copy — three edits make it gr00t-free:

1. No ``gr00t`` imports and no ``BasePolicy`` base class.
2. ``get_action(obs)`` sends ``{"observation": obs, "options": None}`` (the wire key is
   ``"observation"``) and returns ``(action, info)``.
3. ``MsgSerializer._decode_custom`` returns the plain ``as_json`` **dict** for a
   ``ModalityConfig`` payload instead of reconstructing a ``gr00t`` object — so
   ``get_modality_config()`` yields plain dicts (``modality_keys``, ``delta_indices``, ...).

Dependencies: ``pyzmq``, ``msgpack``, ``msgpack_numpy``, ``numpy`` (see
``requirements.txt``). The wire format (msgpack + msgpack_numpy, ``allow_pickle=False``)
matches the server exactly.
"""

from __future__ import annotations

import functools
import io
import json
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


class MsgSerializer:
    """msgpack_numpy serializer with a hard ``allow_pickle=False`` boundary.

    Ported verbatim from GR00T except ``_encode_custom``/``_decode_custom``, which are
    made gr00t-free (a ``ModalityConfig`` payload decodes to its plain ``as_json`` dict).
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        """Serialize ``data`` to msgpack bytes, refusing pickle-bearing payloads."""
        default = functools.partial(
            MsgSerializer._safe_encode, chain=MsgSerializer._encode_custom
        )
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        """Deserialize msgpack ``data``, refusing pickle-bearing payloads."""
        object_hook = functools.partial(
            MsgSerializer._safe_decode, chain=MsgSerializer._decode_custom
        )
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)

    @staticmethod
    def _safe_encode(obj: Any, chain: Any = None) -> Any:
        # Refuse object-dtype ndarrays before mnp.encode would invoke pickle.
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError(
                f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                f"convert to a concrete numeric dtype before sending."
            )
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj: Any, chain: Any = None) -> Any:
        if isinstance(obj, dict):
            marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
            if marker:
                payload = obj.get("as_npy", obj.get(b"as_npy"))
                if payload is None:
                    raise ValueError(
                        "Malformed ndarray payload: marker present but 'as_npy' missing"
                    )
                return np.load(io.BytesIO(payload), allow_pickle=False)
            nd_val = obj.get(b"nd", obj.get("nd"))
            kind_val = obj.get(b"kind", obj.get("kind"))
            if nd_val and kind_val in (b"O", "O"):
                raise ValueError(
                    "Refusing to decode object-dtype ndarray payload (pickle-bearing)."
                )
        return mnp.decode(obj, chain=chain)

    @staticmethod
    def _encode_custom(obj: Any) -> Any:
        # The client never sends a ModalityConfig, so no custom encode is needed.
        return obj

    @staticmethod
    def _decode_custom(obj: Any) -> Any:
        """Decode a server ``ModalityConfig`` payload to its plain ``as_json`` dict."""
        if not isinstance(obj, dict):
            return obj
        has_modality_marker = (
            "__ModalityConfig__" in obj
            or b"__ModalityConfig__" in obj
            or "__ModalityConfig_class__" in obj
            or b"__ModalityConfig_class__" in obj
        )
        if has_modality_marker:
            key = next((k for k in ("as_json", b"as_json") if k in obj), None)
            if key is None:
                raise ValueError("Malformed ModalityConfig payload: 'as_json' missing.")
            payload = obj[key]
            if isinstance(payload, bytes):
                payload = payload.decode()
            if isinstance(payload, str):
                payload = json.loads(payload)
            return payload  # plain dict: {"modality_keys": [...], "delta_indices": [...], ...}
        return obj


class PolicyClient:
    """ZMQ REQ client for a GR00T inference server (``run_gr00t_server``).

    Args:
        host: server host (under the container's ``--network host``, use ``localhost``).
        port: server port (GR00T default 5555).
        timeout_ms: send/recv timeout.
        api_token: optional shared token (GR00T default: no auth).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ) -> None:
        """Connect a REQ socket to the GR00T inference server."""
        self._closed = False
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self) -> None:
        # Close any previous socket first. A timeout leaves the REQ socket unusable and we
        # recreate it here; without this the old one leaks, and context.term() in close()
        # blocks until every socket in the context is closed -- the process hangs on exit.
        old = getattr(self, "socket", None)
        if old is not None:
            try:
                old.close(linger=0)
            except Exception:
                pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """Send one REQ/REP round trip to ``endpoint`` and return the decoded reply."""
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            # Timeout leaves the REQ socket in an invalid state; recreate it.
            self._init_socket()
            raise
        if message == b"ERROR":
            raise RuntimeError(
                "Server error. Is the correct GR00T policy server running?"
            )
        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        """Return True if the server answers, False on any ZMQ-level failure."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            # No _init_socket() here: zmq.error.Again is a ZMQError subclass and
            # call_endpoint already recreated the socket on timeout.
            return False

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Send a nested ``{video, state, language}`` obs dict; return ``(action, info)``.

        ``action`` is a dict keyed by the action modality keys, each ``(B, T_action, D)``.
        """
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # msgpack list -> (action, info)

    def get_modality_config(self) -> dict[str, dict]:
        """Return ``{"video"|"state"|"action"|"language": {modality_keys, delta_indices, ...}}``.

        Values are plain dicts (see ``MsgSerializer._decode_custom``), not gr00t objects.
        """
        return self.call_endpoint("get_modality_config", requires_input=False)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the server-side policy state (clears its action-chunk history)."""
        return self.call_endpoint("reset", {"options": options})

    def close(self) -> None:
        """Close the socket and terminate the ZMQ context. Idempotent."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        socket = getattr(self, "socket", None)
        if socket is not None:
            try:
                socket.close(linger=0)
            except Exception:
                pass
        context = getattr(self, "context", None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass

    def __enter__(self) -> PolicyClient:
        """Enter the context manager, returning this client."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the client on context exit."""
        self.close()

    def __del__(self) -> None:
        """Best-effort close if the client was never explicitly closed."""
        try:
            self.close()
        except Exception:
            pass
