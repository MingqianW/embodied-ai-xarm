from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


class PolicyConnectionError(RuntimeError):
    pass


class PolicyTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True)
class RemotePolicyConfig:
    host: str = "127.0.0.1"
    port: int = 18000
    connect_timeout_s: float = 10.0
    inference_timeout_s: float = 10.0
    api_key: str | None = None

    @property
    def uri(self) -> str:
        base = self.host if self.host.startswith(("ws://", "wss://")) else f"ws://{self.host}"
        return base if self.port is None else f"{base}:{self.port}"


class RemotePolicyClient:
    """Bounded OpenPI WebSocket transport.

    This uses OpenPI's MessagePack codec while adding explicit connection and
    receive timeouts that the upstream convenience client does not expose.
    """

    def __init__(self, config: RemotePolicyConfig = RemotePolicyConfig()) -> None:
        self.config = config
        self._connection: Any | None = None
        self._packer: Any | None = None
        self.server_metadata: dict[str, Any] = {}
        self.last_inference_latency_s: float | None = None

    def connect(self) -> None:
        if self._connection is not None:
            return
        if self.config.connect_timeout_s <= 0 or self.config.inference_timeout_s <= 0:
            raise ValueError("Policy timeout values must be positive")
        try:
            from openpi_client import msgpack_numpy
            import websockets.sync.client
        except ModuleNotFoundError as exc:
            raise PolicyConnectionError(
                "OpenPI client dependencies are unavailable. Install "
                "third_party/openpi/packages/openpi-client."
            ) from exc

        headers = (
            {"Authorization": f"Api-Key {self.config.api_key}"}
            if self.config.api_key
            else None
        )
        try:
            connection = websockets.sync.client.connect(
                self.config.uri,
                compression=None,
                max_size=None,
                additional_headers=headers,
                open_timeout=self.config.connect_timeout_s,
            )
            metadata_raw = connection.recv(timeout=self.config.connect_timeout_s)
            metadata = msgpack_numpy.unpackb(metadata_raw)
        except TimeoutError as exc:
            raise PolicyTimeoutError(
                f"Timed out connecting to policy server {self.config.uri} "
                f"after {self.config.connect_timeout_s:.3f}s"
            ) from exc
        except Exception as exc:
            raise PolicyConnectionError(
                f"Could not connect to policy server {self.config.uri}: {exc}"
            ) from exc
        self._connection = connection
        self._packer = msgpack_numpy.Packer()
        self.server_metadata = dict(metadata) if isinstance(metadata, dict) else {"value": metadata}

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.connect()
        assert self._connection is not None
        assert self._packer is not None
        started = time.perf_counter()
        try:
            self._connection.send(self._packer.pack(observation))
            response = self._connection.recv(timeout=self.config.inference_timeout_s)
        except TimeoutError as exc:
            self.close()
            raise PolicyTimeoutError(
                f"Policy inference timed out after {self.config.inference_timeout_s:.3f}s"
            ) from exc
        except Exception as exc:
            self.close()
            raise PolicyConnectionError(f"Policy inference failed: {exc}") from exc
        self.last_inference_latency_s = time.perf_counter() - started
        if isinstance(response, str):
            raise RuntimeError(f"Policy server returned an error:\n{response}")
        from openpi_client import msgpack_numpy

        result = msgpack_numpy.unpackb(response)
        if not isinstance(result, dict):
            raise TypeError(f"Policy response must be a mapping, got {type(result).__name__}")
        return result

    def reset(self) -> None:
        return None

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None

    def __enter__(self) -> "RemotePolicyClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
