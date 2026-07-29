from __future__ import annotations

import time
import numpy as np

from policy_runtime.remote_policy_client import RemotePolicyClient, RemotePolicyConfig


def main() -> None:
    policy = RemotePolicyClient(
        RemotePolicyConfig(
            host="127.0.0.1",
            port=18000,
        )
    )

    observation = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((7,), dtype=np.float32),
        "prompt": "pick up the object",
    }

    for i in range(3):
        start = time.perf_counter()
        result = policy.infer(observation)
        elapsed = time.perf_counter() - start

        if "actions" not in result:
            raise KeyError(
                f"Server response has no 'actions' key. "
                f"Keys: {list(result.keys())}"
            )

        actions = np.asarray(result["actions"], dtype=np.float32)

        print()
        print(f"========== Inference {i + 1}/3 ==========")
        print("Response keys:", list(result.keys()))
        print("Actions shape:", actions.shape)
        print("Actions dtype:", actions.dtype)
        print("All finite:", bool(np.isfinite(actions).all()))
        print("First action:", actions[0])
        print(f"Total client latency: {elapsed:.3f} s")
        print("Policy timing:", result.get("policy_timing"))
        print("Server timing:", result.get("server_timing"))
    policy.close()


if __name__ == "__main__":
    main()
