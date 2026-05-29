"""Quick inference test with real XR0 LIBERO weights."""

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.models import get_model


def main():
    # Create a dummy image as numpy array (224x224 RGB)
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Config pointing to real weights
    cfg = OmegaConf.create(
        {
            "model_type": "xr0",
            "model_path": "./models/Xiaomi-Robotics-0-LIBERO",
            "precision": "bf16",
            "is_lora": False,
            "action_dim": 32,
            "num_action_chunks": 30,
            "num_steps": 5,
            "xr0": {
                "state_shape": [1, 32],
                "action_shape": [30, 32],
            },
        }
    )

    print("Loading XR0 model from real weights...")
    model = get_model(cfg)
    print(f"Model loaded: {type(model).__name__}")

    # Build env_obs dict matching the expected interface
    env_obs = {
        "main_images": [image],  # list of (H, W, 3) uint8 arrays
        "states": np.zeros((1, 32), dtype=np.float32),
        "task_descriptions": ["pick up the red cup"],
    }

    # Test predict_action_batch
    print("\nRunning predict_action_batch...")
    actions, result = model.predict_action_batch(env_obs, mode="eval")

    print(f"\nResults:")
    print(f"  actions shape: {actions.shape}")
    print(f"  actions dtype: {actions.dtype}")
    print(f"  prev_logprobs: {result.get('prev_logprobs', 'N/A')}")
    print(f"  prev_values: {result.get('prev_values', 'N/A')}")

    # Verify shape
    assert actions.shape == (1, 30, 32), f"Expected (1, 30, 32), got {actions.shape}"
    print("\n✅ Inference test passed!")


if __name__ == "__main__":
    main()
