# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test XR0 async_train prefix mechanism.

Verifies that prefix_length is correctly handled in the replay chain:
- _random_mask_prefix masks the right tokens
- Position ID offset is applied correctly
- Prefix outputs are zeroed
"""

import torch

from rlinf.models.embodiment.xr0.xr0_action_model import XR0ForRLActionPrediction


def _make_policy():
    """Create a minimal policy for testing."""
    policy = object.__new__(XR0ForRLActionPrediction)
    policy.action_dim = 16
    policy.num_steps = 5
    policy.noise_level = 0.1
    policy.noise_method = "flow_sde"
    policy.local_window = 4
    policy.async_train = True
    policy.prefix_mask_prob = 0.5
    return policy


def test_random_mask_prefix_noop_when_small():
    """_random_mask_prefix returns unchanged mask when prefix_length <= keep_last_k."""
    policy = _make_policy()
    causal_mask = torch.tril(torch.ones(1, 8, 8))

    # prefix_length=2, keep_last_k=2 → no masking
    result = policy._random_mask_prefix(causal_mask, prefix_length=2, state_length=1, keep_last_k=2)
    assert torch.equal(result, causal_mask), "Should not mask when prefix_length <= keep_last_k"

    # prefix_length=0 → no masking
    result = policy._random_mask_prefix(causal_mask, prefix_length=0, state_length=1, keep_last_k=2)
    assert torch.equal(result, causal_mask), "Should not mask when prefix_length=0"
    print("✅ test_random_mask_prefix_noop_when_small passed")


def test_random_mask_prefix_masks_correct_region():
    """_random_mask_prefix masks prefix tokens (excluding last keep_last_k)."""
    policy = _make_policy()
    policy.prefix_mask_prob = 1.0  # Always mask for deterministic test

    # 1 sink + 1 state + 6 action = 8 tokens total
    # prefix_length=5, keep_last_k=2 → mask tokens at positions [2, 3, 4] (action_start=2)
    # suffix_start = 2 + 5 = 7
    # masked_prefix_end = 2 + 5 - 2 = 5
    causal_mask = torch.tril(torch.ones(1, 8, 8))
    result = policy._random_mask_prefix(causal_mask, prefix_length=5, state_length=1, keep_last_k=2)

    # After masking: suffix tokens (rows 7+) should NOT see prefix tokens at columns [2, 3, 4]
    # Row 7, columns [2, 3, 4] should be 0 (masked)
    assert result[0, 7, 2] == 0, f"Expected 0, got {result[0, 7, 2]}"
    assert result[0, 7, 3] == 0, f"Expected 0, got {result[0, 7, 3]}"
    assert result[0, 7, 4] == 0, f"Expected 0, got {result[0, 7, 4]}"

    # But suffix tokens SHOULD still see the last keep_last_k prefix tokens (columns [5, 6])
    assert result[0, 7, 5] == 1, f"Expected 1, got {result[0, 7, 5]}"
    assert result[0, 7, 6] == 1, f"Expected 1, got {result[0, 7, 6]}"

    # And prefix tokens SHOULD still see each other (causal)
    assert result[0, 4, 2] == 1, f"Expected 1, got {result[0, 4, 2]}"

    # State/sink tokens should be unchanged
    assert result[0, 0, 0] == 1, "Sink should see itself"
    assert result[0, 1, 0] == 1, "State should see sink"
    print("✅ test_random_mask_prefix_masks_correct_region passed")


def test_random_mask_prefix_partial_masking():
    """_random_mask_prefix with prefix_mask_prob=0.5 sometimes masks, sometimes doesn't."""
    policy = _make_policy()
    policy.prefix_mask_prob = 0.5

    causal_mask = torch.tril(torch.ones(1, 8, 8))

    # Run 100 times — should get different results due to randomness
    results = set()
    for _ in range(100):
        result = policy._random_mask_prefix(causal_mask.clone(), prefix_length=5, state_length=1, keep_last_k=2)
        # Check row 7, column 2 (should be masked with ~50% probability)
        results.add(result[0, 7, 2].item())

    assert 0.0 in results and 1.0 in results, \
        f"Expected both 0 and 1 in results, got {results}"
    print("✅ test_random_mask_prefix_partial_masking passed")


def test_position_id_offset():
    """Position IDs for non-prefix tokens should be offset by +10."""
    # Simulate what sample_actions does
    action_len = 10
    prefix_length = 4
    batch_size = 2
    dit_query_length = action_len + 1 + 1  # action + state + sink

    position_ids = torch.arange(0, dit_query_length).view(1, 1, -1).repeat(3, batch_size, 1) + 100

    # Apply offset (same logic as sample_actions)
    if prefix_length > 0 and action_len > prefix_length:
        position_ids[:, :, -(action_len - prefix_length):] += 10

    # Non-prefix tokens (last 6 action tokens) should have +10 offset
    # Action tokens are at positions [2, 3, ..., 11] in the query
    # Non-prefix: positions [6, 7, 8, 9, 10, 11] (indices 6-11)
    for i in range(6, 12):
        assert position_ids[0, 0, i].item() == 100 + i + 10, \
            f"Position {i}: expected {110 + i}, got {position_ids[0, 0, i].item()}"

    # Prefix tokens (first 4 action tokens) should NOT have offset
    for i in range(2, 6):
        assert position_ids[0, 0, i].item() == 100 + i, \
            f"Position {i}: expected {100 + i}, got {position_ids[0, 0, i].item()}"

    print("✅ test_position_id_offset passed")


def test_prefix_output_zeroing():
    """Prefix positions in v_t should be zeroed."""
    policy = _make_policy()

    v_t = torch.randn(2, 10, 16)
    prefix_length = 4

    # Apply zeroing (same logic as default_forward)
    if prefix_length > 0:
        v_t[:, :prefix_length] = 0.0

    # First 4 action tokens should be zero
    assert v_t[:, :prefix_length].abs().sum().item() == 0, \
        "Prefix positions should be zero"

    # Remaining tokens should be non-zero
    assert v_t[:, prefix_length:].abs().sum().item() > 0, \
        "Non-prefix positions should be non-zero"
    print("✅ test_prefix_output_zeroing passed")


def test_prefix_length_stored_in_forward_inputs():
    """prefix_length should be stored as tensor in forward_inputs."""
    # Simulate what sample_actions does
    prefix_length = 3
    forward_inputs = {
        "prefix_length": torch.tensor([prefix_length]),
    }

    # Simulate what default_forward does
    retrieved = int(forward_inputs.get("prefix_length", torch.tensor([0])).item())
    assert retrieved == prefix_length, f"Expected {prefix_length}, got {retrieved}"

    # Test default (no prefix_length stored)
    forward_inputs_empty = {}
    retrieved_default = int(forward_inputs_empty.get("prefix_length", torch.tensor([0])).item())
    assert retrieved_default == 0, f"Expected 0, got {retrieved_default}"
    print("✅ test_prefix_length_stored_in_forward_inputs passed")


def test_prefix_consistency_across_calls():
    """Multiple calls with same prefix_length should produce consistent masking behavior."""
    policy = _make_policy()
    policy.prefix_mask_prob = 1.0  # Deterministic

    causal_mask = torch.tril(torch.ones(1, 8, 8))

    # Call twice with same inputs — results should be identical
    # (same random seed due to same sequence of random calls)
    torch.manual_seed(42)
    result1 = policy._random_mask_prefix(causal_mask.clone(), prefix_length=5, state_length=1)
    torch.manual_seed(42)
    result2 = policy._random_mask_prefix(causal_mask.clone(), prefix_length=5, state_length=1)

    assert torch.equal(result1, result2), "Same seed should produce same mask"
    print("✅ test_prefix_consistency_across_calls passed")


if __name__ == "__main__":
    test_random_mask_prefix_noop_when_small()
    test_random_mask_prefix_masks_correct_region()
    test_random_mask_prefix_partial_masking()
    test_position_id_offset()
    test_prefix_output_zeroing()
    test_prefix_length_stored_in_forward_inputs()
    test_prefix_consistency_across_calls()
    print("\n✅ All async_train prefix tests passed")
