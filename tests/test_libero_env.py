"""Test LIBERO environment reset and step."""

import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np

def test_libero_basic():
    """Test basic LIBERO environment functionality."""
    print("Testing LIBERO environment...")

    try:
        from libero.libero import benchmark
        print("✅ LIBERO imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LIBERO: {e}")
        return

    try:
        # Get a benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        print(f"Available benchmarks: {list(benchmark_dict.keys())}")
    except Exception as e:
        print(f"❌ Failed to get benchmark: {e}")
        return

    try:
        # Get task suite
        task_suite = benchmark_dict["libero_10"]()
        print(f"✅ Task suite loaded: {task_suite.name}")
        print(f"   Number of tasks: {task_suite.n_tasks}")
    except Exception as e:
        print(f"❌ Failed to load task suite: {e}")
        return

    try:
        # Get task
        task = task_suite.get_task(0)
        task_name = task.name
        task_description = task.language
        task_bddl_file = task.bddl_file
        print(f"✅ Task loaded: {task_name}")
        print(f"   Description: {task_description}")
        print(f"   BDDL file: {task_bddl_file}")
    except Exception as e:
        print(f"❌ Failed to get task: {e}")
        return

    try:
        # Create environment
        from libero.libero.envs import OffScreenRenderEnv
        # Use absolute path for BDDL file
        bddl_dir = "/opt/venv/openvla/libero/libero/libero/bddl_files/libero_10"
        bddl_file = os.path.join(bddl_dir, os.path.basename(task_bddl_file))
        print(f"   BDDL file (absolute): {bddl_file}")
        print(f"   BDDL file exists: {os.path.exists(bddl_file)}")

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env = OffScreenRenderEnv(**env_args)
        print("✅ Environment created")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # Reset environment
        print("Resetting environment...")
        obs = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'not a dict'}")
    except Exception as e:
        print(f"❌ Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # Step environment
        print("Stepping environment...")
        # LIBERO uses 7-DOF action (position + rotation + gripper)
        action = np.zeros(7)
        obs, reward, done, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
    except Exception as e:
        print(f"❌ Failed to step environment: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        env.close()
        print("✅ Environment closed")
    except Exception as e:
        print(f"⚠️ Failed to close environment: {e}")

    print("\n🎉 All LIBERO tests passed!")


if __name__ == "__main__":
    test_libero_basic()
