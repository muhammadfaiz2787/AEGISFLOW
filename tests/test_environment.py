import pandas as pd
import numpy as np

from aegisflow.config import FEATURES
from aegisflow.envs.aegisflow_env import AegisFlowEnv


DATASET_PATH = "data/processed/aegisflow_100k.csv"


def main():

    print("=" * 70)
    print("AEGISFLOW ENVIRONMENT TEST")
    print("=" * 70)

    df = pd.read_csv(
        DATASET_PATH
    )

    env = AegisFlowEnv(
        df,
        episode_length=5,
        seed=42,
    )

    # ========================================================
    # RESET
    # ========================================================

    state, info = env.reset()

    print("\n[1] RESET")

    print(
        "State shape:",
        state.shape,
    )

    print(
        "State dtype:",
        state.dtype,
    )

    print(
        "State min:",
        state.min(),
    )

    print(
        "State max:",
        state.max(),
    )

    print(
        "Scenario:",
        info["scenario"],
    )

    print(
        "Security requirement:",
        info["security_requirement"],
    )

    print(
        "Required level:",
        info["required_level"],
    )

    # ========================================================
    # ASSERT STATE
    # ========================================================

    assert state.shape == (
        len(FEATURES),
    )

    assert state.dtype == np.float32

    assert np.all(
        state >= 0.0
    )

    assert np.all(
        state <= 1.0
    )

    # ========================================================
    # TEST ALL ACTIONS
    # ========================================================

    print("\n[2] ACTION TEST")

    for action in range(4):

        env.current = df.iloc[0]

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        print(
            f"Action {action} "
            f"({info['action_name']}): "
            f"reward={reward:.6f}"
        )

        assert isinstance(
            reward,
            float,
        )

        assert next_state.shape == (
            len(FEATURES),
        )

        assert np.all(
            next_state >= 0.0
        )

        assert np.all(
            next_state <= 1.0
        )

    # ========================================================
    # EPISODE TEST
    # ========================================================

    print("\n[3] EPISODE TEST")

    env.reset()

    for i in range(5):

        _, reward, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )

        print(
            f"Step {i + 1}: "
            f"reward={reward:.6f}, "
            f"terminated={terminated}"
        )

    assert terminated is True

    print("\n" + "=" * 70)
    print("ENVIRONMENT TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()