from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aegisflow.config import FEATURES, ACTIONS
from aegisflow.environment.reward import calculate_reward


class AegisFlowEnv(gym.Env):
    """
    AegisFlow contextual security environment.

    The agent observes a configurable subset of
    data/network context features and selects one of
    four adaptive security levels:

        0 = LOW
        1 = MEDIUM
        2 = HIGH
        3 = CRITICAL

    The environment evaluates the selected policy using
    the centralized AegisFlow reward function.

    Notes
    -----
    - The full environment row always remains available
      to the reward function.
    - The agent observation can use only a subset of
      features for ablation experiments.
    """

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        df,
        episode_length=32,
        seed=None,
        features=None,
    ):
        super().__init__()

        # ====================================================
        # DATASET
        # ====================================================

        self.df = df.reset_index(
            drop=True
        )

        if len(self.df) == 0:
            raise ValueError(
                "AegisFlowEnv received an empty dataframe."
            )

        # ====================================================
        # ACTIVE OBSERVATION FEATURES
        # ====================================================

        self.features = (
            list(FEATURES)
            if features is None
            else list(features)
        )

        if len(self.features) == 0:
            raise ValueError(
                "At least one observation feature is required."
            )

        missing_features = [
            feature
            for feature in self.features
            if feature not in self.df.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing observation features in dataframe: "
                f"{missing_features}"
            )

        # ====================================================
        # EPISODE CONFIGURATION
        # ====================================================

        self.episode_length = int(
            episode_length
        )

        if self.episode_length <= 0:
            raise ValueError(
                "episode_length must be greater than zero."
            )

        # ====================================================
        # RANDOM NUMBER GENERATOR
        # ====================================================

        self.rng = np.random.default_rng(
            seed
        )

        # ====================================================
        # INTERNAL STATE
        # ====================================================

        self.current = None
        self.steps = 0

        # ====================================================
        # OBSERVATION SPACE
        # ====================================================

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                len(self.features),
            ),
            dtype=np.float32,
        )

        # ====================================================
        # ACTION SPACE
        # ====================================================

        self.action_space = spaces.Discrete(
            len(ACTIONS)
        )

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    def _next_row(self):
        """
        Sample one random context from the dataset.
        """

        index = int(
            self.rng.integers(
                0,
                len(self.df),
            )
        )

        return self.df.iloc[
            index
        ]

    def _row_to_obs(
        self,
        row,
    ):
        """
        Convert one dataframe row into the current
        agent observation vector.
        """

        observation = (
            row[
                self.features
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        return observation

    def _observation(self):
        """
        Return the observation corresponding to
        self.current.
        """

        if self.current is None:
            raise RuntimeError(
                "Environment has no current state. "
                "Call reset() first."
            )

        return self._row_to_obs(
            self.current
        )

    def _build_info(
        self,
    ):
        """
        Build metadata for the current context.
        """

        if self.current is None:
            return {}

        return {
            "security_requirement": float(
                self.current[
                    "security_requirement"
                ]
            ),
            "required_level": int(
                self.current[
                    "required_level"
                ]
            ),
            "scenario": self.current.get(
                "scenario",
                None,
            ),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options=None,
    ):
        """
        Start a new episode.

        Returns
        -------
        observation, info
        """

        super().reset(
            seed=seed
        )

        # If Gymnasium supplies a new seed during reset,
        # reinitialize the environment RNG as well.
        if seed is not None:
            self.rng = (
                np.random.default_rng(
                    seed
                )
            )

        self.steps = 0

        self.current = (
            self._next_row()
        )

        observation = (
            self._observation()
        )

        info = (
            self._build_info()
        )

        return (
            observation,
            info,
        )

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action,
    ):
        """
        Apply one security-policy action.
        """

        if self.current is None:
            raise RuntimeError(
                "Environment must be reset before step()."
            )

        action = int(
            action
        )

        if not self.action_space.contains(
            action
        ):
            raise ValueError(
                f"Invalid action: {action}"
            )

        # ----------------------------------------------------
        # Calculate reward on the CURRENT context
        # ----------------------------------------------------

        reward_info = (
            calculate_reward(
                self.current,
                action,
            )
        )

        reward = float(
            reward_info[
                "reward"
            ]
        )

        # ----------------------------------------------------
        # Information about the action that was evaluated
        # ----------------------------------------------------

        info = dict(
            reward_info
        )

        info.update(
            {
                "action":
                    action,

                "action_name":
                    ACTIONS[action],

                "required_level":
                    int(
                        self.current[
                            "required_level"
                        ]
                    ),

                "scenario":
                    self.current.get(
                        "scenario",
                        None,
                    ),

                "security_requirement":
                    float(
                        self.current[
                            "security_requirement"
                        ]
                    ),
            }
        )

        # ----------------------------------------------------
        # Advance episode counter
        # ----------------------------------------------------

        self.steps += 1

        terminated = (
            self.steps
            >=
            self.episode_length
        )

        truncated = False

        # ----------------------------------------------------
        # Sample next independent context
        # ----------------------------------------------------

        self.current = (
            self._next_row()
        )

        next_observation = (
            self._observation()
        )

        return (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        )