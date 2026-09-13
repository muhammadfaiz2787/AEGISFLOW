import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aegisflow.config import (
    FEATURES,
    get_feature_set,
    TrainConfig,
)
from aegisflow.envs.aegisflow_env import AegisFlowEnv
from aegisflow.rl.ddqn import DoubleDQNAgent
from dataclasses import replace

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path(
    "data/processed/splits/train.csv"
)

MODEL_DIR = Path(
    "models"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    """
    Set global random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

def load_dataset():
    """
    Load the frozen AegisFlow training split.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: "
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    if len(df) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    # --------------------------------------------------------
    # Full dataset must still contain all canonical features,
    # even when the agent observes only a subset.
    # --------------------------------------------------------

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing canonical features in dataset: "
            f"{missing}"
        )

    return df


# ============================================================
# MODEL PATH
# ============================================================

def build_model_path(
    model_dir,
    feature_set_name,
    seed,
    gamma,
):
    """
    Build a unique checkpoint name for each
    ablation configuration and random seed.
    """

    feature_tag = (
        feature_set_name
        .replace(
            "-",
            "_",
        )
    )

    gamma_tag = str(gamma).replace(
        ".",
        "p",
    )

    filename = (
        f"aegisflow_ddqn_"
        f"{feature_tag}_"
        f"gamma{gamma_tag}_"
        f"seed{seed}.pt"
    )

    return (
        model_dir
        / filename
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train AegisFlow Double DQN"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override training random seed. "
            "Default uses TrainConfig.seed."
        ),

    )

    parser.add_argument(
         "--gamma",
        type=float,
        default=None,
        help=(
            "Override discount factor gamma. "
            "Default uses TrainConfig.gamma."
        ),
    )

    parser.add_argument(
        "--feature-set",
        type=str,
        default="full",
        choices=[
            "full",
            "no-threat",
            "no-trust",
            "no-latency",
        ],
        help=(
            "Observation feature configuration "
            "for feature-ablation experiments."
        ),
    )

    return parser.parse_args()


# ============================================================
# TRAINING
# ============================================================

def train(
    seed_override=None,
    feature_set_name="full",
    gamma_override=None,
):

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = TrainConfig()

    seed = (
        config.seed
        if seed_override is None
        else int(seed_override)
    )

    gamma = (
        config.gamma
        if gamma_override is None
        else float(gamma_override)
    )

    config = replace(
        config,
        gamma=gamma,
    )

    active_features = list(
        get_feature_set(
            feature_set_name
        )
    )

    state_dim = len(
        active_features
    )

    if state_dim <= 0:
        raise ValueError(
            "Active feature set cannot be empty."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the resolved seed, not config.seed.
    # --------------------------------------------------------

    set_seed(
        seed
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 78)
    print(
        "AEGISFLOW DOUBLE DQN TRAINING"
    )
    print("=" * 78)

    print(
        "Dataset:",
        DATASET_PATH,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"Dataset size: "
        f"{len(df):,}"
    )

    print(
        "Experiment: TRAIN-SPLIT DDQN"
    )

    print(
        "Training seed:",
        seed,
    )

    print(
        "Feature set:",
        feature_set_name,
    )

    print(
        "State dimension:",
        state_dim,
    )

    print(
        "Gamma:",
        config.gamma,
    )

    print(
        "Active features:"
    )

    for index, feature in enumerate(
        active_features,
        start=1,
    ):
        print(
            f"  {index:2d}. "
            f"{feature}"
        )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = AegisFlowEnv(
        df=df,
        episode_length=(
            config.steps_per_episode
        ),
        seed=seed,
        features=active_features,
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = DoubleDQNAgent(
        config,
        state_dim=state_dim,
    )

    print()
    print(
        "Device:",
        agent.device,
    )

    print(
        "Action dimension:",
        config.action_dim,
    )

    print(
        "Hidden dimension:",
        config.hidden_dim,
    )

    print()
    print(
        "Episodes:",
        config.episodes,
    )

    print(
        "Steps / episode:",
        config.steps_per_episode,
    )

    print(
        "Total environment steps:",
        (
            config.episodes
            * config.steps_per_episode
        ),
    )

    # ========================================================
    # STORAGE
    # ========================================================

    episode_rewards = []
    episode_losses = []

    global_step = 0

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for episode in range(
        config.episodes
    ):

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        state, _ = env.reset()

        total_reward = 0.0

        losses = []

        # ----------------------------------------------------
        # Episode
        # ----------------------------------------------------

        for _ in range(
            config.steps_per_episode
        ):

            # ------------------------------------------------
            # Select action
            # ------------------------------------------------

            action = (
                agent.select_action(
                    state,
                    training=True,
                )
            )

            # ------------------------------------------------
            # Environment transition
            # ------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(
                action
            )

            done = (
                terminated
                or truncated
            )

            # ------------------------------------------------
            # Replay buffer
            # ------------------------------------------------

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            # ------------------------------------------------
            # Learning
            # ------------------------------------------------

            loss = (
                agent.train_step()
            )

            if loss is not None:
                losses.append(
                    loss
                )

            # ------------------------------------------------
            # State transition
            # ------------------------------------------------

            state = (
                next_state
            )

            total_reward += (
                reward
            )

            global_step += 1

            # ------------------------------------------------
            # Target network
            # ------------------------------------------------

            if (
                global_step
                % config.target_update_every
                == 0
            ):
                agent.update_target()

            # ------------------------------------------------
            # Epsilon decay
            # ------------------------------------------------

            agent.decay_epsilon()

            if done:
                break

        # ====================================================
        # EPISODE METRICS
        # ====================================================

        episode_rewards.append(
            float(total_reward)
        )

        if losses:

            mean_loss = float(
                np.mean(
                    losses
                )
            )

            episode_losses.append(
                mean_loss
            )

        else:

            mean_loss = None

            episode_losses.append(
                None
            )

        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            episode + 1
        ) % 10 == 0:

            recent_rewards = (
                episode_rewards[
                    -25:
                ]
            )

            mean_reward = float(
                np.mean(
                    recent_rewards
                )
            )

            loss_text = (
                f"{mean_loss:.6f}"
                if mean_loss is not None
                else "N/A"
            )

            print(
                f"Episode "
                f"{episode + 1:4d} | "
                f"Mean Reward: "
                f"{mean_reward:8.4f} | "
                f"Epsilon: "
                f"{agent.epsilon:7.4f} | "
                f"Replay: "
                f"{len(agent.replay):6d} | "
                f"Loss: "
                f"{loss_text}"
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()
    print("=" * 78)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 78)

    final_reward = float(
        np.mean(
            episode_rewards[
                -25:
            ]
        )
    )

    print(
        "Feature set:",
        feature_set_name,
    )

    print(
        "State dimension:",
        state_dim,
    )

    print(
        "Training seed:",
        seed,
    )

    print(
        "Gamma:",
        config.gamma,
    )

    print(
        f"Final mean reward "
        f"(last 25): "
        f"{final_reward:.6f}"
    )

    print(
        f"Final epsilon: "
        f"{agent.epsilon:.6f}"
    )

    print(
        f"Replay size: "
        f"{len(agent.replay):,}"
    )

    print(
        "Global steps:",
        global_step,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        build_model_path(
            model_dir=MODEL_DIR,
            feature_set_name=(
                feature_set_name
            ),
            seed=seed,
            gamma=config.gamma,
        )
    )

    checkpoint = {

        # ----------------------------------------------------
        # Neural networks
        # ----------------------------------------------------

        "online_net":
            agent.online_net.state_dict(),

        "target_net":
            agent.target_net.state_dict(),

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        "optimizer":
            agent.optimizer.state_dict(),

        # ----------------------------------------------------
        # Training state
        # ----------------------------------------------------

        "epsilon":
            float(
                agent.epsilon
            ),

        "global_step":
            int(
                global_step
            ),

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        "seed":
            int(
                seed
            ),

        "gamma": float(
            config.gamma
        ),

        # ----------------------------------------------------
        # Ablation metadata
        # ----------------------------------------------------

        "feature_set":
            feature_set_name,

        "features":
            list(
                active_features
            ),

        "state_dim":
            int(
                state_dim
            ),

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        "config":
            dict(
                config.__dict__
            ),

        # ----------------------------------------------------
        # Training history
        # ----------------------------------------------------

        "episode_rewards":
            episode_rewards,

        "episode_losses":
            episode_losses,

        # ----------------------------------------------------
        # Dataset metadata
        # ----------------------------------------------------

        "dataset_path":
            str(
                DATASET_PATH
            ),

        "dataset_size":
            int(
                len(df)
            ),
    }

    torch.save(
        checkpoint,
        model_path,
    )

    print()
    print(
        "Model saved to:"
    )

    print(
        model_path
    )

    return (
        model_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    train(
        seed_override=args.seed,
        feature_set_name=args.feature_set,
        gamma_override=args.gamma,
    )