import numpy as np
import pandas as pd
import torch

from aegisflow.config import FEATURES, TrainConfig, ACTIONS
from aegisflow.rl.networks import QNetwork
from aegisflow.environment.reward import calculate_reward


DATASET_PATH = "data/processed/aegisflow_100k.csv"
ORACLE_PATH = "data/processed/aegisflow_oracle_100k.csv"
MODEL_PATH = "models/aegisflow_ddqn.pt"


def load_model(config):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = QNetwork(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["online_net"]
    )

    model.eval()

    return model, device


def evaluate():

    print("=" * 70)
    print("AEGISFLOW DDQN EVALUATION")
    print("=" * 70)

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------

    df = pd.read_csv(DATASET_PATH)
    oracle = pd.read_csv(ORACLE_PATH)

    if len(df) != len(oracle):
        raise ValueError(
            "Dataset and Oracle dataset sizes do not match."
        )

    print(f"Dataset size: {len(df):,}")

    # -------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------

    config = TrainConfig()

    model, device = load_model(config)

    print("Device:", device)
    print("Model:", MODEL_PATH)

    # -------------------------------------------------
    # STATE MATRIX
    # -------------------------------------------------

    states = torch.tensor(
        df[FEATURES].values,
        dtype=torch.float32,
        device=device,
    )

    # -------------------------------------------------
    # DDQN PREDICTION
    # -------------------------------------------------

    with torch.no_grad():

        q_values = model(states)

        predicted_actions = torch.argmax(
            q_values,
            dim=1,
        ).cpu().numpy()

    df["ddqn_action"] = predicted_actions

    df["oracle_action"] = oracle["oracle_action"].values
    df["oracle_reward"] = oracle["oracle_reward"].values

    # -------------------------------------------------
    # AGREEMENT
    # -------------------------------------------------

    df["agreement"] = (
        df["ddqn_action"]
        == df["oracle_action"]
    )

    agreement = df["agreement"].mean() * 100

    print()
    print("Oracle agreement:")
    print(
        f"{agreement:.2f}%"
    )

    # -------------------------------------------------
    # ACTION DISTRIBUTION
    # -------------------------------------------------

    print()
    print("DDQN action distribution:")

    distribution = (
        df["ddqn_action"]
        .value_counts()
        .sort_index()
    )

    print(distribution)

    print()
    print("DDQN action percentage:")

    print(
        (
            df["ddqn_action"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(3)
    )

    # -------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------

    print()
    print("DDQN vs Oracle:")

    confusion = pd.crosstab(
        df["oracle_action"],
        df["ddqn_action"],
        rownames=["Oracle"],
        colnames=["DDQN"],
    )

    print(confusion)

    print()
    print("DDQN vs Oracle (%):")

    confusion_pct = pd.crosstab(
        df["oracle_action"],
        df["ddqn_action"],
        normalize="index",
    ) * 100

    print(
        confusion_pct.round(2)
    )

    # -------------------------------------------------
    # DDQN REWARD
    # -------------------------------------------------

    ddqn_rewards = []

    for _, row in df.iterrows():

        action = int(
            row["ddqn_action"]
        )

        result = calculate_reward(
            row,
            action,
        )

        ddqn_rewards.append(
            result["reward"]
        )

    df["ddqn_reward"] = ddqn_rewards

    # -------------------------------------------------
    # REGRET
    # -------------------------------------------------

    df["regret"] = (
        df["oracle_reward"]
        - df["ddqn_reward"]
    )

    print()
    print("Reward statistics:")

    print(
        df[
            [
                "ddqn_reward",
                "oracle_reward",
                "regret",
            ]
        ].describe()
    )

    # -------------------------------------------------
    # EXACT ORACLE REWARD
    # -------------------------------------------------

    print()
    print("Reward comparison:")

    print(
        f"Mean DDQN reward: "
        f"{df['ddqn_reward'].mean():.6f}"
    )

    print(
        f"Mean Oracle reward: "
        f"{df['oracle_reward'].mean():.6f}"
    )

    print(
        f"Mean regret: "
        f"{df['regret'].mean():.6f}"
    )

    # -------------------------------------------------
    # AGREEMENT BY REQUIRED LEVEL
    # -------------------------------------------------

    print()
    print(
        "Agreement by required level:"
    )

    level_agreement = (
        df.groupby("required_level")[
            "agreement"
        ]
        .mean()
        * 100
    )

    print(
        level_agreement.round(2)
    )

    # -------------------------------------------------
    # AGREEMENT BY SCENARIO
    # -------------------------------------------------

    print()
    print(
        "Agreement by scenario:"
    )

    scenario_agreement = (
        df.groupby("scenario")[
            "agreement"
        ]
        .mean()
        * 100
    )

    print(
        scenario_agreement
        .sort_values()
        .round(2)
    )

    # -------------------------------------------------
    # REGRET BY REQUIRED LEVEL
    # -------------------------------------------------

    print()
    print(
        "Mean regret by required level:"
    )

    regret_level = (
        df.groupby("required_level")[
            "regret"
        ]
        .mean()
    )

    print(
        regret_level.round(6)
    )

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------

    output_path = (
        "data/processed/"
        "aegisflow_ddqn_evaluation_100k.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "Evaluation dataset saved to:"
    )

    print(output_path)


if __name__ == "__main__":
    evaluate()