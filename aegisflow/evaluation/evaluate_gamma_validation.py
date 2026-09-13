from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aegisflow.config import (
    FEATURES,
    ACTIONS,
    TrainConfig,
)
from aegisflow.rl.networks import QNetwork
from aegisflow.environment.reward import calculate_reward


VALIDATION_PATH = Path(
    "data/processed/splits/validation.csv"
)

MODEL_GAMMA_095 = Path(
    "models/aegisflow_ddqn_full_seed42.pt"
)

MODEL_GAMMA_000 = Path(
    "models/aegisflow_ddqn_full_gamma0p0_seed42.pt"
)


def calculate_oracle(df):

    oracle_actions = []
    oracle_rewards = []

    for _, row in df.iterrows():

        action_rewards = []

        for action in range(
            len(ACTIONS)
        ):

            result = calculate_reward(
                row,
                action,
            )

            action_rewards.append(
                result["reward"]
            )

        best_action = int(
            np.argmax(
                action_rewards
            )
        )

        oracle_actions.append(
            best_action
        )

        oracle_rewards.append(
            action_rewards[
                best_action
            ]
        )

    return (
        np.asarray(
            oracle_actions,
            dtype=np.int64,
        ),
        np.asarray(
            oracle_rewards,
            dtype=np.float64,
        ),
    )


def load_model(
    model_path,
    config,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = QNetwork(
        state_dim=len(FEATURES),
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["online_net"]
    )

    model.eval()

    return (
        model,
        device,
        checkpoint,
    )


def predict_actions(
    model,
    device,
    df,
):

    states = torch.as_tensor(
        df[FEATURES].to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        q_values = model(
            states
        )

        actions = torch.argmax(
            q_values,
            dim=1,
        )

    return (
        actions
        .cpu()
        .numpy()
        .astype(np.int64)
    )


def evaluate_policy(
    df,
    actions,
    oracle_actions,
    oracle_rewards,
):

    rewards = []
    violations = []
    risks = []
    latencies = []
    compute_costs = []
    bandwidth_costs = []

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        result = calculate_reward(
            row,
            int(actions[i]),
        )

        rewards.append(
            result["reward"]
        )

        violations.append(
            result["violation"]
        )

        risks.append(
            result["risk"]
        )

        latencies.append(
            result["latency_ms"]
        )

        compute_costs.append(
            result["compute_cost"]
        )

        bandwidth_costs.append(
            result["bandwidth_cost"]
        )

    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    )

    violations = np.asarray(
        violations,
        dtype=np.float64,
    )

    risks = np.asarray(
        risks,
        dtype=np.float64,
    )

    latencies = np.asarray(
        latencies,
        dtype=np.float64,
    )

    compute_costs = np.asarray(
        compute_costs,
        dtype=np.float64,
    )

    bandwidth_costs = np.asarray(
        bandwidth_costs,
        dtype=np.float64,
    )

    regret = (
        oracle_rewards
        - rewards
    )

    agreement = (
        actions
        == oracle_actions
    )

    under = (
        actions
        < oracle_actions
    )

    over = (
        actions
        > oracle_actions
    )

    return {
        "agreement_pct":
            agreement.mean() * 100,

        "mean_reward":
            rewards.mean(),

        "mean_regret":
            regret.mean(),

        "p95_regret":
            np.quantile(
                regret,
                0.95,
            ),

        "under_selection_pct":
            under.mean() * 100,

        "over_selection_pct":
            over.mean() * 100,

        "mean_violation":
            violations.mean(),

        "p95_violation":
            np.quantile(
                violations,
                0.95,
            ),

        "mean_risk":
            risks.mean(),

        "mean_latency_ms":
            latencies.mean(),

        "mean_compute_cost":
            compute_costs.mean(),

        "mean_bandwidth_cost":
            bandwidth_costs.mean(),
    }


def main():

    print("=" * 78)
    print(
        "AEGISFLOW GAMMA ABLATION VALIDATION"
    )
    print("=" * 78)

    df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation size: "
        f"{len(df):,}"
    )

    print()
    print(
        "Calculating Oracle..."
    )

    (
        oracle_actions,
        oracle_rewards,
    ) = calculate_oracle(
        df
    )

    config = TrainConfig()

    experiments = {
        "gamma_0.95": MODEL_GAMMA_095,
        "gamma_0.00": MODEL_GAMMA_000,
    }

    rows = []

    for name, model_path in experiments.items():

        print()
        print("-" * 78)

        print(
            "Experiment:",
            name,
        )

        print(
            "Model:",
            model_path,
        )

        if not model_path.exists():
            raise FileNotFoundError(
                model_path
            )

        (
            model,
            device,
            checkpoint,
        ) = load_model(
            model_path,
            config,
        )

        checkpoint_gamma = (
            checkpoint.get(
                "gamma",
                checkpoint
                .get("config", {})
                .get("gamma", None),
            )
        )

        print(
            "Checkpoint gamma:",
            checkpoint_gamma,
        )

        actions = predict_actions(
            model,
            device,
            df,
        )

        metrics = evaluate_policy(
            df,
            actions,
            oracle_actions,
            oracle_rewards,
        )

        rows.append(
            {
                "experiment": name,
                **metrics,
            }
        )

        print(
            f"Agreement: "
            f"{metrics['agreement_pct']:.2f}%"
        )

        print(
            f"Mean reward: "
            f"{metrics['mean_reward']:.6f}"
        )

        print(
            f"Mean regret: "
            f"{metrics['mean_regret']:.6f}"
        )

        print(
            f"Under-selection: "
            f"{metrics['under_selection_pct']:.2f}%"
        )

        print(
            f"Over-selection: "
            f"{metrics['over_selection_pct']:.2f}%"
        )

        print(
            f"Mean violation: "
            f"{metrics['mean_violation']:.6f}"
        )

        print(
            f"Mean risk: "
            f"{metrics['mean_risk']:.6f}"
        )

    results = pd.DataFrame(
        rows
    )

    print()
    print("=" * 78)
    print(
        "GAMMA ABLATION RESULTS"
    )
    print("=" * 78)

    print(
        results.to_string(
            index=False
        )
    )

    output_path = Path(
        "data/processed/"
        "aegisflow_gamma_ablation_validation.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "Saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()