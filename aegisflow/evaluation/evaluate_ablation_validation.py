from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aegisflow.config import (
    ACTIONS,
    TrainConfig,
    get_feature_set,
)
from aegisflow.rl.networks import QNetwork
from aegisflow.environment.reward import calculate_reward


VALIDATION_PATH = Path(
    "data/processed/splits/validation.csv"
)

MODEL_DIR = Path("models")

OUTPUT_PATH = Path(
    "data/processed/aegisflow_ablation_validation.csv"
)

SEED = 42

EXPERIMENTS = {
    "full": {
        "feature_set": "full",
        "model": "aegisflow_ddqn_full_seed42.pt",
    },

    "no-threat": {
        "feature_set": "no-threat",
        "model": "aegisflow_ddqn_no_threat_seed42.pt",
    },

    "no-trust": {
        "feature_set": "no-trust",
        "model": "aegisflow_ddqn_no_trust_seed42.pt",
    },

    "no-latency": {
        "feature_set": "no-latency",
        "model": "aegisflow_ddqn_no_latency_seed42.pt",
    },
}


def calculate_oracle(df):
    oracle_actions = []
    oracle_rewards = []

    for _, row in df.iterrows():
        rewards = []

        for action in range(len(ACTIONS)):
            result = calculate_reward(
                row,
                action,
            )

            rewards.append(
                result["reward"]
            )

        best_action = int(
            np.argmax(rewards)
        )

        oracle_actions.append(
            best_action
        )

        oracle_rewards.append(
            rewards[best_action]
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
    state_dim,
    config,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = QNetwork(
        state_dim=state_dim,
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
    features,
):
    states = torch.as_tensor(
        df[features].to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        q_values = model(states)

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
        "AEGISFLOW FEATURE ABLATION VALIDATION"
    )
    print("=" * 78)

    df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation size: "
        f"{len(df):,}"
    )

    print(
        "Seed:",
        SEED,
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

    print(
        "Oracle complete."
    )

    config = TrainConfig()

    rows = []

    for experiment_name, spec in (
        EXPERIMENTS.items()
    ):
        print()
        print("-" * 78)

        feature_set_name = (
            spec["feature_set"]
        )

        features = get_feature_set(
            feature_set_name
        )

        state_dim = len(
            features
        )

        model_path = (
            MODEL_DIR
            / spec["model"]
        )

        print(
            "Experiment:",
            experiment_name,
        )

        print(
            "State dimension:",
            state_dim,
        )

        print(
            "Features:",
            features,
        )

        print(
            "Model:",
            model_path,
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: "
                f"{model_path}"
            )

        (
            model,
            device,
            checkpoint,
        ) = load_model(
            model_path,
            state_dim,
            config,
        )

        print(
            "Device:",
            device,
        )

        checkpoint_feature_set = (
            checkpoint.get(
                "feature_set",
                None,
            )
        )

        checkpoint_state_dim = (
            checkpoint.get(
                "state_dim",
                None,
            )
        )

        if checkpoint_feature_set is not None:
            print(
                "Checkpoint feature set:",
                checkpoint_feature_set,
            )

        if checkpoint_state_dim is not None:
            print(
                "Checkpoint state dimension:",
                checkpoint_state_dim,
            )

        actions = predict_actions(
            model,
            device,
            df,
            features,
        )

        metrics = evaluate_policy(
            df,
            actions,
            oracle_actions,
            oracle_rewards,
        )

        rows.append(
            {
                "experiment":
                    experiment_name,

                "state_dim":
                    state_dim,

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

    # ========================================================
    # DELTA VS FULL MODEL
    # ========================================================

    full_row = (
        results[
            results[
                "experiment"
            ]
            == "full"
        ]
        .iloc[0]
    )

    results[
        "delta_agreement_vs_full"
    ] = (
        results[
            "agreement_pct"
        ]
        - full_row[
            "agreement_pct"
        ]
    )

    results[
        "delta_reward_vs_full"
    ] = (
        results[
            "mean_reward"
        ]
        - full_row[
            "mean_reward"
        ]
    )

    results[
        "delta_regret_vs_full"
    ] = (
        results[
            "mean_regret"
        ]
        - full_row[
            "mean_regret"
        ]
    )

    results[
        "delta_risk_vs_full"
    ] = (
        results[
            "mean_risk"
        ]
        - full_row[
            "mean_risk"
        ]
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 78)
    print(
        "ABLATION RESULTS"
    )
    print("=" * 78)

    columns = [
        "experiment",
        "state_dim",
        "agreement_pct",
        "mean_reward",
        "mean_regret",
        "under_selection_pct",
        "over_selection_pct",
        "mean_violation",
        "p95_violation",
        "mean_risk",
        "mean_latency_ms",
        "mean_compute_cost",
        "mean_bandwidth_cost",
    ]

    print(
        results[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "CHANGE RELATIVE TO FULL MODEL"
    )
    print("=" * 78)

    delta_columns = [
        "experiment",
        "delta_agreement_vs_full",
        "delta_reward_vs_full",
        "delta_regret_vs_full",
        "delta_risk_vs_full",
    ]

    print(
        results[
            delta_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "ABLATION EVALUATION COMPLETE"
    )
    print("=" * 78)

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()