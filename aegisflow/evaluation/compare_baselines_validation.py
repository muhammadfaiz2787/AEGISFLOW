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

from aegisflow.environment.reward import (
    calculate_reward,
)


# ============================================================
# PATHS
# ============================================================

VALIDATION_PATH = Path(
    "data/processed/splits/validation.csv"
)

MODEL_PATH = Path(
    "models/aegisflow_ddqn_train_split.pt"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "aegisflow_validation_baseline_comparison.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(config):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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


# ============================================================
# DDQN PREDICTION
# ============================================================

def predict_ddqn(
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


# ============================================================
# CALCULATE ORACLE
# ============================================================

def calculate_oracle(df):

    actions = []
    rewards = []

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

        actions.append(
            best_action
        )

        rewards.append(
            action_rewards[
                best_action
            ]
        )

    return (
        np.asarray(
            actions,
            dtype=np.int64,
        ),
        np.asarray(
            rewards,
            dtype=np.float64,
        ),
    )


# ============================================================
# EVALUATE ONE POLICY
# ============================================================

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

        action = int(
            actions[i]
        )

        result = calculate_reward(
            row,
            action,
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

    actions = np.asarray(
        actions,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Regret
    # --------------------------------------------------------

    regret = (
        oracle_rewards
        - rewards
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    agreement = (
        actions
        == oracle_actions
    )

    # --------------------------------------------------------
    # Under / over selection
    # --------------------------------------------------------

    under_selection = (
        actions
        < oracle_actions
    )

    over_selection = (
        actions
        > oracle_actions
    )

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {
        "mean_reward":
            rewards.mean(),

        "mean_regret":
            regret.mean(),

        "oracle_agreement_pct":
            agreement.mean() * 100,

        "under_selection_pct":
            under_selection.mean() * 100,

        "over_selection_pct":
            over_selection.mean() * 100,

        "mean_violation":
            violations.mean(),

        "median_violation":
            np.median(
                violations
            ),

        "p95_violation":
            np.quantile(
                violations,
                0.95,
            ),

        "nonzero_violation_pct":
            (
                violations > 1e-9
            ).mean() * 100,

        "mean_risk":
            risks.mean(),

        "mean_latency_ms":
            latencies.mean(),

        "mean_compute_cost":
            compute_costs.mean(),

        "mean_bandwidth_cost":
            bandwidth_costs.mean(),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "AEGISFLOW VALIDATION BASELINE COMPARISON"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation size: "
        f"{len(df):,}"
    )

    print(
        "Dataset:",
        VALIDATION_PATH,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    config = TrainConfig()

    model, device = load_model(
        config
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DDQN
    # --------------------------------------------------------

    ddqn_actions = predict_ddqn(
        model,
        device,
        df,
    )

    # ========================================================
    # POLICIES
    # ========================================================

    n = len(df)

    policies = {
        "Always LOW":
            np.zeros(
                n,
                dtype=np.int64,
            ),

        "Always MEDIUM":
            np.ones(
                n,
                dtype=np.int64,
            ),

        "Always HIGH":
            np.full(
                n,
                2,
                dtype=np.int64,
            ),

        "Always CRITICAL":
            np.full(
                n,
                3,
                dtype=np.int64,
            ),

        "Required Level":
            df[
                "required_level"
            ]
            .to_numpy(
                dtype=np.int64
            ),

        "AegisFlow DDQN":
            ddqn_actions,

        "Oracle":
            oracle_actions,
    }

    # ========================================================
    # EVALUATE
    # ========================================================

    rows = []

    for policy_name, actions in (
        policies.items()
    ):

        print(
            f"Evaluating: "
            f"{policy_name}"
        )

        metrics = evaluate_policy(
            df,
            actions,
            oracle_actions,
            oracle_rewards,
        )

        metrics[
            "policy"
        ] = policy_name

        rows.append(
            metrics
        )

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    results = results[
        [
            "policy",

            "mean_reward",
            "mean_regret",
            "oracle_agreement_pct",

            "under_selection_pct",
            "over_selection_pct",

            "mean_violation",
            "median_violation",
            "p95_violation",
            "nonzero_violation_pct",

            "mean_risk",

            "mean_latency_ms",
            "mean_compute_cost",
            "mean_bandwidth_cost",
        ]
    ]

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 78)
    print("REWARD / DECISION QUALITY")
    print("=" * 78)

    print(
        results[
            [
                "policy",
                "mean_reward",
                "mean_regret",
                "oracle_agreement_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("SELECTION BEHAVIOR")
    print("=" * 78)

    print(
        results[
            [
                "policy",
                "under_selection_pct",
                "over_selection_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("SECURITY METRICS")
    print("=" * 78)

    print(
        results[
            [
                "policy",
                "mean_violation",
                "median_violation",
                "p95_violation",
                "nonzero_violation_pct",
                "mean_risk",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("PERFORMANCE COST")
    print("=" * 78)

    print(
        results[
            [
                "policy",
                "mean_latency_ms",
                "mean_compute_cost",
                "mean_bandwidth_cost",
            ]
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
    print("BASELINE COMPARISON COMPLETE")
    print("=" * 78)

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()