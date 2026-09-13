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
    "aegisflow_validation_evaluation.csv"
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
# CALCULATE ORACLE
# ============================================================

def calculate_validation_oracle(df):

    oracle_actions = []
    oracle_rewards = []

    print()
    print("Calculating validation Oracle...")

    for _, row in df.iterrows():

        action_rewards = []

        for action in range(len(ACTIONS)):

            result = calculate_reward(
                row,
                action,
            )

            action_rewards.append(
                result["reward"]
            )

        best_action = int(
            np.argmax(action_rewards)
        )

        best_reward = float(
            action_rewards[best_action]
        )

        oracle_actions.append(
            best_action
        )

        oracle_rewards.append(
            best_reward
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


# ============================================================
# MODEL PREDICTION
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


# ============================================================
# REWARD DETAILS
# ============================================================

def calculate_policy_metrics(
    df,
    actions,
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

    return {
        "reward": np.asarray(
            rewards,
            dtype=np.float64,
        ),

        "violation": np.asarray(
            violations,
            dtype=np.float64,
        ),

        "risk": np.asarray(
            risks,
            dtype=np.float64,
        ),

        "latency_ms": np.asarray(
            latencies,
            dtype=np.float64,
        ),

        "compute_cost": np.asarray(
            compute_costs,
            dtype=np.float64,
        ),

        "bandwidth_cost": np.asarray(
            bandwidth_costs,
            dtype=np.float64,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def evaluate():

    print("=" * 70)
    print("AEGISFLOW VALIDATION EVALUATION")
    print("=" * 70)

    print(
        "Validation dataset:",
        VALIDATION_PATH,
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation size: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    config = TrainConfig()

    model, device = load_model(
        config
    )

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    (
        oracle_actions,
        oracle_rewards,
    ) = calculate_validation_oracle(
        df
    )

    # --------------------------------------------------------
    # DDQN prediction
    # --------------------------------------------------------

    ddqn_actions = predict_ddqn(
        model,
        device,
        df,
    )

    # --------------------------------------------------------
    # Policy metrics
    # --------------------------------------------------------

    ddqn_metrics = (
        calculate_policy_metrics(
            df,
            ddqn_actions,
        )
    )

    oracle_metrics = (
        calculate_policy_metrics(
            df,
            oracle_actions,
        )
    )

    # --------------------------------------------------------
    # Add results
    # --------------------------------------------------------

    result_df = df.copy()

    result_df[
        "oracle_action"
    ] = oracle_actions

    result_df[
        "ddqn_action"
    ] = ddqn_actions

    result_df[
        "oracle_reward"
    ] = oracle_rewards

    result_df[
        "ddqn_reward"
    ] = ddqn_metrics[
        "reward"
    ]

    result_df[
        "regret"
    ] = (
        result_df["oracle_reward"]
        - result_df["ddqn_reward"]
    )

    result_df[
        "agreement"
    ] = (
        result_df["oracle_action"]
        == result_df["ddqn_action"]
    )

    # --------------------------------------------------------
    # Under / over selection
    # --------------------------------------------------------

    result_df[
        "under_selection"
    ] = (
        result_df["ddqn_action"]
        <
        result_df["oracle_action"]
    )

    result_df[
        "over_selection"
    ] = (
        result_df["ddqn_action"]
        >
        result_df["oracle_action"]
    )

    # --------------------------------------------------------
    # Reward details
    # --------------------------------------------------------

    result_df[
        "ddqn_violation"
    ] = ddqn_metrics[
        "violation"
    ]

    result_df[
        "ddqn_risk"
    ] = ddqn_metrics[
        "risk"
    ]

    result_df[
        "ddqn_latency_ms"
    ] = ddqn_metrics[
        "latency_ms"
    ]

    result_df[
        "ddqn_compute_cost"
    ] = ddqn_metrics[
        "compute_cost"
    ]

    result_df[
        "ddqn_bandwidth_cost"
    ] = ddqn_metrics[
        "bandwidth_cost"
    ]

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    agreement = (
        result_df[
            "agreement"
        ].mean()
        * 100
    )

    under_rate = (
        result_df[
            "under_selection"
        ].mean()
        * 100
    )

    over_rate = (
        result_df[
            "over_selection"
        ].mean()
        * 100
    )

    print()
    print("=" * 70)
    print("OVERALL PERFORMANCE")
    print("=" * 70)

    print(
        f"Oracle agreement: "
        f"{agreement:.2f}%"
    )

    print(
        f"Mean DDQN reward: "
        f"{result_df['ddqn_reward'].mean():.6f}"
    )

    print(
        f"Mean Oracle reward: "
        f"{result_df['oracle_reward'].mean():.6f}"
    )

    print(
        f"Mean regret: "
        f"{result_df['regret'].mean():.6f}"
    )

    print(
        f"Under-selection rate: "
        f"{under_rate:.2f}%"
    )

    print(
        f"Over-selection rate: "
        f"{over_rate:.2f}%"
    )

    # ========================================================
    # ACTION DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("DDQN ACTION DISTRIBUTION")
    print("=" * 70)

    action_counts = (
        result_df[
            "ddqn_action"
        ]
        .value_counts()
        .sort_index()
    )

    action_pct = (
        result_df[
            "ddqn_action"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print()
    print("Counts:")
    print(action_counts)

    print()
    print("Percentage:")
    print(
        action_pct.round(3)
    )

    # ========================================================
    # ORACLE DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("ORACLE ACTION DISTRIBUTION")
    print("=" * 70)

    oracle_pct = (
        result_df[
            "oracle_action"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        oracle_pct.round(3)
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 70)
    print("DDQN VS ORACLE")
    print("=" * 70)

    confusion = pd.crosstab(
        result_df[
            "oracle_action"
        ],
        result_df[
            "ddqn_action"
        ],
        rownames=["Oracle"],
        colnames=["DDQN"],
    )

    print()
    print(confusion)

    print()
    print("Row normalized (%):")

    confusion_pct = pd.crosstab(
        result_df[
            "oracle_action"
        ],
        result_df[
            "ddqn_action"
        ],
        normalize="index",
    ) * 100

    print(
        confusion_pct.round(2)
    )

    # ========================================================
    # REQUIRED LEVEL
    # ========================================================

    print()
    print("=" * 70)
    print("AGREEMENT BY REQUIRED LEVEL")
    print("=" * 70)

    level_agreement = (
        result_df
        .groupby(
            "required_level"
        )["agreement"]
        .mean()
        * 100
    )

    print(
        level_agreement.round(2)
    )

    print()
    print(
        "Mean regret by "
        "required level:"
    )

    level_regret = (
        result_df
        .groupby(
            "required_level"
        )["regret"]
        .mean()
    )

    print(
        level_regret.round(6)
    )

    # ========================================================
    # SCENARIO
    # ========================================================

    print()
    print("=" * 70)
    print("AGREEMENT BY SCENARIO")
    print("=" * 70)

    scenario_agreement = (
        result_df
        .groupby(
            "scenario"
        )["agreement"]
        .mean()
        * 100
    )

    print(
        scenario_agreement
        .sort_values()
        .round(2)
    )

    # ========================================================
    # SECURITY / PERFORMANCE
    # ========================================================

    print()
    print("=" * 70)
    print("SECURITY AND PERFORMANCE METRICS")
    print("=" * 70)

    print(
        f"Mean security violation: "
        f"{result_df['ddqn_violation'].mean():.6f}"
    )

    violation_rate = (
        result_df[
            "ddqn_violation"
        ]
        .gt(1e-9)
        .mean()
        * 100
    )

    print(
        f"Security violation rate: "
        f"{violation_rate:.2f}%"
    )

    print(
        f"Mean environment-adjusted risk: "
        f"{result_df['ddqn_risk'].mean():.6f}"
    )

    print(
        f"Mean latency: "
        f"{result_df['ddqn_latency_ms'].mean():.6f} ms"
    )

    print(
        f"Mean compute cost: "
        f"{result_df['ddqn_compute_cost'].mean():.6f}"
    )

    print(
        f"Mean bandwidth cost: "
        f"{result_df['ddqn_bandwidth_cost'].mean():.6f}"
    )

    # ========================================================
    # REGRET DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("REGRET STATISTICS")
    print("=" * 70)

    print(
        result_df[
            "regret"
        ].describe()
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    evaluate()