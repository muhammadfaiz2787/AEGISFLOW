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


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_PATH = Path(
    "data/processed/splits/validation.csv"
)

MODEL_DIR = Path(
    "models"
)

OUTPUT_DIR = Path(
    "data/processed/multiseed"
)

SEEDS = [
    11,
    22,
    42,
    84,
    123,
]

EXPECTED_GAMMA = 0.0
EXPECTED_FEATURE_SET = "full"
EXPECTED_STATE_DIM = len(FEATURES)

PER_SEED_OUTPUT = (
    OUTPUT_DIR
    / "validation_gamma0_multiseed_per_seed.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "validation_gamma0_multiseed_summary.csv"
)


# ============================================================
# MODEL PATH
# ============================================================

def build_model_path(seed):

    return (
        MODEL_DIR
        / (
            f"aegisflow_ddqn_full_"
            f"gamma0p0_"
            f"seed{seed}.pt"
        )
    )


# ============================================================
# LOAD MODEL
# ============================================================

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
        state_dim=EXPECTED_STATE_DIM,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if "online_net" not in checkpoint:
        raise KeyError(
            f"'online_net' not found in checkpoint: "
            f"{model_path}"
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


# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

def validate_checkpoint(
    checkpoint,
    expected_seed,
    model_path,
):

    checkpoint_seed = checkpoint.get(
        "seed",
        None,
    )

    checkpoint_gamma = checkpoint.get(
        "gamma",
        checkpoint
        .get(
            "config",
            {},
        )
        .get(
            "gamma",
            None,
        ),
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

    checkpoint_features = (
        checkpoint.get(
            "features",
            None,
        )
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    if checkpoint_seed is None:
        raise ValueError(
            f"Checkpoint does not contain seed metadata: "
            f"{model_path}"
        )

    if int(checkpoint_seed) != int(expected_seed):
        raise ValueError(
            f"Seed mismatch in {model_path}. "
            f"Expected {expected_seed}, "
            f"found {checkpoint_seed}."
        )

    # --------------------------------------------------------
    # Gamma
    # --------------------------------------------------------

    if checkpoint_gamma is None:
        raise ValueError(
            f"Checkpoint does not contain gamma metadata: "
            f"{model_path}"
        )

    if not np.isclose(
        float(checkpoint_gamma),
        EXPECTED_GAMMA,
        atol=1e-12,
    ):
        raise ValueError(
            f"Gamma mismatch in {model_path}. "
            f"Expected {EXPECTED_GAMMA}, "
            f"found {checkpoint_gamma}."
        )

    # --------------------------------------------------------
    # Feature set
    # --------------------------------------------------------

    if checkpoint_feature_set is None:
        raise ValueError(
            f"Checkpoint does not contain "
            f"feature_set metadata: {model_path}"
        )

    if checkpoint_feature_set != EXPECTED_FEATURE_SET:
        raise ValueError(
            f"Feature-set mismatch in {model_path}. "
            f"Expected '{EXPECTED_FEATURE_SET}', "
            f"found '{checkpoint_feature_set}'."
        )

    # --------------------------------------------------------
    # State dimension
    # --------------------------------------------------------

    if checkpoint_state_dim is None:
        raise ValueError(
            f"Checkpoint does not contain "
            f"state_dim metadata: {model_path}"
        )

    if int(checkpoint_state_dim) != EXPECTED_STATE_DIM:
        raise ValueError(
            f"State-dimension mismatch in {model_path}. "
            f"Expected {EXPECTED_STATE_DIM}, "
            f"found {checkpoint_state_dim}."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    if checkpoint_features is None:
        raise ValueError(
            f"Checkpoint does not contain "
            f"features metadata: {model_path}"
        )

    if list(checkpoint_features) != list(FEATURES):
        raise ValueError(
            f"Feature mismatch in {model_path}.\n"
            f"Expected: {list(FEATURES)}\n"
            f"Found:    {list(checkpoint_features)}"
        )

    return {
        "seed": int(checkpoint_seed),
        "gamma": float(checkpoint_gamma),
        "feature_set": checkpoint_feature_set,
        "state_dim": int(checkpoint_state_dim),
    }


# ============================================================
# ORACLE
# ============================================================

def calculate_oracle(df):

    oracle_actions = []
    oracle_rewards = []

    for _, row in df.iterrows():

        rewards = []

        for action in range(
            len(ACTIONS)
        ):

            result = calculate_reward(
                row,
                action,
            )

            rewards.append(
                float(
                    result["reward"]
                )
            )

        best_action = int(
            np.argmax(
                rewards
            )
        )

        oracle_actions.append(
            best_action
        )

        oracle_rewards.append(
            rewards[
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


# ============================================================
# DDQN PREDICTION
# ============================================================

def predict_actions(
    model,
    device,
    df,
):

    states = torch.as_tensor(
        df[
            FEATURES
        ].to_numpy(
            dtype=np.float32
        ),
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
        .astype(
            np.int64
        )
    )


# ============================================================
# POLICY METRICS
# ============================================================

def calculate_metrics(
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
            float(
                result["reward"]
            )
        )

        violations.append(
            float(
                result["violation"]
            )
        )

        risks.append(
            float(
                result["risk"]
            )
        )

        latencies.append(
            float(
                result["latency_ms"]
            )
        )

        compute_costs.append(
            float(
                result["compute_cost"]
            )
        )

        bandwidth_costs.append(
            float(
                result["bandwidth_cost"]
            )
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

    regret = (
        oracle_rewards
        - rewards
    )

    # Numerical safeguard.
    regret = np.maximum(
        regret,
        0.0,
    )

    return {

        "agreement_pct":
            float(
                agreement.mean()
                * 100
            ),

        "mean_reward":
            float(
                rewards.mean()
            ),

        "mean_regret":
            float(
                regret.mean()
            ),

        "median_regret":
            float(
                np.median(
                    regret
                )
            ),

        "p95_regret":
            float(
                np.quantile(
                    regret,
                    0.95,
                )
            ),

        "max_regret":
            float(
                regret.max()
            ),

        "under_selection_pct":
            float(
                under.mean()
                * 100
            ),

        "over_selection_pct":
            float(
                over.mean()
                * 100
            ),

        "mean_violation":
            float(
                violations.mean()
            ),

        "median_violation":
            float(
                np.median(
                    violations
                )
            ),

        "p95_violation":
            float(
                np.quantile(
                    violations,
                    0.95,
                )
            ),

        "nonzero_violation_pct":
            float(
                (
                    violations
                    > 1e-9
                ).mean()
                * 100
            ),

        "mean_risk":
            float(
                risks.mean()
            ),

        "p95_risk":
            float(
                np.quantile(
                    risks,
                    0.95,
                )
            ),

        "mean_latency_ms":
            float(
                latencies.mean()
            ),

        "mean_compute_cost":
            float(
                compute_costs.mean()
            ),

        "mean_bandwidth_cost":
            float(
                bandwidth_costs.mean()
            ),
    }


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

def action_distribution(
    actions,
):

    counts = np.bincount(
        actions,
        minlength=len(ACTIONS),
    )

    percentages = (
        counts
        / len(actions)
        * 100
    )

    return {

        "low_pct":
            float(
                percentages[0]
            ),

        "medium_pct":
            float(
                percentages[1]
            ),

        "high_pct":
            float(
                percentages[2]
            ),

        "critical_pct":
            float(
                percentages[3]
            ),
    }


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    results_df,
):

    numeric_columns = [
        column
        for column
        in results_df.columns
        if column
        not in {
            "seed",
            "checkpoint_seed",
            "gamma",
            "state_dim",
        }
    ]

    summary_rows = []

    for metric in numeric_columns:

        values = (
            results_df[
                metric
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        summary_rows.append(
            {
                "metric":
                    metric,

                "mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "std":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),

                "min":
                    float(
                        np.min(
                            values
                        )
                    ),

                "max":
                    float(
                        np.max(
                            values
                        )
                    ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "AEGISFLOW GAMMA 0.0 MULTI-SEED "
        "VALIDATION EVALUATION"
    )
    print("=" * 78)

    print(
        "Expected gamma:",
        EXPECTED_GAMMA,
    )

    print(
        "Expected feature set:",
        EXPECTED_FEATURE_SET,
    )

    print(
        "Expected state dimension:",
        EXPECTED_STATE_DIM,
    )

    print(
        "Seeds:",
        SEEDS,
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    if not VALIDATION_PATH.exists():

        raise FileNotFoundError(
            f"Validation dataset not found: "
            f"{VALIDATION_PATH}"
        )

    df = pd.read_csv(
        VALIDATION_PATH
    )

    if len(df) == 0:

        raise ValueError(
            "Validation dataset is empty."
        )

    missing_features = [
        feature
        for feature
        in FEATURES
        if feature
        not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Validation dataset is missing "
            f"features: {missing_features}"
        )

    print()

    print(
        f"Validation size: "
        f"{len(df):,}"
    )

    print(
        "Validation dataset:",
        VALIDATION_PATH,
    )

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    print()
    print(
        "Calculating validation Oracle..."
    )

    (
        oracle_actions,
        oracle_rewards,
    ) = calculate_oracle(
        df
    )

    print(
        "Oracle calculation complete."
    )

    print(
        f"Oracle mean reward: "
        f"{oracle_rewards.mean():.6f}"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = TrainConfig()

    results = []

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # EACH SEED
    # ========================================================

    for seed in SEEDS:

        print()
        print("-" * 78)

        model_path = (
            build_model_path(
                seed
            )
        )

        print(
            f"Evaluating seed {seed}"
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
            config,
        )

        metadata = (
            validate_checkpoint(
                checkpoint,
                expected_seed=seed,
                model_path=model_path,
            )
        )

        print(
            "Device:",
            device,
        )

        print(
            "Checkpoint seed:",
            metadata["seed"],
        )

        print(
            "Checkpoint gamma:",
            metadata["gamma"],
        )

        print(
            "Checkpoint feature set:",
            metadata[
                "feature_set"
            ],
        )

        print(
            "Checkpoint state dimension:",
            metadata[
                "state_dim"
            ],
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        actions = predict_actions(
            model,
            device,
            df,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            df,
            actions,
            oracle_actions,
            oracle_rewards,
        )

        distribution = (
            action_distribution(
                actions
            )
        )

        row = {

            "seed":
                int(seed),

            "checkpoint_seed":
                metadata["seed"],

            "gamma":
                metadata["gamma"],

            "state_dim":
                metadata[
                    "state_dim"
                ],

            **metrics,

            **distribution,
        }

        results.append(
            row
        )

        print()

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
            f"P95 regret: "
            f"{metrics['p95_regret']:.6f}"
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

    # ========================================================
    # DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SAVE PER-SEED
    # ========================================================

    results_df.to_csv(
        PER_SEED_OUTPUT,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = (
        create_summary(
            results_df
        )
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # DISPLAY PER-SEED
    # ========================================================

    print()
    print("=" * 78)
    print(
        "GAMMA 0.0 PER-SEED RESULTS"
    )
    print("=" * 78)

    display_columns = [
        "seed",
        "agreement_pct",
        "mean_reward",
        "mean_regret",
        "p95_regret",
        "under_selection_pct",
        "over_selection_pct",
        "mean_violation",
        "mean_risk",
        "mean_latency_ms",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print()
    print("=" * 78)
    print(
        "GAMMA 0.0 MULTI-SEED SUMMARY"
    )
    print("=" * 78)

    important_metrics = [
        "agreement_pct",
        "mean_reward",
        "mean_regret",
        "p95_regret",
        "under_selection_pct",
        "over_selection_pct",
        "mean_violation",
        "p95_violation",
        "nonzero_violation_pct",
        "mean_risk",
        "p95_risk",
        "mean_latency_ms",
        "mean_compute_cost",
        "mean_bandwidth_cost",
    ]

    important_summary = (
        summary_df[
            summary_df[
                "metric"
            ].isin(
                important_metrics
            )
        ]
        .copy()
    )

    print(
        important_summary
        .to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    agreement_mean = float(
        results_df[
            "agreement_pct"
        ].mean()
    )

    agreement_std = float(
        results_df[
            "agreement_pct"
        ].std(
            ddof=1
        )
    )

    reward_mean = float(
        results_df[
            "mean_reward"
        ].mean()
    )

    reward_std = float(
        results_df[
            "mean_reward"
        ].std(
            ddof=1
        )
    )

    regret_mean = float(
        results_df[
            "mean_regret"
        ].mean()
    )

    regret_std = float(
        results_df[
            "mean_regret"
        ].std(
            ddof=1
        )
    )

    risk_mean = float(
        results_df[
            "mean_risk"
        ].mean()
    )

    risk_std = float(
        results_df[
            "mean_risk"
        ].std(
            ddof=1
        )
    )

    print()
    print("=" * 78)
    print(
        "FINAL GAMMA 0.0 MULTI-SEED REPORT"
    )
    print("=" * 78)

    print(
        f"Seeds evaluated: "
        f"{len(SEEDS)}"
    )

    print(
        f"Oracle agreement: "
        f"{agreement_mean:.3f}% "
        f"+/- "
        f"{agreement_std:.3f}%"
    )

    print(
        f"Mean reward: "
        f"{reward_mean:.6f} "
        f"+/- "
        f"{reward_std:.6f}"
    )

    print(
        f"Mean regret: "
        f"{regret_mean:.6f} "
        f"+/- "
        f"{regret_std:.6f}"
    )

    print(
        f"Mean risk: "
        f"{risk_mean:.6f} "
        f"+/- "
        f"{risk_std:.6f}"
    )

    print()

    print(
        "Per-seed results saved to:"
    )

    print(
        PER_SEED_OUTPUT
    )

    print()

    print(
        "Summary saved to:"
    )

    print(
        SUMMARY_OUTPUT
    )


if __name__ == "__main__":
    main()