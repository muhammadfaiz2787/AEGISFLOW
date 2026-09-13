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

MODEL_DIR = Path("models")

SEEDS = [
    11,
    22,
    42,
    84,
    123,
]

OUTPUT_DIR = Path(
    "data/processed/multiseed"
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
        state_dim=config.state_dim,
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


# ============================================================
# DDQN PREDICTION
# ============================================================

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


# ============================================================
# POLICY DETAILS
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

    return {
        "agreement_pct":
            agreement.mean() * 100,

        "mean_reward":
            rewards.mean(),

        "mean_regret":
            regret.mean(),

        "median_regret":
            np.median(regret),

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


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

def action_distribution(actions):

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
            percentages[0],

        "medium_pct":
            percentages[1],

        "high_pct":
            percentages[2],

        "critical_pct":
            percentages[3],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "AEGISFLOW MULTI-SEED VALIDATION EVALUATION"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation size: "
        f"{len(df):,}"
    )

    print(
        "Validation dataset:",
        VALIDATION_PATH,
    )

    # --------------------------------------------------------
    # Oracle only needs to be calculated once
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

    # --------------------------------------------------------
    # Config
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
            MODEL_DIR
            / f"aegisflow_ddqn_full_seed{seed}.pt"
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

        print(
            "Device:",
            device,
        )

        # ----------------------------------------------------
        # Optional checkpoint seed check
        # ----------------------------------------------------

        checkpoint_seed = (
            checkpoint.get(
                "seed",
                None,
            )
        )

        if checkpoint_seed is not None:

            print(
                "Checkpoint seed:",
                checkpoint_seed,
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
            "seed": seed,
            **metrics,
            **distribution,
        }

        results.append(
            row
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

    # ========================================================
    # DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save per seed
    # --------------------------------------------------------

    per_seed_path = (
        OUTPUT_DIR
        / "validation_multiseed_per_seed.csv"
    )

    results_df.to_csv(
        per_seed_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    numeric_columns = [
        column
        for column
        in results_df.columns
        if column != "seed"
    ]

    summary_rows = []

    for metric in numeric_columns:

        values = (
            results_df[
                metric
            ].to_numpy()
        )

        summary_rows.append(
            {
                "metric": metric,

                "mean":
                    np.mean(values),

                "std":
                    np.std(
                        values,
                        ddof=1,
                    ),

                "min":
                    np.min(values),

                "max":
                    np.max(values),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "validation_multiseed_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 78)
    print("PER-SEED RESULTS")
    print("=" * 78)

    display_columns = [
        "seed",
        "agreement_pct",
        "mean_reward",
        "mean_regret",
        "under_selection_pct",
        "over_selection_pct",
        "mean_violation",
        "mean_risk",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("MULTI-SEED SUMMARY")
    print("=" * 78)

    important_metrics = [
        "agreement_pct",
        "mean_reward",
        "mean_regret",
        "under_selection_pct",
        "over_selection_pct",
        "mean_violation",
        "mean_risk",
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

    agreement_mean = (
        results_df[
            "agreement_pct"
        ].mean()
    )

    agreement_std = (
        results_df[
            "agreement_pct"
        ].std(
            ddof=1
        )
    )

    reward_mean = (
        results_df[
            "mean_reward"
        ].mean()
    )

    reward_std = (
        results_df[
            "mean_reward"
        ].std(
            ddof=1
        )
    )

    regret_mean = (
        results_df[
            "mean_regret"
        ].mean()
    )

    regret_std = (
        results_df[
            "mean_regret"
        ].std(
            ddof=1
        )
    )

    print()
    print("=" * 78)
    print("FINAL MULTI-SEED REPORT")
    print("=" * 78)

    print(
        "Oracle agreement:"
    )

    print(
        f"{agreement_mean:.3f}% "
        f"+/- "
        f"{agreement_std:.3f}%"
    )

    print()
    print(
        "Mean reward:"
    )

    print(
        f"{reward_mean:.6f} "
        f"+/- "
        f"{reward_std:.6f}"
    )

    print()
    print(
        "Mean regret:"
    )

    print(
        f"{regret_mean:.6f} "
        f"+/- "
        f"{regret_std:.6f}"
    )

    print()
    print(
        "Per-seed results saved to:"
    )

    print(
        per_seed_path
    )

    print()
    print(
        "Summary saved to:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()