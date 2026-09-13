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


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_PATH = Path(
    "data/processed/splits/validation.csv"
)

MODEL_DIR = Path(
    "models"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "aegisflow_ablation_gamma0_validation.csv"
)

SEED = 42
EXPECTED_GAMMA = 0.0


EXPERIMENTS = {
    "full": {
        "feature_set": "full",
        "model":
            "aegisflow_ddqn_full_gamma0p0_seed42.pt",
    },

    "no-threat": {
        "feature_set": "no-threat",
        "model":
            "aegisflow_ddqn_no_threat_gamma0p0_seed42.pt",
    },

    "no-trust": {
        "feature_set": "no-trust",
        "model":
            "aegisflow_ddqn_no_trust_gamma0p0_seed42.pt",
    },

    "no-latency": {
        "feature_set": "no-latency",
        "model":
            "aegisflow_ddqn_no_latency_gamma0p0_seed42.pt",
    },
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
# LOAD MODEL
# ============================================================

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

    if "online_net" not in checkpoint:
        raise KeyError(
            f"'online_net' not found in "
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
    expected_feature_set,
    expected_features,
    expected_state_dim,
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

    # Seed check
    if checkpoint_seed is None:
        raise ValueError(
            f"Checkpoint seed metadata "
            f"missing: {model_path}"
        )

    if int(checkpoint_seed) != SEED:
        raise ValueError(
            f"Seed mismatch in "
            f"{model_path}. "
            f"Expected {SEED}, "
            f"found {checkpoint_seed}."
        )

    # Gamma check
    if checkpoint_gamma is None:
        raise ValueError(
            f"Checkpoint gamma metadata "
            f"missing: {model_path}"
        )

    if not np.isclose(
        float(checkpoint_gamma),
        EXPECTED_GAMMA,
        atol=1e-12,
    ):
        raise ValueError(
            f"Gamma mismatch in "
            f"{model_path}. "
            f"Expected {EXPECTED_GAMMA}, "
            f"found {checkpoint_gamma}."
        )

    # Feature-set check
    if checkpoint_feature_set is None:
        raise ValueError(
            f"Feature-set metadata "
            f"missing: {model_path}"
        )

    if (
        checkpoint_feature_set
        != expected_feature_set
    ):
        raise ValueError(
            f"Feature-set mismatch in "
            f"{model_path}. "
            f"Expected "
            f"'{expected_feature_set}', "
            f"found "
            f"'{checkpoint_feature_set}'."
        )

    # State dimension check
    if checkpoint_state_dim is None:
        raise ValueError(
            f"State-dim metadata "
            f"missing: {model_path}"
        )

    if (
        int(checkpoint_state_dim)
        != expected_state_dim
    ):
        raise ValueError(
            f"State-dim mismatch in "
            f"{model_path}. "
            f"Expected "
            f"{expected_state_dim}, "
            f"found "
            f"{checkpoint_state_dim}."
        )

    # Feature list check
    if checkpoint_features is None:
        raise ValueError(
            f"Features metadata "
            f"missing: {model_path}"
        )

    if (
        list(checkpoint_features)
        != list(expected_features)
    ):
        raise ValueError(
            f"Feature mismatch in "
            f"{model_path}.\n"
            f"Expected: "
            f"{list(expected_features)}\n"
            f"Found: "
            f"{list(checkpoint_features)}"
        )

    return {
        "seed":
            int(checkpoint_seed),

        "gamma":
            float(checkpoint_gamma),

        "feature_set":
            checkpoint_feature_set,

        "state_dim":
            int(checkpoint_state_dim),
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_actions(
    model,
    device,
    df,
    features,
):

    states = torch.as_tensor(
        df[
            features
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
# METRICS
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

        result = calculate_reward(
            row,
            int(
                actions[i]
            ),
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
                result[
                    "bandwidth_cost"
                ]
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

    regret = (
        oracle_rewards
        - rewards
    )

    regret = np.maximum(
        regret,
        0.0,
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
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "AEGISFLOW GAMMA 0.0 "
        "FEATURE ABLATION VALIDATION"
    )
    print("=" * 78)

    print(
        "Expected gamma:",
        EXPECTED_GAMMA,
    )

    print(
        "Seed:",
        SEED,
    )

    # --------------------------------------------------------
    # Dataset
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
        "Calculating Oracle..."
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

    config = TrainConfig()

    rows = []

    # ========================================================
    # EACH ABLATION CONFIGURATION
    # ========================================================

    for (
        experiment_name,
        spec,
    ) in EXPERIMENTS.items():

        print()
        print("-" * 78)

        feature_set_name = (
            spec["feature_set"]
        )

        features = list(
            get_feature_set(
                feature_set_name
            )
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
            "Model:",
            model_path,
        )

        print(
            "Feature set:",
            feature_set_name,
        )

        print(
            "State dimension:",
            state_dim,
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

        metadata = (
            validate_checkpoint(
                checkpoint,
                expected_feature_set=(
                    feature_set_name
                ),
                expected_features=(
                    features
                ),
                expected_state_dim=(
                    state_dim
                ),
                model_path=(
                    model_path
                ),
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

    results = pd.DataFrame(
        rows
    )

    # ========================================================
    # DELTAS VS FULL
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
        "delta_violation_vs_full"
    ] = (
        results[
            "mean_violation"
        ]
        - full_row[
            "mean_violation"
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

    results[
        "delta_latency_vs_full"
    ] = (
        results[
            "mean_latency_ms"
        ]
        - full_row[
            "mean_latency_ms"
        ]
    )

    # ========================================================
    # DISPLAY MAIN RESULTS
    # ========================================================

    print()
    print("=" * 78)
    print(
        "GAMMA 0.0 ABLATION RESULTS"
    )
    print("=" * 78)

    display_columns = [
        "experiment",
        "state_dim",
        "agreement_pct",
        "mean_reward",
        "mean_regret",
        "p95_regret",
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
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # DISPLAY DELTAS
    # ========================================================

    print()
    print("=" * 78)
    print(
        "CHANGE RELATIVE TO "
        "FULL GAMMA 0.0 MODEL"
    )
    print("=" * 78)

    delta_columns = [
        "experiment",
        "delta_agreement_vs_full",
        "delta_reward_vs_full",
        "delta_regret_vs_full",
        "delta_violation_vs_full",
        "delta_risk_vs_full",
        "delta_latency_vs_full",
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
        "GAMMA 0.0 FEATURE "
        "ABLATION COMPLETE"
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