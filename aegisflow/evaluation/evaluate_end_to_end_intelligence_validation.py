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
# PATHS
# ============================================================

VALIDATION_CONTEXT_PATH = Path(
    "data/processed/splits/validation.csv"
)

INTELLIGENCE_PATH = Path(
    "data/telemetry/"
    "telemetry_validation_intelligence_final.csv"
)

POLICY_MODEL_PATH = Path(
    "models/"
    "aegisflow_ddqn_full_gamma0p0_seed42.pt"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "aegisflow_end_to_end_intelligence_validation.csv"
)


# ============================================================
# CONFIG
# ============================================================

EXPECTED_GAMMA = 0.0
EXPECTED_FEATURE_SET = "full"


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
# POLICY MODEL
# ============================================================

def load_policy_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = TrainConfig()

    checkpoint = torch.load(
        POLICY_MODEL_PATH,
        map_location=device,
        weights_only=False,
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

    if checkpoint_gamma is None:
        raise ValueError(
            "Policy checkpoint gamma "
            "metadata missing."
        )

    if not np.isclose(
        float(checkpoint_gamma),
        EXPECTED_GAMMA,
        atol=1e-12,
    ):
        raise ValueError(
            f"Expected gamma "
            f"{EXPECTED_GAMMA}, "
            f"found {checkpoint_gamma}."
        )

    if (
        checkpoint_feature_set
        != EXPECTED_FEATURE_SET
    ):
        raise ValueError(
            f"Expected feature set "
            f"'{EXPECTED_FEATURE_SET}', "
            f"found "
            f"'{checkpoint_feature_set}'."
        )

    if (
        int(checkpoint_state_dim)
        != len(FEATURES)
    ):
        raise ValueError(
            f"Expected state dim "
            f"{len(FEATURES)}, "
            f"found "
            f"{checkpoint_state_dim}."
        )

    if (
        list(checkpoint_features)
        != list(FEATURES)
    ):
        raise ValueError(
            "Policy checkpoint feature "
            "ordering mismatch."
        )

    model = QNetwork(
        state_dim=len(FEATURES),
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "online_net"
        ]
    )

    model.eval()

    return (
        model,
        device,
        checkpoint,
    )


# ============================================================
# ACTION PREDICTION
# ============================================================

def predict_actions(
    model,
    device,
    df,
):

    x = (
        df[
            FEATURES
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    x = torch.as_tensor(
        x,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        q_values = model(
            x
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
    reward_df,
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
        reward_df.iterrows()
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

        "p95_regret":
            float(
                np.quantile(
                    regret,
                    0.95,
                )
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

        "p95_violation":
            float(
                np.quantile(
                    violations,
                    0.95,
                )
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
        "AEGISFLOW END-TO-END "
        "AI INTELLIGENCE VALIDATION"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    context_df = pd.read_csv(
        VALIDATION_CONTEXT_PATH
    )

    intelligence_df = pd.read_csv(
        INTELLIGENCE_PATH
    )

    print(
        f"Context rows: "
        f"{len(context_df):,}"
    )

    print(
        f"Intelligence rows: "
        f"{len(intelligence_df):,}"
    )

    if (
        len(context_df)
        != len(intelligence_df)
    ):

        raise ValueError(
            "Context and intelligence "
            "row counts do not match."
        )

    if (
        "predicted_network_threat"
        not in intelligence_df.columns
    ):

        raise ValueError(
            "'predicted_network_threat' "
            "not found."
        )

    if (
        "reference_network_threat"
        not in intelligence_df.columns
    ):

        raise ValueError(
            "'reference_network_threat' "
            "not found."
        )

    # ========================================================
    # CRITICAL ALIGNMENT CHECK
    # ========================================================

    original_threat = (
        context_df[
            "network_threat"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    telemetry_reference = (
        intelligence_df[
            "reference_network_threat"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.allclose(
        original_threat,
        telemetry_reference,
        atol=1e-10,
    ):

        max_difference = float(
            np.max(
                np.abs(
                    original_threat
                    - telemetry_reference
                )
            )
        )

        raise ValueError(
            "Validation rows are not aligned. "
            f"Maximum threat difference: "
            f"{max_difference}"
        )

    print(
        "Row alignment check: PASSED"
    )

    # ========================================================
    # THREAT ESTIMATION METRICS
    # ========================================================

    predicted_threat = (
        intelligence_df[
            "predicted_network_threat"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    threat_error = (
        predicted_threat
        - original_threat
    )

    threat_mae = float(
        np.mean(
            np.abs(
                threat_error
            )
        )
    )

    threat_rmse = float(
        np.sqrt(
            np.mean(
                threat_error ** 2
            )
        )
    )

    threat_corr = float(
        np.corrcoef(
            original_threat,
            predicted_threat,
        )[0, 1]
    )

    print()
    print(
        "Threat estimator:"
    )

    print(
        f"MAE: "
        f"{threat_mae:.6f}"
    )

    print(
        f"RMSE: "
        f"{threat_rmse:.6f}"
    )

    print(
        f"Correlation: "
        f"{threat_corr:.6f}"
    )

    # ========================================================
    # MODE A
    #
    # Ground-truth / simulator threat
    # ========================================================

    original_context_df = (
        context_df.copy()
    )

    # ========================================================
    # MODE B
    #
    # AI-predicted threat
    # ========================================================

    ai_context_df = (
        context_df.copy()
    )

    ai_context_df[
        "network_threat"
    ] = np.clip(
        predicted_threat,
        0.0,
        1.0,
    )

    # ========================================================
    # ORACLE
    #
    # IMPORTANT:
    # Oracle is calculated against original
    # environment context.
    # ========================================================

    print()
    print(
        "Calculating Oracle using "
        "original validation context..."
    )

    (
        oracle_actions,
        oracle_rewards,
    ) = calculate_oracle(
        context_df
    )

    print(
        f"Oracle mean reward: "
        f"{oracle_rewards.mean():.6f}"
    )

    # ========================================================
    # LOAD POLICY
    # ========================================================

    (
        policy_model,
        device,
        checkpoint,
    ) = load_policy_model()

    print()
    print(
        "Policy device:",
        device,
    )

    print(
        "Policy gamma:",
        checkpoint[
            "gamma"
        ],
    )

    # ========================================================
    # MODE A PREDICTION
    # ========================================================

    original_actions = (
        predict_actions(
            policy_model,
            device,
            original_context_df,
        )
    )

    # ========================================================
    # MODE B PREDICTION
    # ========================================================

    ai_actions = (
        predict_actions(
            policy_model,
            device,
            ai_context_df,
        )
    )

    # ========================================================
    # IMPORTANT:
    #
    # Both modes are scored against the REAL /
    # original simulator context, not predicted threat.
    #
    # Otherwise Mode B would evaluate itself.
    # ========================================================

    original_metrics = (
        evaluate_policy(
            context_df,
            original_actions,
            oracle_actions,
            oracle_rewards,
        )
    )

    ai_metrics = (
        evaluate_policy(
            context_df,
            ai_actions,
            oracle_actions,
            oracle_rewards,
        )
    )

    # ========================================================
    # POLICY CONSISTENCY
    # ========================================================

    policy_consistency = float(
        (
            original_actions
            == ai_actions
        ).mean()
        * 100
    )

    changed_policy = (
        original_actions
        != ai_actions
    )

    changed_policy_pct = float(
        changed_policy.mean()
        * 100
    )

    print()
    print("=" * 78)
    print(
        "POLICY CONSISTENCY"
    )
    print("=" * 78)

    print(
        f"Original vs AI-threat "
        f"policy agreement: "
        f"{policy_consistency:.3f}%"
    )

    print(
        f"Changed decisions: "
        f"{changed_policy_pct:.3f}%"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = pd.DataFrame(
        [
            {
                "mode":
                    "original_threat",

                **original_metrics,
            },

            {
                "mode":
                    "ai_predicted_threat",

                **ai_metrics,
            },
        ]
    )

    print()
    print("=" * 78)
    print(
        "END-TO-END RESULTS"
    )
    print("=" * 78)

    display_columns = [
        "mode",
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
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # DELTAS
    # ========================================================

    delta_agreement = (
        ai_metrics[
            "agreement_pct"
        ]
        - original_metrics[
            "agreement_pct"
        ]
    )

    delta_reward = (
        ai_metrics[
            "mean_reward"
        ]
        - original_metrics[
            "mean_reward"
        ]
    )

    delta_regret = (
        ai_metrics[
            "mean_regret"
        ]
        - original_metrics[
            "mean_regret"
        ]
    )

    delta_risk = (
        ai_metrics[
            "mean_risk"
        ]
        - original_metrics[
            "mean_risk"
        ]
    )

    print()
    print("=" * 78)
    print(
        "IMPACT OF AI THREAT ESTIMATION"
    )
    print("=" * 78)

    print(
        f"Agreement delta: "
        f"{delta_agreement:+.6f}"
    )

    print(
        f"Reward delta: "
        f"{delta_reward:+.6f}"
    )

    print(
        f"Regret delta: "
        f"{delta_regret:+.6f}"
    )

    print(
        f"Risk delta: "
        f"{delta_risk:+.6f}"
    )

    # ========================================================
    # SAVE ROW-LEVEL OUTPUT
    # ========================================================

    output_df = (
        context_df.copy()
    )

    output_df[
        "original_network_threat"
    ] = original_threat

    output_df[
        "ai_predicted_network_threat"
    ] = predicted_threat

    output_df[
        "threat_absolute_error"
    ] = np.abs(
        threat_error
    )

    output_df[
        "oracle_action"
    ] = oracle_actions

    output_df[
        "original_threat_action"
    ] = original_actions

    output_df[
        "ai_threat_action"
    ] = ai_actions

    output_df[
        "policy_changed"
    ] = changed_policy.astype(
        np.int64
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "END-TO-END VALIDATION COMPLETE"
    )
    print("=" * 78)

    print(
        "Row-level output saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()