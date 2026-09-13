from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from aegisflow.config import (
    FEATURES,
    ACTIONS,
    TrainConfig,
)
from aegisflow.rl.networks import (
    QNetwork,
)
from aegisflow.environment.reward import (
    calculate_reward,
)


TEST_PATH = Path(
    "data/processed/splits/test.csv"
)

INTELLIGENCE_PATH = Path(
    "data/telemetry/"
    "telemetry_test_intelligence.csv"
)

POLICY_MODEL_PATH = Path(
    "models/"
    "aegisflow_ddqn_full_gamma0p0_seed42.pt"
)

OUTPUT_DIR = Path(
    "data/processed/final_test"
)


EXPECTED_GAMMA = 0.0
EXPECTED_SEED = 42
EXPECTED_FEATURE_SET = "full"


def calculate_oracle(
    df,
):

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
                    result[
                        "reward"
                    ]
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


def load_policy():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        POLICY_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    if (
        int(
            checkpoint[
                "seed"
            ]
        )
        != EXPECTED_SEED
    ):
        raise ValueError(
            "Unexpected policy seed."
        )

    if not np.isclose(
        float(
            checkpoint[
                "gamma"
            ]
        ),
        EXPECTED_GAMMA,
        atol=1e-12,
    ):
        raise ValueError(
            "Unexpected policy gamma."
        )

    if (
        checkpoint[
            "feature_set"
        ]
        != EXPECTED_FEATURE_SET
    ):
        raise ValueError(
            "Unexpected policy feature set."
        )

    if (
        int(
            checkpoint[
                "state_dim"
            ]
        )
        != len(FEATURES)
    ):
        raise ValueError(
            "Unexpected policy state_dim."
        )

    if (
        list(
            checkpoint[
                "features"
            ]
        )
        != list(FEATURES)
    ):
        raise ValueError(
            "Policy feature ordering mismatch."
        )

    config = TrainConfig()

    model = QNetwork(
        state_dim=len(
            FEATURES
        ),
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
    )


def predict_actions(
    model,
    device,
    df,
):

    x_np = (
        df[
            FEATURES
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    x = torch.as_tensor(
        x_np,
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


def policy_metrics(
    reward_df,
    actions,
    oracle_actions,
    oracle_rewards,
):

    rewards = []
    violations = []
    risks = []
    latency = []
    compute = []
    bandwidth = []

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
                result[
                    "reward"
                ]
            )
        )

        violations.append(
            float(
                result[
                    "violation"
                ]
            )
        )

        risks.append(
            float(
                result[
                    "risk"
                ]
            )
        )

        latency.append(
            float(
                result[
                    "latency_ms"
                ]
            )
        )

        compute.append(
            float(
                result[
                    "compute_cost"
                ]
            )
        )

        bandwidth.append(
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

    latency = np.asarray(
        latency,
        dtype=np.float64,
    )

    compute = np.asarray(
        compute,
        dtype=np.float64,
    )

    bandwidth = np.asarray(
        bandwidth,
        dtype=np.float64,
    )

    regret = np.maximum(
        oracle_rewards
        - rewards,
        0.0,
    )

    return {
        "agreement_pct":
            float(
                (
                    actions
                    == oracle_actions
                ).mean()
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
                (
                    actions
                    < oracle_actions
                ).mean()
                * 100
            ),

        "over_selection_pct":
            float(
                (
                    actions
                    > oracle_actions
                ).mean()
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
                latency.mean()
            ),

        "mean_compute_cost":
            float(
                compute.mean()
            ),

        "mean_bandwidth_cost":
            float(
                bandwidth.mean()
            ),
    }


def baseline_actions(
    name,
    df,
):

    n = len(
        df
    )

    if name == "always_low":
        return np.zeros(
            n,
            dtype=np.int64,
        )

    if name == "always_medium":
        return np.ones(
            n,
            dtype=np.int64,
        )

    if name == "always_high":
        return np.full(
            n,
            2,
            dtype=np.int64,
        )

    if name == "always_critical":
        return np.full(
            n,
            3,
            dtype=np.int64,
        )

    if name == "required_level":
        return (
            df[
                "required_level"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

    raise ValueError(
        name
    )


def main():

    print("=" * 78)
    print(
        "AEGISFLOW FINAL HELD-OUT TEST"
    )
    print("=" * 78)

    test_df = pd.read_csv(
        TEST_PATH
    )

    intelligence_df = pd.read_csv(
        INTELLIGENCE_PATH
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    if (
        len(test_df)
        != len(intelligence_df)
    ):
        raise ValueError(
            "Test/intelligence size mismatch."
        )

    original_threat = (
        test_df[
            "network_threat"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    reference_threat = (
        intelligence_df[
            "reference_network_threat"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.allclose(
        original_threat,
        reference_threat,
        atol=1e-10,
    ):
        raise ValueError(
            "Test/intelligence rows not aligned."
        )

    print(
        "Row alignment: PASSED"
    )

    # ========================================================
    # THREAT INTELLIGENCE METRICS
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

    ss_res = float(
        np.sum(
            threat_error ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                original_threat
                - original_threat.mean()
            ) ** 2
        )
    )

    threat_r2 = float(
        1.0
        - (
            ss_res
            / max(
                ss_tot,
                1e-12,
            )
        )
    )

    # ========================================================
    # ANOMALY METRICS
    # ========================================================

    anomaly_true = (
        intelligence_df[
            "reference_anomaly"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    anomaly_error = (
        intelligence_df[
            "reconstruction_error"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    anomaly_pred = (
        intelligence_df[
            "predicted_anomaly"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    anomaly_roc_auc = float(
        roc_auc_score(
            anomaly_true,
            anomaly_error,
        )
    )

    anomaly_pr_auc = float(
        average_precision_score(
            anomaly_true,
            anomaly_error,
        )
    )

    tp = int(
        (
            (anomaly_true == 1)
            & (anomaly_pred == 1)
        ).sum()
    )

    tn = int(
        (
            (anomaly_true == 0)
            & (anomaly_pred == 0)
        ).sum()
    )

    fp = int(
        (
            (anomaly_true == 0)
            & (anomaly_pred == 1)
        ).sum()
    )

    fn = int(
        (
            (anomaly_true == 1)
            & (anomaly_pred == 0)
        ).sum()
    )

    anomaly_precision = float(
        tp
        / max(
            tp + fp,
            1,
        )
    )

    anomaly_recall = float(
        tp
        / max(
            tp + fn,
            1,
        )
    )

    anomaly_f1 = float(
        (
            2
            * anomaly_precision
            * anomaly_recall
        )
        / max(
            anomaly_precision
            + anomaly_recall,
            1e-12,
        )
    )

    # ========================================================
    # ORACLE + POLICY
    # ========================================================

    print(
        "Calculating final-test Oracle..."
    )

    (
        oracle_actions,
        oracle_rewards,
    ) = calculate_oracle(
        test_df
    )

    model, device = load_policy()

    original_actions = predict_actions(
        model,
        device,
        test_df,
    )

    ai_context_df = (
        test_df.copy()
    )

    ai_context_df[
        "network_threat"
    ] = np.clip(
        predicted_threat,
        0.0,
        1.0,
    )

    ai_actions = predict_actions(
        model,
        device,
        ai_context_df,
    )

    # ========================================================
    # POLICY RESULTS
    # ========================================================

    results = []

    original_metrics = policy_metrics(
        test_df,
        original_actions,
        oracle_actions,
        oracle_rewards,
    )

    results.append(
        {
            "policy":
                "aegisflow_original_threat",

            **original_metrics,
        }
    )

    ai_metrics = policy_metrics(
        test_df,
        ai_actions,
        oracle_actions,
        oracle_rewards,
    )

    results.append(
        {
            "policy":
                "aegisflow_ai_threat",

            **ai_metrics,
        }
    )

    for baseline in [
        "always_low",
        "always_medium",
        "always_high",
        "always_critical",
        "required_level",
    ]:

        actions = baseline_actions(
            baseline,
            test_df,
        )

        metrics = policy_metrics(
            test_df,
            actions,
            oracle_actions,
            oracle_rewards,
        )

        results.append(
            {
                "policy":
                    baseline,

                **metrics,
            }
        )

    results.append(
        {
            "policy":
                "oracle",

            "agreement_pct":
                100.0,

            "mean_reward":
                float(
                    oracle_rewards.mean()
                ),

            "mean_regret":
                0.0,

            "median_regret":
                0.0,

            "p95_regret":
                0.0,

            "max_regret":
                0.0,

            "under_selection_pct":
                0.0,

            "over_selection_pct":
                0.0,
        }
    )

    results_df = pd.DataFrame(
        results
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

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "THREAT ESTIMATOR FINAL TEST"
    )
    print("=" * 78)

    print(
        f"MAE: "
        f"{threat_mae:.6f}"
    )

    print(
        f"RMSE: "
        f"{threat_rmse:.6f}"
    )

    print(
        f"R2: "
        f"{threat_r2:.6f}"
    )

    print(
        f"Correlation: "
        f"{threat_corr:.6f}"
    )

    print()
    print("=" * 78)
    print(
        "ANOMALY DETECTOR FINAL TEST"
    )
    print("=" * 78)

    print(
        f"ROC-AUC: "
        f"{anomaly_roc_auc:.6f}"
    )

    print(
        f"PR-AUC: "
        f"{anomaly_pr_auc:.6f}"
    )

    print(
        f"Precision: "
        f"{anomaly_precision:.6f}"
    )

    print(
        f"Recall: "
        f"{anomaly_recall:.6f}"
    )

    print(
        f"F1: "
        f"{anomaly_f1:.6f}"
    )

    print(
        f"TN={tn:,}, FP={fp:,}, "
        f"FN={fn:,}, TP={tp:,}"
    )

    print()
    print("=" * 78)
    print(
        "POLICY FINAL TEST"
    )
    print("=" * 78)

    print(
        results_df[
            [
                "policy",
                "agreement_pct",
                "mean_reward",
                "mean_regret",
                "p95_regret",
                "under_selection_pct",
                "over_selection_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Original-threat vs AI-threat "
        f"policy consistency: "
        f"{policy_consistency:.3f}%"
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "final_test_policy_summary.csv",
        index=False,
    )

    intelligence_summary = (
        pd.DataFrame(
            [
                {
                    "threat_mae":
                        threat_mae,

                    "threat_rmse":
                        threat_rmse,

                    "threat_r2":
                        threat_r2,

                    "threat_correlation":
                        threat_corr,

                    "anomaly_roc_auc":
                        anomaly_roc_auc,

                    "anomaly_pr_auc":
                        anomaly_pr_auc,

                    "anomaly_precision":
                        anomaly_precision,

                    "anomaly_recall":
                        anomaly_recall,

                    "anomaly_f1":
                        anomaly_f1,

                    "policy_consistency_pct":
                        policy_consistency,
                }
            ]
        )
    )

    intelligence_summary.to_csv(
        OUTPUT_DIR
        / "final_test_intelligence_summary.csv",
        index=False,
    )

    sample_output = (
        test_df.copy()
    )

    sample_output[
        "ai_predicted_network_threat"
    ] = predicted_threat

    sample_output[
        "oracle_action"
    ] = oracle_actions

    sample_output[
        "original_threat_action"
    ] = original_actions

    sample_output[
        "ai_threat_action"
    ] = ai_actions

    sample_output[
        "policy_changed"
    ] = (
        original_actions
        != ai_actions
    ).astype(
        np.int64
    )

    sample_output.to_csv(
        OUTPUT_DIR
        / "final_test_samples.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "FINAL TEST COMPLETE — CONFIGURATION LOCKED"
    )
    print("=" * 78)

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()