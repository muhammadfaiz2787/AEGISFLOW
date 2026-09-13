import pandas as pd

from aegisflow.environment.reward import calculate_reward
from aegisflow.environment.policy import get_level_name


DATASET_PATH = "data/processed/aegisflow_100k.csv"


def evaluate_row(row):
    """
    Evaluate all security actions for a single state
    and return the optimal action according to the reward function.
    """

    results = {}

    for action in range(4):
        results[action] = calculate_reward(
            row,
            action,
        )

    best_action = max(
        results,
        key=lambda action: results[action]["reward"],
    )

    return {
        "oracle_action": best_action,
        "oracle_level": get_level_name(best_action),
        "oracle_reward": results[best_action]["reward"],
        "action_rewards": {
            action: results[action]["reward"]
            for action in range(4)
        },
    }


def main():

    print("=" * 70)
    print("AEGISFLOW ORACLE EVALUATION")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset size: {len(df):,}")

    oracle_actions = []
    oracle_rewards = []
    oracle_margins = []

    # ========================================================
    # ORACLE EVALUATION
    # ========================================================

    for _, row in df.iterrows():

        result = evaluate_row(row)

        oracle_actions.append(
            result["oracle_action"]
        )

        oracle_rewards.append(
            result["oracle_reward"]
        )

        # ----------------------------------------------------
        # Decision margin
        # ----------------------------------------------------

        sorted_rewards = sorted(
            result["action_rewards"].values(),
            reverse=True,
        )

        margin = (
            sorted_rewards[0]
            - sorted_rewards[1]
        )

        oracle_margins.append(
            margin
        )

    # ========================================================
    # SAVE RESULTS TO DATAFRAME
    # ========================================================

    df["oracle_action"] = oracle_actions
    df["oracle_reward"] = oracle_rewards
    df["oracle_margin"] = oracle_margins

    # ========================================================
    # ORACLE ACTION DISTRIBUTION
    # ========================================================

    print("\nOracle action distribution:")

    print(
        df["oracle_action"]
        .value_counts()
        .sort_index()
    )

    print("\nOracle action percentage:")

    print(
        (
            df["oracle_action"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(3)
    )

    # ========================================================
    # REQUIRED LEVEL VS ORACLE
    # ========================================================

    print("\nRequired level vs Oracle level:")

    comparison = pd.crosstab(
        df["required_level"],
        df["oracle_action"],
    )

    print(comparison)

    # ========================================================
    # REQUIRED LEVEL VS ORACLE (%)
    # ========================================================

    print("\nOracle action vs required level (%):")

    comparison_pct = pd.crosstab(
        df["required_level"],
        df["oracle_action"],
        normalize="index",
    ) * 100

    print(
        comparison_pct.round(2)
    )

    # ========================================================
    # ORACLE REWARD
    # ========================================================

    print("\nOracle reward statistics:")

    print(
        df["oracle_reward"].describe()
    )

    # ========================================================
    # DECISION MARGIN
    # ========================================================

    print("\nOracle decision margin statistics:")

    print(
        df["oracle_margin"].describe()
    )

    # ========================================================
    # NEAR-TIE ANALYSIS
    # ========================================================

    print("\nNear-tie decisions:")

    for threshold in [0.01, 0.05, 0.10, 0.25]:

        count = (
            df["oracle_margin"] < threshold
        ).sum()

        percentage = (
            count / len(df) * 100
        )

        print(
            f"margin < {threshold:.2f}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # SAVE ORACLE DATASET
    # ========================================================

    print("\nSaving Oracle dataset...")

    output_path = (
        "data/processed/"
        "aegisflow_oracle_100k.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()