from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/processed/aegisflow_100k.csv"
)

OUTPUT_DIR = Path(
    "data/processed/splits"
)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42


# ============================================================
# VALIDATION
# ============================================================

def validate_ratios():

    total = (
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
    )

    if not np.isclose(total, 1.0):

        raise ValueError(
            "Train/validation/test ratios "
            f"must sum to 1.0, got {total}"
        )


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(df):

    rng = np.random.default_rng(SEED)

    train_parts = []
    val_parts = []
    test_parts = []

    # --------------------------------------------------------
    # Stratify using:
    #
    # scenario + required_level
    #
    # This preserves both scenario composition and
    # security-level distribution.
    # --------------------------------------------------------

    grouped = df.groupby(
        [
            "scenario",
            "required_level",
        ],
        sort=True,
    )

    for _, group in grouped:

        indices = group.index.to_numpy().copy()

        rng.shuffle(indices)

        n = len(indices)

        n_train = int(
            np.floor(
                n * TRAIN_RATIO
            )
        )

        n_val = int(
            np.floor(
                n * VAL_RATIO
            )
        )

        train_idx = indices[
            :n_train
        ]

        val_idx = indices[
            n_train:
            n_train + n_val
        ]

        test_idx = indices[
            n_train + n_val:
        ]

        train_parts.append(
            df.loc[train_idx]
        )

        val_parts.append(
            df.loc[val_idx]
        )

        test_parts.append(
            df.loc[test_idx]
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    val_df = pd.concat(
        val_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Shuffle each split again
    # --------------------------------------------------------

    train_df = train_df.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1.0,
        random_state=SEED + 1,
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1.0,
        random_state=SEED + 2,
    ).reset_index(drop=True)

    return (
        train_df,
        val_df,
        test_df,
    )


# ============================================================
# AUDIT
# ============================================================

def print_distribution(
    name,
    df,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Rows: {len(df):,}"
    )

    print()
    print(
        "Required level distribution:"
    )

    level_distribution = (
        df["required_level"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(
        level_distribution.round(3)
    )

    print()
    print(
        "Scenario distribution:"
    )

    scenario_distribution = (
        df["scenario"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(
        scenario_distribution.round(3)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    validate_ratios()

    print("=" * 70)
    print("AEGISFLOW DATASET SPLITTING")
    print("=" * 70)

    print(
        "Input:",
        INPUT_PATH,
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Original dataset: "
        f"{len(df):,} rows"
    )

    # --------------------------------------------------------
    # Add immutable identifier
    # --------------------------------------------------------

    df = df.copy()

    df.insert(
        0,
        "sample_id",
        np.arange(
            len(df),
            dtype=np.int64,
        ),
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df,
    ) = stratified_split(df)

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    train_ids = set(
        train_df["sample_id"]
    )

    val_ids = set(
        val_df["sample_id"]
    )

    test_ids = set(
        test_df["sample_id"]
    )

    assert train_ids.isdisjoint(
        val_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert val_ids.isdisjoint(
        test_ids
    )

    assert (
        len(train_df)
        + len(val_df)
        + len(test_df)
        ==
        len(df)
    )

    all_ids = (
        train_ids
        | val_ids
        | test_ids
    )

    assert len(all_ids) == len(df)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        OUTPUT_DIR
        / "train.csv"
    )

    val_path = (
        OUTPUT_DIR
        / "validation.csv"
    )

    test_path = (
        OUTPUT_DIR
        / "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    val_df.to_csv(
        val_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_distribution(
        "TRAIN SET",
        train_df,
    )

    print_distribution(
        "VALIDATION SET",
        val_df,
    )

    print_distribution(
        "TEST SET",
        test_df,
    )

    print()
    print("=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print(
        "Train:",
        train_path,
    )

    print(
        "Validation:",
        val_path,
    )

    print(
        "Test:",
        test_path,
    )

    print()
    print(
        "Overlap check: PASSED"
    )

    print(
        "Total rows:",
        (
            len(train_df)
            + len(val_df)
            + len(test_df)
        ),
    )


if __name__ == "__main__":
    main()