from pathlib import Path

import pandas as pd

from aegisflow.intelligence.telemetry_generator import (
    generate_telemetry,
)


TEST_CONTEXT_PATH = Path(
    "data/processed/splits/test.csv"
)

OUTPUT_PATH = Path(
    "data/telemetry/telemetry_test.csv"
)

TEST_TELEMETRY_SEED = 2026


def main():

    print("=" * 78)
    print(
        "AEGISFLOW FINAL TEST TELEMETRY GENERATION"
    )
    print("=" * 78)

    if not TEST_CONTEXT_PATH.exists():
        raise FileNotFoundError(
            TEST_CONTEXT_PATH
        )

    test_df = pd.read_csv(
        TEST_CONTEXT_PATH
    )

    print(
        f"Test context rows: "
        f"{len(test_df):,}"
    )

    telemetry_df = generate_telemetry(
        context_df=test_df,
        seed=TEST_TELEMETRY_SEED,
    )

    if (
        len(telemetry_df)
        != len(test_df)
    ):
        raise ValueError(
            "Telemetry/test row count mismatch."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    telemetry_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved {len(telemetry_df):,} rows to:"
    )

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "Reference anomaly distribution:"
    )

    print(
        telemetry_df[
            "reference_anomaly"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print()
    print("=" * 78)
    print(
        "TEST TELEMETRY GENERATION COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()