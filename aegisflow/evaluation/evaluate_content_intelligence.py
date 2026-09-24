from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from aegisflow.context.content_classifier import ContentClassifier


CATEGORIES = (
    "public",
    "general",
    "personal",
    "financial",
    "medical",
    "credentials",
    "iot",
)
BENIGN_CATEGORIES = {"public", "general"}
SENSITIVE_CATEGORIES = {"personal", "financial", "medical", "credentials"}


def iter_samples(root: Path):
    for label in CATEGORIES:
        folder = root / label
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name != "README.md":
                yield label, path


def evaluate(root: Path, output_dir: Path, enable_vision: bool = True) -> None:
    samples = list(iter_samples(root))
    if not samples:
        raise SystemExit(
            f"No validation files found under {root}. "
            "Create one folder per class first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    classifier = ContentClassifier(enable_vision=enable_vision)

    rows = []
    print("=" * 88)
    print("AEGISFLOW CONTENT INTELLIGENCE VALIDATION")
    print("=" * 88)
    print(f"Dataset root : {root}")
    print(f"Samples      : {len(samples)}")
    print(f"Vision       : {'enabled' if enable_vision else 'disabled'}")
    print()

    for index, (true_label, path) in enumerate(samples, start=1):
        content = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        result = classifier.classify(
            filename=path.name,
            content=content,
            mime_type=mime,
        )
        predicted = result.category
        rows.append(
            {
                "path": str(path),
                "true_label": true_label,
                "predicted_label": predicted,
                "correct": predicted == true_label,
                "context_confidence": result.confidence,
                "mime": result.detected_mime,
                "analysis_mode": result.analysis_mode,
                "vision_status": result.vision_status,
                "signals": " | ".join(result.signals),
            }
        )
        print(
            f"{index:4d}/{len(samples)} | "
            f"true={true_label:11s} | pred={predicted:11s} | {path.name}"
        )

    df = pd.DataFrame(rows)
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()

    labels_present = [
        label for label in CATEGORIES
        if label in set(y_true) or label in set(y_pred)
    ]
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    matrix = confusion_matrix(y_true, y_pred, labels=labels_present)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true:{label}" for label in labels_present],
        columns=[f"pred:{label}" for label in labels_present],
    )

    accuracy = float((df["true_label"] == df["predicted_label"]).mean())
    macro_f1 = float(report_dict.get("macro avg", {}).get("f1-score", 0.0))

    benign_mask = df["true_label"].isin(BENIGN_CATEGORIES)
    sensitive_prediction = df["predicted_label"].isin(SENSITIVE_CATEGORIES)
    if benign_mask.any():
        false_sensitive_escalation_rate = float(
            (benign_mask & sensitive_prediction).sum() / benign_mask.sum()
        )
    else:
        false_sensitive_escalation_rate = float("nan")

    sensitive_mask = df["true_label"].isin(SENSITIVE_CATEGORIES)
    benign_prediction = df["predicted_label"].isin(BENIGN_CATEGORIES)
    if sensitive_mask.any():
        sensitive_miss_rate = float(
            (sensitive_mask & benign_prediction).sum() / sensitive_mask.sum()
        )
    else:
        sensitive_miss_rate = float("nan")

    summary_df = pd.DataFrame(
        [
            {"metric": "samples", "value": len(df)},
            {"metric": "accuracy", "value": accuracy},
            {"metric": "macro_f1", "value": macro_f1},
            {
                "metric": "false_sensitive_escalation_rate",
                "value": false_sensitive_escalation_rate,
            },
            {"metric": "sensitive_miss_rate", "value": sensitive_miss_rate},
        ]
    )

    predictions_path = output_dir / "content_validation_predictions.csv"
    report_path = output_dir / "content_validation_classification_report.csv"
    matrix_path = output_dir / "content_validation_confusion_matrix.csv"
    summary_path = output_dir / "content_validation_summary.csv"

    df.to_csv(predictions_path, index=False)
    report_df.to_csv(report_path)
    matrix_df.to_csv(matrix_path)
    summary_df.to_csv(summary_path, index=False)

    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"Accuracy                       : {accuracy:.4f}")
    print(f"Macro F1                       : {macro_f1:.4f}")
    if benign_mask.any():
        print(
            "False sensitive escalation rate: "
            f"{false_sensitive_escalation_rate:.4f}"
        )
    else:
        print("False sensitive escalation rate: n/a (no benign samples)")
    if sensitive_mask.any():
        print(f"Sensitive miss rate            : {sensitive_miss_rate:.4f}")
    else:
        print("Sensitive miss rate            : n/a (no sensitive samples)")
    print()
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {predictions_path}")
    print(f"  {report_path}")
    print(f"  {matrix_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AegisFlow automatic content classification."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/content_validation"),
        help="Dataset root containing one folder per class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/content_validation/results"),
        help="Directory for CSV evaluation outputs.",
    )
    parser.add_argument(
        "--disable-vision",
        action="store_true",
        help="Evaluate only metadata/text fallback without local image vision.",
    )
    args = parser.parse_args()
    evaluate(
        root=args.dataset,
        output_dir=args.output,
        enable_vision=not args.disable_vision,
    )


if __name__ == "__main__":
    main()
