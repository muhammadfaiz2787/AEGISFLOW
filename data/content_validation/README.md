# AegisFlow Content Intelligence Validation Dataset

This folder is intentionally kept separate from the synthetic policy-training data.

Place real validation files into the class folders below:

```text
data/content_validation/
  public/
  general/
  personal/
  financial/
  medical/
  credentials/
  iot/
```

Recommended examples:

- **public**: event posters, flyers, public announcements, advertisements.
- **general**: vector icons, logos, ordinary photos, BTS/tower photos, landscapes, diagrams.
- **personal**: ID cards, passports, private personal documents or images.
- **financial**: receipts, invoices, bank statements, payment documents.
- **medical**: prescriptions, laboratory results, medical records.
- **credentials**: screenshots/files containing API keys, passwords, tokens, private keys.
- **iot**: telemetry dashboards, sensor/control interfaces, IoT screenshots.

Do not commit private or sensitive real documents to a public repository. For sensitive
classes, use redacted, synthetic, or explicitly authorized samples. The evaluator only
needs the files locally.

Run:

```powershell
python -m aegisflow.evaluation.evaluate_content_intelligence
```

Optional metadata/text-only baseline:

```powershell
python -m aegisflow.evaluation.evaluate_content_intelligence --disable-vision
```

Outputs are written to `data/content_validation/results/`:

- `content_validation_summary.csv`
- `content_validation_predictions.csv`
- `content_validation_classification_report.csv`
- `content_validation_confusion_matrix.csv`

The most important deployment metric for benign files is
**false_sensitive_escalation_rate**: the fraction of PUBLIC/GENERAL samples that are
incorrectly classified as PERSONAL/FINANCIAL/MEDICAL/CREDENTIALS. This directly
captures failures such as a harmless vector icon being escalated into a sensitive
security context.

The displayed "Context Confidence" in the frontend is currently heuristic and is not
an accuracy probability. Use this validation dataset to measure real accuracy, per-class
precision/recall/F1, and security-escalation errors.
