# Adaptive ECG Quality Assessment ? Final Experiment Summary

## 1. Objective

Evaluate whether adapting preprocessing, continuous wavelet transform (CWT)
frequency settings, or Tucker decomposition rank according to ECG signal
characteristics improves unacceptable-record detection.

## 2. Dataset and evaluation

- Dataset: PhysioNet/CinC Challenge 2011, Set A.
- Label meaning: Acceptable versus Unacceptable ECG.
- Evaluation: five-fold group-aware cross-validation was used in the
  experiment scripts.
- Positive class: Unacceptable ECG.
- Sensitivity measures detection of unacceptable ECG records.
- Specificity measures correct identification of acceptable ECG records.
- Pooled out-of-fold metrics are reported from the saved experiment outputs.

The experiments do not all contain the same number of records. E1 and E3
contain 995 records; E2 contains 998 records. Their metrics should not be
treated as a perfectly matched three-way comparison.

## 3. E1 ? Adaptive preprocessing

| Metric | Result |
|---|---:|
| Records | 995 |
| Accuracy | 89.15% |
| Precision | 89.58% |
| Sensitivity | 58.11% |
| Specificity | 98.06% |
| F1-score | 70.49% |
| TN / FP / FN / TP | 758 / 15 / 93 / 129 |

The current adaptive preprocessing run achieved an F1-score of
70.49% and sensitivity of 58.11%.
A matched E0 baseline generated through the same feature-generation function
is needed before attributing the difference specifically to preprocessing.

## 4. E2 ? Adaptive CWT frequency

| Metric | Fixed 40 Hz | Adaptive 40/50 Hz |
|---|---:|---:|
| Accuracy | 91.08% | 90.68% |
| Precision | 85.05% | 85.87% |
| Sensitivity | 73.33% | 70.22% |
| Specificity | 96.25% | 96.64% |
| F1-score | 78.76% | 77.26% |
| FN / TP | 60 / 165 | 67 / 158 |

Adaptive CWT changed F1 by
-1.50 percentage points and
sensitivity by
-3.11 percentage
points. In this run, adaptive CWT did not improve the pooled F1-score.

## 5. E3 ? Adaptive Tucker rank

| Metric | Fixed rank | Adaptive rank |
|---|---:|---:|
| Accuracy | 90.85% | 91.06% |
| Precision | 85.03% | 82.13% |
| Sensitivity | 71.62% | 76.58% |
| Specificity | 96.38% | 95.21% |
| F1-score | 77.75% | 79.25% |
| TN / FP / FN / TP | 745 / 28 / 63 / 159 | 736 / 37 / 52 / 170 |

Within E3, adaptive rank changed F1 by
+1.50 percentage points and
sensitivity by
+4.95
percentage points. It detected
11 additional unacceptable records,
with 9 additional false positives.

This is the most promising result among these three experiments, but it is
a trade-off: sensitivity improves while specificity decreases.

## 6. Interpretation and limitations

1. Adaptive methods do not automatically outperform fixed settings.
2. E2 did not improve pooled F1 in the saved comparison.
3. E1 requires a matched baseline using the same feature-generation pipeline.
4. E3 should be compared against its own internal fixed-rank reference.
5. A separate fixed-core baseline has different results from E3's internal
   fixed-rank result. These are not interchangeable controls.
6. The results are cross-validation estimates on this dataset, not proof of
   generalization to other ECG datasets or clinical settings.
7. No statistical-significance claim is made here.

## 7. Conclusion

Adaptive Tucker rank improved pooled F1-score and sensitivity relative to
E3's internal fixed-rank reference. Adaptive CWT did not improve pooled F1
in its experiment. Adaptive preprocessing achieved lower sensitivity in its
current run, and a matched control is needed for a fair conclusion.

The overall contribution should be described as an investigation of
signal-adaptive ECG quality assessment, with mixed results across adaptive
components and a promising sensitivity/F1 improvement from adaptive rank.
