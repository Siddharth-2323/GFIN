# Filled Tables from claude3.py Run

## Table 5. Baseline classifier performance (threshold=0.50)
| Model | Acc (%) | Prec (%) | Rec (%) | F1 (%) | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 79.00 | 80.85 | 76.00 | 78.35 | 0.8666 |
| Random Forest (200 trees) | 88.00 | 84.55 | 93.00 | 88.57 | 0.9376 |
| Hist. Gradient Boosting | 88.00 | 83.33 | 95.00 | 88.79 | 0.9240 |
| K-Nearest Neighbours | 82.00 | 76.23 | 93.00 | 83.78 | 0.8943 |
| Naive Bayes | 76.00 | 83.33 | 65.00 | 73.03 | 0.8596 |

## Table 7. Per-class classification report (Final Fused on test set)
| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| Class 0 (Non-Diabetic) | 92.39 | 85.00 | 88.54 | 100 |
| Class 1 (Diabetic) | 86.11 | 93.00 | 89.42 | 100 |
| Macro Average | 89.25 | 89.00 | 88.98 | 200 |
