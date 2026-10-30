# HASTIKA @ ICON 2026: Kannada-English Hate Speech Detection

This repository contains the implementation and the System Description Paper for the **HASTIKA Shared Task** at **ICON 2026**. The project addresses hate speech detection and target category identification in Kannada-English code-mixed (Kanglish) social media comments.

## 📌 Task Overview

The shared task consists of two subtasks:
*   **Task A (Binary Hate Detection):** Classifying a comment as `Hate` or `Non-Hate`.
*   **Task B (Target Category Identification):** Classifying hateful comments into one of six categories: `Gender`, `Geo-political`, `Others`, `Political`, `Religion`, or `Violence`. 
*   *Note:* Task B is highly imbalanced (e.g., 1,362 Gender samples vs. 186 Geo-political samples).

## ⚙️ Methodology

We designed a hybrid pipeline combining traditional machine learning with deep learning to handle romanized spelling variations and class imbalance.

1.  **Text Preprocessing:** 
    *   Repaired UTF-8 mojibake (recovered damaged Kannada Unicode).
    *   Converted emojis to text, removed HTML/URLs, and normalized repeated characters.
2.  **Classical Stacking:** 
    *   Extracted Word-level and Character-level TF-IDF features along with 8 hand-crafted meta-features.
    *   Trained a Stacking Ensemble (LinearSVC, Logistic Regression, ComplementNB) with a Logistic Regression meta-learner.
    *   *Strict leakage control: Vectorizers were fitted strictly inside each cross-validation fold.*
3.  **Transformer Fine-Tuning:**
    *   Fine-tuned `xlm-roberta-base` and `google/muril-base-cased`.
    *   Applied **R-Drop regularization** and **Label Smoothing** to prevent overfitting on the small dataset.
4.  **Ensemble Strategy:**
    *   **Task A:** Weighted probability ensemble of Classical (0.45), XLM-R (0.25), and MuRIL (0.30).
    *   **Task B:** Transformers struggled with minority classes due to imbalance, so the final prediction relied solely on the Classical Stacking model.

## 🏆 Official Test Results

| Task | System | Dev F1 | Test Macro-F1 | Test Accuracy | Official Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task A** | Classical + XLM-R + MuRIL | 0.8288 | **0.7977** | 0.7977 | 18 / 30 |
| **Task B** | Classical Stacking Only | 0.6648 | **0.6246** | 0.6869 | 15 / 29 |

## 📂 Repository Structure

*   `nlp-hastika.ipynb`: The complete training and inference notebook (executed on Kaggle).
*   `paper.pdf`: The final System Description Paper submitted to ICON 2026.
*   `README.md`: Project documentation.

## 📜 Citation

If you use this work, please cite the original HASTIKA dataset paper:

```bibtex
@article{kavatagi2025hastika,
  title={HASTIKA: hate speech and target identification in Kannada-English code-mixed text},
  author={Kavatagi, Sanjana and Rachh, Rashmi},
  journal={Language Resources and Evaluation},
  volume={59},
  number={3},
  pages={2811--2856},
  year={2025},
  publisher={Springer}
}
