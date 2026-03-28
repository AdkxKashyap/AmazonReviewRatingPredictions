# Amazon Product Rating Predictions

A machine learning project that predicts the star rating (1–5) of Amazon product reviews using both classical NLP methods and deep learning models.

---

## Problem Statement

Given the text of an Amazon product review, predict the corresponding star rating on a scale of 1 to 5. This is a **5-class text classification** problem where the classes are naturally ordered and balanced across the dataset.

---

## Dataset

- **Source:** Amazon product reviews (text format)
- **Size:** ~650,000 reviews
- **Classes:** Star ratings 1 through 5 (~20% each — balanced)
- **Split:** 80% train / 10% validation / 10% test

> The dataset file is not included in this repository due to size constraints (278 MB). Download it separately and place it under `Datasets/amazon_review_small.txt`.

---

## Project Structure

```
AmazonReviewRatingPredictions/
├── Review_rating_predictions.ipynb   # Main notebook
├── Datasets/                          # Local dataset folder (git-ignored)
│   └── amazon_review_small.txt
└── README.md
```

---

## Approach

### Section 1–6: Classical ML with TF-IDF + SVM

Multiple iterations of a TF-IDF + LinearSVC pipeline were built and progressively tuned:

| Iteration | Changes | Val Accuracy |
|---|---|---|
| Baseline | CountVectorizer, Naive Bayes | ~55% |
| Iteration 1 | TF-IDF, LogisticRegression | ~60% |
| Iteration 2 | Tuned TF-IDF params | ~62% |
| Iteration 3 | LinearSVC, class_weight='balanced' | ~63% |
| Iteration 4 | max_features 5k→10k, ngram (1,2)→(1,3), min_df=2 | ~64% |

**Iteration 4 final config:**
- `TfidfVectorizer(max_features=10000, ngram_range=(1,3), min_df=2)`
- `LinearSVC(class_weight="balanced", max_iter=10000, random_state=42)`

---

### Section 7: Deep Learning with LSTM

#### 7.1 Tokenization & Preprocessing
- Used `keras_preprocessing.text.Tokenizer` with `MAX_VOCAB_SIZE=30,000`
- Sequences padded/truncated to `MAX_SEQUENCE_LENGTH=100` tokens (post-padding)
- Labels converted from 1–5 to 0–4 as NumPy int32 arrays
- Trained on a **25,000-sample subset** for rapid experimentation

#### 7.2 LSTM with Hyperparameter Tuning

**v1 hyperparameters** caused heavy overfitting:
- `embedding_dim=128`, `lstm_units=64`, `max_len=200`, `dropout=0.5`
- Result: train accuracy 80%+, val accuracy only **~44%**

**v2 hyperparameters** reduced model capacity and added EarlyStopping:
- `embedding_dim=64`, `lstm_units=32`, `max_len=100`, `dropout=0.3`
- `EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)`
- Result: train accuracy **61.32%**, val accuracy **56.95%** — overfitting significantly reduced

#### 7.3 Bidirectional LSTM on Full Training Data

Upgraded to a Bidirectional LSTM trained on the **full training set** (~416k samples):
- Same v2 hyperparameters
- `Bidirectional(LSTM(...))` reads sequences forward and backward, capturing richer context
- Result: train accuracy **63.16%**, best val accuracy **57.72%** (val_loss: 1.0024)

---

## Results Summary

| Model | Training Accuracy | Validation Accuracy |
|---|---|---|
| TF-IDF + LinearSVC (Iter 4) | — | ~64% |
| LSTM v1 (overfit) | 80%+ | 44% |
| LSTM v2 (tuned, 25k) | 61.32% | 56.95% |
| BiLSTM v2 (full data) | 63.16% | **57.72%** |

The classical TF-IDF + LinearSVC pipeline outperformed the LSTM-based models on this dataset. This is a known pattern for shorter text classification tasks where n-gram features are highly informative and the dataset is too large to fully exploit deep learning without more training time or a pretrained language model.

---

## Tech Stack

- **Python 3.x**
- **TensorFlow 2.21 / Keras 3.x**
- **scikit-learn**
- **pandas**, **NumPy**
- **keras-preprocessing** (for `Tokenizer` compatibility with Keras 3)
- Jupyter Notebook

## Setup

Install dependencies from the repository root:

```bash
source venv/bin/activate
pip install -r requirement.txt
venv/bin/python scripts/install_gpu_hook.py
```

For Linux/WSL GPU setups, this project uses `tensorflow[and-cuda]==2.21.0` and installs a virtualenv startup hook that preloads the CUDA libraries bundled inside the environment before TensorFlow imports. This avoids the common `Could not find cuda drivers` / `Cannot dlopen some GPU libraries` startup failure when the GPU driver is installed but the Python environment cannot see the CUDA runtime libraries.

---

## Key Lessons

- `mask_zero=True` in the Embedding layer is critical for padded sequence models — without it, the LSTM treats padding zeros as real tokens and fails to learn
- Labels must be converted to NumPy arrays before Keras training to avoid pandas index misalignment in mini-batches
- For 5-class fine-grained sentiment on shorter text, TF-IDF + linear classifiers are strong baselines that are hard to beat with shallow LSTMs
