# 🎯 Smart MCQ Solver Challenge

### Deep Learning & Generative AI Project · Diploma in BS Data Science and Applications

**Author:** Shamanth V (`23f2004250`)\
**Course:** Deep Learning and Generative AI : Diploma Level, BS in Data Science and Applications\
**Competition:** [Smart MCQ Solver Challenge]([https://www.kaggle.com/](https://www.kaggle.com/competitions/smart-mcq-solver-challenge)) (Kaggle)  
**Metric:** MAP@3

---

## 📌 Problem Statement

Each question in the dataset provides a **prompt** and **five candidate answers (A to E)**, exactly one of which is correct. The task is to predict the **top-3 most likely correct answers, ranked by confidence**, so that the true answer is retrieved as early as possible in the ranking.

Performance is scored with **Mean Average Precision @ 3 (MAP@3)**, which rewards the correct label appearing higher in the predicted list, getting it in position 1 is worth more than position 3, but position 3 still counts for something.

- **Training set:** 2,000 labeled questions (5 options each, 1 correct label, no missing values)
- **Test set:** 500 unlabeled questions, same format
- **Submission format:** `id, A B C` : three letters, space-separated, ranked by confidence

---

## 🧭 Project Rubric Coverage

| # | Rubric Requirement | Model | Notebook Section |
|---|---|---|---|
| 1 | Classical / statistical baseline | **LightGBM** on TF-IDF similarity + engineered statistical features | Model 1 |
| 2 | **From-scratch model** | **Feed-forward MLP** (pure PyTorch, no pretrained weights) on TF-IDF/SVD embeddings | Model 2 |
| 3 | **Pretrained model**, fine-tuned | **RoBERTa-base**, fully fine-tuned end-to-end on `(prompt, option)` pairs | Model 3 |
| 4 | **Model of choice / bonus** | **LoRA-tuned RoBERTa** — parameter-efficient fine-tuning via `peft` | Model 4 |
| 5 | Ensembling milestone | Weighted probability average of the best 2–3 models by OOF MAP@3 | Ensemble |

Every model is evaluated with **GroupKFold cross-validation**, grouped by question `id`, so all 5 options belonging to the same question always stay together on one side of the split, this prevents leakage between options of the same MCQ.

---

## 🌿 Branching Strategy

Per course requirements, this repo follows a strict milestone-branch workflow:

- **`main`** always contains the latest working notebook, final report, and README — the most stable, up-to-date snapshot of the project.
```
project-name/
├── notebooks/
│   └── smart_mcq_solver.ipynb        # full pipeline, milestones 1–5
├── reports/
│   └── final-report.pdf
├── requirements.txt
└── README.md
```

---

## 🔎 Exploratory Data Analysis

Two strong, exploitable signals shaped the entire pipeline:

**1. Position bias.** Correct answers are not evenly distributed across A–E (uniform baseline would be 20% each):

| Option | % Correct |
|---|---|
| B | 24.5% |
| C | 22.95% |
| A | 18.45% |
| D | 17.9% |
| E | 16.2% |

This was encoded directly as a `position_prior` feature (learned only from train, applied to both splits, no leakage).

**2. Length bias.** Prompts average 18 words; options average 26 words. Correct options run longer on average (28.66 words) than incorrect ones (25.68 words), and **the single longest option in a question is correct 34.95% of the time** , well above the 20% random baseline. This was the single strongest signal in the dataset and drove most of the feature engineering.

No data augmentation was used, the existing signals were strong enough that augmentation risked diluting them.

---

## 🛠️ Approach

The dataset was reshaped from **wide format** (one row per question, 5 answer columns) into **long format** (one row per `(question, option)` pair with a binary `correct / not correct` label). This turns the ranking problem into a binary classification problem that every model below solves in the same way, each producing a per-option probability that is then ranked into a top-3 submission.

### 1️⃣ Feature Engineering

Before modeling, each option is described three ways: on its own, relative to the prompt, and relative to the other 4 options in the same question.

- **Length-based:** character/word counts, rank of this option's length among its siblings, z-score relative to the question's mean/std, "is longest" / "is shortest" flags
- **Positional prior:** empirical probability that a given letter (A to E) is correct, learned from train only
- **Prompt overlap:** word-level overlap between option text and the question prompt
- **Inter-option overlap:** lexical similarity of an option to the other options in the same question
- **Surface cues:** comma/digit counts, negation-word presence
- **TF-IDF cosine similarity** between option text and prompt (vectorizer fit once, jointly across train + test, 30,000 features, unigrams + bigrams)
- **SVD-compressed embeddings** (64-dim) of TF-IDF vectors, used as dense input for the MLP

### 2️⃣ Model 1 : LightGBM (Classical Baseline)

A gradient-boosted decision tree trained as a binary classifier on the engineered features above. Fast, interpretable, and used to sanity-check which engineered signals matter most via feature importance (length and TF-IDF similarity dominated, matching the EDA).

- 31 leaves, learning rate 0.03, up to 2,000 estimators, early stopping after 100 rounds without improvement
- Evaluated with GroupKFold cross-validation, OOF predictions used for both scoring and the ensemble

### 3️⃣ Model 2 : From-Scratch MLP

Satisfies the **mandatory from-scratch model** requirement: a plain feed-forward network in pure PyTorch (`Linear → ReLU → Dropout`, two hidden layers of 128 then 64 units), with **no pretrained weights**. Inputs are the SVD-compressed TF-IDF embeddings plus the same engineered features used by LightGBM, standardized before training.

- Adam optimizer, learning rate 1e-3, up to 60 epochs
- Early stopping based on **validation MAP@3** directly (not just loss), so the stopping criterion matches the competition metric

### 4️⃣ Models 3 & 4 : RoBERTa Fine-Tune + LoRA (Bonus)

One reusable function implements both, toggled by a `use_lora` flag. Both tokenize `(prompt, option)` as a sentence pair (RoBERTa's byte-level BPE tokenizer, truncated to 256 tokens) and train with `BCEWithLogitsLoss`, keeping them directly comparable to Models 1 & 2 via OOF MAP@3.

- **Model 3 (pretrained fine-tune):** `roberta-base` with a sequence classification head, **all weights updated**. AdamW, learning rate 2e-5, 2 epochs, batch size 16.
- **Model 4 (bonus / model of choice):** same backbone, but with **LoRA adapters** (`peft`, rank=16, alpha=32) attached to the attention query/value projections, only the low-rank adapters are trained, everything else frozen. Only ~0.94% of parameters trainable. Higher learning rate (1e-4), 3 epochs, since only a small fraction of the model is learning.

### 5️⃣ Ensemble

The best 2–3 models by OOF MAP@3 are combined via a **weighted probability average** (weights proportional to each model's own OOF score), then re-ranked per question for the final top-3 submission, this is what typically pushes the leaderboard score above any single model.

---

## 📊 Results

All models evaluated with GroupKFold cross-validation (grouped by question `id`). Precision/recall/F1/accuracy computed at a 0.5 probability cutoff; MAP@3 is the competition metric.

| Model | MAP@3 | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| LightGBM | **0.9928** | 0.9918 | 0.9710 | 0.9813 | 0.9926 |
| MLP (from scratch) | 0.9896 | 0.9894 | 0.9330 | 0.9604 | 0.9846 |
| RoBERTa (fine-tuned) | 0.7650 | 0.8487 | 0.1430 | 0.2448 | 0.8235 |
| RoBERTa + LoRA | 0.6531 | 0.0000 | 0.0000 | 0.0000 | 0.8000 |
| **Ensemble** (LightGBM + MLP + RoBERTa) | **0.9932** | — | — | — | — |

**Key finding:** the feature-based models won clearly. LightGBM and the MLP both exploit the same length/overlap signals surfaced during EDA and scored near-perfectly, while the transformers had to rediscover correctness cues from raw text with only 2,000 training questions, too little data for that class of model to shine. LoRA, training under 1% of parameters, never crossed the 0.5 cutoff in the given epoch budget (hence zero precision/recall), though its raw ranking still earned partial MAP@3 credit. LightGBM was also far cheaper to train (seconds vs. several minutes per fold for the transformers), making it both the most accurate and the most efficient model here. The ensemble improved marginally over LightGBM alone (0.9932 vs 0.9928), showing the other models still contributed a small amount of independent signal.

**Kaggle leaderboard:** 0.75394 (public) / 0.76094 (private), final rank **109th** on the private leaderboard (up 439 spots from the public rank).

---

## 🧪 Experiment Tracking

All model runs (TF-IDF/LightGBM baseline, MLP, RoBERTa fine-tune, LoRA, ensemble) are logged to **Weights & Biases** under the project `23f2004250-t22026`, comparing accuracy and F1 across runs per the course requirement.

---

## ▶️ Running It

Designed to run inside a Kaggle notebook (T4 GPU recommended)

```bash
# inside the Kaggle notebook, milestone branch checked out
!pip install -q transformers accelerate datasets peft lightgbm

# clone / pull the latest branch
!git clone https://github.com/Shamanth-V/DL-and-GenAI-Project.git
%cd DL-and-GenAI-Project
!git checkout milestone-5
```

From there, running the notebook top to bottom:

1. Loads and reshapes the data (wide → long)
2. Builds engineered + TF-IDF/SVD features
3. Trains Models 1 to 4, each reporting its own OOF MAP@3
4. Scores and ranks all trained models, selects the top 2 to 3 for the ensemble
5. Builds the final ensembled top-3 predictions and writes `submission.csv` in the `id,A B C` format the competition expects

For a quick local/CPU-only test, set `run_transformers=False` in `main()` to only run the LightGBM + MLP baseline pair.

Local dependencies (for anything run outside Kaggle) are listed in `requirements.txt`.

---

## ⚠️ Notes for Future

- **GroupKFold is non-negotiable here** : grouping by question `id` is the only thing stopping options from the same question leaking across train/validation.
- **`peft` probes for an optional `torchao` backend on import**, which can conflict with the installed version, resolved with an uninstall/reinstall fix in the notebook rather than pinning a specific version.
- **`transformers` renamed `tokenizer` → `processing_class` in `Trainer`** in newer versions, and `TrainingArguments` dropped `group_by_length` in some releases, both handled defensively so the notebook doesn't break depending on which version Kaggle has installed that day.
- LoRA's low trainable-parameter count means it needs a *higher* learning rate and more epochs than full fine-tuning to converge in a comparable wall-clock budget, and even then it underperformed here.
- The classical/from-scratch models beating the transformers is a direct consequence of dataset size (2,000 examples), with more data, the balance would likely tip back toward the transformer models.

---

## 🔭 Future Work

- Give the transformer models more training time and a learning-rate warmup schedule
- Try a pretrained model architecture better suited to this kind of pairwise classification task
- Combine engineered features with transformer embeddings directly in one model, rather than only at the ensemble stage
- Increase cross-validation folds for a more reliable score estimate, at the cost of extra training time

---
