# Smart MCQ Solver : DL & GenAI Course Project

**Author:** Shamanth V (23f2004250)
**Course:** Deep Learning and Generative AI : Diploma Level, BS in Data Science and Applications
**Competition:** Smart MCQ Solver Challenge (Kaggle)

---

## What this project is about

This repo holds all my work for the DL & GenAI course project. The task itself is
pretty straightforward to state and a lot harder to actually do well: given a
question (`prompt`) and five answer options (`A`:`E`), predict the top 3 most
likely correct answers, ranked by confidence. The competition is scored using
**MAP@3** (Mean Average Precision @ 3), so getting the right answer into position
1 is worth more than getting it into position 3, but getting it in there at all
still counts for something.

Dataset is roughly 2000 labeled training questions and 500 unlabeled test
questions. Submission format is just `id,A B C` style rows : three letters,
space separated, ranked.

I've gone through this in stages rather than trying to jump straight to a
fine-tuned transformer, mainly because the milestone structure of the course
pushes you that way, but honestly it also just made debugging easier. Every
stage is a separate branch (see below) so the progression is visible in the
git history rather than just squashed into one giant commit at the end.

---


Branches follow `milestone-1`, `milestone-2`, ... `milestone-5`, each merged
into `main` once that stage is stable. None of the milestone branches get
deleted after merging, keeping them around is part of the evaluation
requirement (progress has to be auditable, not just the final state).

---

## How I approached each milestone

**Milestone 1 : Baselines**
Started with basic text cleaning and TF-IDF vectorization of the
prompt + options. Computed cosine similarity between the prompt and each
option as a first pass "does this even make sense" check, then wrote a MAP@3
scorer from scratch to sanity-check against sklearn's ranking metrics. This
gave me a real (if not great) baseline submission on Kaggle to compare
everything else against.

**Milestone 2 : Transformers enter the picture**
Moved to `bert-base-uncased` for contextual embeddings and actually looked at
the attention weights to see what the model was picking up on across
question/option pairs,useful for building intuition, less useful for actual
score improvement at this stage. Also tried `all-MiniLM-L6-v2` for
sentence-level similarity (much faster, similar-ish performance), zero-shot
classification with `facebook/bart-large-mnli`, and a generative approach
using `google/flan-t5-base` just to see what a purely generative answer
selection would look like.

**Milestone 3 : RAG**
Added a retrieval layer: FAISS for semantic search over a small context
corpus, followed by cross-encoder reranking
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) to tighten up the retrieved context
before it gets fed alongside the prompt and options. There's a
`USE_RAG_CONTEXT` toggle in the code so the pipeline still runs (and doesn't
silently break) if retrieval isn't wanted for a given run — this ended up
being handy for A/B comparisons more than anything else.

**Milestones 4 & 5 : Fine-tuning and ensembling**
This is where most of the actual score improvement came from. Fine-tuned
`microsoft/deberta-v3-base` using `AutoModelForMultipleChoice`, which forces a
specific `(batch, num_choices, seq_len)` input shape and a custom data
collator — that reshape isn't a style choice, it's just how the HF
multiple-choice head expects its inputs. Ran 5-fold cross-validation and
stacked out-of-fold predictions, and separately built a from-scratch
TF-IDF + feedforward model to have a genuinely "not-pretrained" model in the
mix (needed for the "3 unique models" requirement anyway, one from scratch,
one pretrained, one of choice). Final predictions come from a small grid
search over ensemble blend weights between the DeBERTa CV models and the
from-scratch model.

---

## A couple of things worth flagging (mostly to future-me during viva prep)

- FAISS is picky about dtype, everything going in needs to be `float32` and
  contiguous, or it fails in ways that aren't obvious from the error message.
- Mapping predicted labels back to letters (A–E) has to be done positionally,
  not by matching option text, if two options happen to have identical text,
  a text-based reverse lookup silently gives the wrong letter.
- `transformers` renamed `tokenizer` → `processing_class` in `Trainer`
  somewhere along the way, and `TrainingArguments` dropped `group_by_length`
  in some versions. Both are wrapped in try/except so the notebook doesn't
  break depending on which version Kaggle happens to have installed that day.
- Training was done on Kaggle's free T4 GPUs — the P100 option exists too but
  its older `sm_60` CUDA compute capability doesn't play nicely with newer
  PyTorch builds, so T4 was the safer pick throughout.

---

## Running it

Everything is designed to run inside a Kaggle notebook (T4 GPU). Rough order:

```bash
# inside the Kaggle notebook, milestone branch checked out
!pip install -q transformers accelerate datasets sentence-transformers faiss-cpu peft

# clone / pull latest branch
!git clone https://github.com/Shamanth-V/DL-and-GenAI-Project.git
%cd DL-and-GenAI-Project
!git checkout milestone-4
```

From there, the notebook cells handle training, cross-validation, and
generating `submission.csv` in the `id,Prediction` format the competition
expects.

Local dependencies (for anything run outside Kaggle) are listed in
`requirements.txt`.

---

## Experiment tracking

All model runs are logged to Weights & Biases under the project
`23f2004250-t22026`. At least three runs (TF-IDF baseline, DeBERTa fine-tune,
ensemble) are compared there on accuracy and F1, per the course requirement.

---

## Status

Milestones 1 through 5 are done and merged into `main`. Final report and
Kaggle submission polishing are still in progress ahead of the last
submission deadline.
