# Beyond Answer Generation: Diagnosing Narrative Reasoning in Transformers

**Course:** NLP Final Project | April 2026

---

## Abstract

We investigate whether language models can perform genuine narrative reasoning or merely exploit surface-level cues. Using the ROC Stories corpus and the Story Cloze Task, we generate controlled incorrect story endings spanning four reasoning types — temporal, causal, state, and sentiment — using GPT-4o-mini. We evaluate four models: a TF-IDF cosine similarity baseline, Sentence-BERT (SBERT), and two fine-tuned transformers (DistilBERT and RoBERTa). Our EDA reveals that the Story Cloze dataset contains weak shallow signals that partially explain baseline performance. Our central finding is a large gap between fake-detection accuracy on LLM-generated negatives (Val C: up to 99.7%) and accuracy on human-authored Story Cloze alternatives (Val B: 50–59%). Fine-tuned models appear to learn GPT-4o-mini's writing style rather than genuine reasoning. RoBERTa partially escapes this, achieving 57.0% on Val B (p < 0.001). Temporal reasoning is the hardest category across all models.

---

## 1. Introduction

The Story Cloze Task (Mostafazadeh et al., 2016) asks models to select the correct ending to a four-sentence story from two human-authored candidates. While modern transformers achieve high accuracy, it remains unclear whether they understand narrative logic or exploit shallow statistical patterns.

This project takes a diagnostic approach: rather than optimizing for benchmark accuracy, we ask *what types of reasoning* models actually perform. We construct four categories of controlled incorrect endings and probe each model's ability to detect violations of temporal order, causal logic, world-state consistency, and emotional tone.

**Research questions:**
1. What shallow signals exist in the Story Cloze dataset?
2. Can models distinguish gold endings from controlled LLM-generated negatives?
3. Do models that succeed on fake detection also succeed on the original Cloze task?
4. Which reasoning type is hardest?

---

## 2. Exploratory Data Analysis

### 2.1 Dataset Overview

| Dataset | Size |
|---------|------|
| ROC Stories (full) | 52,665 stories |
| ROC Stories (used, after filtering) | 4,841 stories |
| Story Cloze Validation Set | 1,571 stories |

ROC stories follow a five-sentence structure with average sentence lengths increasing across positions (S1: 7.7 words → S5: 9.1 words). The Cloze validation set is nearly balanced: ending 1 is correct 51.1% of the time.

### 2.2 Shallow Signal Analysis (Nitya's EDA)

**Experiment 1 — Lexical Overlap Bias:**  
Correct endings share slightly more words with the context (mean overlap: 0.447) vs wrong endings (0.435), but correct has higher overlap only 46% of the time. A lexical overlap classifier achieves only **51.9%** — barely above random.

**Experiment 2 — Sentiment Alignment Bias:**  
A sentiment-only classifier (VADER) achieves **59.8%** accuracy. Correct endings are emotionally closer to the context than wrong ones, inflated by easy cases with obvious emotional mismatches.

**Experiment 3 — Hard Sentiment Subset:**  
Isolating 390 stories (24.8%) where both endings have similar sentiment, the sentiment baseline drops to **51.3%** — near chance. Sentiment is a surface shortcut, not a reasoning signal.

### 2.3 Feature Comparison Table (Aurelia's EDA)

| Feature | Correct Mean | Wrong Mean | Difference |
|---------|-------------|------------|------------|
| Ending word count | 7.59 | 7.32 | +0.26 |
| Lexical overlap | 0.447 | 0.436 | +0.012 |
| Jaccard similarity | 0.102 | 0.098 | +0.005 |
| New entity count | 0.255 | 0.192 | **+0.064** |
| Has pronoun | 62.8% | 57.7% | +5.1% |
| Has temporal cue | 15.0% | 9.8% | **+5.2%** |
| Has negation | 3.6% | 9.2% | **−5.6%** |

Correct endings introduce more new entities and temporal cues — natural story progression. Wrong endings use negation far more (9.2% vs 3.6%), suggesting some are constructed as simple contradictions. Surface differences are small — not strong enough for reliable classification.

**EDA conclusion:** The Story Cloze task contains weak shallow signals. A model must go beyond word overlap and sentiment to reliably solve it.

---

## 3. Data Generation

### 3.1 Negative Generation with GPT-4o-mini

We generate controlled incorrect endings using **GPT-4o-mini** with four constrained prompts, one per reasoning type. Each prompt instructs the model to:
- Use the same characters, setting, and topic
- Write a fluent, natural, single-sentence ending
- Violate *only* the target reasoning type
- Avoid introducing new characters or locations
- Not overlap with other reasoning types

| Type | What it violates |
|------|-----------------|
| **Temporal** | Event ordering — something happens too early, too late, or in wrong sequence |
| **Causal** | Cause-effect logic — outcome contradicts what earlier events should produce |
| **State** | World-state — contradicts an established fact, possession, or condition |
| **Sentiment** | Emotional tone — reverses the emotional direction of the story |

### 3.2 Quality Filtering

Three filters applied to each generated negative: (1) length 4–25 words, (2) single sentence, (3) no repetition of gold ending.

| Stage | ROC Stories | Cloze Val |
|-------|------------|-----------|
| Raw generated rows | 20,240 | 6,284 |
| Failed length filter | 211 | — |
| Failed sentence filter | 24 | — |
| **Final clean rows** | **19,364** | **6,284** |
| **Stories retained** | **4,841** | **1,571** |

Zero generation errors recorded across all batches.

---

## 4. Models

### 4.1 Baseline: TF-IDF Cosine Similarity

TF-IDF vectors represent context and each ending as weighted word-frequency vectors. The ending with higher cosine similarity to the context is selected. Vectorizer fit only on training text. Fully unsupervised — no labels used.

### 4.2 Sentence-BERT (SBERT)

`all-MiniLM-L6-v2` encodes context and endings as dense semantic vectors. The ending with higher cosine similarity is selected. No fine-tuning. Captures semantic meaning beyond word overlap — still unsupervised.

### 4.3 Pairwise DistilBERT

Fine-tunes `distilbert-base-uncased` on story ending pairs. Input: `[context] + [SEP] + [ending_a]` paired with `[ending_b]`. The model sees both endings simultaneously and predicts which is correct. Trained with both orderings for augmentation. **Config:** 3 epochs, batch size 16, lr 2e-5, dynamic padding.

### 4.4 Pairwise RoBERTa

Same pairwise architecture using `roberta-base`. RoBERTa was pretrained with dynamic masking and more data — stronger at sentence-pair tasks.

### 4.5 Data Splits

| Split | Source | Size | Purpose |
|-------|--------|------|---------|
| Train (80%) | ROC + GPT negatives | 3,872 stories / 15,488 pairs | Train Models 5 & 6 |
| Monitor (20%) | ROC + GPT negatives | 969 stories / 3,876 pairs | Early stopping |
| Val B | Cloze Val (human endings) | 1,571 stories | 2-way accuracy |
| Val C | Cloze Val + GPT negatives | 1,571 × 4 types | Per-type fake detection |

---

## 5. Results

### 5.1 Val B: Original Cloze 2-way Accuracy

![Val B bar chart showing accuracy per model](val_b_bar.png)

| Model | Val B Accuracy | p-value | Significant? |
|-------|---------------|---------|-------------|
| TF-IDF Cosine (Baseline) | 51.7% | 0.095 | No |
| SBERT | **59.4%** | < 0.001 | **Yes** |
| DistilBERT | 50.6% | 0.325 | No |
| RoBERTa | 57.0% | < 0.001 | **Yes** |

*Significance: one-sided binomial test, H₀: acc = 0.5, n = 1,571.*

SBERT achieves the highest Val B accuracy (59.4%). RoBERTa is the only fine-tuned model to significantly beat chance. DistilBERT performs near-chance despite near-perfect training performance.

### 5.2 Val C: Per-type Fake Detection Accuracy

| Model | Temporal | Causal | State | Sentiment | Overall |
|-------|----------|--------|-------|-----------|---------|
| TF-IDF Cosine | 31.0% | 48.0% | 47.4% | 55.3% | 45.4% |
| SBERT | 23.8% | 41.1% | 44.3% | 54.2% | 40.9% |
| DistilBERT | 99.3% | 99.8% | 98.6% | 98.8% | 99.1% |
| RoBERTa | **99.7%** | **99.9%** | **99.5%** | **99.7%** | **99.7%** |

### 5.3 Training Curves (Fine-tuned Models)

**DistilBERT:**

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| 1 | 0.1230 | 0.0329 | 99.15% |
| 2 | 0.0267 | 0.0244 | 99.45% |
| 3 | 0.0109 | 0.0289 | 99.41% |

**RoBERTa:**

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| 1 | 0.0768 | 0.0374 | 99.33% |
| 2 | 0.0156 | 0.0300 | 99.47% |
| 3 | 0.0039 | 0.0332 | 99.55% |

Both models converge rapidly to near-perfect monitor accuracy, consistent with learning an easy signal (GPT style) rather than complex reasoning.

---

## 6. Analysis

### 6.1 The Val B vs Val C Gap — Central Finding

![Val B vs Val C scatter plot showing the gap per model](val_b_vs_val_c.png)

| Model | Val B | Val C | Gap (C − B) |
|-------|-------|-------|-------------|
| TF-IDF Cosine | 51.7% | 45.4% | −6.2% |
| SBERT | 59.4% | 40.9% | −18.5% |
| DistilBERT | 50.6% | 99.1% | **+48.5%** |
| RoBERTa | 57.0% | 99.7% | **+42.7%** |

Fine-tuned models score 99%+ on Val C but only 50–57% on Val B. This reveals that **DistilBERT and RoBERTa learned to detect GPT-4o-mini's writing style** rather than narrative reasoning — since both training and Val C negatives come from the same generator. When tested on human-written alternatives, performance collapses.

RoBERTa's smaller gap (42.7% vs 48.5%) suggests its stronger pretrained representations encode some transferable reasoning signal beyond style detection.

### 6.2 Per-type Difficulty

Across unsupervised models (TF-IDF and SBERT), temporal is consistently the hardest and sentiment the easiest:

| Model | Temporal Error | Causal Error | State Error | Sentiment Error |
|-------|---------------|-------------|------------|----------------|
| TF-IDF Cosine | **69.0%** | 52.0% | 52.6% | 44.7% |
| SBERT | **76.2%** | 58.9% | 55.7% | 45.8% |

Temporal relationships require understanding event ordering — something lexical features and semantic embeddings fail to capture. Sentiment is easier because emotional tone is reflected directly in word choice.

### 6.3 Qualitative Error Analysis

Errors from the TF-IDF baseline highlight where GPT-generated negatives fool lexical matching:

**Temporal** — subtle event reordering:
> *Context:* "Laverne follows a brownie recipe closely. She tests one brownie."  
> *Gold:* "The brownies are so delicious Laverne eats two of them."  
> *Negative:* "Laverne tests one brownie **after she has already eaten all of them**."

**Causal** — implausible but lexically plausible outcome:
> *Context:* "John and Billy won a beer pong contest. The next level sent them to Vegas."  
> *Gold:* "In Vegas, they competed against eighty contestants."  
> *Negative:* "In Vegas, they **decided to quit beer pong and take up professional knitting**."

**State** — contradicts established world state:
> *Context:* "Ron finishes landscaping the mayor's yard early and is ecstatic."  
> *Gold:* "His boss commends him for a job well done."  
> *Negative:* "His boss **is unhappy and fires Ron on the spot**."

---

## 7. Discussion

Our results surface a fundamental limitation: when training and Val C negatives share the same generator (GPT-4o-mini), fine-tuned models can achieve excellent scores by learning stylistic fingerprints — sentence length, vocabulary patterns, hedging language — rather than reasoning. The Val B vs Val C gap is a direct diagnostic of this.

RoBERTa partially breaks the pattern — 57.0% on Val B, statistically significant — but the 42.7% gap indicates style exploitation remains substantial. The most honest signal comes from SBERT: 59.4% on Val B without any training, confirming that semantic similarity is a competitive baseline for the original task.

---

## 8. Conclusion

We present a diagnostic framework for evaluating narrative reasoning using four categories of controlled negatives. Fine-tuned models achieve near-perfect accuracy on LLM-generated negatives but near-chance on human-authored alternatives, revealing they learn generator style rather than reasoning. SBERT and RoBERTa are the only models to significantly exceed chance on the real Story Cloze task. Temporal reasoning is consistently the hardest category. Future work should explore human-authored negatives and cross-generator evaluation to close the style exploitation gap.

---

## Appendix A: TF-IDF Models

### A.1 Model 1 — Lexical Overlap

Counts shared words between context and ending and picks the higher-overlap ending. No TF-IDF weighting — purely raw word count.

**Results:**
- Val B: 51.7% (p = 0.095, not significant)
- Val C: Temporal 6.6%, Causal 9.5%, State 11.3%, Sentiment 13.4%, Overall 10.2%

Val C scores are well below chance — the model systematically picks the LLM-generated negative because it reuses more topic words from the context than the gold ending does.

### A.2 Model 2 — TF-IDF Cosine Similarity

Represents context and endings as TF-IDF vectors. Cosine similarity picks the most relevant ending. Vectorizer fit only on training text.

**Results:**
- Val B: 51.7% (p = 0.095, not significant)
- Val C: Temporal 31.0%, Causal 48.0%, State 47.4%, Sentiment 55.3%, Overall 45.4%
- Val C significance: Only sentiment is significant (p < 0.001)

### A.3 Model 3 — TF-IDF + Logistic Regression

Trains a binary classifier on (context + [SEP] + ending) TF-IDF features.

**Results:**
- Train accuracy: 90.6% (30,976 pairs)
- Monitor accuracy: 80.4% (7,752 pairs)
- Val B: 59.3% (significant, p < 0.001)
- Val C: Temporal 98.3%, Causal 97.7%, State 96.1%, Sentiment 97.3%, Overall 97.4%

Model 3 shows the same Val B vs Val C gap as the transformer models — confirming even logistic regression can exploit GPT writing style.

---

## Appendix B: Full Results Table

| Model | Val B | C-Temp | C-Causal | C-State | C-Sent | C-Overall |
|-------|-------|--------|---------|--------|-------|----------|
| Lexical Overlap | 51.7% | 6.6% | 9.5% | 11.3% | 13.4% | 10.2% |
| TF-IDF Cosine | 51.7% | 31.0% | 48.0% | 47.4% | 55.3% | 45.4% |
| TF-IDF + LR | 59.3% | 98.3% | 97.7% | 96.1% | 97.3% | 97.4% |
| SBERT | 59.4% | 23.8% | 41.1% | 44.3% | 54.2% | 40.9% |
| DistilBERT | 50.6% | 99.3% | 99.8% | 98.6% | 98.8% | 99.1% |
| RoBERTa | 57.0% | 99.7% | 99.9% | 99.5% | 99.7% | 99.7% |
