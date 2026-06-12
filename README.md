# NLP-articles-summarization
### Text Summarization: Extractive and Abstractive Approaches

A structured NLP pipeline that implements and compares two text summarization techniques — **TF-IDF extractive summarization** and **BART-based abstractive summarization** — evaluated using ROUGE metrics.

---

### 🔍 Overview

This project builds an end-to-end text summarization system that explores both classical and deep learning approaches to automatically condense long-form text into concise summaries. The pipeline covers the full workflow: data loading, preprocessing, model inference, and quantitative evaluation.

---

### 📂 Project Structure

The notebook is organized into 5 parts:

1. **Setup & Preprocessing** — Installs dependencies, loads up to 50,000 records from a CSV dataset, performs EDA (length distributions, readability scores, novelty ratios), and applies text cleaning (lowercasing, URL removal, emoji stripping, contraction expansion). An 80/20 train-test split is applied.

2. **TF-IDF Extractive Summarization** — Scores sentences using TF-IDF weights and selects the top-N most informative sentences, preserving original document order for coherence.

3. **BART Abstractive Summarization** — Uses the `facebook/bart-large-cnn` transformer model (fine-tuned on news summarization) to generate novel, fluent summaries from unmodified input text via beam search decoding.

4. **Pipeline Integration** — Wraps both approaches in a unified `summarize_text(text, method)` interface, allowing seamless switching between `'tfidf'` and `'bart'` modes.

5. **ROUGE Evaluation** — Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores on a 100-sample evaluation subset, with a side-by-side bar chart comparing both models.

---

### 🛠️ Technologies & Libraries

- **Python**, **Jupyter Notebook**
- `transformers` (Hugging Face) — BART model
- `torch` — PyTorch (CPU build)
- `scikit-learn` — TF-IDF vectorization
- `nltk` — Sentence tokenization & stopwords
- `rouge-score` — ROUGE evaluation
- `textstat` — Readability scoring
- `pandas`, `numpy`, `matplotlib`, `seaborn`

---

### 🚀 Getting Started

1. Run **cell 1.0** to install dependencies, then **restart the kernel**.
2. Run all remaining cells in order.
3. Place your dataset as `data.csv` in the working directory with `Content` and `Summary` columns.

---

### 📊 Evaluation

Both models are evaluated on the same 100-sample test subset using ROUGE F1 scores. Results are displayed in a formatted table and visualized as a grouped bar chart for direct comparison.
