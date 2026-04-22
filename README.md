# Comparison of RNN, LSTM, & Transformers for Music Generation

**Zeynep Lal Celkbilek, Sam Castelein, Milo Margolis**

## Overview

Music can be represented as sequences of notes, rests, and characters. This symbolic representation lends itself naturally to language modeling, where models learn to predict the next token in a sequence. This project implements and compares three architectures — Vanilla RNN, LSTM, and Transformer — for symbolic music generation using ABC notation (and/or MIDI) datasets.

We evaluate the models using both quantitative metrics (perplexity, pitch distribution similarity, step/duration statistics) and qualitative listening tests.

## Project Structure

```
├── src/
│   ├── dataset.py           # Data loading and preprocessing
│   ├── train.py             # Training loop
│   ├── generate.py          # Inference / music generation
│   └── models/
│       ├── rnn.py           # Vanilla RNN
│       ├── lstm.py          # LSTM
│       └── transformer.py   # Transformer decoder
├── requirements.txt
├── run.sh
└── README.md
```

## Datasets

We use datasets containing symbolic music in ABC notation or MIDI format.

**ABC notation:**
- [ABC Notation Music for RNN](https://www.kaggle.com/) (Kaggle)
- [ABC Notation for Irish Folk Tunes](https://huggingface.co/) (Hugging Face)
- [ABC Notation Examples](https://zenodo.org/) (Zenodo)

**MIDI (plan B):**
- Classical Music MIDI
- Video Game Music MIDI

ABC notation example:
```
X:1
T:The Butterfly
M:9/8
L:1/8
K:Emin
|:B2E G2E F3|B2E G2E FED|B2E G2E F3|B3 B3 B3:|
```

## Models

| Model | Role |
|---|---|
| Vanilla RNN | Baseline — simple but limited on long-range dependencies |
| LSTM | Captures longer musical structure via gating mechanisms |
| Transformer | Captures long-range patterns via self-attention with causal masking |

### What we implement ourselves
- **Data pipeline** — custom `torch.utils.data.Dataset` for tokenization and vocabulary building
- **Architecture logic** — LSTM recurrence and Transformer causal masking
- **Training & decoding** — full training loop and inference engine
- **Evaluation suite** — perplexity, pitch distribution comparison, step/duration statistics

## Evaluation

**Quantitative:** training/validation loss curves, pitch distribution similarity vs. training data, step/duration statistics.

**Qualitative:** listening tests among team members for overall quality and musicality.

**Visualizations:** loss curves (including component-level pitch/step/duration loss), distribution plots (generated vs. training data), and a summary comparison table across all metrics.

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

Train a single model:
```bash
python3 src/train.py lstm
```

Train multiple models:
```bash
python3 src/train.py lstm rnn transformer
```

Outputs are saved to `outputs/` with the model name as a prefix (e.g. `lstm_model.pt`, `lstm_loss_curves.png`, `lstm_generated.mid`).

## Team & Timeline

| Week | Focus |
|---|---|
| 1 | Data preprocessing, note extraction pipeline, Vanilla RNN baseline |
| 2 | LSTM implementation and training |
| 3 | Transformer implementation and training |
| 4 | Evaluation, comparison, and write-up |

| Member | Responsibilities |
|---|---|
| **Sam** | Data preprocessing pipeline (tokenization, vocab, Dataset class), ABC/MIDI ingestion, Vanilla RNN |
| **Lal** | LSTM model, custom recurrence logic, hyperparameter tuning, audio rendering (FluidSynth/music21) |
| **Milo** | Transformer decoder with causal masking, inference/decoding, evaluation suite |
| **All** | Visualizations, comparison table, listening tests, write-up |

## References

"Generating Music Using ABC Notation." *GeeksforGeeks*, 23 July 2025, www.geeksforgeeks.org/nlp/generating-music-using-abc-notation/. Accessed 22 Apr. 2026.

MITDeepLearning. "PT_Part2_Music_Generation.ipynb." *GitHub*, github.com/MITDeepLearning/introtodeeplearning/blob/master/lab1/PT_Part2_Music_Generation.ipynb. Accessed 22 Apr. 2026.

We used LLMs to format this README.md (the table and diagram)