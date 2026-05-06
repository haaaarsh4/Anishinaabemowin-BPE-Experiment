# Anishinaabemowin BPE Experiment

This project tests how well BPE tokenization handles Anishinaabemowin (Ojibwe), an Indigenous polysynthetic language, by comparing BPE's output against real morpheme boundaries from the Ojibwe People's Dictionary.

## Files

| File | What it does |
|---|---|
| `BPE.py` | BPE tokenizer |
| `experiment.py` | Runs the experiment and saves results to CSV |
| `observations.py` | Analyses results and saves a summary CSV |
| `VAIO_CNJ.csv` | Ojibwe verb forms with gold standard morpheme splits |

## How to Run

```bash
python3 experiment.py
python3 observations.py
```

Change `VOCAB_SIZE` in `experiment.py` to 50, 100, or 200 to test different settings.

## Data

Data is from [OjibweMorph](https://github.com/ELF-Lab/OjibweMorph) by the ELF Lab and the Ojibwe People's Dictionary, licensed under CC BY-NC-SA 4.0.
