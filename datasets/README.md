# Downloaded Datasets

This directory contains datasets for the research project. Data files are not
committed to git due to size. Follow the instructions below to reproduce locally.

## Dataset 1: CommonGen

### Overview
- **Source**: Hugging Face `common_gen`
- **Size**: train 67,389 / validation 4,018 / test 1,497
- **Format**: HuggingFace Dataset (Arrow on disk)
- **Task**: constrained text generation (creativity/novelty-sensitive)
- **Splits**: train/validation/test
- **License**: Check dataset card on Hugging Face

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("common_gen")
dataset.save_to_disk("datasets/common_gen")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/common_gen")
print(dataset["train"][0])
```

### Sample Data
Saved sample file:
- `datasets/common_gen/samples/train_10.json`

### Notes
- Useful for testing whether model outputs are merely fluent vs genuinely novel.
- Includes concept constraints, enabling controlled generation evaluation.

## Dataset 2: MovieLens Latest Small

### Overview
- **Source**: GroupLens (`https://files.grouplens.org/datasets/movielens/`)
- **Size**: 100,836 ratings; 9,742 movies
- **Format**: CSV inside ZIP
- **Task**: recommendation/ranking; supports novelty/serendipity analysis
- **Splits**: no fixed splits (user-defined)
- **License**: GroupLens usage terms apply

### Download Instructions

**Direct download:**
```bash
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -O datasets/ml-latest-small.zip
unzip datasets/ml-latest-small.zip -d datasets/movielens_small/
```

### Loading the Dataset
```python
import pandas as pd

ratings = pd.read_csv("datasets/movielens_small/ml-latest-small/ratings.csv")
movies = pd.read_csv("datasets/movielens_small/ml-latest-small/movies.csv")
```

### Sample Data
Saved sample files:
- `datasets/movielens_small/samples/ratings_head20.csv`
- `datasets/movielens_small/samples/movies_head20.csv`

### Notes
- Supports constructing an "underrated" target by combining average rating with popularity (#ratings).
- Suitable for novelty@k / serendipity-style recommendation evaluations.

## Validation Snapshot
A quick validation summary is stored in:
- `datasets/validation_summary.json`
