Here is your exact text, properly formatted for a GitHub README without changing the wording:

---

# MachineLearningProject

# How to get started for linux or wsl

1. setup a venv (optional)

```bash
python -m venv .venv
```

load venv (needed on every start)

```bash
source .venv/bin/activate
```

2. install dependencies

install kagglehub

```bash
pip install kagglehub
```

install pytorch (check [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

install scikit-learn

```bash
pip install scikit-learn
```

3. download dataset into local folder

run:

```bash
python downloadDataset.py
```

4. Run CNN sample

run:

```bash
python train-cnn.py
```
