Here is your exact text, properly formatted for a GitHub README without changing the wording:

---

# MachineLearningProject

# How to get started for linux or wsl - Setup

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

install matplotlib

```bash
pip install matplotlib
```

3. download dataset into local folder

run:

```bash
python downloadDataset.py
```

# How to run

1. Load venv
run

```bash
source .venv/bin/activate
```

2. Run scenarios

run:

```bash
python scenarios.py
```
This will run all scenarios and will generate .pth for every model for each scenario as well as a txt file for every scenerio which contains the runtime data.

To generate the performance metrics for every model in every scenario run

Minor adjustments may be needed as the path for folder containing the .pth files is hardcoded

run:

```bash
python testTrainedModels.py
```

The following loads and test a single model
See the file itself for more detailed instructions

run:

```bash
python loadAndTestModel.py [MODELNAME] [.pth FILE]
```
