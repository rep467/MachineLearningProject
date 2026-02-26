# MachineLearningProject


# How to get started for linux or wsl
1. setup a venv (optional)
   python -m venv .venv
load venv (needed on every start)
   source .venv/bin/activate
2. install dependencies
install kagglehub
   pip install kagglehub
install pytorch (check **https://pytorch.org/get-started/locally/**)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

3. download dataset into local folder
run:
   python downloadDataset.py
