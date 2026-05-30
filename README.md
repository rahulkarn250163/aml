# Advanced Machine Learning (STW7085CEM)

Coursework for the Advanced Machine Learning module, by Rahul Karn (250163) and Rashant Nayaju (250117).
The repository is split into two self-contained tasks.

```
task1/   - Topic discovery and escalation-risk prediction (Python)
task2/   - Fuzzy logic control, genetic optimisation and CEC'2005 comparison (MATLAB)
```

The compiled reports are kept local and are not committed; the repository holds the dataset, code,
figures and result tables for both tasks.

---

## Task 1 - LDA topic discovery and GP escalation-risk prediction (Python)

Topic discovery and escalation-risk prediction on consumer-finance complaints, using Latent Dirichlet
Allocation (LDA) for unsupervised topics (compared against LSA) and a Gaussian Process (GP) classifier,
implemented by thresholding a GP regression output, for escalation risk, with an SVM baseline. The
headline result is a calibration comparison: the GP yields better-calibrated probabilities and usable
per-case uncertainty.

```
task1/notebooks/task1_pipeline.ipynb  - end-to-end notebook (sections A-K)
task1/src/                            - preprocess, topics, features, evaluate modules
task1/data/raw/                       - the complaint dataset (CSV)
task1/results/figures/                - generated figures (EDA, LDA selection, ROC, calibration, ...)
task1/results/tables/                 - metrics and model-selection tables (CSV)
task1/requirements.txt                - Python dependencies (pinned, Python 3.10)
```

Reproduce:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r task1/requirements.txt
python -m ipykernel install --user --name task1-aml
jupyter nbconvert --to notebook --execute task1/notebooks/task1_pipeline.ipynb
```

---

## Task 2 - Fuzzy logic controller, GA optimisation and CEC'2005 comparison (MATLAB)

A Mamdani fuzzy logic controller for thermal comfort in an assistive-care room, its membership functions
optimised by a hand-coded genetic algorithm, and a comparison of three optimisers (GA, PSO, SA) on two
CEC'2005 benchmark functions. Built in MATLAB R2026a (Fuzzy Logic Toolbox + Global Optimization Toolbox).

```
task2/flc/        - Mamdani FLC: build_flc.m -> room_flc.fis, evidence figures
task2/gaflc/      - GA optimisation of the FLC membership functions (dataset, encoding, fitness)
task2/cec2005/    - CEC'2005 functions and hand-coded GA/PSO/SA + toolbox cross-check
task2/figures/    - generated figures for the report
task2/realss/     - MATLAB Fuzzy Logic Designer screenshots (Part 1 evidence)
task2/results/    - result tables (CSV/TXT)
task2/report/     - report source (Task2_Report.tex)
task2/run_all.m   - reproduces every figure and table end to end
```

Reproduce (from the `task2/` folder, in MATLAB):

```matlab
run_all
```

Key results: the GA cuts the controller fitting error by 81 percent (MSE 300.9 -> 57.2) and beats the
built-in `ga` solver on the same problem; on the CEC'2005 benchmarks the GA is the most robust optimiser
on the multimodal Rastrigin function, PSO is strongest on the smooth Sphere, and SA is weakest at high
dimension. A Global Optimization Toolbox cross-check confirms the pattern.

---

## Authors

Group of two: Rahul Karn (250163) and Rashant Nayaju (250117). Per-task author contributions are recorded
in each task's report.
