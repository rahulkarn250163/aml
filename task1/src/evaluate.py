"""
Task 1 — §G/§H/§I: GP classification (via thresholded regression), SVM baseline, and
the comparative evaluation (metrics + calibration).

GP classification is implemented the assignment-mandated way: fit a GaussianProcess
*Regressor* on the float 0/1 target, predict a continuous mean (with std = uncertainty),
then THRESHOLD the mean to obtain class labels.  This is NOT GaussianProcessClassifier.
"""
from __future__ import annotations

import numpy as np

RANDOM_STATE = 42


# ----------------------------------------------------------------------------- sampling / scaling

def stratified_sample(X, y, n, random_state=RANDOM_STATE):
    """Stratified down-sample to ~n rows (GP is O(n^3); a 70K GP is infeasible)."""
    from sklearn.model_selection import train_test_split

    if len(y) <= n:
        return X, y
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=random_state
    )
    return X_s.reset_index(drop=True), y_s.reset_index(drop=True)


def fit_scaler(X_train):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(X_train)


# ----------------------------------------------------------------------------- Gaussian Process

def make_kernels():
    """RBF and Matern kernels, each with ConstantKernel + WhiteKernel (learned noise)."""
    from sklearn.gaussian_process.kernels import (
        RBF, ConstantKernel, Matern, WhiteKernel,
    )

    rbf = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    matern = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
    return {"RBF": rbf, "Matern": matern}


def fit_gp(X_train, y_train, kernel, n_restarts_optimizer=0, maxiter=50):
    """Fit a GP regressor with a capped L-BFGS optimiser.

    scikit-learn's default optimiser allows up to 15000 L-BFGS iterations; each iteration
    Cholesky-factorises the n x n kernel and forms an n x n x n_params gradient, so on a few
    thousand points the marginal-likelihood optimisation can run for many minutes. Capping at
    `maxiter` bounds GP fit time predictably with negligible effect on the fitted kernel.
    """
    from scipy.optimize import minimize
    from sklearn.gaussian_process import GaussianProcessRegressor

    def capped_optimizer(obj_func, initial_theta, bounds):
        res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True,
                       bounds=bounds, options={"maxiter": maxiter})
        return res.x, res.fun

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        optimizer=capped_optimizer,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=RANDOM_STATE,
    )
    gp.fit(X_train, y_train.astype(float))
    return gp


def gp_predict_proba(gp, X):
    """Return (prob, std): continuous mean clipped to [0,1] as P(relief), and uncertainty."""
    mean, std = gp.predict(X, return_std=True)
    prob = np.clip(mean, 0.0, 1.0)
    return prob, std


def tune_threshold(y_true, prob, metric="f1"):
    """Pick the threshold on the validation scores that maximises macro-F1."""
    from sklearn.metrics import f1_score

    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (prob >= t).astype(int), average="macro", zero_division=0)
              for t in grid]
    best = grid[int(np.argmax(scores))]
    return float(best), float(np.max(scores))


# ----------------------------------------------------------------------------- SVM baseline

def fit_svm(X_train, y_train, cv=3):
    """Tune C/gamma on the same split, then refit with probability=True."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced"),
        param_grid={"C": [1.0, 10.0], "gamma": ["scale", 0.1]},
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best = SVC(kernel="rbf", class_weight="balanced", probability=True,
               random_state=RANDOM_STATE, **grid.best_params_)
    best.fit(X_train, y_train)
    return best, grid.best_params_


# ----------------------------------------------------------------------------- metrics / calibration

def expected_calibration_error(y_true, prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_metrics(y_true, prob, y_pred):
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
    )

    y_true = np.asarray(y_true)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),  # better than ROC-AUC under imbalance
        "brier": brier_score_loss(y_true, prob),
        "ece": expected_calibration_error(y_true, prob),
    }


def risk_coverage(y_true, prob, std, threshold, n_points=20):
    """Selective prediction: keep only the most-confident (lowest-std) cases.

    Returns (coverage, accuracy, macro_f1) as coverage decreases from 1.0. If the GP's
    uncertainty is meaningful, accuracy/F1 should rise as we abstain on high-std cases.
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true)
    order = np.argsort(std)                 # most confident first
    yt, pr = y_true[order], np.asarray(prob)[order]
    pred = (pr >= threshold).astype(int)
    covs = np.linspace(1.0, 0.1, n_points)
    accs, f1s = [], []
    for c in covs:
        m = max(20, int(c * len(yt)))
        accs.append(accuracy_score(yt[:m], pred[:m]))
        f1s.append(f1_score(yt[:m], pred[:m], average="macro", zero_division=0))
    return covs, np.array(accs), np.array(f1s)


def cross_validate_eval(X, y, kernel_name="Matern", n_splits=5):
    """Stratified k-fold CV for the GP (thresholded) and SVM; returns per-fold metrics.

    A fresh kernel and scaler are fitted per fold (scaler on the training fold only).
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    recs = []
    for i, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        scaler = fit_scaler(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        gp = fit_gp(X_tr_s, y_tr, make_kernels()[kernel_name])
        prob_tr, _ = gp_predict_proba(gp, X_tr_s)
        prob_te, _ = gp_predict_proba(gp, X_te_s)
        thr, _ = tune_threshold(y_tr.values, prob_tr)
        m = compute_metrics(y_te, prob_te, (prob_te >= thr).astype(int))
        m.update(model=f"GP-{kernel_name}", fold=i)
        recs.append(m)

        svm, _ = fit_svm(X_tr_s, y_tr)
        sp = svm.predict_proba(X_te_s)[:, 1]
        m = compute_metrics(y_te, sp, svm.predict(X_te_s))
        m.update(model="SVM", fold=i)
        recs.append(m)
        print(f"    fold {i+1}/{n_splits} done", flush=True)
    return pd.DataFrame(recs)
