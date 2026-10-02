"""
common.py
=========

Shared building blocks for the three experiments in this repository:

* data preparation (Mackey-Glass generation, train/test splitting, sliding windows)
* model builders (Echo State Network, plus ridge / SVR / LSTM baselines)
* the hyperparameter-search objective used with ReservoirPy's ``research`` helper
* plotting helpers that save every figure to a results folder

Background (see README for the full explanation)
------------------------------------------------
An Echo State Network (ESN) is a type of reservoir computer:

    input u(t) --W_in--> [ reservoir of N recurrently-connected neurons ] --W_out--> output y(t)

* ``W_in`` (input weights) and ``W`` (reservoir weights) are generated randomly
  and then FIXED - they are never trained.
* Only the readout ``W_out`` is learned, using ridge regression
  (a single closed-form linear solve, no back-propagation).

The reservoir state is updated as

    x(t+1) = (1 - lr) * x(t) + lr * tanh(W_in u(t+1) + W x(t))

where ``lr`` is the leaking rate. The spectral radius (largest absolute
eigenvalue of ``W``) controls how long past inputs "echo" around the reservoir.

Attribution
-----------
Several helpers in this file are adapted from the official ReservoirPy
tutorials (https://github.com/reservoirpy/reservoirpy, MIT License,
Copyright (c) 2018 neuronalX). Each adapted function is marked with
``Adapted from ReservoirPy ...`` in its docstring. Everything else
(baselines, sliding-window pipeline, comparison plots) is my own work.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
from reservoirpy.datasets import mackey_glass, to_forecasting
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
from sklearn.linear_model import Ridge as RidgeSklearn
from sklearn.svm import SVR

# Folder where every script writes its figures, e.g. results/mackey_glass_multistep/
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Fraction of the series used for training (the rest is the test set).
TRAIN_FRACTION = 0.8


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def results_folder(name):
    """Create (if needed) and return ``results/<name>/``."""
    folder = RESULTS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_figure(folder, filename):
    """Save the current matplotlib figure as PNG and close it to free memory."""
    plt.savefig(Path(folder) / f"{filename}.png", bbox_inches="tight")
    plt.close("all")


def rescale(series):
    """Min-max rescale a series to the range [-1, 1].

    Reservoir neurons use a tanh activation, whose useful range is about
    [-1, 1], so inputs are normalised to that range before training.
    """
    return 2 * (series - series.min()) / (series.max() - series.min()) - 1


def analyse(y_true, y_pred, label=""):
    """Print the R^2 score and NRMSE of a prediction.

    NRMSE = RMSE / (max(y_true) - min(y_true)). 0 means a perfect prediction.
    """
    if label:
        print(label)
    print(f"  R-squared: {rsquare(y_true, y_pred):.4f}")
    print(f"  NRMSE:     {nrmse(y_true, y_pred):.4f}\n")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_mackey_glass(timesteps=2510, tau=20):
    """Generate the Mackey-Glass chaotic time series, rescaled to [-1, 1].

    tau = 20 (> 16.8) puts the system in its chaotic regime.
    2510 samples with an 80/20 split gives 2008 training and 502 test points.

    Adapted from ReservoirPy Tutorial 3 ("General Introduction to Reservoir
    Computing"), which uses the same generator and rescaling with tau = 17.
    """
    series = mackey_glass(timesteps, tau=tau)
    return rescale(series)


def split_multistep(series, forecast, train_fraction=TRAIN_FRACTION):
    """Build input/target pairs for a k-step-ahead forecasting task.

    ``to_forecasting`` shifts the series so that the target is the input
    ``forecast`` steps in the future: y(t) = u(t + forecast).
    The split index is computed on the ORIGINAL series length (as in the
    report), so the test set has ``len(series) - forecast - split_index`` points.

    Returns X_train, y_train, X_test, y_test, split_index.
    """
    x, y = to_forecasting(series, forecast=forecast)
    split_index = int(train_fraction * len(series))
    return x[:split_index], y[:split_index], x[split_index:], y[split_index:], split_index


def split_sliding_window(series, train_fraction=TRAIN_FRACTION):
    """Prepare data for the sliding-window (one-step-ahead) experiments.

    Each input row is the previous ``window_size`` values of the series and
    the target is the very next value. The window size is set equal to the
    length of the test period (e.g. 502 for Mackey-Glass, 252 for Amazon).

    Note: at test time each window contains the TRUE past values (the model's
    own predictions are not fed back in), so every test point is a
    one-step-ahead prediction. See "Limitations" in the README.

    Returns X_train, y_train, X_test, y_test, train_data, split_index, window_size.
    """
    split_index = int(train_fraction * len(series))
    window_size = len(series) - split_index
    train_data = series[:split_index]
    # Test windows start ``window_size`` steps before the split, so the first
    # test window is filled with the last training values.
    test_data = series[split_index - window_size:]

    X_train, y_train = [], []
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i - window_size:i, 0])
        y_train.append(train_data[i, 0])
    X_train = np.array(X_train).reshape(-1, window_size)
    y_train = np.array(y_train).reshape(-1, 1)

    X_test = []
    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i - window_size:i, 0])
    X_test = np.array(X_test).reshape(-1, window_size)
    y_test = np.array(series[split_index:])

    return X_train, y_train, X_test, y_test, train_data, split_index, window_size


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Default ("non-optimised") ESN hyperparameters.
# Starting values taken from ReservoirPy Tutorial 3; units changed from 100 to 200.
DEFAULT_ESN = dict(
    units=200,               # number of reservoir neurons
    lr=0.3,                  # leaking rate: how quickly the state forgets (1 = no memory of x(t))
    sr=1.25,                 # spectral radius of W: >1 means longer, less stable echoes
    input_scaling=1.0,       # gain applied to the inputs
    rc_connectivity=0.1,     # fraction of non-zero connections inside the reservoir
    input_connectivity=0.2,  # fraction of non-zero input -> neuron connections
    seed=1234,               # makes the random reservoir reproducible
)
DEFAULT_RIDGE = 1e-8         # ridge (L2) regularisation for the readout


def build_esn(units, lr, sr, input_scaling, rc_connectivity, input_connectivity,
              seed=1234, input_dim=None, ridge=DEFAULT_RIDGE):
    """Create an Echo State Network: a fixed random reservoir + a trained ridge readout.

    ``reservoir >> readout`` chains the two nodes, so ``esn.fit(X, y)`` runs
    the inputs through the reservoir and fits only the readout weights W_out:

        W_out = Y X^T (X X^T + ridge * I)^-1

    ``input_dim`` is 1 for the multi-step task (one value per time step) and
    ``window_size`` for the sliding-window task (one window per time step).

    Adapted from ``reset_esn()`` in ReservoirPy Tutorial 3.
    """
    reservoir = Reservoir(
        units,
        input_scaling=input_scaling,
        sr=sr,
        lr=lr,
        rc_connectivity=rc_connectivity,
        input_connectivity=input_connectivity,
        seed=seed,
        input_dim=input_dim,
    )
    readout = Ridge(1, ridge=ridge)  # 1 output dimension
    return reservoir >> readout


def ridge_baseline(X_train, y_train, X_test):
    """Baseline WITHOUT a reservoir: ridge regression straight on the raw inputs.

    Comparing against this shows how much the reservoir itself adds.
    """
    model = RidgeSklearn(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict(X_test).reshape(-1, 1)  # column vector, same shape as y_test


def svr_baseline(X_train, y_train, X_test):
    """Support Vector Regression baseline (RBF kernel).

    C, gamma and epsilon were tuned by hand (see report appendix A).
    """
    model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train.ravel())
    return model.predict(X_test).reshape(-1, 1)


def lstm_baseline(X_train, y_train, X_test, layers=(128, 64)):
    """LSTM baseline trained with back-propagation through time (Adam, MSE loss).

    Keras expects inputs shaped (samples, timesteps, features), so each
    scalar input is reshaped to (1, 1). 1 epoch with batch size 1 gave the
    best balance between accuracy and over-fitting in the original tests.

    Returns None if TensorFlow is not installed (the LSTM is optional
    because TensorFlow is a large dependency).
    """
    try:
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential
    except ImportError:
        print("TensorFlow not installed - skipping LSTM baseline.")
        return None

    X_train = np.asarray(X_train).reshape(-1, 1, 1)
    X_test = np.asarray(X_test).reshape(-1, 1, 1)

    model = Sequential()
    for i, n_units in enumerate(layers):
        last = i == len(layers) - 1
        model.add(LSTM(n_units, return_sequences=not last, activation="relu",
                       input_shape=(1, 1)))
    model.add(Dense(25))
    if len(layers) > 1:
        model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0)
    return model.predict(X_test, verbose=0)


# ---------------------------------------------------------------------------
# Hyperparameter optimisation
# ---------------------------------------------------------------------------
def objective(dataset, config, *, iss, N, sr, lr, icntvt, rcntvt, idim, ridge, seed):
    """Loss function minimised by hyperopt (via ``reservoirpy.hyper.research``).

    For one set of hyperparameters, build ``instances_per_trial`` ESNs with
    different random seeds, train and evaluate each, and return the mean NRMSE.
    Averaging over seeds stops a single lucky reservoir from winning.

    Parameter names (short names are what appear in the hyperopt reports):
        iss    - input scaling          N      - number of neurons
        sr     - spectral radius        lr     - leaking rate
        icntvt - input connectivity     rcntvt - recurrent connectivity
        idim   - input dimension        ridge  - readout regularisation
        seed   - base random seed

    Note: in the experiments the "validation" set passed in is the TEST set
    (there was no separate validation split), so the optimised scores are
    optimistic. See "Limitations" in the README.

    Adapted from the objective function in ReservoirPy Tutorial 4
    ("Understand and optimise hyperparameters"); the connectivity and
    input-dimension parameters were added for this project.
    """
    (X_train, y_train), (X_val, y_val) = dataset

    instances = config["instances_per_trial"]
    variable_seed = seed

    losses, r2s = [], []
    for _ in range(instances):
        model = build_esn(N, lr=lr, sr=sr, input_scaling=iss,
                          rc_connectivity=rcntvt, input_connectivity=icntvt,
                          seed=variable_seed, input_dim=idim, ridge=ridge)
        predictions = model.fit(X_train, y_train).run(X_val)

        losses.append(nrmse(y_val, predictions, norm_value=np.ptp(X_train)))
        r2s.append(rsquare(y_val, predictions))

        variable_seed += 1  # a different random reservoir for the next instance

    return {"loss": np.mean(losses), "r2": np.mean(r2s)}


def hyperopt_config(name, window_size, max_evals=200):
    """Search space used for both the Mackey-Glass and Amazon optimisations.

    * 200 random-search trials x 3 reservoirs each = 600 ESNs trained.
      Random search beats grid search for this kind of problem
      (Bergstra & Bengio, 2012).
    * "loguniform" samples evenly on a log scale, good for values that
      span several orders of magnitude.
    * "choice" with a single value fixes that parameter.

    Structure adapted from ReservoirPy Tutorial 4.
    """
    return {
        "exp": name,
        "hp_max_evals": max_evals,
        "hp_method": "random",
        "seed": 69,
        "instances_per_trial": 3,
        "hp_space": {
            "N": ["choice", 500],
            "sr": ["loguniform", 1e-2, 10],
            "lr": ["loguniform", 1e-3, 1],
            "iss": ["loguniform", 1e-2, 10],
            "ridge": ["choice", 1e-7],
            "seed": ["choice", 1234],
            "icntvt": ["loguniform", 1e-2, 1],
            "rcntvt": ["loguniform", 1e-2, 1],
            "idim": ["choice", window_size],
        },
    }


def optimise_esn(name, X_train, y_train, X_test, y_test, window_size, out_dir, max_evals=200):
    """Run the random hyperparameter search and return the best parameter dict.

    Also saves hyperopt's summary figure (loss / R^2 against each parameter).
    In the returned dict, "choice" parameters (N, ridge, seed, idim) are given
    as the INDEX of the chosen option (always 0 here), not the value itself.
    """
    import json

    from reservoirpy.hyper import plot_hyperopt_report, research

    config = hyperopt_config(name, window_size, max_evals)
    config_path = Path(out_dir) / f"{name}.config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    dataset = ((X_train, y_train), (X_test, y_test))
    best, _trials = research(objective, dataset, str(config_path), str(out_dir))

    plot_hyperopt_report(str(Path(out_dir) / name), ("iss", "sr", "lr", "icntvt", "rcntvt"),
                         metric="r2")
    save_figure(out_dir, f"Hyperparameter search - {name}")

    print("Best hyperparameters found:", best)
    return best


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_mackey_glass(X, sample, tau, out_dir):
    """Plot the Mackey-Glass series and its phase diagram P(t) vs P(t - tau).

    Taken from ReservoirPy Tutorial 3 with only the save-to-file change.
    """
    plt.figure(figsize=(13, 5))
    N = sample

    ax = plt.subplot(121)
    t = np.linspace(0, N, N)
    for i in range(N - 1):
        ax.plot(t[i:i + 2], X[i:i + 2], color=plt.cm.magma(255 * i // N), lw=1.0)
    plt.title(f"Timeseries - {N} timesteps")
    plt.xlabel("$t$")
    plt.ylabel("$P(t)$")

    ax2 = plt.subplot(122)
    ax2.margins(0.05)
    for i in range(N - 1):
        ax2.plot(X[i:i + 2], X[i + tau:i + tau + 2], color=plt.cm.magma(255 * i // N), lw=1.0)
    plt.title("Phase diagram: $P(t) = f(P(t-\\tau))$")
    plt.xlabel("$P(t-\\tau)$")
    plt.ylabel("$P(t)$")

    plt.tight_layout()
    save_figure(out_dir, "Time series and phase diagram")


def plot_train_test(X_train, y_train, X_test, y_test, forecast, out_dir):
    """Show the last 500 training points and the test set, inputs vs targets.

    Adapted from ReservoirPy Tutorial 3 (added colours, labels and title).
    """
    sample = 500
    test_len = X_test.shape[0]
    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(0, sample), X_train[-sample:], color="red", label="Training data")
    plt.plot(np.arange(0, sample), y_train[-sample:], color="orange", label="Training ground truth")
    plt.plot(np.arange(sample, sample + test_len), X_test, color="blue", label="Testing data")
    plt.plot(np.arange(sample, sample + test_len), y_test, color="green", label="Testing ground truth")
    plt.title(f"Data preprocessing for {forecast} timesteps", fontsize=20)
    plt.xlabel("Number of days", fontsize=16)
    plt.ylabel("Timeseries - Mackey Glass", fontsize=16)
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.35, color="gray")
    save_figure(out_dir, f"Data preprocessing for {forecast} timesteps")


def plot_readout(readout, out_dir):
    """Bar chart of the trained readout weights W_out (one bar per neuron).

    Taken from ReservoirPy Tutorial 3. Not called by default; useful for
    checking that no single neuron dominates the output.
    """
    Wout = np.r_[readout.bias, readout.Wout]
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    ax.grid(axis="y")
    ax.set_ylabel("Coefs. of $W_{out}$")
    ax.set_xlabel("reservoir neurons index")
    ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])
    save_figure(out_dir, "Readout weights")


def plot_prediction(y_pred, y_true, title, out_dir, label="Prediction"):
    """Plot a single model's prediction against the true values.

    Adapted from ``plot_results()`` in ReservoirPy Tutorial 3.
    """
    plt.figure(figsize=(15, 7.5))
    plt.plot(y_pred, lw=3, label=label)
    plt.plot(y_true, linestyle="--", lw=2, label="True value")
    plt.title(title)
    plt.xlabel("Number of days")
    plt.ylabel("Timeseries - Mackey Glass")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.35, color="gray")
    save_figure(out_dir, title)


def plot_all_models(y_test, preds, forecast, out_dir):
    """Overlay every model's prediction (ESN, ridge, LSTM, SVR) on the true values.

    ``preds`` maps a label (e.g. "ESN") to its prediction array; models that
    were skipped (value None) are left out.
    """
    colours = {"ESN": "blue", "Ridge": "orange", "LSTM": "green", "SVR": "red"}
    plt.figure(figsize=(22.5, 15))
    for name, pred in preds.items():
        if pred is not None:
            plt.plot(pred, lw=5, color=colours.get(name), label=f"{name} prediction")
    plt.plot(y_test, linestyle="--", lw=15, alpha=0.3, color="red", label="True value")
    plt.title(f"Results for {forecast} timesteps", fontsize=30)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Number of days", fontsize=30)
    plt.ylabel("Timeseries - Mackey Glass", fontsize=30)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, color="gray")
    save_figure(out_dir, f"Predictions for {forecast} timesteps (all models)")


def plot_with_without_reservoir(y_test, y_pred_esn, y_pred_ridge, forecast, out_dir):
    """Three stacked panels: (a) ridge only, (b) ESN, (c) both together."""
    plt.figure(figsize=(22.5, 15))

    ax_a = plt.subplot(311)
    ax_a.plot(y_pred_ridge, lw=3, label="Ridge prediction")
    ax_a.plot(y_test, linestyle="--", lw=2, label="True value")
    ax_a.set_title(f"Results for {forecast} timesteps without reservoir", fontsize=20)

    ax_b = plt.subplot(312)
    ax_b.plot(y_pred_esn, lw=3, label="ESN prediction")
    ax_b.plot(y_test, linestyle="--", lw=2, label="True value")
    ax_b.set_title(f"Results for {forecast} timesteps with reservoir", fontsize=20)
    ax_b.set_ylabel("Timeseries - Mackey Glass / $P(t)$", fontsize=25)

    ax_c = plt.subplot(313)
    ax_c.plot(y_pred_esn, lw=3, label="ESN prediction")
    ax_c.plot(y_pred_ridge, lw=3, label="Ridge prediction")
    ax_c.plot(y_test, linestyle="--", lw=2, label="True value")
    ax_c.set_title(f"Results for {forecast} timesteps with and without reservoir", fontsize=20)
    ax_c.set_xlabel("Number of days", fontsize=20)

    for ax in (ax_a, ax_b, ax_c):
        ax.legend(fontsize=18)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(axis="y", linestyle="--", linewidth=0.35, color="gray")

    save_figure(out_dir, f"Predictions for {forecast} timesteps with and without reservoir")


def plot_optimisation_comparison(series, train_data, y_test, split_index, window_size,
                                 y_ridge, y_esn, y_esn_opt, ylabel, out_dir, filename):
    """Three stacked panels comparing ridge-only, default ESN and optimised ESN.

    Each panel shows the end of the training data, the true test values and
    the prediction, with its NRMSE in the title.
    """
    plt.figure(figsize=(17, 21))
    k = int(window_size * 0.5)  # how much training data to show before the split
    train_x = range(split_index - k, split_index)
    test_x = range(split_index, len(series))

    panels = [
        (y_ridge, "Prediction without reservoir", "Ridge regression"),
        (y_esn, "Prediction with non-optimised hyperparameters", "RC before optimisation"),
        (y_esn_opt, "Prediction with optimised hyperparameters", "RC after optimisation"),
    ]
    for i, (pred, label, name) in enumerate(panels):
        ax = plt.subplot(311 + i)
        ax.plot(train_x, train_data[split_index - k:], lw=3, label="Training data")
        ax.plot(test_x, y_test, lw=3, label="True value")
        ax.plot(test_x, pred, label=label)
        ax.set_title(f"{name}, NRMSE = {nrmse(y_test, pred):.3f}", fontsize=25)
        ax.grid(axis="y", linestyle="--", linewidth=0.35, color="gray")
        ax.legend(fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        if i == 0:
            ax.text(0.5, 1.12,
                    f"One-step-ahead predictions over the {window_size}-step test period "
                    f"(window = {window_size})",
                    fontsize=24, fontweight="bold", ha="center", va="bottom",
                    transform=ax.transAxes)
        if i == 1:
            ax.set_ylabel(ylabel, fontsize=30)
        if i == 2:
            ax.set_xlabel("Number of days", fontsize=25)

    save_figure(out_dir, filename)


def plot_nrmse_bar(y_test, y_ridge, y_esn, y_esn_opt, out_dir, filename):
    """Bar chart of NRMSE for ridge-only, default ESN and optimised ESN."""
    plt.figure(figsize=(14, 9))
    plt.bar(["Ridge"], [nrmse(y_test, y_ridge)], label="Without reservoir")
    plt.bar(["Non-optimised\nhyperparameters (RC)", "Optimised\nhyperparameters (RC)"],
            [nrmse(y_test, y_esn), nrmse(y_test, y_esn_opt)],
            label="With reservoir", color="magenta")
    plt.xlabel("Prediction", fontsize=30)
    plt.ylabel("NRMSE", fontsize=30)
    plt.title("NRMSE against predictions", fontsize=30)
    plt.grid(linestyle="--", linewidth=0.35, color="gray")
    plt.legend(fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    save_figure(out_dir, filename)
