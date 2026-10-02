# Shared code for the three experiments: data prep, models, hyperparameter search, plots.
#
# Some functions are adapted from the ReservoirPy tutorials
# (https://github.com/reservoirpy/reservoirpy, MIT License, (c) 2018 neuronalX).
# These are marked "Adapted from ReservoirPy" below.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
from reservoirpy.datasets import mackey_glass, to_forecasting
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
from sklearn.linear_model import Ridge as RidgeSklearn
from sklearn.svm import SVR

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TRAIN_FRACTION = 0.8  # 80% train, 20% test


def results_folder(name):
    folder = RESULTS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_figure(folder, filename):
    plt.savefig(Path(folder) / f"{filename}.png", bbox_inches="tight")
    plt.close("all")


def rescale(series):
    # scale to [-1, 1] to match the range of the tanh neurons
    return 2 * (series - series.min()) / (series.max() - series.min()) - 1


def analyse(y_true, y_pred, label=""):
    """Print R^2 and NRMSE (RMSE / (max - min) of the true values)."""
    if label:
        print(label)
    print(f"  R-squared: {rsquare(y_true, y_pred):.4f}")
    print(f"  NRMSE:     {nrmse(y_true, y_pred):.4f}\n")


# ---------------------------------------------------------------- data

def load_mackey_glass(timesteps=2510, tau=20):
    # Adapted from ReservoirPy Tutorial 3 (they use tau=17).
    # tau > 16.8 makes the series chaotic, so I used 20.
    return rescale(mackey_glass(timesteps, tau=tau))


def split_multistep(series, forecast, train_fraction=TRAIN_FRACTION):
    """Inputs u(t) and targets u(t + forecast), split into train/test."""
    x, y = to_forecasting(series, forecast=forecast)
    split_index = int(train_fraction * len(series))
    return x[:split_index], y[:split_index], x[split_index:], y[split_index:], split_index


def split_sliding_window(series, train_fraction=TRAIN_FRACTION):
    """Each input is the previous window_size values, target is the next value.

    window_size = length of the test set. The test windows use the real past
    values, so this is one-step-ahead prediction.
    """
    split_index = int(train_fraction * len(series))
    window_size = len(series) - split_index
    train_data = series[:split_index]
    test_data = series[split_index - window_size:]  # first test window = end of training data

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


# ---------------------------------------------------------------- models

# Default ESN settings (from ReservoirPy Tutorial 3, with 200 neurons instead of 100)
DEFAULT_ESN = dict(
    units=200,               # number of neurons
    lr=0.3,                  # leaking rate
    sr=1.25,                 # spectral radius
    input_scaling=1.0,
    rc_connectivity=0.1,     # density of connections inside the reservoir
    input_connectivity=0.2,  # density of input connections
    seed=1234,
)
DEFAULT_RIDGE = 1e-8


def build_esn(units, lr, sr, input_scaling, rc_connectivity, input_connectivity,
              seed=1234, input_dim=None, ridge=DEFAULT_RIDGE):
    # Adapted from reset_esn() in ReservoirPy Tutorial 3.
    # Reservoir weights are random and fixed; only the Ridge readout gets trained.
    reservoir = Reservoir(units, input_scaling=input_scaling, sr=sr, lr=lr,
                          rc_connectivity=rc_connectivity,
                          input_connectivity=input_connectivity,
                          seed=seed, input_dim=input_dim)
    readout = Ridge(1, ridge=ridge)
    return reservoir >> readout


def ridge_baseline(X_train, y_train, X_test):
    # same readout but no reservoir - shows what the reservoir adds
    model = RidgeSklearn(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict(X_test).reshape(-1, 1)


def svr_baseline(X_train, y_train, X_test):
    # C, gamma, epsilon picked by trial and error
    model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train.ravel())
    return model.predict(X_test).reshape(-1, 1)


def lstm_baseline(X_train, y_train, X_test, layers=(128, 64)):
    # TensorFlow is optional because it's a big install
    try:
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential
    except ImportError:
        print("TensorFlow not installed - skipping LSTM.")
        return None

    # Keras wants (samples, timesteps, features)
    X_train = np.asarray(X_train).reshape(-1, 1, 1)
    X_test = np.asarray(X_test).reshape(-1, 1, 1)

    model = Sequential()
    for i, n_units in enumerate(layers):
        last = i == len(layers) - 1
        model.add(LSTM(n_units, return_sequences=not last, activation="relu", input_shape=(1, 1)))
    model.add(Dense(25))
    if len(layers) > 1:
        model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0)  # 1 epoch, batch 1 worked best
    return model.predict(X_test, verbose=0)


# ---------------------------------------------------------------- hyperparameter search

def objective(dataset, config, *, iss, N, sr, lr, icntvt, rcntvt, idim, ridge, seed):
    # Adapted from ReservoirPy Tutorial 4 - I added the connectivity and input dim parameters.
    # Builds a few ESNs with different seeds and returns the average NRMSE,
    # so one lucky random reservoir doesn't win.
    # NOTE: the "validation" data passed in is actually the test set (see README).
    (X_train, y_train), (X_val, y_val) = dataset

    variable_seed = seed
    losses, r2s = [], []
    for _ in range(config["instances_per_trial"]):
        model = build_esn(N, lr=lr, sr=sr, input_scaling=iss, rc_connectivity=rcntvt,
                          input_connectivity=icntvt, seed=variable_seed,
                          input_dim=idim, ridge=ridge)
        predictions = model.fit(X_train, y_train).run(X_val)
        losses.append(nrmse(y_val, predictions, norm_value=np.ptp(X_train)))
        r2s.append(rsquare(y_val, predictions))
        variable_seed += 1

    return {"loss": np.mean(losses), "r2": np.mean(r2s)}


def hyperopt_config(name, window_size, max_evals=200):
    # Random search, 200 trials x 3 reservoirs. "choice" with one value = fixed.
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
    """Run the search and return the best hyperparameters.

    For "choice" parameters hyperopt returns the index (0), not the value.
    """
    import json

    from reservoirpy.hyper import plot_hyperopt_report, research

    config = hyperopt_config(name, window_size, max_evals)
    config_path = Path(out_dir) / f"{name}.config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    dataset = ((X_train, y_train), (X_test, y_test))
    best, _ = research(objective, dataset, str(config_path), str(out_dir))

    plot_hyperopt_report(str(Path(out_dir) / name), ("iss", "sr", "lr", "icntvt", "rcntvt"),
                         metric="r2")
    save_figure(out_dir, f"Hyperparameter search - {name}")

    print("Best hyperparameters:", best)
    return best


# ---------------------------------------------------------------- plots

def plot_mackey_glass(X, sample, tau, out_dir):
    # From ReservoirPy Tutorial 3 (only changed to save the figure)
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
    # Adapted from ReservoirPy Tutorial 3
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
    # From ReservoirPy Tutorial 3 - bar chart of the trained W_out (not used by default)
    Wout = np.r_[readout.bias, readout.Wout]
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    ax.grid(axis="y")
    ax.set_ylabel("Coefs. of $W_{out}$")
    ax.set_xlabel("reservoir neurons index")
    ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])
    save_figure(out_dir, "Readout weights")


def plot_prediction(y_pred, y_true, title, out_dir, label="Prediction"):
    # Adapted from plot_results() in ReservoirPy Tutorial 3
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
    # all models on one plot; preds = {"ESN": ..., "Ridge": ..., ...}, None = skipped
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
    # (a) ridge only, (b) ESN, (c) both
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
    # ridge vs ESN before tuning vs ESN after tuning, NRMSE in each title
    plt.figure(figsize=(17, 21))
    k = int(window_size * 0.5)  # how much training data to show before the test period
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
            ax.text(0.5, 1.12, f"One-step-ahead predictions, window = {window_size}",
                    fontsize=24, fontweight="bold", ha="center", va="bottom",
                    transform=ax.transAxes)
        if i == 1:
            ax.set_ylabel(ylabel, fontsize=30)
        if i == 2:
            ax.set_xlabel("Number of days", fontsize=25)

    save_figure(out_dir, filename)


def plot_nrmse_bar(y_test, y_ridge, y_esn, y_esn_opt, out_dir, filename):
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
