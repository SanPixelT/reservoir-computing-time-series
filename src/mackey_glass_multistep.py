"""
Experiment 1 - Mackey-Glass multi-step-ahead forecasting
=========================================================

Question: given the current value of the Mackey-Glass series, how well can
different models predict its value k steps into the future?

Models compared
---------------
* ESN   - Echo State Network (reservoir + ridge readout)
* Ridge - the same ridge readout with NO reservoir (shows what the reservoir adds)
* LSTM  - recurrent network trained with back-propagation (optional, needs TensorFlow)
* SVR   - support vector regression with an RBF kernel

Steps
-----
1. Generate 2510 Mackey-Glass samples (tau = 20) and rescale to [-1, 1].
2. Forecast 10 steps ahead with every model; plot and print R^2 / NRMSE.
3. Repeat for 100 steps ahead.
4. Sweep the horizon k = 1 ... 99 and plot NRMSE against k for every model.

Report results (NRMSE): 10 steps -> ESN 0.0014 vs ridge 0.2316,
                       100 steps -> ESN 0.0660 vs ridge 0.2519.

Usage
-----
    python src/mackey_glass_multistep.py            # full run, as in the report
    python src/mackey_glass_multistep.py --quick    # fast check (coarse sweep, no LSTM)

Figures are saved to results/mackey_glass_multistep/.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from reservoirpy.observables import nrmse

from common import (DEFAULT_ESN, analyse, build_esn, load_mackey_glass, lstm_baseline,
                    plot_all_models, plot_mackey_glass, plot_prediction, plot_train_test,
                    plot_with_without_reservoir, results_folder, ridge_baseline,
                    save_figure, split_multistep, svr_baseline)

TAU = 20


def run_horizon(esn, X, forecast, out_dir, use_lstm=True):
    """Train and evaluate every model for a single forecast horizon.

    ``esn`` is re-used between horizons (as in the original code). Only its
    readout is re-trained; the reservoir keeps its internal state from the
    previous run, which acts as a warm-up. See main() for details.
    """
    X_train, y_train, X_test, y_test, _ = split_multistep(X, forecast)
    plot_train_test(X_train, y_train, X_test, y_test, forecast, out_dir)

    # Echo State Network: fit() trains only the ridge readout W_out
    y_esn = esn.fit(X_train, y_train).run(X_test)
    plot_prediction(y_esn, y_test, f"Prediction for {forecast} timesteps with reservoir (ESN)",
                    out_dir, label="ESN prediction")

    # Baselines
    y_ridge = ridge_baseline(X_train, y_train, X_test)
    plot_prediction(y_ridge, y_test, f"Prediction for {forecast} timesteps without reservoir (ridge)",
                    out_dir, label="Ridge prediction")

    y_svr = svr_baseline(X_train, y_train, X_test)
    plot_prediction(y_svr, y_test, f"Prediction for {forecast} timesteps (SVR)",
                    out_dir, label="SVR prediction")

    y_lstm = lstm_baseline(X_train, y_train, X_test) if use_lstm else None
    if y_lstm is not None:
        plot_prediction(y_lstm, y_test, f"Prediction for {forecast} timesteps (LSTM)",
                        out_dir, label="LSTM prediction")

    # Comparison figures
    plot_all_models(y_test, {"ESN": y_esn, "Ridge": y_ridge, "LSTM": y_lstm, "SVR": y_svr},
                    forecast, out_dir)
    plot_with_without_reservoir(y_test, y_esn, y_ridge, forecast, out_dir)

    print(f"===== {forecast} timesteps ahead =====")
    analyse(y_test, y_esn, "With reservoir (ESN):")
    analyse(y_test, y_ridge, "Without reservoir (ridge):")
    analyse(y_test, y_svr, "SVR:")
    if y_lstm is not None:
        analyse(y_test, y_lstm, "LSTM:")


def nrmse_sweep(X, horizons, out_dir, use_lstm=True):
    """NRMSE of every model for each forecast horizon k (report Figure 7.4).

    Shows how accuracy degrades as we predict further ahead. Without a
    reservoir the error oscillates with k: the prediction lags (is phase
    shifted from) the true signal, and how much that lag hurts depends on k.
    """
    scores = {"ESN": [], "Ridge": [], "LSTM": [], "SVR": []}

    for forecast in horizons:
        print(f"Sweep: k = {forecast}")
        X_train, y_train, X_test, y_test, _ = split_multistep(X, forecast)

        y_esn = build_esn(**DEFAULT_ESN).fit(X_train, y_train).run(X_test)
        scores["ESN"].append(nrmse(y_test, y_esn))
        scores["Ridge"].append(nrmse(y_test, ridge_baseline(X_train, y_train, X_test)))
        scores["SVR"].append(nrmse(y_test, svr_baseline(X_train, y_train, X_test)))

        if use_lstm:
            # A smaller single-layer LSTM (100 units) keeps the 99-model sweep affordable.
            y_lstm = lstm_baseline(X_train, y_train, X_test, layers=(100,))
            if y_lstm is not None:
                scores["LSTM"].append(nrmse(y_test, y_lstm))

    plt.figure(figsize=(22.5, 15))
    for name, values in scores.items():
        if values:
            plt.plot(list(horizons), values, lw=5, marker="x", markersize=15,
                     label=f"{name} prediction")
    plt.title("NRMSE for prediction of the Mackey-Glass series $k$ timesteps in the future",
              fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=20)
    plt.xlabel("$k$ timesteps", fontsize=20)
    plt.ylabel("NRMSE", fontsize=20)
    plt.grid(linestyle="--", linewidth=0.5)
    save_figure(out_dir, f"NRMSE for {horizons[0]}-{horizons[-1]} timesteps ahead")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="coarse k-sweep and no LSTM, for a fast check that everything runs")
    parser.add_argument("--no-lstm", action="store_true", help="skip the LSTM baseline")
    args = parser.parse_args()
    use_lstm = not (args.quick or args.no_lstm)

    out_dir = results_folder("mackey_glass_multistep")
    X = load_mackey_glass(tau=TAU)
    plot_mackey_glass(X, 500, TAU, out_dir)

    # One ESN (default hyperparameters) is shared by the 10- and 100-step runs.
    # The first call pushes a single sample through so the reservoir and
    # readout weights get initialised. Because the reservoir state is NOT reset
    # between calls, the 100-step model starts from the state left by the
    # 10-step test run. This reproduces the report's numbers exactly
    # (NRMSE 0.0014 and 0.0660); a brand-new ESN gives 0.0731 for 100 steps.
    esn = build_esn(**DEFAULT_ESN)
    esn(X[0])

    run_horizon(esn, X, 10, out_dir, use_lstm)
    run_horizon(esn, X, 100, out_dir, use_lstm)

    horizons = list(range(1, 100, 14)) if args.quick else list(range(1, 100))
    nrmse_sweep(X, horizons, out_dir, use_lstm)

    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    np.random.seed(1234)
    main()
