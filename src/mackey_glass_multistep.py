# Experiment 1: predict the Mackey-Glass series k steps ahead.
# Compares ESN vs ridge (no reservoir) vs LSTM vs SVR at 10 and 100 steps,
# then sweeps k from 1 to 99.
#
#   python src/mackey_glass_multistep.py          (full run)
#   python src/mackey_glass_multistep.py --quick  (fast check, no LSTM)

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
    X_train, y_train, X_test, y_test, _ = split_multistep(X, forecast)
    plot_train_test(X_train, y_train, X_test, y_test, forecast, out_dir)

    # fit() only trains the readout
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
    # NRMSE vs k for every model (Figure 7.4 in the report)
    scores = {"ESN": [], "Ridge": [], "LSTM": [], "SVR": []}

    for forecast in horizons:
        print(f"Sweep: k = {forecast}")
        X_train, y_train, X_test, y_test, _ = split_multistep(X, forecast)

        y_esn = build_esn(**DEFAULT_ESN).fit(X_train, y_train).run(X_test)
        scores["ESN"].append(nrmse(y_test, y_esn))
        scores["Ridge"].append(nrmse(y_test, ridge_baseline(X_train, y_train, X_test)))
        scores["SVR"].append(nrmse(y_test, svr_baseline(X_train, y_train, X_test)))

        if use_lstm:
            # smaller LSTM here so the 99 runs don't take forever
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="coarse sweep, no LSTM")
    parser.add_argument("--no-lstm", action="store_true", help="skip the LSTM baseline")
    args = parser.parse_args()
    use_lstm = not (args.quick or args.no_lstm)

    out_dir = results_folder("mackey_glass_multistep")
    X = load_mackey_glass(tau=TAU)
    plot_mackey_glass(X, 500, TAU, out_dir)

    # Same ESN used for 10 and 100 steps, like in my original code. The reservoir
    # state isn't reset in between, so the 100-step run starts from where the
    # 10-step run ended. A fresh ESN gives 0.0731 instead of 0.0660 (see README).
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
