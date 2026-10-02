"""
Experiment 2 - Mackey-Glass sliding-window prediction + hyperparameter optimisation
====================================================================================

Question: using a window of past values, how accurately can an ESN predict
the next value of the series, and how much does tuning its hyperparameters help?

Setup
-----
* 2510 Mackey-Glass samples, first 2008 for training, last 502 for testing.
* Each input is the previous 502 values (window size = test length) and the
  target is the next value, so every test point is a one-step-ahead prediction
  made from TRUE past values (see "Limitations" in the README).

Steps
-----
1. Ridge regression on the raw windows (no reservoir).
2. ESN with default hyperparameters.
3. Random search over 200 hyperparameter sets x 3 reservoirs (hyperopt).
4. ESN with the best hyperparameters; compare all three.

Report results (NRMSE): ridge 0.014, default ESN 0.162, optimised ESN 0.003
(78.57% lower error than ridge).

Usage
-----
    python src/mackey_glass_sliding_window.py           # full run (~7 min search)
    python src/mackey_glass_sliding_window.py --quick   # 5-trial search, fast check

Figures are saved to results/mackey_glass_sliding_window/.
"""

import argparse

from common import (DEFAULT_ESN, analyse, build_esn, load_mackey_glass, optimise_esn,
                    plot_nrmse_bar, plot_optimisation_comparison, results_folder,
                    ridge_baseline, split_sliding_window)

OPTIMISED_UNITS = 500        # N was fixed to 500 during the search
OPTIMISED_RIDGE = 1e-8       # note: the search itself used ridge = 1e-7


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="only 5 hyperparameter trials, for a fast check that everything runs")
    args = parser.parse_args()

    out_dir = results_folder("mackey_glass_sliding_window")
    X = load_mackey_glass(tau=20)
    X_train, y_train, X_test, y_test, train_data, split_index, window_size = split_sliding_window(X)

    # 1. Baseline without a reservoir
    y_ridge = ridge_baseline(X_train, y_train, X_test)

    # 2. ESN with default hyperparameters (input_dim = window size)
    esn = build_esn(**DEFAULT_ESN, input_dim=window_size)
    y_esn = esn.fit(X_train, y_train).run(X_test)

    print("===== Before optimisation =====")
    analyse(y_test, y_ridge, "Without reservoir (ridge):")
    analyse(y_test, y_esn, "With reservoir (default ESN):")

    # 3. Hyperparameter search
    best = optimise_esn("hyperopt-mackeyGlass", X_train, y_train, X_test, y_test,
                        window_size, out_dir, max_evals=5 if args.quick else 200)

    # 4. Rebuild the ESN with the best values found
    esn_opt = build_esn(OPTIMISED_UNITS, lr=best["lr"], sr=best["sr"],
                        input_scaling=best["iss"], rc_connectivity=best["rcntvt"],
                        input_connectivity=best["icntvt"], seed=1234,
                        input_dim=window_size, ridge=OPTIMISED_RIDGE)
    y_esn_opt = esn_opt.fit(X_train, y_train).run(X_test)

    print("===== After optimisation =====")
    analyse(y_test, y_esn_opt, "With optimised reservoir:")

    plot_optimisation_comparison(X, train_data, y_test, split_index, window_size,
                                 y_ridge, y_esn, y_esn_opt,
                                 ylabel="Timeseries - Mackey Glass / $P(t)$", out_dir=out_dir,
                                 filename="Sliding-window results before and after optimisation")
    plot_nrmse_bar(y_test, y_ridge, y_esn, y_esn_opt, out_dir,
                   filename="NRMSE bar chart before and after optimisation")

    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
