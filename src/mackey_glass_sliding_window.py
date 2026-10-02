# Experiment 2: predict the next Mackey-Glass value from the previous 502 values,
# then tune the ESN hyperparameters with a random search.
#
#   python src/mackey_glass_sliding_window.py          (full run, ~7 min search)
#   python src/mackey_glass_sliding_window.py --quick  (5 trials only)

import argparse

from common import (DEFAULT_ESN, analyse, build_esn, load_mackey_glass, optimise_esn,
                    plot_nrmse_bar, plot_optimisation_comparison, results_folder,
                    ridge_baseline, split_sliding_window)

OPTIMISED_UNITS = 500
OPTIMISED_RIDGE = 1e-8  # the search used 1e-7, I used 1e-8 for the final model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="5 search trials only")
    args = parser.parse_args()

    out_dir = results_folder("mackey_glass_sliding_window")
    X = load_mackey_glass(tau=20)
    X_train, y_train, X_test, y_test, train_data, split_index, window_size = split_sliding_window(X)

    # no reservoir
    y_ridge = ridge_baseline(X_train, y_train, X_test)

    # ESN with default settings
    esn = build_esn(**DEFAULT_ESN, input_dim=window_size)
    y_esn = esn.fit(X_train, y_train).run(X_test)

    print("===== Before optimisation =====")
    analyse(y_test, y_ridge, "Without reservoir (ridge):")
    analyse(y_test, y_esn, "With reservoir (default ESN):")

    best = optimise_esn("hyperopt-mackeyGlass", X_train, y_train, X_test, y_test,
                        window_size, out_dir, max_evals=5 if args.quick else 200)

    # ESN with the best values from the search
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
