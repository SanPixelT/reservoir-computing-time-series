"""
Experiment 3 - Amazon opening stock price prediction
=====================================================

Applies the sliding-window pipeline from Experiment 2 to real data: daily
Amazon (AMZN) stock prices from 2014 to 2019.

Unlike Mackey-Glass, stock prices are NON-stationary (the average level
keeps rising), so this is a harder test of the reservoir.

Setup
-----
* 1259 trading days of opening prices, rescaled to [-1, 1].
* First 1007 days for training, last 252 days for testing.
* Window size = 252, i.e. one-step-ahead prediction from the previous 252 days.

Report results (NRMSE): ridge 0.070, default ESN 0.514, optimised ESN 0.053
(24.29% lower error than ridge).

Data
----
Download the dataset and place ``AMZNtrain.csv`` in the ``data/`` folder
(see data/README.md). Only the ``Open`` column is used for modelling.

Usage
-----
    python src/amazon_stock.py           # full run
    python src/amazon_stock.py --quick   # 5-trial search, fast check

Figures are saved to results/amazon_stock/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (analyse, build_esn, optimise_esn, plot_nrmse_bar,
                    plot_optimisation_comparison, rescale, results_folder,
                    ridge_baseline, save_figure, split_sliding_window)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "AMZNtrain.csv"

# Hand-picked starting hyperparameters for the stock data (before optimisation)
AMAZON_DEFAULT_ESN = dict(
    units=500,
    lr=0.3,
    sr=0.07,
    input_scaling=0.05,
    rc_connectivity=0.05,
    input_connectivity=0.05,
    seed=1234,
)
OPTIMISED_RIDGE = 1e-8  # note: the search itself used ridge = 1e-7


def plot_price_columns(df, out_dir):
    """Save a plot of each raw price column (Open, Close, High, Low, Adj Close)."""
    for column in ["Open", "Close", "High", "Low", "Adj Close"]:
        if column not in df.columns:
            continue
        ax = df[column].plot(figsize=(10, 6), xlabel="Number of days",
                             ylabel=f"{column} price / $",
                             title=f"Amazon {column} stock price, 2014-2019")
        ax.grid(linestyle=":", linewidth=0.30, color="gray")
        save_figure(out_dir, f"Amazon {column} price 2014-2019")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="only 5 hyperparameter trials, for a fast check that everything runs")
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="path to the AMZN CSV file")
    args = parser.parse_args()

    if not args.data.exists():
        sys.exit(f"Data file not found: {args.data}\nSee data/README.md for download instructions.")

    out_dir = results_folder("amazon_stock")
    df = pd.read_csv(args.data)
    plot_price_columns(df, out_dir)

    # Opening price as a column vector, rescaled to [-1, 1]
    X_open = rescale(np.vstack(df["Open"].to_numpy(dtype=float)))
    X_train, y_train, X_test, y_test, train_data, split_index, window_size = \
        split_sliding_window(X_open)

    # 1. Baseline without a reservoir
    y_ridge = ridge_baseline(X_train, y_train, X_test)

    # 2. ESN with hand-picked starting values
    esn = build_esn(**AMAZON_DEFAULT_ESN, input_dim=window_size)
    y_esn = esn.fit(X_train, y_train).run(X_test)

    print("===== Before optimisation =====")
    analyse(y_test, y_ridge, "Without reservoir (ridge):")
    analyse(y_test, y_esn, "With reservoir (default ESN):")

    # 3. Hyperparameter search (same search space as Mackey-Glass)
    best = optimise_esn("hyperopt-amazonOpening", X_train, y_train, X_test, y_test,
                        window_size, out_dir, max_evals=5 if args.quick else 200)

    # 4. ESN with the best values found
    esn_opt = build_esn(500, lr=best["lr"], sr=best["sr"], input_scaling=best["iss"],
                        rc_connectivity=best["rcntvt"], input_connectivity=best["icntvt"],
                        seed=1234, input_dim=window_size, ridge=OPTIMISED_RIDGE)
    y_esn_opt = esn_opt.fit(X_train, y_train).run(X_test)

    print("===== After optimisation =====")
    analyse(y_test, y_esn_opt, "With optimised reservoir:")

    plot_optimisation_comparison(X_open, train_data, y_test, split_index, window_size,
                                 y_ridge, y_esn, y_esn_opt,
                                 ylabel="Normalised Amazon opening price", out_dir=out_dir,
                                 filename="Amazon results before and after optimisation")
    plot_nrmse_bar(y_test, y_ridge, y_esn, y_esn_opt, out_dir,
                   filename="Amazon NRMSE bar chart before and after optimisation")
    plt.close("all")

    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
