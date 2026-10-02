# Reservoir Computing for Time-Series Prediction

An **Echo State Network (ESN)**, a type of reservoir computer, built in Python to forecast a chaotic benchmark signal (Mackey-Glass) and real stock-market data (Amazon, 2014–2019). It is compared against ridge regression, LSTM and SVR baselines, and its hyperparameters are tuned with a random search.

> 3rd-year individual project, MEng Electronic & Electrical Engineering, UCL (2023).
> Supervisor: Hidekazu Kurebayashi.

## Highlights

| Task | Without reservoir (ridge) | ESN | Improvement |
|---|---|---|---|
| Mackey-Glass, 10 steps ahead | 0.2316 | **0.0014** | 99.4 % lower error |
| Mackey-Glass, 100 steps ahead | 0.2519 | **0.0660** | 73.8 % lower error |
| Mackey-Glass, sliding window (optimised ESN) | 0.014 | **0.003** | 78.6 % lower error |
| Amazon opening price, sliding window (optimised ESN) | 0.070 | **0.053** | 24.3 % lower error |

*Metric: normalised root-mean-squared error (NRMSE); lower is better, 0 is perfect. Please read the [Limitations](#limitations-and-what-id-do-differently) section before comparing these numbers with other work.*

- The ESN beat the LSTM and SVR baselines at every horizon from 1 to 99 steps ahead.
- Only the linear readout is trained, so training is a single matrix solve with no back-propagation.
- 600 reservoirs were evaluated in a 200-trial random hyperparameter search (about 7 minutes on a laptop CPU).

## How an Echo State Network works

<p align="center"><img src="docs/figures/esn_architecture.png" width="520" alt="ESN architecture: input layer, reservoir, output layer"></p>

1. **Input layer → reservoir (`W_in`, fixed, random).** Each input value is projected into a large pool of neurons.
2. **Reservoir (`W`, fixed, random, recurrent).** The neurons are randomly connected to each other. Their state keeps a fading "echo" of past inputs (memory) and mixes them non-linearly through `tanh`:

   ```
   x(t+1) = (1 − lr)·x(t) + lr·tanh(W_in·u(t+1) + W·x(t))
   ```

3. **Readout (`W_out`, the only trained weights).** The output is a linear combination of the neuron states. It is found in closed form with ridge regression:

   ```
   W_out = Y·Xᵀ·(X·Xᵀ + λI)⁻¹
   ```

Because the hard part (the recurrent dynamics) is never trained, an ESN avoids the slow, unstable gradient training that RNNs and LSTMs need. That is why reservoir computing is attractive for real-time and **physical/hardware** reservoirs, the long-term motivation of this project.

The main hyperparameters:

| Name | Meaning |
|---|---|
| `units` (N) | number of reservoir neurons |
| `sr` | spectral radius of `W`; below 1 means the reservoir is stable with no input |
| `lr` | leaking rate; how much the previous state is kept at each step |
| `input_scaling` | gain on the inputs; higher pushes `tanh` into its non-linear region |
| `rc_connectivity` / `input_connectivity` | fraction of non-zero weights in `W` / `W_in` |
| `ridge` | L2 regularisation of the readout |

## Repository structure

```
├── src/
│   ├── common.py                       # data prep, ESN + baseline models, hyperopt objective, plots
│   ├── mackey_glass_multistep.py       # Experiment 1: k-steps-ahead forecasting, ESN vs ridge/LSTM/SVR
│   ├── mackey_glass_sliding_window.py  # Experiment 2: sliding-window prediction + hyperparameter search
│   └── amazon_stock.py                 # Experiment 3: same pipeline on Amazon stock prices
├── data/README.md                      # where to download the Amazon dataset
├── docs/figures/                       # figures used in this README
├── results/                            # created when you run the scripts (git-ignored)
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/SanPixelT/reservoir-computing-time-series.git
cd reservoir-computing-time-series

python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install tensorflow          # optional, only for the LSTM baseline
```

Run the experiments from the repository root:

```bash
python src/mackey_glass_multistep.py         # Experiment 1
python src/mackey_glass_sliding_window.py    # Experiment 2 (~7 min hyperparameter search)
python src/amazon_stock.py                   # Experiment 3 (needs data/AMZNtrain.csv)
```

Add `--quick` to any script for a fast check that everything runs (5 search trials, coarse sweep, no LSTM). Figures are written to `results/<experiment>/`, and R² / NRMSE scores are printed to the terminal.

Experiments 1 and 2 reproduce the report's ESN and ridge numbers exactly. Seeds are fixed for the reservoirs (`1234`) and the search (`69`). LSTM results vary slightly between runs.

## Experiments and results

### 1. Mackey-Glass: predicting k steps ahead

The [Mackey-Glass](https://en.wikipedia.org/wiki/Mackey%E2%80%93Glass_equations) delay equation with τ = 20 is chaotic but stationary, which makes it a standard benchmark for learning dynamical systems. The series (2510 samples, rescaled to [−1, 1]) is split 80/20 into training and test sets. Each model gets the current value and must output the value `k` steps later.

<p align="center"><img src="docs/figures/mackey_glass_100_steps.png" width="720" alt="100-steps-ahead prediction with and without reservoir"></p>

*100 steps ahead: (a) ridge regression on its own lags and flattens the signal; (b) the ESN tracks it closely; (c) both overlaid.*

Sweeping the horizon from 1 to 99 shows the ESN error staying far below every baseline:

<p align="center"><img src="docs/figures/nrmse_vs_horizon.png" width="620" alt="NRMSE against forecast horizon for ESN, ridge, LSTM and SVR"></p>

The baselines' error **oscillates** with `k`. Their predictions are phase-shifted relative to the true signal, so the error depends on how that lag lines up with the signal's period. The reservoir's memory of past inputs removes this lag.

### 2. Mackey-Glass: sliding window + hyperparameter optimisation

Each input is a window of the previous 502 values, and the target is the next value. The ESN was tuned with [hyperopt](https://github.com/hyperopt/hyperopt) random search: 200 hyperparameter sets × 3 random reservoirs each, minimising mean NRMSE. Random search was chosen over grid search following Bergstra & Bengio (2012).

<p align="center"><img src="docs/figures/mackey_glass_sliding_window.png" width="560" alt="Ridge vs default ESN vs optimised ESN on Mackey-Glass"></p>

The default ESN (NRMSE 0.162) is worse than plain ridge (0.014), which shows that **hyperparameters are task-specific**. After tuning, the ESN reaches 0.003. Best values found: spectral radius 0.22, leaking rate 0.54, input scaling 0.013, input connectivity 0.037, recurrent connectivity 0.11.

### 3. Amazon opening stock price (2014–2019)

Real financial data is noisy and **non-stationary** (its average level keeps rising), so it is a harder test. Training data is 1007 trading days; test data is 252 days with a 252-day window.

<p align="center"><img src="docs/figures/amazon_sliding_window.png" width="560" alt="Ridge vs default ESN vs optimised ESN on Amazon stock"></p>

The untuned ESN (b) cannot follow the upward trend: its memory fades too quickly for this data. After optimisation (c), with a much smaller spectral radius (0.07) and leaking rate (0.17), it beats ridge regression (0.053 vs 0.070 NRMSE).

## What makes a good reservoir?

From the hyperparameter study (full discussion in the report):

- **Echo state property / fading memory.** The reservoir must forget its initial state but remember recent inputs. A spectral radius below 1 makes it stable when there is no input; both tuned models ended up well below 1.
- **Memory vs non-linearity trade-off.** Higher input scaling or connectivity pushes `tanh` into saturation. That gives more non-linearity (better separation of inputs) but less linear memory. Both tuned models chose very small input scaling (about 0.01), which favours memory.
- **Size.** More neurons give higher dimensionality and memory capacity, at a hardware or compute cost. N was fixed at 500 for the search.

## Limitations and what I'd do differently

I wrote this project as an undergraduate. Looking back with fresh eyes, these points matter when reading the results:

1. **Hyperparameters were tuned on the test set.** There was no separate validation split, so the "optimised" scores (0.003 and 0.053) are optimistic. *Fix:* split train / validation / test (e.g. 70/10/20), tune on validation, and report test once at the end.
2. **The sliding-window results are one-step-ahead.** Each test prediction uses the *true* previous 502 (or 252) values, not the model's own earlier predictions. This explains why plain ridge regression already does well. *Fix:* add a closed-loop (generative) forecast that feeds predictions back in.
3. **No reservoir warm-up.** Reservoir states are not "washed out" before training. In Experiment 1 the reported 100-step score (0.0660) relies on the state left over from the 10-step run; a freshly initialised ESN scores 0.0731. *Fix:* discard the first ~100 states (`warmup=` in ReservoirPy) and reset state between experiments.
4. **Baselines were only lightly tuned.** The LSTM and SVR settings were chosen by hand, while the ESN got a 200-trial search. A fair comparison would tune every model with the same budget.
5. **Single dataset per task, single seed for the final model.** Results should be averaged over several seeds and, for stocks, several tickers and time periods.

## Acknowledgements

- **[ReservoirPy](https://github.com/reservoirpy/reservoirpy)** (Trouvain et al., Inria, MIT License) provides the ESN implementation, the Mackey-Glass generator and the hyperopt helpers. Several parts of `src/common.py` are adapted from its official tutorials, and each is marked in its docstring:
  - *Tutorial 3, "General Introduction to Reservoir Computing"*: the Mackey-Glass setup and rescaling, `plot_mackey_glass` (used almost unchanged, including the Mackey-Glass time-series/phase-diagram figure), `plot_train_test`, `plot_readout`, `plot_prediction` (from the tutorial's `plot_results`), the default ESN hyperparameters, and the ESN builder.
  - *Tutorial 4, "Understand and optimise hyperparameters"*: the structure of the `objective` function and the hyperopt configuration.

  The ridge/LSTM/SVR baselines, the horizon sweep, the sliding-window pipeline, the Amazon experiment and the comparison plots are my own work.
- Amazon data: P. Kottarathil, [Amazon Stock Price 2014–2019](https://www.kaggle.com/datasets/prasoonkottarathil/amazon-stock-price-20142019), Kaggle.
- Key references: H. Jaeger, *The "echo state" approach to analysing and training recurrent neural networks* (2001); M. Lukoševičius & H. Jaeger, *Reservoir computing approaches to recurrent neural network training* (2009); J. Bergstra & Y. Bengio, *Random search for hyper-parameter optimization* (2012).

## Author

**Muhammad Alief bin Azman**, MEng Electronic & Electrical Engineering, UCL.

Licensed under the MIT License (see [LICENSE](LICENSE)).
