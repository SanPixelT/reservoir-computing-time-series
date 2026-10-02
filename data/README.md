# Data

## Mackey-Glass (Experiments 1 and 2)

Nothing to download: the series is generated in code with
`reservoirpy.datasets.mackey_glass` (2510 samples, tau = 20).

## Amazon stock price 2014–2019 (Experiment 3)

The dataset isn't included in this repo. Download it from Kaggle:

**[Amazon Stock Price 2014–2019](https://www.kaggle.com/datasets/prasoonkottarathil/amazon-stock-price-20142019)** (P. Kottarathil, 2019)

Save the training file in this folder as:

```
data/AMZNtrain.csv
```

The script only needs an `Open` column with one row per trading day,
oldest first. The original project used 1259 trading days, which gives
1007 training days and 252 test days with the 80/20 split. Any other daily
OHLC price CSV with an `Open` column also works:

```bash
python src/amazon_stock.py --data path/to/your_file.csv
```
