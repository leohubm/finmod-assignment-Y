# Trend-filtered global ETF portfolio

Group project for **2270 Financial Modeling**, NOVA SBE.
An investor prospectus (PDF) plus a reproducible Python backtest (Jupyter notebook), with an optional Streamlit app.

## Strategy (draft)

- Multi-asset ETF universe (US equity, international equity, bonds, real estate, commodities), with cash as the safe asset.
- Hold an asset only when its month-end price is above its 10-month moving average; otherwise that slice goes to cash.
- Monthly rebalancing; the signal is computed at month-end *t* and applied to returns in *t+1* (no look-ahead).
- Benchmarks: 60/40 (SPY/AGG) and the same assets equal-weighted without the trend filter.

## Planned layout

```
strategy.py        core backtest and metrics, shared by notebook and app
notebook.ipynb     runs end to end in a fresh Google Colab session
app.py             Streamlit app
requirements.txt
```

## Run

Instructions to be added once the notebook and app exist.

## Links

- Streamlit app: _to be added_

## Group

Moodle group name and members: _to be added_
