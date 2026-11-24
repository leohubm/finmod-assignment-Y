# US sector momentum: holding the top 3 SPDR sector ETFs

Group project for **2270 Financial Modeling**, NOVA SBE: an investor prospectus (PDF) and a reproducible Python backtest (Jupyter notebook).

## Strategy

- **Universe:** the Select Sector SPDR ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY since 1998; XLRE and XLC join the ranking once they have enough history).
- **Signal:** 12-1 month momentum, i.e. the return from month-end t-12 to t-1, skipping the latest month.
- **Portfolio:** at each month-end the three ETFs with the highest signal are held with equal weight (1/3 each) during the next month.
- **Timing:** the signal uses prices up to month-end t and is applied to returns in month t+1, so there is no look-ahead.
- **Benchmarks:** SPY, and an equal-weight portfolio of all ETFs in the ranking universe.
- **Costs:** one-way trading costs on the traded value, shown at 0, 10, 25 and 50 bp.

## Notebook

`notebook.ipynb` reproduces every number, table and figure of the prospectus. It runs top to bottom in a fresh Google Colab session. All data is downloaded through APIs when it runs (prices from Yahoo Finance via `yfinance`, the 3-month Treasury yield from FRED); no data files are stored in this repository.

To run it locally: Python 3 with `numpy`, `pandas`, `matplotlib`, `statsmodels` and `yfinance`, then open the notebook and run all cells. Figures are saved to a `figures/` folder.

## Layout

```
notebook.ipynb     data, strategy, performance, costs, robustness, limitations
app.py             Streamlit app (to be added)
requirements.txt   dependencies of the app (to be added)
```

## Links

- Streamlit app: _to be added_

## Group

Moodle group name and members: _to be added_
