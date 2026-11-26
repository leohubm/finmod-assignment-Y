"""Streamlit app for the US sector momentum strategy (2270 Financial Modeling, NOVA SBE).

The calculation functions in the block "Calculations copied from notebook.ipynb" are identical to the
notebook's. `check_consistency.py` asserts this and compares the app's default results with the
notebook's tables. Never edit that block here only: change the notebook first, then copy.
"""
import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------------------------
# Configuration (same constants as the notebook)
# ---------------------------------------------------------------------------------------------
START_DATE = "1998-12-01"   # the nine original SPDR sector ETFs start trading on 22 Dec 1998
END_DATE = "2026-10-31"     # last full month before the valuation date (inclusive)

SECTOR_NAMES = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
SECTORS = list(SECTOR_NAMES)
BENCHMARK = "SPY"
RF_SERIES = "DGS3MO"   # FRED: 3-month Treasury constant-maturity yield, investment basis, % p.a.

# defaults = the strategy in the prospectus
N_HOLD = 3
LOOKBACK = 12
SKIP = 1
COST_BPS = 10
PERIODS_PER_YEAR = 12

COST_LEVELS_BPS = [0, 10, 25, 50]
SATELLITE_DEFAULT = 15   # percent of the portfolio held in the strategy

NET, GROSS, EW = "Strategy (net)", "Strategy (gross)", "Equal-weight sectors (net)"


# ---------------------------------------------------------------------------------------------
# Calculations copied from notebook.ipynb (unchanged)
# ---------------------------------------------------------------------------------------------
def download_prices(tickers, start, end):
    """Daily adjusted close from Yahoo Finance; `end` is inclusive."""
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=start, end=end_exclusive, auto_adjust=True, progress=False)["Close"]
    return px[tickers]


def download_fred(series_id, start, end):
    """Daily FRED series from the public CSV endpoint (no API key needed); `end` is inclusive."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    series = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
    return series.loc[start:end]


def momentum_signal(prices, lookback, skip):
    """Return from month-end t-lookback to month-end t-skip, known at month-end t."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def top_n_weights(signal, n):
    """Weight 1/n on the n ETFs with the highest signal, 0 elsewhere; NaN while fewer than n have a signal."""
    rank = signal.rank(axis=1, ascending=False, method="first")
    weights = (rank <= n) / n
    enough = signal.notna().sum(axis=1) >= n
    return weights.loc[enough].reindex(signal.index)


def equal_weights(signal):
    """Weight 1/k on each of the k ETFs that have a signal (the ranking universe); NaN while none has."""
    eligible = signal.notna()
    return eligible.div(eligible.sum(axis=1), axis=0)


def run_strategy(returns, weights):
    """Gross monthly return and turnover of a portfolio rebalanced to `weights` at every month-end.

    Weights set at month-end t are held during t+1. Before each rebalance the previous weights have
    drifted with the returns; turnover is the traded value (buys + sells) as a share of the portfolio.
    The first month builds the portfolio from cash, so its turnover is 1.
    """
    held = weights.shift(1).dropna(how="all")
    r = returns.loc[held.index, held.columns]
    assert not (held.gt(0) & r.isna()).any().any(), "a held ETF has no return"
    r = r.fillna(0)
    gross = (held * r).sum(axis=1)
    drifted = (held * (1 + r)).div(1 + gross, axis=0)
    turnover = (held - drifted.shift(1).fillna(0)).abs().sum(axis=1)
    return gross.rename("gross"), turnover.rename("turnover")


def net_returns(gross, turnover, cost_bps):
    """Gross return minus the one-way cost on the traded value."""
    return gross - turnover * cost_bps / 10_000


def sharpe_ratio(r, rf):
    """Annualized Sharpe ratio of monthly returns `r`."""
    excess = r - rf.loc[r.index]
    return PERIODS_PER_YEAR * excess.mean() / (np.sqrt(PERIODS_PER_YEAR) * excess.std())


def alpha_beta(r, bench, rf):
    """OLS of excess return on benchmark excess return: annualized alpha (decimal), beta, t-stat of alpha."""
    rf = rf.loc[r.index]
    assert rf.notna().all(), "missing risk-free rate"
    fit = sm.OLS(r - rf, sm.add_constant(bench.loc[r.index] - rf)).fit()
    return PERIODS_PER_YEAR * fit.params.iloc[0], fit.params.iloc[1], fit.tvalues.iloc[0]


def wealth_index(rets):
    """Growth of 1; a starting value of 1 is added one month before the first return."""
    wealth = (1 + rets).cumprod()
    wealth.loc[rets.index[0] - pd.offsets.MonthEnd(1)] = 1.0
    return wealth.sort_index()


def drawdowns(rets):
    """Fall from the previous peak of the cumulative return, as a negative fraction."""
    wealth = wealth_index(rets)
    return wealth / wealth.cummax() - 1


def performance_table(rets, rf, benchmark):
    """Performance measures, one column per series in `rets`, measured against column `benchmark`."""
    k = PERIODS_PER_YEAR
    table = {}
    for name, r in rets.items():
        if name == benchmark:
            alpha, beta, t_alpha, info = np.nan, np.nan, np.nan, np.nan
        else:
            alpha, beta, t_alpha = alpha_beta(r, rets[benchmark], rf)
            active = r - rets[benchmark]
            info = k * active.mean() / (np.sqrt(k) * active.std())
        table[name] = {
            "Average return (% p.a.)": 100 * k * r.mean(),
            "CAGR (% p.a.)": 100 * ((1 + r).prod() ** (k / len(r)) - 1),
            "Volatility (% p.a.)": 100 * np.sqrt(k) * r.std(),
            "Sharpe ratio": sharpe_ratio(r, rf),
            "Alpha (% p.a.)": 100 * alpha,
            "Alpha t-stat": t_alpha,
            "Beta": beta,
            "Information ratio": info,
            "Max drawdown (%)": 100 * drawdowns(r).min(),
        }
    return pd.DataFrame(table)


def show(table, precision=2):
    """Format a table as text with a dash for undefined entries."""
    def fmt(x):
        if isinstance(x, str):
            return x
        if pd.isna(x):
            return "–"
        return f"{x:d}" if isinstance(x, (int, np.integer)) else f"{x:.{precision}f}"
    return table.map(fmt)


# ---------------------------------------------------------------------------------------------
# App-specific code: data loading, model assembly, charts, layout
# ---------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_data():
    """Month-end prices, simple monthly returns and the lagged monthly risk-free return (as in the notebook)."""
    daily = download_prices(SECTORS + [BENCHMARK], START_DATE, END_DATE)
    yield_daily = download_fred(RF_SERIES, START_DATE, END_DATE)
    prices = daily.resample("ME").last()
    returns = prices / prices.shift(1) - 1
    yield_month_end = yield_daily.resample("ME").last().reindex(prices.index)
    rf = ((1 + yield_month_end / 100) ** (1 / 12) - 1).shift(1).rename("rf")
    return prices, returns, rf


def evaluation_months(prices, returns):
    """Months of the prospectus strategy (12-1, three ETFs). Every setting is evaluated on these same months."""
    base = top_n_weights(momentum_signal(prices[SECTORS], LOOKBACK, SKIP), N_HOLD)
    return run_strategy(returns, base)[0].index


def build_model(prices, returns, n, lookback, skip, sample):
    """Signal, weights and monthly gross returns and turnover of the strategy and the equal-weight sectors."""
    signal = momentum_signal(prices[SECTORS], lookback, skip)
    weights = top_n_weights(signal, n)
    gross, turnover = run_strategy(returns, weights)
    ew_gross, ew_turnover = run_strategy(returns, equal_weights(signal))
    return {
        "signal": signal,
        "weights": weights,
        "gross": gross.loc[sample],
        "turnover": turnover.loc[sample],
        "ew_gross": ew_gross.loc[sample],
        "ew_turnover": ew_turnover.loc[sample],
    }


def monthly_returns(model, returns, sample, cost_bps):
    """The notebook's `monthly` table: net and gross strategy, equal-weight sectors (net) and SPY."""
    monthly = pd.DataFrame({
        NET: net_returns(model["gross"], model["turnover"], cost_bps),
        GROSS: model["gross"],
        EW: net_returns(model["ew_gross"], model["ew_turnover"], cost_bps),
        BENCHMARK: returns[BENCHMARK].loc[sample],
    })
    assert monthly.notna().all().all(), "missing return in the evaluation sample"
    return monthly


# --- palette (Apple system colours) -----------------------------------------------------------
ACCENT, INK, GREY, LIGHT = "#80241E", "#1D1D1F", "#8E8E93", "#C7C7CC"   # accent as in the notebook (Bridgewater red)
FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Helvetica Neue", Arial, sans-serif'

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"], .stApp {{ font-family: {FONT}; color: {INK}; -webkit-font-smoothing: antialiased; }}
.stApp {{ background: #F5F5F7; }}
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"] {{ display: none; }}
.block-container {{ max-width: 1120px; padding: 3.5rem 1.5rem 4rem; }}

.hero {{ display: flex; align-items: center; justify-content: space-between; gap: 2.5rem; }}
.hero-text {{ flex: 1; min-width: 0; }}
.logo {{ width: auto; display: block; flex-shrink: 0; }}
.eyebrow {{ color: {GREY}; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.02em; margin-bottom: 0.4rem; }}
@media (max-width: 700px) {{ .hero {{ flex-direction: column-reverse; align-items: flex-start; gap: 1.2rem; }} .logo {{ height: 64px !important; }} }}
.hero h1 {{ font-size: 3.1rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.05; margin: 0 0 0.7rem; padding: 0; }}
.hero p {{ color: #6E6E73; font-size: 1.25rem; line-height: 1.4; max-width: 46rem; margin: 0 0 2rem; }}

[class*="st-key-card"] {{ background: #FFFFFF; border: none !important; border-radius: 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 28px rgba(0,0,0,0.04); padding: 1.1rem 1.4rem; }}
.label {{ color: #6E6E73; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.15rem; }}

.tile {{ background: #FFFFFF; border-radius: 20px; padding: 1.1rem 1.2rem 1rem; min-height: 8.4rem; box-sizing: border-box;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 28px rgba(0,0,0,0.04); }}
.tile .name {{ color: #6E6E73; font-size: 0.8rem; font-weight: 600; }}
.tile .value {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; margin: 0.2rem 0 0.25rem; }}
.tile .note {{ color: {GREY}; font-size: 0.78rem; line-height: 1.3; }}

h3 {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; padding: 0.2rem 0 0.4rem; }}
.caption {{ color: #6E6E73; font-size: 0.88rem; line-height: 1.45; }}

/* tabs as a segmented control */
.stTabs [role="tablist"] {{ background: #E8E8ED; border-radius: 12px; padding: 3px; gap: 2px; width: fit-content; border: none !important; box-shadow: none; max-width: 100%; overflow-x: auto; }}
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before {{ display: none !important; }}
.stTabs [role="tab"] {{ height: 2.1rem; padding: 0 1.1rem; border-radius: 9px; background: transparent; border: none; position: relative; }}
.stTabs [role="tab"] p {{ font-size: 0.9rem; font-weight: 500; color: #3A3A3C; position: relative; z-index: 1; }}
.stTabs [role="tab"][aria-selected="true"] {{ background: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
.stTabs [role="tab"][aria-selected="true"] p {{ font-weight: 600; color: {INK}; }}
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [role="tabpanel"] {{ padding-top: 1.2rem; }}

/* clean table */
table.clean {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; font-variant-numeric: tabular-nums; }}
table.clean th, table.clean td {{ border-left: none !important; border-right: none !important; border-top: none !important; }}
table.clean th {{ color: #6E6E73; font-weight: 600; font-size: 0.78rem; text-align: right; padding: 0.55rem 0.6rem; border-bottom: 1px solid #E5E5EA; }}
table.clean td {{ text-align: right; padding: 0.6rem 0.6rem; border-bottom: 1px solid #F2F2F7; }}
table.clean th:first-child, table.clean td:first-child {{ text-align: left; color: {INK}; font-weight: 500; }}
table.clean tr:last-child td {{ border-bottom: none; }}
table.clean td.left, table.clean th.left {{ text-align: left; }}
table.clean td.hl {{ color: {ACCENT}; font-weight: 600; }}

[data-testid="stExpander"] {{ border: none; background: #FFFFFF; border-radius: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
.stButton button {{ border-radius: 980px; border: none; background: #E8E8ED; color: {INK}; font-weight: 500; }}
.stButton button:hover {{ background: #DCDCE1; color: {INK}; }}
</style>
"""


def html_table(table, highlight=None, left=()):
    """Render a table of strings (see `show`) as a clean HTML table; the column `highlight` is set in blue."""
    head = "".join(f'<th class="{"left" if c in left else ""}">{html.escape(str(c))}</th>' for c in [""] + list(table.columns))
    body = ""
    for label, row in table.iterrows():
        cells = "".join(f'<td class="{"hl" if c == highlight else "left" if c in left else ""}">{html.escape(str(v))}</td>'
                        for c, v in row.items())
        body += f"<tr><td>{html.escape(str(label))}</td>{cells}</tr>"
    return f'<div style="overflow-x:auto"><table class="clean"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def tile(name, value, notes, min_height=None):
    """Metric card; `notes` is a string or a list of lines."""
    notes = "<br>".join([notes] if isinstance(notes, str) else notes)
    height = f' style="min-height:{min_height}"' if min_height else ""
    return f'<div class="tile"{height}><div class="name">{name}</div><div class="value">{value}</div><div class="note">{notes}</div></div>'


def style_figure(fig, height=380, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=36, b=0), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT, color=INK, size=13), hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)"), showlegend=legend)
    fig.update_xaxes(showgrid=False, showline=True, linecolor=LIGHT, ticks="outside", tickcolor=LIGHT)
    fig.update_yaxes(gridcolor="#EDEDF0", zeroline=False)
    return fig


# line styles as in the notebook: accent for the strategy, black dashed for SPY, grey dash-dot for equal-weight
LINES = {
    NET: dict(color=ACCENT, width=2.6),
    GROSS: dict(color=ACCENT, width=1.4, dash="dot"),
    EW: dict(color="#7F7F7F", width=1.8, dash="dashdot"),
    BENCHMARK: dict(color="black", width=1.8, dash="dash"),
}
BAR_COLORS = {NET: ACCENT, BENCHMARK: "#9A9A9A", EW: "#D0D0D0"}
BENCH_OPTIONS = {"SPY": BENCHMARK, "Equal-weight": EW}   # label in the app -> column


def plot_growth(rets):
    """Growth of 1 on a log axis; every series starts at 1."""
    wealth = wealth_index(rets)
    fig = go.Figure()
    for column in wealth:
        fig.add_scatter(x=wealth.index, y=wealth[column], name=column, line=LINES[column],
                        hovertemplate="$%{y:.2f}")
    ticks = [t for t in (0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100) if wealth.min().min() * 0.9 <= t <= wealth.max().max() * 1.1]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"${t:g}" for t in ticks], title="Value of $1 invested")
    return style_figure(fig)


def with_alpha(hex_color, alpha):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def plot_drawdown(rets):
    """Underwater chart: drawdown from the previous peak, the strategy shaded and the benchmarks as lines."""
    dd = 100 * drawdowns(rets)
    fig = go.Figure()
    for k, column in enumerate(dd):
        shade = dict(fill="tozeroy", fillcolor=with_alpha(ACCENT, 0.25)) if k == 0 else {}
        fig.add_scatter(x=dd.index, y=dd[column], name=column, line=LINES[column], hovertemplate="%{y:.1f}%", **shade)
    fig.update_yaxes(title="Drawdown (%)", ticksuffix="%", range=[1.05 * dd.min().min(), 0])
    return style_figure(fig, height=300)


def plot_years(rets):
    """Return of every calendar year as grouped bars; the last year may be partial."""
    yearly = 100 * ((1 + rets).groupby(rets.index.year).prod() - 1)
    labels = [str(y) for y in yearly.index]
    if rets.index[-1].month != 12:
        labels[-1] += "*"
    fig = go.Figure()
    for column in yearly:
        fig.add_bar(x=labels, y=yearly[column], name=column, marker_color=BAR_COLORS[column],
                    hovertemplate="%{y:.1f}%")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(title="Return in the year (%)", ticksuffix="%")
    fig.update_xaxes(tickangle=-90)
    return style_figure(fig), yearly


def plot_signal(signal_row, held):
    """Latest momentum signal of every ETF in the ranking, the held ones in blue."""
    row = signal_row.dropna().sort_values()
    fig = go.Figure(go.Bar(
        x=100 * row, y=[f"{SECTOR_NAMES[t]} ({t})" for t in row.index], orientation="h",
        marker_color=[ACCENT if t in held else LIGHT for t in row.index], hovertemplate="%{x:.1f}%"))
    fig.update_xaxes(ticksuffix="%", title="Momentum signal")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(hovermode="closest")
    return style_figure(fig, height=420, legend=False)


def plot_relative(rets, reference):
    """Growth of 1 of each portfolio divided by the growth of 1 of the reference portfolio (as Figure 7)."""
    wealth = wealth_index(rets)
    relative = wealth.div(wealth[reference], axis=0).drop(columns=reference)
    fig = go.Figure()
    for column in relative:
        line = LINES.get(column, dict(color="#4D4D4D", width=2.2, dash="dashdot"))
        fig.add_scatter(x=relative.index, y=relative[column], name=column, line=line, hovertemplate="%{y:.2f}")
    fig.add_hline(y=1, line=dict(color="black", width=1))
    fig.update_yaxes(title="Growth relative to SPY only")
    return style_figure(fig, height=340)


def signed(x, digits=1, unit="%"):
    return f"{x:+.{digits}f}{unit}"


def reset_settings():
    st.session_state.update(n=N_HOLD, lookback=LOOKBACK, skip=bool(SKIP), cost=COST_BPS, bench=["SPY"])


# ---------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------
LOGO_FILE = Path(__file__).parent / "assets" / "bridgeriver.png"   # trimmed copy with a transparent background


def logo_html(height_px=64):
    """The logo as an inline image, or nothing if the file is missing."""
    if not LOGO_FILE.exists():
        return ""
    data = base64.b64encode(LOGO_FILE.read_bytes()).decode()
    return f'<img class="logo" alt="BRIDGERIVER" src="data:image/png;base64,{data}" style="height:{height_px}px">'


st.set_page_config(page_title="Sector Momentum", page_icon="📈", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    f'<div class="hero"><div class="hero-text"><div class="eyebrow">2270 Financial Modeling, NOVA SBE</div>'
    "<h1>Sector Momentum</h1>"
    "<p>Each month, hold the SPDR sector ETFs with the strongest trailing return. "
    "Change an assumption and see how performance and the comparison with the benchmarks respond.</p></div>"
    f"{logo_html(120)}</div>",
    unsafe_allow_html=True)

try:
    with st.spinner("Loading prices from Yahoo Finance and FRED…"):
        prices, returns, rf = load_data()
except Exception as error:   # network problems on the first load
    st.error(f"Could not download the data ({error}). Please reload the page in a moment.")
    st.stop()

sample = evaluation_months(prices, returns)

# --- controls ---------------------------------------------------------------------------------
for key, default in dict(n=N_HOLD, lookback=LOOKBACK, skip=bool(SKIP), cost=COST_BPS, bench=["SPY"]).items():
    st.session_state.setdefault(key, default)

with st.container(key="card_controls"):
    c1, c2, c3, c4 = st.columns([1.35, 1.2, 1.2, 1.1], gap="large")
    with c1:
        st.markdown('<div class="label">Sectors held</div>', unsafe_allow_html=True)
        n_hold = st.segmented_control("Sectors held", [2, 3, 4, 5], key="n", label_visibility="collapsed") or N_HOLD
        skip = int(st.toggle("Skip the latest month", key="skip"))
    with c2:
        st.markdown('<div class="label">Lookback (months)</div>', unsafe_allow_html=True)
        lookback = st.slider("Lookback", 3, 12, key="lookback", label_visibility="collapsed")
    with c3:
        st.markdown('<div class="label">One-way trading cost (bp)</div>', unsafe_allow_html=True)
        cost_bps = st.slider("Cost", 0, 50, step=5, key="cost", label_visibility="collapsed")
    with c4:
        st.markdown('<div class="label">Benchmarks</div>', unsafe_allow_html=True)
        bench_choice = st.segmented_control("Benchmarks", list(BENCH_OPTIONS), selection_mode="multi", key="bench",
                                            label_visibility="collapsed") or ["SPY"]
        st.button("Reset to prospectus", on_click=reset_settings)

benches = [BENCH_OPTIONS[label] for label in BENCH_OPTIONS if label in bench_choice]   # SPY first
short = {BENCHMARK: "SPY", EW: "Equal-weight"}
signal_label = f"{lookback}-{skip}"

model = build_model(prices, returns, n_hold, lookback, skip, sample)
monthly = monthly_returns(model, returns, sample, cost_bps)
# one table per benchmark: the strategy's measures against it, and the benchmark's own measures
tables = {b: performance_table(monthly[[NET, b]], rf, b) for b in benches}
first = benches[0]

st.markdown(
    f'<p class="caption" style="margin:0.9rem 0 0.9rem">{signal_label} momentum, top {n_hold} sector ETFs, equal weight, '
    f"{monthly.index[0]:%b %Y} to {monthly.index[-1]:%b %Y} ({len(monthly)} months), net of {cost_bps} bp, "
    f"benchmarks: {', '.join(short[b] for b in benches)}</p>", unsafe_allow_html=True)

# --- the six required measures ----------------------------------------------------------------
def own(measure, fmt):
    """The benchmarks' own value of a measure, one line each."""
    return [f"{short[b]} {fmt(tables[b][b][measure])}" for b in benches]


def versus(measure, fmt, first_note=None):
    """The strategy's value against the first benchmark, and against each further one as a line."""
    lines = [f"vs {short[first]}" + (f", {first_note}" if first_note else "")]
    return lines + [f"vs {short[b]} {fmt(tables[b][NET][measure])}" for b in benches[1:]]


pct, two = (lambda x: f"{x:.1f}%"), (lambda x: f"{x:.2f}")
head = tables[first][NET]
tiles = [
    ("Average return", pct(head["Average return (% p.a.)"]), ["per year"] + own("Average return (% p.a.)", pct)),
    ("Volatility", pct(head["Volatility (% p.a.)"]), ["per year"] + own("Volatility (% p.a.)", pct)),
    ("Sharpe ratio", two(head["Sharpe ratio"]), own("Sharpe ratio", two)),
    ("Alpha", signed(head["Alpha (% p.a.)"]), versus("Alpha (% p.a.)", signed, f"t-stat {head['Alpha t-stat']:.1f}")),
    ("Beta", two(head["Beta"]), versus("Beta", two)),
    ("Information ratio", two(head["Information ratio"]), versus("Information ratio", two)),
]
tile_height = f"{8.4 + 1.1 * (len(benches) - 1)}rem"
for column, (name, value, notes) in zip(st.columns(6, gap="small"), tiles):
    column.markdown(tile(name, value, notes, tile_height), unsafe_allow_html=True)
st.write("")

# --- tabs -------------------------------------------------------------------------------------
tab_perf, tab_years, tab_costs, tab_hold, tab_sat = st.tabs(["Performance", "Calendar years", "Costs", "Holdings", "Satellite"])

chart_columns = [NET] + benches
names = {NET: "the strategy", BENCHMARK: "SPY", EW: "equal-weight sectors"}

with tab_perf:
    with st.container(key="card_growth"):
        st.markdown("### Growth of \\$1")
        st.plotly_chart(plot_growth(monthly[chart_columns]), width="stretch", config={"displayModeBar": False})
        final = wealth_index(monthly[chart_columns]).iloc[-1]
        grew = [f"<b>&#36;{final[c]:.2f}</b> in {names[c]}" for c in chart_columns]
        st.markdown(f'<p class="caption">&#36;1 invested in {monthly.index[0]:%B %Y} grew to {", ".join(grew[:-1])} and {grew[-1]}.</p>',
                    unsafe_allow_html=True)
    st.write("")
    with st.container(key="card_drawdown"):
        st.markdown("### Drawdowns")
        st.plotly_chart(plot_drawdown(monthly[chart_columns]), width="stretch", config={"displayModeBar": False})
    st.write("")
    with st.container(key="card_table"):
        st.markdown("### All measures")
        full = performance_table(monthly[[NET, GROSS, BENCHMARK, EW]], rf, BENCHMARK)
        st.markdown(html_table(show(full), highlight=NET), unsafe_allow_html=True)
        st.markdown(f'<p class="caption" style="margin-top:0.8rem">Alpha, beta and information ratio are measured against {BENCHMARK}. '
                    "Averages are monthly means × 12, volatility is the monthly standard deviation × √12, and the "
                    "Sharpe ratio uses the 3-month Treasury yield (FRED DGS3MO), known at the start of each month. "
                    "Beta is not annualized.</p>", unsafe_allow_html=True)

with tab_years:
    with st.container(key="card_years"):
        st.markdown("### Return in each calendar year")
        fig, yearly = plot_years(monthly[chart_columns])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        beat = [f"{names[b]} in <b>{(yearly[NET] > yearly[b]).sum()} of {len(yearly)}</b>" for b in benches]
        st.markdown(f'<p class="caption">The strategy beat {" and ".join(beat)} calendar years. * year to date.</p>',
                    unsafe_allow_html=True)

with tab_costs:
    with st.container(key="card_costs"):
        st.markdown("### Net of trading costs")
        cost_returns = pd.DataFrame({f"{bps} bp": net_returns(model["gross"], model["turnover"], bps) for bps in COST_LEVELS_BPS})
        cost_returns[BENCHMARK] = monthly[BENCHMARK]
        rows = ["Average return (% p.a.)", "Sharpe ratio", "Alpha (% p.a.)", "Alpha t-stat", "Information ratio"]
        st.markdown(html_table(show(performance_table(cost_returns, rf, BENCHMARK).loc[rows]),
                               highlight=f"{cost_bps} bp"), unsafe_allow_html=True)
        annual_turnover = PERIODS_PER_YEAR * model["turnover"].mean()
        breakeven_bps = (model["gross"].mean() - monthly[BENCHMARK].mean()) / model["turnover"].mean() * 10_000
        st.write("")
        a, b, c = st.columns(3)
        a.markdown(tile("Annual turnover", f"{annual_turnover:.1f}×", "buys plus sells, in portfolio values"), unsafe_allow_html=True)
        b.markdown(tile("Cost drag", f"{100 * annual_turnover * cost_bps / 10_000:.2f}%", f"per year at {cost_bps} bp"), unsafe_allow_html=True)
        c.markdown(tile("Break-even cost", f"{breakeven_bps:.0f} bp", f"average return falls to {BENCHMARK}'s"), unsafe_allow_html=True)
        st.markdown(f'<p class="caption" style="margin-top:1.3rem">The one-way cost is charged on the traded value of every monthly rebalance. '
                    f"{BENCHMARK} is buy-and-hold and pays no costs. The table is measured against {BENCHMARK}.</p>", unsafe_allow_html=True)

with tab_hold:
    latest = model["weights"].iloc[-1]
    held_now = list(latest[latest > 0].index)
    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        with st.container(key="card_signal"):
            st.markdown("### Ranking")
            st.plotly_chart(plot_signal(model["signal"].iloc[-1], held_now), width="stretch", config={"displayModeBar": False})
    with right:
        with st.container(key="card_holdings"):
            st.markdown(f"### Held in {(model['signal'].index[-1] + pd.offsets.MonthEnd(1)):%B %Y}")
            current = pd.DataFrame({"Name": pd.Series(SECTOR_NAMES)[held_now],
                                    "Signal": [f"{100 * model['signal'].iloc[-1][t]:.1f}%" for t in held_now],
                                    "Weight": [f"{100 * latest[t]:.0f}%" for t in held_now]}, index=held_now)
            st.markdown(html_table(current, left=("Name",)), unsafe_allow_html=True)
            st.markdown(f'<p class="caption" style="margin-top:0.8rem">Chosen with prices up to {model["signal"].index[-1]:%d %B %Y}. '
                        "Real Estate and Communication Services join the ranking once they have enough history.</p>", unsafe_allow_html=True)

with tab_sat:
    with st.container(key="card_satellite"):
        st.markdown("### As a satellite next to a core S&P 500 position")
        weight = st.slider("Share of the portfolio in the strategy", 0, 50, SATELLITE_DEFAULT, step=5, format="%d%%") / 100
        blend = pd.DataFrame({
            "SPY only": monthly[BENCHMARK],
            f"{1 - weight:.0%} SPY + {weight:.0%} strategy": (1 - weight) * monthly[BENCHMARK] + weight * monthly[NET],
        })
        rows = ["Average return (% p.a.)", "Volatility (% p.a.)", "Sharpe ratio", "Max drawdown (%)", "Alpha (% p.a.)", "Information ratio"]
        sat = performance_table(blend, rf, "SPY only").loc[rows]
        sat.loc["Tracking error (% p.a.)"] = [
            np.nan if name == "SPY only" else 100 * np.sqrt(PERIODS_PER_YEAR) * (r - blend["SPY only"]).std()
            for name, r in blend.items()]
        st.markdown(html_table(show(sat), highlight=blend.columns[1]), unsafe_allow_html=True)
        st.markdown('<p class="caption" style="margin-top:0.8rem">The blend is rebalanced every month. The cost of that second rebalance is not included.</p>',
                    unsafe_allow_html=True)
    st.write("")
    with st.container(key="card_relative"):
        st.markdown("### Growth relative to SPY alone")
        st.plotly_chart(plot_relative(blend.assign(**{NET: monthly[NET]}), "SPY only"), width="stretch",
                        config={"displayModeBar": False})
        final = wealth_index(blend).iloc[-1]
        st.markdown(f'<p class="caption">Above 1, the portfolio is ahead of SPY alone since {monthly.index[0]:%B %Y}. '
                    f"&#36;1 invested grew to <b>&#36;{final.iloc[1]:.2f}</b> in the blend and <b>&#36;{final.iloc[0]:.2f}</b> in SPY alone.</p>",
                    unsafe_allow_html=True)

st.markdown(
    '<p class="caption" style="margin-top:2.2rem; text-align:center">Data: Yahoo Finance (adjusted close, USD) and FRED. '
    f"Sample {START_DATE[:7]} to {END_DATE[:7]}, monthly. Past performance does not predict future returns. "
    "Educational project, not investment advice.</p>", unsafe_allow_html=True)
