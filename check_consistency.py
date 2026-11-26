"""Check that app.py and notebook.ipynb give the same results.

1. Every calculation function shared with the notebook has identical source code.
2. The app's results at the default settings equal the notebook's tables (Tables 1, 4 and 7).
3. At non-default settings the app equals the notebook's `variant_returns`.

Run from the repository root: python check_consistency.py (downloads data, needs internet).
"""
import ast
import json
import os
import tempfile

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

SHARED = ["download_prices", "download_fred", "momentum_signal", "top_n_weights", "equal_weights",
          "run_strategy", "net_returns", "sharpe_ratio", "alpha_beta", "wealth_index", "drawdowns",
          "performance_table", "show"]


def functions(source):
    return {node.name: ast.get_source_segment(source, node)
            for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def main():
    root = os.getcwd()
    cells = [("".join(c["source"])) for c in json.load(open("notebook.ipynb"))["cells"] if c["cell_type"] == "code"]
    notebook_functions = {}
    for cell in cells:
        notebook_functions.update(functions(cell))
    app_source = open("app.py").read()
    app_functions = functions(app_source)

    # 1. identical source
    for name in SHARED:
        assert app_functions[name] == notebook_functions[name], f"{name} differs between app.py and the notebook"
    print(f"1. {len(SHARED)} shared functions are identical in app.py and the notebook")

    # run the notebook in a temporary folder so that its figures do not overwrite the project's
    namespace = {"__name__": "notebook"}
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        for cell in cells:
            exec(compile(cell, "notebook", "exec"), namespace)
        os.chdir(root)

    # load the app without running its page code
    code = app_source.split("# Page\n")[0]
    app = {"__name__": "app"}
    exec(compile(code, "app.py", "exec"), app)
    load_data = app["load_data"].__wrapped__ if hasattr(app["load_data"], "__wrapped__") else app["load_data"]
    prices, returns, rf = load_data()
    sample = app["evaluation_months"](prices, returns)

    def close(a, b, what):
        pd.testing.assert_frame_equal(pd.DataFrame(a).astype(float), pd.DataFrame(b).astype(float),
                                      rtol=1e-9, atol=1e-12, check_names=False, obj=what)

    # 2. defaults
    model = app["build_model"](prices, returns, app["N_HOLD"], app["LOOKBACK"], app["SKIP"], sample)
    monthly = app["monthly_returns"](model, returns, sample, app["COST_BPS"])
    close(monthly, namespace["monthly"], "monthly returns")
    app_table1 = app["performance_table"](monthly[[app["NET"], app["GROSS"], app["BENCHMARK"], app["EW"]]], rf, app["BENCHMARK"])
    close(app_table1, namespace["table1"], "Table 1")
    cost_returns = pd.DataFrame({f"{b} bp": app["net_returns"](model["gross"], model["turnover"], b) for b in app["COST_LEVELS_BPS"]})
    cost_returns[app["BENCHMARK"]] = monthly[app["BENCHMARK"]]
    rows = ["Average return (% p.a.)", "Sharpe ratio", "Alpha (% p.a.)", "Alpha t-stat", "Information ratio"]
    close(app["performance_table"](cost_returns, rf, app["BENCHMARK"]).loc[rows], namespace["table3"], "Table 4 (costs)")
    blend = pd.DataFrame({"SPY only": monthly[app["BENCHMARK"]],
                          "85% SPY + 15% strategy": 0.85 * monthly[app["BENCHMARK"]] + 0.15 * monthly[app["NET"]]})
    theirs = namespace["blend_returns"][["SPY only", "85% SPY + 15% strategy"]]
    close(app["performance_table"](blend, rf, "SPY only"), namespace["performance_table"](theirs, rf, "SPY only"), "satellite blend")
    print("2. default settings: monthly returns, Table 1, the cost table and the 15% satellite blend equal the notebook's")

    # 3. other settings against the notebook's variant_returns, on the common months
    for n, lookback, skip, cost in [(2, 6, 1, 25), (4, 9, 0, 0), (5, 3, 1, 50), (3, 12, 0, 10)]:
        mine = app["build_model"](prices, returns, n, lookback, skip, sample)
        mine_net = app["net_returns"](mine["gross"], mine["turnover"], cost)
        theirs = namespace["variant_returns"](n, lookback, skip, cost_bps=cost).loc[sample]
        assert np.allclose(mine_net, theirs, rtol=1e-9, atol=1e-12), f"mismatch for n={n}, lookback={lookback}, skip={skip}"
    print("3. four non-default settings equal the notebook's variant_returns")
    print("OK: the app is consistent with the notebook")


if __name__ == "__main__":
    main()
