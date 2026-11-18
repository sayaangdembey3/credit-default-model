"""Helper functions for credit_default_model.ipynb.

The notebook tells the story; this file holds the plumbing: reading Freddie Mac's files,
building the default flag, cleaning features, fitting models, computing metrics and
drawing charts. Every function returns aggregate tables or fitted models; nothing here
writes loan-level data anywhere except the gitignored data/ folder.
"""
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# File layout (Freddie Mac General User Guide, July 2026)
# --------------------------------------------------------------------------------------
ORIG_COLS = ["credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date", "msa", "mi_pct",
             "num_units", "occupancy", "cltv", "dti", "orig_upb", "ltv", "orig_rate", "channel", "ppm_flag",
             "amortization_type", "state", "property_type", "postal_code", "loan_id", "loan_purpose", "orig_term",
             "num_borrowers", "seller_name", "super_conforming", "pre_harp_loan_id", "program", "harp_indicator",
             "valuation_method", "io_indicator", "vantage_score"]
# Performance file columns used. 29 = borrower assistance status (F = forbearance, R = repayment plan, T = trial period)
PERF_USECOLS = {0: "loan_id", 1: "period", 3: "delinquency", 8: "zero_balance_code", 29: "assistance"}
LOAN_ID_PATTERN = re.compile(r"^[A-Z]\d{2}Q[1-4]\w+$")

# Model inputs: only fields known on the day the loan is made (nothing from the performance file)
NUMERIC = ["credit_score", "ltv", "cltv", "dti", "orig_rate", "orig_upb", "orig_term", "mi_pct", "num_units", "num_borrowers"]
CATEGORICAL = ["loan_purpose", "occupancy", "property_type", "channel", "first_time_homebuyer", "state"]
FEATURES = NUMERIC + CATEGORICAL
ALLOW_LIST = set(FEATURES)
MONOTONIC = {"credit_score": -1, "ltv": 1, "cltv": 1, "dti": 1}   # boosting: risk falls with score, rises with LTV/CLTV/DTI

MODELED = "Modeled: standard loans, outcome fully observed"
POP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated categorical palette, fixed order

plt.rcParams.update({"figure.figsize": (7, 4), "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------------------
# 1. Reading the files and building the default flag
# --------------------------------------------------------------------------------------
def _find_member(zf, year, kinds):
    names = [n for n in zf.namelist() if str(year) in n and any(k in n.lower() for k in kinds)]
    if not names:
        raise FileNotFoundError(f"No {kinds} file for {year} in {zf.filename}: {zf.namelist()}")
    return names[0]


def load_origination(data_dir, years):
    """One row per loan, as described on the day it was made. Stops if the column layout looks wrong."""
    frames = []
    for year in years:
        with zipfile.ZipFile(Path(data_dir) / f"sample_{year}.zip") as zf:
            with zf.open(_find_member(zf, year, ["orig"])) as f:
                raw = pd.read_csv(f, sep="|", header=None, dtype=str)
        raw = raw.iloc[:, :len(ORIG_COLS)]
        raw.columns = ORIG_COLS
        share_ok = raw["loan_id"].str.match(LOAN_ID_PATTERN).mean()
        if share_ok < 0.99:
            raise ValueError(f"{year}: column 20 doesn't look like a Loan Identifier ({share_ok:.0%} match). "
                             "The file layout may differ from the July 2026 guide; check the column order.")
        raw["vintage"] = year
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def _month_index(yyyymm):
    v = pd.to_numeric(yyyymm, errors="coerce")
    return (v // 100) * 12 + (v % 100)


def load_outcomes(data_dir, years, orig, window_months):
    """Summarize each loan's monthly payment history into a few outcome fields.

    For each loan: the first month (counted from first payment) it was 90+ days delinquent or had a
    credit loss, the same excluding months in forbearance, the first month it left the data for a
    non-credit reason, how many months were seen inside the window, and when reporting began.
    """
    first_pay = pd.Series(_month_index(orig["first_payment_date"]).to_numpy(), index=orig["loan_id"])
    summaries = []
    for year in years:
        events = []
        with zipfile.ZipFile(Path(data_dir) / f"sample_{year}.zip") as zf:
            with zf.open(_find_member(zf, year, ["perf", "svcg"])) as f:
                for chunk in pd.read_csv(f, sep="|", header=None, dtype=str, usecols=list(PERF_USECOLS), chunksize=2_000_000):
                    chunk = chunk.rename(columns=PERF_USECOLS)
                    unexpected = set(chunk["assistance"].dropna().str.strip()) - {"", "F", "R", "T"}
                    if unexpected:
                        raise ValueError(f"{year}: borrower assistance column holds unexpected values {unexpected}; check the file layout")
                    month = _month_index(chunk["period"]) - chunk["loan_id"].map(first_pay)
                    first_reported = month.groupby(chunk["loan_id"]).min().rename("first_reported_month")
                    chunk = chunk.assign(month=month)[(month >= 0) & (month < window_months)]
                    dq = chunk["delinquency"].str.strip()
                    zb = pd.to_numeric(chunk["zero_balance_code"], errors="coerce")
                    is_default = (pd.to_numeric(dq, errors="coerce") >= 3) | (dq == "RA") | zb.isin([2, 3, 9])
                    is_excluded = zb.isin([15, 16, 96])
                    is_default_ex_forb = is_default & ~chunk["assistance"].str.strip().eq("F")
                    g = chunk.assign(default_month=chunk["month"].where(is_default),
                                     exclude_month=chunk["month"].where(is_excluded),
                                     default_ex_forb_month=chunk["month"].where(is_default_ex_forb)).groupby("loan_id")
                    events.append(g.agg(default_month=("default_month", "min"), exclude_month=("exclude_month", "min"),
                                        default_ex_forb_month=("default_ex_forb_month", "min"),
                                        months_seen=("month", "count")).join(first_reported, how="outer"))
        summaries.append(pd.concat(events).groupby(level=0).agg(
            {"default_month": "min", "exclude_month": "min", "default_ex_forb_month": "min",
             "months_seen": "sum", "first_reported_month": "min"}))
    return pd.concat(summaries)


def build_population(orig, outcomes, max_first_report_month, last_dev_year):
    """Attach the default flag and set aside loans the model shouldn't learn from.

    Returns the kept loans plus aggregate tables explaining what was set aside and why.
    """
    loans = orig.merge(outcomes, left_on="loan_id", right_index=True, how="left")
    loans["default"] = loans["default_month"].notna().astype(int)
    loans["default_ex_forbearance"] = loans["default_ex_forb_month"].notna().astype(int)
    loans["harp"] = loans["harp_indicator"].str.strip().eq("Y")

    rules = {
        "HARP refinance loans": loans["harp"],
        f"Reporting starts more than {max_first_report_month} months after first payment":
            loans["first_reported_month"].isna() | (loans["first_reported_month"] > max_first_report_month),
        "No performance records in the 24-month window": loans["months_seen"].fillna(0) == 0,
        "Removed for non-credit reasons (15/16/96) before any default":
            loans["exclude_month"].notna() & (loans["default_month"].isna() | (loans["exclude_month"] < loans["default_month"])),
    }

    # Why HARP is set aside: compare HARP and standard loans (development years, outcome fully observed)
    observed = ~pd.concat(list(rules.values())[1:], axis=1).any(axis=1) & (loans["vintage"] <= last_dev_year)
    dti_num, ltv_num = pd.to_numeric(loans["dti"], errors="coerce"), pd.to_numeric(loans["ltv"], errors="coerce")
    harp_tbl = loans[observed].assign(dti_missing=(dti_num.isna() | (dti_num >= 999))[observed],
                                      ltv_over_100=((ltv_num > 100) & (ltv_num < 999))[observed]) \
        .groupby(loans["harp"].map({True: "HARP", False: "Standard"}))[["default", "dti_missing", "ltv_over_100"]].mean().mul(100).round(2)
    harp_tbl.columns = ["default rate %", "% missing DTI", "% LTV over 100"]
    harp_tbl.index.name = None
    harp_share = (loans.groupby("vintage")["harp"].mean() * 100).round(1).rename("% HARP loans").to_frame().T

    keep = pd.Series(True, index=loans.index)
    group = pd.Series(MODELED, index=loans.index)
    steps = []
    for name, mask in rules.items():
        group[keep & mask] = name
        steps.append({"set aside (in this order)": name, "loans removed": int((keep & mask).sum()),
                      "loans remaining": int((keep & ~mask).sum())})
        keep &= ~mask
    steps_tbl = pd.DataFrame(steps).set_index("set aside (in this order)")
    population = pd.crosstab(loans["vintage"], group).reindex(columns=[MODELED] + list(rules), fill_value=0)
    return loans[keep].copy(), steps_tbl, harp_tbl, harp_share, population


def clean_features(loans):
    """Turn Freddie Mac's 'unknown' codes (e.g. credit score 9999, DTI 999) into proper missing values."""
    d = loans.copy()
    for col in NUMERIC:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d.loc[~d["credit_score"].between(300, 850), "credit_score"] = np.nan
    d.loc[d["dti"] > 65, "dti"] = np.nan
    for col in ["ltv", "cltv", "mi_pct"]:
        d.loc[d[col] >= 999, col] = np.nan
    for col in ["num_units", "num_borrowers"]:
        d.loc[d[col] >= 99, col] = np.nan
    for col in CATEGORICAL:
        d[col] = d[col].fillna("Unknown").str.strip().replace({"9": "Unknown", "99": "Unknown", "": "Unknown"})
    return d


# --------------------------------------------------------------------------------------
# 2. Models and calibration
# --------------------------------------------------------------------------------------
def make_logit():
    """Logistic regression: fill missing values (with a 'was missing' flag), scale numbers, one-hot categories."""
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL)])),
        ("model", LogisticRegression(max_iter=3000)),
    ])


def make_gbm(seed, **params):
    """Gradient boosting with monotonic constraints and early stopping."""
    prep = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)])
    return Pipeline([("prep", prep), ("model", HistGradientBoostingClassifier(
        categorical_features=[False] * len(NUMERIC) + [True] * len(CATEGORICAL),
        monotonic_cst=[MONOTONIC.get(c, 0) for c in NUMERIC] + [0] * len(CATEGORICAL),
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, max_iter=1000,
        random_state=seed, **params))])


def tune_gbm(X_tr, y_tr, X_va, y_va, seed):
    """Try four boosting settings; keep the one with the best AUC on the validation year."""
    search = []
    for leaves in [15, 31]:
        for min_leaf in [200, 1000]:
            m = make_gbm(seed, learning_rate=0.05, max_leaf_nodes=leaves, min_samples_leaf=min_leaf, l2_regularization=1.0).fit(X_tr, y_tr)
            search.append({"learning_rate": 0.05, "max_leaf_nodes": leaves, "min_samples_leaf": min_leaf,
                           "trees": m["model"].n_iter_, "valid AUC": roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]), "model": m})
    best = max(search, key=lambda r: r["valid AUC"])
    table = pd.DataFrame(search).drop(columns="model").sort_values("valid AUC", ascending=False)
    return best["model"], {k: best[k] for k in ["learning_rate", "max_leaf_nodes", "min_samples_leaf", "trees"]}, table


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    """Maps raw model scores to probabilities: 'platt' (smooth curve) or 'isotonic' (step-shaped)."""
    def __init__(self, kind):
        self.kind = kind

    def fit(self, p, y):
        self.m = (IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y) if self.kind == "isotonic"
                  else LogisticRegression(C=1e6).fit(_logit(p), y))
        return self

    def transform(self, p):
        return self.m.predict(p) if self.kind == "isotonic" else self.m.predict_proba(_logit(p))[:, 1]


def choose_calibration(p_valid, y_valid, seed):
    """Fit Platt and isotonic on one half of the validation year, score on the other, refit the winner on all of it."""
    y_valid = np.asarray(y_valid)
    half = np.random.default_rng(seed).random(len(y_valid)) < 0.5
    rows = []
    for kind in ["platt", "isotonic"]:
        c = Calibrator(kind).fit(p_valid[half], y_valid[half])
        rows.append({"method": kind, "Brier on held-out half": brier_score_loss(y_valid[~half], c.transform(p_valid[~half]))})
    rows.append({"method": "none (raw model)", "Brier on held-out half": brier_score_loss(y_valid[~half], p_valid[~half])})
    table = pd.DataFrame(rows).set_index("method")
    kind = table.drop("none (raw model)")["Brier on held-out half"].idxmin()
    return Calibrator(kind).fit(p_valid, y_valid), kind, table


# --------------------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------------------
def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def top_capture(y, p, share=0.2):
    """Share of all defaults found among the riskiest `share` of loans."""
    top = p >= np.quantile(p, 1 - share)
    return y[top].sum() / max(y.sum(), 1)


def metrics(y, p):
    auc = roc_auc_score(y, p)
    return {"ROC-AUC": auc, "Gini": 2 * auc - 1, "KS": ks_stat(y, p), "PR-AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "avg predicted PD %": p.mean() * 100, "actual default rate %": y.mean() * 100}


def bootstrap_auc_gap(y, p_a, p_b, seed, n=300):
    """AUC(a) - AUC(b), with a 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        gaps.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    return roc_auc_score(y, p_a) - roc_auc_score(y, p_b), np.percentile(gaps, [2.5, 97.5])


def risk_band_table(y, p):
    """Ten equal groups by predicted PD (band 1 = riskiest 10%)."""
    band = pd.qcut(pd.Series(p).rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = pd.DataFrame({"band": band, "p": p, "y": y}).groupby("band", observed=True).agg(
        loans=("y", "size"), avg_pd=("p", "mean"), actual=("y", "mean"), defaults=("y", "sum"))
    t["share"] = t["defaults"] / t["defaults"].sum()
    t["cumulative"] = t["share"].cumsum()
    out = t.drop(columns="defaults").copy()
    for col in ["avg_pd", "actual", "share", "cumulative"]:
        out[col] = (out[col] * 100).round(2)
    return out.rename(columns={"avg_pd": "avg predicted PD %", "actual": "actual default rate %",
                               "share": "% of all defaults", "cumulative": "cumulative % of defaults"})


# --------------------------------------------------------------------------------------
# 4. Choosing a cutoff
# --------------------------------------------------------------------------------------
def cutoff_table(y, p, shares=(0.02, 0.05, 0.10, 0.20, 0.30)):
    """For each share of applicants flagged as riskiest: what is caught, and what it costs in good borrowers."""
    rows = []
    for k in shares:
        cut = np.quantile(p, 1 - k)
        flag = p >= cut
        good_flagged = int((flag & (y == 0)).sum())
        rows.append({"flag the riskiest": f"{k:.0%}", "PD cutoff %": cut * 100,
                     "defaults caught %": y[flag].sum() / y.sum() * 100,
                     "good borrowers flagged %": good_flagged / (y == 0).sum() * 100,
                     "good borrowers per default caught": good_flagged / max(y[flag].sum(), 1),
                     "default rate, flagged %": y[flag].mean() * 100,
                     "default rate, approved %": y[~flag].mean() * 100})
    return pd.DataFrame(rows).set_index("flag the riskiest")


def breakeven_table(y, p, balance, lgd_options, margin_options):
    """Decline only when expected loss beats expected profit: PD x LGD > (1 - PD) x margin.

    The breakeven PD is margin / (margin + LGD). LGD and margin are assumptions, not data.
    """
    rows = []
    for lgd in lgd_options:
        for margin in margin_options:
            be = margin / (margin + lgd)
            flag = p >= be
            value = ((y * lgd - (1 - y) * margin) * balance)[flag].sum()
            rows.append({"loss if default (LGD) %": lgd * 100, "profit if repaid, 2 yrs %": margin * 100,
                         "breakeven PD %": be * 100, "applicants above it %": flag.mean() * 100,
                         "defaults caught": int(y[flag].sum()), "defaults in total": int(y.sum()),
                         "net saving vs approving all, $m": value / 1e6})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. Stress test and leakage check
# --------------------------------------------------------------------------------------
def stress_table(loans, years, scenarios, champion, calibrator, challenger):
    rows = []
    for year in years:
        d = loans[loans.vintage == year]
        y, y_ex = d["default"].to_numpy(), d["default_ex_forbearance"].to_numpy()
        raw = champion.predict_proba(d[FEATURES])[:, 1]
        rows.append({
            "year": year, "scenario": scenarios.get(year, "Normal (test)"), "loans": len(d), "defaults": int(y.sum()),
            "actual default rate %": y.mean() * 100, "avg predicted PD %": calibrator.transform(raw).mean() * 100,
            "AUC champion": roc_auc_score(y, raw), "AUC boosting": roc_auc_score(y, challenger.predict_proba(d[FEATURES])[:, 1]),
            "KS champion": ks_stat(y, raw), "top-20% capture %": top_capture(y, raw) * 100,
            "actual rate excl. forbearance %": y_ex.mean() * 100,
            "AUC excl. forbearance": roc_auc_score(y_ex, raw) if 0 < y_ex.sum() < len(y_ex) else np.nan,
            "median rate %": d["orig_rate"].median()})
    return pd.DataFrame(rows).set_index("year")


def overlap_row(model, cal_X, cal_y, n_defaults, X_test, y_test):
    cal = Calibrator("platt").fit(model.predict_proba(cal_X)[:, 1], np.asarray(cal_y))
    raw = model.predict_proba(X_test)[:, 1]
    p = cal.transform(raw)
    return {"training defaults": n_defaults, "2017 AUC": roc_auc_score(y_test, raw),
            "avg predicted PD %": p.mean() * 100, "actual default rate %": y_test.mean() * 100,
            "Brier": brier_score_loss(y_test, p)}


# --------------------------------------------------------------------------------------
# 6. Charts (all aggregates)
# --------------------------------------------------------------------------------------
def plot_population(population, stress_years, last_dev_year, max_first_report_month, path):
    labels = [MODELED, "HARP refinance", f"Reporting starts >{max_first_report_month} months late",
              "No records in 24-month window", "Removed for non-credit reasons"]
    pop = population.copy()
    pop.columns = labels
    share = pop.div(pop.sum(axis=1), axis=0) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    x = np.arange(len(share))
    bottom = np.zeros(len(share))
    for label, color in zip(labels[1:], POP_COLORS[1:]):
        ax1.bar(x, share[label], bottom=bottom, width=0.72, color=color, edgecolor="white", linewidth=1.5)
        bottom += share[label].to_numpy()
    for xi, total_pct in zip(x, bottom):
        ax1.text(xi, total_pct + 0.6, f"{total_pct:.0f}%" if total_pct >= 1 else f"{total_pct:.1f}%",
                 ha="center", fontsize=8.5, color="#52514e")
    ax1.set_xticks(x, [f"{y}\nstress" if y in stress_years else str(y) for y in share.index], fontsize=9)
    ax1.set_ylim(0, bottom.max() * 1.12)
    ax1.set_ylabel("% of that year's loans set aside")
    ax1.set_title("Loans set aside, by origination year", loc="left")

    dev_counts = pop.loc[pop.index <= last_dev_year].sum()
    total = int(dev_counts.sum())
    rows = [(f"Loans read, 2010-{last_dev_year}", 0, total, "#b5b4ad", f"{total:,}")]
    level = total
    for label, color in zip(labels[1:], POP_COLORS[1:]):
        level -= dev_counts[label]
        rows.append((label, level, dev_counts[label], color, f"-{dev_counts[label]:,}"))
    rows.append(("Loans modeled", 0, dev_counts[MODELED], POP_COLORS[0], f"{dev_counts[MODELED]:,}"))
    ypos = np.arange(len(rows))[::-1]
    for yi, (_, left, width, color, text) in zip(ypos, rows):
        ax2.barh(yi, width, left=left, height=0.58, color=color)
        ax2.text(left + width + total * 0.012, yi, text, va="center", fontsize=9, color="#0b0b0b")
    ax2.set_yticks(ypos, [r[0] for r in rows], fontsize=9)
    ax2.set_xlim(0, total * 1.2)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k"))
    ax2.set_xlabel("loans")
    ax2.set_title("Development years: from loans read to loans modeled", loc="left")

    fig.legend([Patch(color=c) for c in POP_COLORS], labels, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_risk_profile(loans, by_year, stress_years, last_dev_year, path):
    dev = loans[loans.vintage <= last_dev_year]
    bands = {
        "credit_score": ([300, 620, 660, 700, 740, 780, 851], ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]),
        "ltv": ([0, 60, 70, 80, 90, 95, 1000], ["<=60", "61-70", "71-80", "81-90", "91-95", ">95"]),
        "dti": ([0, 20, 30, 36, 43, 50, 66], ["<20", "20-29", "30-35", "36-42", "43-49", "50+"]),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].bar(by_year.index.astype(str), by_year["default_rate"] * 100,
                   color=["#E45756" if y in stress_years else "#4C78A8" for y in by_year.index])
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].set_title("Default rate by origination year (red: stress years)")
    axes[0, 0].set_ylabel("% defaulting within 24 months")
    for ax, (col, (edges, labels)) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], bands.items()):
        b = pd.cut(dev[col], edges, labels=labels, right=False)
        rate = dev.groupby(b, observed=True)["default"].mean() * 100
        ax.bar(rate.index.astype(str), rate.values, color="#F58518")
        ax.set_title(f"Default rate by {col.replace('_', ' ')} band, 2010–{last_dev_year}")
        ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_roc_calibration(y, p_logit, p_gbm, p_champ, champion_name, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p, color in [("Logistic regression", p_logit, "#BAB0AC"), ("Gradient boosting", p_gbm, "#4C78A8")]:
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve, 2017 test loans")
    axes[0].legend(loc="lower right")
    for name, p, color in [("Logistic regression (raw)", p_logit, "#BAB0AC"), ("Gradient boosting (raw)", p_gbm, "#F58518"),
                           (f"{champion_name} (calibrated)", p_champ, "#4C78A8")]:
        b = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
        grp = pd.DataFrame({"p": p, "y": y, "b": b}).groupby("b").mean()
        axes[1].plot(grp["p"] * 100, grp["y"] * 100, "o-", color=color, label=name)
    lim = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, lim], [0, lim], "--", color="gray", lw=1, label="Perfect calibration")
    axes[1].set_xlabel("Predicted PD % (decile average)")
    axes[1].set_ylabel("Actual default rate %")
    axes[1].set_title("Calibration, 2017 test loans")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_cutoffs(y, p, cut_tbl, path):
    """Share of defaults caught vs share of applicants flagged, with the cutoff table's points marked."""
    order = np.argsort(-p)
    flagged = np.arange(1, len(p) + 1) / len(p) * 100
    caught = np.cumsum(y[order]) / y.sum() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(flagged, caught, color="#2a78d6", lw=2, label="Model")
    ax.plot([0, 100], [0, 100], "--", color="gray", lw=1, label="No model (random)")
    for label, row in cut_tbl.iterrows():
        share = float(label.rstrip("%"))
        ax.plot(share, row["defaults caught %"], "o", color="#2a78d6", ms=8, mec="white", mew=2)
        ax.annotate(f"flag {label}: catch {row['defaults caught %']:.0f}%\n({row['good borrowers per default caught']:.0f} good per bad)",
                    (share, row["defaults caught %"]), xytext=(10, -26), textcoords="offset points", fontsize=8.5, color="#0b0b0b")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of applicants flagged as riskiest")
    ax.set_ylabel("% of all defaults caught")
    ax.set_title("The trade-off: catching defaults vs flagging good borrowers (2017)", loc="left")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_stress(stress_tbl, test_year, champion_name, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(stress_tbl))
    w = 0.27
    labels = [f"{y}\n{s}" for y, s in zip(stress_tbl.index, stress_tbl["scenario"])]
    axes[0].bar(x - w, stress_tbl["actual default rate %"], w, label="Actual", color="#E45756")
    axes[0].bar(x, stress_tbl["actual rate excl. forbearance %"], w, label="Actual, excl. forbearance", color="#F2A5A4")
    axes[0].bar(x + w, stress_tbl["avg predicted PD %"], w, label="Predicted (frozen model)", color="#4C78A8")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("% defaulting within 24 months")
    axes[0].set_title("Level: predicted vs actual default rate")
    axes[0].legend()
    axes[1].plot(x, stress_tbl["AUC champion"], "o-", color="#4C78A8", label=f"Champion ({champion_name})")
    axes[1].plot(x, stress_tbl["AUC boosting"], "s--", color="#F58518", label="Gradient boosting")
    axes[1].axhline(stress_tbl.loc[test_year, "AUC champion"], color="gray", lw=1, ls=":")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Ranking: AUC by origination year")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_drivers(model, X, y, model_name, seed, importance_path, dependence_path):
    """Permutation importance and partial dependence for credit score, LTV and DTI. Returns the importances."""
    perm = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=3, random_state=seed)
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_xlabel("Drop in test AUC when the feature is shuffled")
    ax.set_title(f"Which features the champion ({model_name}) relies on")
    fig.savefig(importance_path, dpi=120, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    PartialDependenceDisplay.from_estimator(model, X.sample(min(5000, len(X)), random_state=seed),
                                            ["credit_score", "ltv", "dti"], ax=axes, grid_resolution=30)
    fig.suptitle("Partial dependence: predicted risk moves the expected way", y=1.03)
    fig.savefig(dependence_path, dpi=120, bbox_inches="tight")
    plt.show()
    return imp
