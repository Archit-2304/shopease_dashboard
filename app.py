"""
FOUNDATIONS OF BIG DATA ANALYTICS WITH PYTHON (FBDA)
Dynamic Analytical Dashboard with Python - FORE School of Management

Group: GR6 | Roll suffixes: 066_100_102
Fixed randomization seed: 066100102
Required analytical sample: 2,500 random records

Run locally / in Colab/Streamlit:
    streamlit run app.py

The application intentionally uses the libraries specified in the FBDA project brief:
Pandas, NumPy, Random, Matplotlib, Seaborn, SciPy, Statsmodels and Streamlit.
"""

from __future__ import annotations

import io
import os
import random
import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PROJECT CONSTANTS - based on the supplied FBDA project brief + notebook name
# -----------------------------------------------------------------------------
GROUP_ID = "066_100_102"
RANDOM_SEED = 66100102
SAMPLE_SIZE = 2500
DATASET_HINT = "shopease_raw_orders (1).csv"

st.set_page_config(
    page_title="ShopEase | FBDA Analytical Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# STYLING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { padding-top: 1rem; }
        .block-container { max-width: 1450px; padding-top: 1.2rem; }
        .metric-card {
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(128,128,128,.25);
            background: rgba(128,128,128,.06);
        }
        .small-note { font-size: 0.85rem; opacity: .75; }
        h1, h2, h3 { letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column labels without changing the underlying meaning."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return out


def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV robustly from uploaded bytes."""
    errors = []
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return clean_column_names(pd.read_csv(io.BytesIO(file_bytes), encoding=encoding))
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not read the CSV file. Attempts: " + " | ".join(errors))


def infer_date_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in ["date", "time", "timestamp", "created", "ordered", "ship"]):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() >= 0.60:
                candidates.append(c)
    return candidates


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_column_names(df)
    # Convert obvious date/time columns where conversion is reliable.
    for c in infer_date_columns(out):
        converted = pd.to_datetime(out[c], errors="coerce")
        if converted.notna().mean() >= 0.60:
            out[c] = converted
    return out


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def datetime_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").dropna()


def safe_name_match(columns: List[str], keywords: List[str]) -> Optional[str]:
    for c in columns:
        lc = c.lower().replace("_", " ")
        if any(k in lc for k in keywords):
            return c
    return None


def guess_sales_column(df: pd.DataFrame) -> Optional[str]:
    return safe_name_match(
        numeric_columns(df),
        ["sales", "revenue", "amount", "order value", "order_value", "gmv", "total"]
    )


def guess_quantity_column(df: pd.DataFrame) -> Optional[str]:
    return safe_name_match(numeric_columns(df), ["quantity", "qty", "units", "items"])


def guess_discount_column(df: pd.DataFrame) -> Optional[str]:
    return safe_name_match(numeric_columns(df), ["discount", "markdown", "rebate"])


def guess_profit_column(df: pd.DataFrame) -> Optional[str]:
    return safe_name_match(numeric_columns(df), ["profit", "margin", "contribution"])


def format_number(x, decimals=2):
    if pd.isna(x):
        return "—"
    if abs(float(x)) >= 1_000_000:
        return f"{float(x)/1_000_000:.{decimals}f}M"
    if abs(float(x)) >= 1_000:
        return f"{float(x)/1_000:.{decimals}f}K"
    return f"{float(x):,.{decimals}f}"


def confidence_interval_mean(series: pd.Series, confidence: float = 0.95):
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = x.mean()
    se = stats.sem(x)
    lo, hi = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)
    return mean, lo, hi


def p_value_text(p):
    if pd.isna(p):
        return "—"
    return f"{p:.6f}"


def significance_text(p, alpha=0.05):
    if pd.isna(p):
        return "Not available"
    return "Statistically significant at α = 0.05" if p < alpha else "Not statistically significant at α = 0.05"


def detect_binary(series: pd.Series) -> Tuple[bool, Optional[pd.Series]]:
    s = series.dropna()
    unique = list(pd.unique(s))
    if len(unique) != 2:
        return False, None
    return True, s


def make_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_read_file(file_bytes: bytes) -> pd.DataFrame:
    return prepare_dataframe(load_csv_from_bytes(file_bytes))


st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.caption("FBDA • FORE School of Management")

uploaded_file = st.sidebar.file_uploader(
    "Upload the ShopEase CSV dataset",
    type=["csv"],
    help=f"Expected dataset based on the supplied notebook: {DATASET_HINT}",
)

# Try the expected local file if it exists; otherwise require upload.
default_candidates = [
    DATASET_HINT,
    "shopease_raw_orders.csv",
    os.path.join("data", DATASET_HINT),
    os.path.join("data", "shopease_raw_orders.csv"),
]

raw_df = None
source_label = None

if uploaded_file is not None:
    try:
        raw_df = cached_read_file(uploaded_file.getvalue())
        source_label = f"Uploaded: {uploaded_file.name}"
    except Exception as exc:
        st.error(f"Could not load the uploaded dataset: {exc}")
        st.stop()
else:
    for candidate in default_candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "rb") as f:
                    raw_df = cached_read_file(f.read())
                source_label = f"Local file: {candidate}"
                break
            except Exception:
                pass

if raw_df is None:
    st.title("📊 ShopEase — Dynamic Analytical Dashboard")
    st.info(
        "Upload the CSV dataset from the left sidebar to start. "
        "The application will then create the required reproducible 2,500-record analytical sample "
        f"using random seed {RANDOM_SEED:,}."
    )
    st.markdown(
        "### Project configuration\n"
        f"- **Group ID:** `{GROUP_ID}`\n"
        f"- **Random seed / random_state:** `{RANDOM_SEED}`\n"
        f"- **Required analytical sample:** `{SAMPLE_SIZE:,}` records\n"
        "- **Dataset expected from the supplied Colab notebook:** ShopEase raw orders CSV"
    )
    st.stop()

# -----------------------------------------------------------------------------
# FIXED RANDOM SAMPLE - PROJECT REQUIREMENT
# -----------------------------------------------------------------------------
if len(raw_df) < SAMPLE_SIZE:
    st.error(
        f"The uploaded dataset contains only {len(raw_df):,} records, but the FBDA brief requires "
        f"a random sample of {SAMPLE_SIZE:,} records. Please verify that you uploaded the complete dataset."
    )
    st.stop()

# Use both random.seed and random_state exactly as required by the project brief.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

sample_df = raw_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

# -----------------------------------------------------------------------------
# SESSION-LEVEL DATA SUMMARY
# -----------------------------------------------------------------------------
num_cols = numeric_columns(sample_df)
cat_cols = categorical_columns(sample_df)
date_cols = datetime_columns(sample_df)

sales_col = guess_sales_column(sample_df)
quantity_col = guess_quantity_column(sample_df)
discount_col = guess_discount_column(sample_df)
profit_col = guess_profit_column(sample_df)

# Sidebar page navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Executive Overview",
        "🔎 Data Quality & Dictionary",
        "📈 Descriptive Analytics",
        "📊 Categorical Analytics",
        "🔗 Correlation & Visuals",
        "🧪 Inferential Statistics",
        "📐 Regression Analysis",
        "📥 Sample Data",
    ],
)

st.sidebar.divider()
st.sidebar.caption(f"Source: {source_label}")
st.sidebar.caption(f"Original records: {len(raw_df):,}")
st.sidebar.caption(f"Analytical sample: {len(sample_df):,}")
st.sidebar.caption(f"Fixed random_state: {RANDOM_SEED:,}")

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("📊 ShopEase — Dynamic Analytical Dashboard")
st.caption(
    "Foundations of Big Data Analytics with Python (FBDA) • FORE School of Management • "
    f"Group {GROUP_ID}"
)

# -----------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
if page == "🏠 Executive Overview":
    st.subheader("Executive Overview")
    st.write(
        "This dashboard analyses the fixed 2,500-record random sample required by the FBDA project brief. "
        "The sample is reproducible using the group's prescribed randomization seed."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sample Records", f"{len(sample_df):,}")
    c2.metric("Variables", f"{sample_df.shape[1]:,}")
    c3.metric("Numeric Variables", f"{len(num_cols):,}")
    c4.metric("Categorical Variables", f"{len(cat_cols):,}")
    c5.metric("Missing Cells", f"{int(sample_df.isna().sum().sum()):,}")

    if sales_col:
        st.divider()
        st.subheader(f"Primary Business Metric: {sales_col}")
        sales = safe_numeric(sample_df, sales_col)
        a, b, c, d = st.columns(4)
        a.metric("Total", format_number(sales.sum()))
        b.metric("Mean", format_number(sales.mean()))
        c.metric("Median", format_number(sales.median()))
        d.metric("Std. Deviation", format_number(sales.std()))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.histplot(sales, kde=True, ax=ax)
        ax.set_title(f"Distribution of {sales_col}")
        ax.set_xlabel(sales_col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    st.subheader("Automatically detected analytical fields")
    detected = pd.DataFrame(
        {
            "Analytical Role": ["Sales / Revenue", "Quantity", "Discount", "Profit / Margin"],
            "Detected Variable": [sales_col or "Not detected", quantity_col or "Not detected", discount_col or "Not detected", profit_col or "Not detected"],
        }
    )
    st.dataframe(detected, use_container_width=True, hide_index=True)

    st.info(
        "Variable detection is only a convenience for the dashboard. All analytical conclusions should be based on the actual variable definitions and data-quality checks shown in the report."
    )

# -----------------------------------------------------------------------------
# PAGE 2: DATA QUALITY & DICTIONARY
# -----------------------------------------------------------------------------
elif page == "🔎 Data Quality & Dictionary":
    st.subheader("Data Quality & Variable Dictionary")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Original Rows", f"{len(raw_df):,}")
    q2.metric("Sample Rows", f"{len(sample_df):,}")
    q3.metric("Columns", f"{sample_df.shape[1]:,}")
    q4.metric("Duplicate Rows", f"{int(sample_df.duplicated().sum()):,}")

    st.markdown("### Missing Data")
    missing = pd.DataFrame({
        "Variable": sample_df.columns,
        "Missing Count": sample_df.isna().sum().values,
        "Missing %": (sample_df.isna().mean().values * 100),
        "Data Type": sample_df.dtypes.astype(str).values,
        "Unique Values": sample_df.nunique(dropna=True).values,
    }).sort_values("Missing Count", ascending=False)
    st.dataframe(missing, use_container_width=True, hide_index=True)

    st.markdown("### Variable Dictionary")
    dictionary = pd.DataFrame({
        "Variable": sample_df.columns,
        "Python Data Type": [str(sample_df[c].dtype) for c in sample_df.columns],
        "Classification": [
            "Non-Categorical" if pd.api.types.is_numeric_dtype(sample_df[c]) else ("Date/Time" if pd.api.types.is_datetime64_any_dtype(sample_df[c]) else "Categorical")
            for c in sample_df.columns
        ],
        "Unique Values": [sample_df[c].nunique(dropna=True) for c in sample_df.columns],
        "Missing Values": [sample_df[c].isna().sum() for c in sample_df.columns],
        "Example Value": [sample_df[c].dropna().iloc[0] if sample_df[c].notna().any() else "—" for c in sample_df.columns],
    })
    st.dataframe(dictionary, use_container_width=True, hide_index=True)

    st.markdown("### Numerical Summary")
    if num_cols:
        st.dataframe(sample_df[num_cols].describe().T, use_container_width=True)
    else:
        st.warning("No numeric variables were detected.")

# -----------------------------------------------------------------------------
# PAGE 3: DESCRIPTIVE ANALYTICS
# -----------------------------------------------------------------------------
elif page == "📈 Descriptive Analytics":
    st.subheader("Descriptive Statistics — Non-Categorical Data")

    if not num_cols:
        st.warning("No numeric variables were detected in the sample.")
    else:
        selected = st.selectbox("Select numerical variable", num_cols)
        x = safe_numeric(sample_df, selected)

        mean, ci_lo, ci_hi = confidence_interval_mean(x)
        mode_series = x.mode()
        mode_value = mode_series.iloc[0] if not mode_series.empty else np.nan

        stats_table = pd.DataFrame(
            {
                "Statistic": [
                    "Count", "Minimum", "Maximum", "25th Percentile", "Median",
                    "75th Percentile", "Mean", "Mode", "Range", "Standard Deviation",
                    "Skewness", "Kurtosis", "95% CI — Lower", "95% CI — Upper"
                ],
                "Value": [
                    len(x), x.min(), x.max(), x.quantile(.25), x.median(), x.quantile(.75),
                    x.mean(), mode_value, x.max() - x.min(), x.std(),
                    stats.skew(x, bias=False) if len(x) > 2 else np.nan,
                    stats.kurtosis(x, bias=False) if len(x) > 3 else np.nan,
                    ci_lo, ci_hi
                ]
            }
        )
        st.dataframe(stats_table, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.histplot(x, kde=True, ax=ax)
            ax.set_title(f"Histogram — {selected}")
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.boxplot(x=x, ax=ax)
            ax.set_title(f"Box-Whisker Plot — {selected}")
            st.pyplot(fig, clear_figure=True, use_container_width=True)

        if len(num_cols) >= 2:
            st.markdown("### Pairwise Scatter Plot")
            xcol, ycol = st.columns(2)
            x_var = xcol.selectbox("X variable", num_cols, index=0)
            y_var = ycol.selectbox("Y variable", num_cols, index=min(1, len(num_cols)-1))
            plot_df = sample_df[[x_var, y_var]].dropna()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=plot_df, x=x_var, y=y_var, ax=ax)
            ax.set_title(f"{y_var} vs {x_var}")
            st.pyplot(fig, clear_figure=True, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 4: CATEGORICAL ANALYTICS
# -----------------------------------------------------------------------------
elif page == "📊 Categorical Analytics":
    st.subheader("Categorical Data Analysis")

    if not cat_cols:
        st.warning("No categorical variables were detected.")
    else:
        cat = st.selectbox("Select categorical variable", cat_cols)
        counts = sample_df[cat].fillna("Missing").astype(str).value_counts()
        rel = counts / counts.sum()
        summary = pd.DataFrame({
            "Category": counts.index,
            "Frequency": counts.values,
            "Relative Frequency": rel.values,
            "Relative Frequency %": rel.values * 100,
        })

        c1, c2, c3 = st.columns(3)
        c1.metric("Categories", f"{len(counts):,}")
        c2.metric("Highest Frequency", str(counts.index[0]))
        c3.metric("Lowest Frequency", str(counts.index[-1]))

        st.dataframe(summary, use_container_width=True, hide_index=True)

        top_n = st.slider("Number of categories to display", 3, min(25, len(counts)), min(10, len(counts)))
        chart_df = summary.head(top_n).sort_values("Frequency")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=chart_df, x="Frequency", y="Category", ax=ax)
        ax.set_title(f"Top {top_n} Categories by Frequency — {cat}")
        st.pyplot(fig, clear_figure=True, use_container_width=True)

        if len(counts) <= 10:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"Relative Frequency — {cat}")
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.caption("Pie chart is hidden when there are more than 10 categories to preserve readability.")

# -----------------------------------------------------------------------------
# PAGE 5: CORRELATION & VISUALS
# -----------------------------------------------------------------------------
elif page == "🔗 Correlation & Visuals":
    st.subheader("Correlation, Heat Map & Multivariate Visualisation")

    if len(num_cols) < 2:
        st.warning("At least two numeric variables are required for correlation analysis.")
    else:
        method = st.radio("Correlation method", ["Pearson", "Spearman"], horizontal=True)
        corr = sample_df[num_cols].corr(method=method.lower())
        st.dataframe(corr.round(4), use_container_width=True)

        fig, ax = plt.subplots(figsize=(11, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"{method} Correlation Heat Map")
        st.pyplot(fig, clear_figure=True, use_container_width=True)

        if len(num_cols) <= 8:
            st.markdown("### Pair Plot")
            pair_df = sample_df[num_cols].dropna()
            # Limit rows for responsiveness while retaining a reproducible subset.
            if len(pair_df) > 1000:
                pair_df = pair_df.sample(1000, random_state=RANDOM_SEED)
            g = sns.pairplot(pair_df)
            st.pyplot(g.figure, clear_figure=True, use_container_width=True)
        else:
            st.info("Pair plot is suppressed because there are more than 8 numeric variables. Use the heat map instead.")

# -----------------------------------------------------------------------------
# PAGE 6: INFERENTIAL STATISTICS
# -----------------------------------------------------------------------------
elif page == "🧪 Inferential Statistics":
    st.subheader("Inferential Statistics")
    st.caption("Select a test appropriate to the structure of the data and the business question.")

    test = st.selectbox(
        "Select statistical procedure",
        [
            "Mean — 95% Confidence Interval",
            "Independent Samples t-test",
            "One-way ANOVA",
            "Levene Test for Equality of Variances",
            "Pearson Correlation Significance Test",
            "Spearman Correlation Significance Test",
            "Normality — Shapiro-Wilk",
            "Normality — Jarque-Bera",
            "Chi-square Test of Independence",
            "Mann-Whitney U",
            "Kruskal-Wallis",
        ],
    )

    alpha = st.number_input("Significance level (α)", min_value=0.001, max_value=0.20, value=0.05, step=0.01)

    if test == "Mean — 95% Confidence Interval":
        var = st.selectbox("Numerical variable", num_cols)
        x = safe_numeric(sample_df, var)
        mean, lo, hi = confidence_interval_mean(x, confidence=1-alpha)
        st.metric("Sample Mean", format_number(mean))
        st.write(f"{(1-alpha)*100:.1f}% confidence interval: **[{lo:.4f}, {hi:.4f}]**")
        st.write("Interpretation: the interval estimates the population mean under the usual random-sampling assumptions.")

    elif test in ["Independent Samples t-test", "Mann-Whitney U"]:
        if len(num_cols) == 0 or len(cat_cols) == 0:
            st.warning("This test requires at least one numeric and one categorical variable.")
        else:
            var = st.selectbox("Numerical outcome", num_cols)
            group = st.selectbox("Two-group categorical variable", cat_cols)
            work = sample_df[[var, group]].dropna()
            levels = work[group].value_counts().index.tolist()
            if len(levels) != 2:
                st.warning(f"Selected variable '{group}' has {len(levels)} observed groups. Select a categorical variable with exactly two groups.")
            else:
                g1, g2 = levels[0], levels[1]
                a = pd.to_numeric(work.loc[work[group] == g1, var], errors="coerce").dropna()
                b = pd.to_numeric(work.loc[work[group] == g2, var], errors="coerce").dropna()
                if test == "Independent Samples t-test":
                    result = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                else:
                    result = stats.mannwhitneyu(a, b, alternative="two-sided")
                st.write(f"**Group 1:** {g1} (n={len(a):,})")
                st.write(f"**Group 2:** {g2} (n={len(b):,})")
                st.metric("Test statistic", f"{result.statistic:.4f}")
                st.metric("p-value", p_value_text(result.pvalue))
                st.write(significance_text(result.pvalue, alpha))

    elif test == "One-way ANOVA":
        if not num_cols or not cat_cols:
            st.warning("ANOVA requires numeric and categorical variables.")
        else:
            var = st.selectbox("Numerical outcome", num_cols)
            group = st.selectbox("Grouping variable", cat_cols)
            work = sample_df[[var, group]].dropna()
            groups = [pd.to_numeric(g[var], errors="coerce").dropna() for _, g in work.groupby(group)]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                st.warning("At least two groups with sufficient observations are required.")
            else:
                result = stats.f_oneway(*groups)
                st.metric("F-statistic", f"{result.statistic:.4f}")
                st.metric("p-value", p_value_text(result.pvalue))
                st.write(significance_text(result.pvalue, alpha))

    elif test == "Levene Test for Equality of Variances":
        if not num_cols or not cat_cols:
            st.warning("Levene's test requires numeric and categorical variables.")
        else:
            var = st.selectbox("Numerical variable", num_cols)
            group = st.selectbox("Grouping variable", cat_cols)
            work = sample_df[[var, group]].dropna()
            groups = [pd.to_numeric(g[var], errors="coerce").dropna() for _, g in work.groupby(group)]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                st.warning("At least two groups are required.")
            else:
                result = stats.levene(*groups, center="median")
                st.metric("Levene statistic", f"{result.statistic:.4f}")
                st.metric("p-value", p_value_text(result.pvalue))
                st.write(significance_text(result.pvalue, alpha))

    elif test in ["Pearson Correlation Significance Test", "Spearman Correlation Significance Test"]:
        if len(num_cols) < 2:
            st.warning("At least two numeric variables are required.")
        else:
            xcol = st.selectbox("X variable", num_cols, index=0)
            ycol = st.selectbox("Y variable", num_cols, index=min(1, len(num_cols)-1))
            work = sample_df[[xcol, ycol]].dropna()
            if test.startswith("Pearson"):
                result = stats.pearsonr(work[xcol], work[ycol])
            else:
                result = stats.spearmanr(work[xcol], work[ycol])
            st.metric("Correlation", f"{result.statistic:.4f}")
            st.metric("p-value", p_value_text(result.pvalue))
            st.write(significance_text(result.pvalue, alpha))

    elif test.startswith("Normality"):
        if not num_cols:
            st.warning("No numeric variables detected.")
        else:
            var = st.selectbox("Numerical variable", num_cols)
            x = safe_numeric(sample_df, var)
            # Shapiro is generally intended for smaller samples; use a reproducible subset if needed.
            if test.endswith("Shapiro-Wilk"):
                test_x = x if len(x) <= 5000 else x.sample(5000, random_state=RANDOM_SEED)
                result = stats.shapiro(test_x)
            else:
                result = stats.jarque_bera(x)
            st.metric("Test statistic", f"{result.statistic:.4f}")
            st.metric("p-value", p_value_text(result.pvalue))
            st.write(significance_text(result.pvalue, alpha))

    elif test == "Chi-square Test of Independence":
        if len(cat_cols) < 2:
            st.warning("At least two categorical variables are required.")
        else:
            c1 = st.selectbox("Categorical variable 1", cat_cols, index=0)
            c2 = st.selectbox("Categorical variable 2", cat_cols, index=min(1, len(cat_cols)-1))
            contingency = pd.crosstab(sample_df[c1].fillna("Missing"), sample_df[c2].fillna("Missing"))
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            st.dataframe(contingency, use_container_width=True)
            st.metric("Chi-square statistic", f"{chi2:.4f}")
            st.metric("Degrees of freedom", f"{dof}")
            st.metric("p-value", p_value_text(p))
            st.write(significance_text(p, alpha))

    elif test == "Kruskal-Wallis":
        if not num_cols or not cat_cols:
            st.warning("Kruskal-Wallis requires numeric and categorical variables.")
        else:
            var = st.selectbox("Numerical outcome", num_cols)
            group = st.selectbox("Grouping variable", cat_cols)
            work = sample_df[[var, group]].dropna()
            groups = [pd.to_numeric(g[var], errors="coerce").dropna() for _, g in work.groupby(group)]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                st.warning("At least two groups are required.")
            else:
                result = stats.kruskal(*groups)
                st.metric("H-statistic", f"{result.statistic:.4f}")
                st.metric("p-value", p_value_text(result.pvalue))
                st.write(significance_text(result.pvalue, alpha))

# -----------------------------------------------------------------------------
# PAGE 7: REGRESSION
# -----------------------------------------------------------------------------
elif page == "📐 Regression Analysis":
    st.subheader("Regression Analysis")
    st.write(
        "This section provides an OLS regression for a continuous outcome and a logistic regression option for a binary outcome. "
        "Categorical predictors are converted to dummy variables."
    )

    regression_type = st.radio("Model type", ["OLS — Continuous Outcome", "Logistic — Binary Outcome"], horizontal=True)

    if regression_type == "OLS — Continuous Outcome":
        if len(num_cols) < 2:
            st.warning("At least two numeric variables are needed for OLS regression.")
        else:
            default_target = sales_col if sales_col in num_cols else num_cols[0]
            target = st.selectbox("Dependent variable (Y)", num_cols, index=num_cols.index(default_target))
            candidate_predictors = [c for c in sample_df.columns if c != target and (c in num_cols or c in cat_cols)]
            predictors = st.multiselect(
                "Independent variables (X)",
                candidate_predictors,
                default=[c for c in [quantity_col, discount_col, profit_col] if c and c in candidate_predictors][:3],
            )

            if not predictors:
                st.info("Select at least one predictor.")
            else:
                model_df = sample_df[[target] + predictors].copy().dropna()
                y = pd.to_numeric(model_df[target], errors="coerce")
                X_raw = model_df[predictors].copy()
                X = pd.get_dummies(X_raw, drop_first=True, dtype=float)
                X = X.apply(pd.to_numeric, errors="coerce")
                valid = y.notna() & X.notna().all(axis=1)
                y = y.loc[valid]
                X = X.loc[valid]
                if len(y) < max(30, len(X.columns) + 10):
                    st.warning("Insufficient complete observations for a stable regression model.")
                else:
                    X = sm.add_constant(X, has_constant="add")
                    try:
                        model = sm.OLS(y, X).fit()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Observations", f"{int(model.nobs):,}")
                        c2.metric("R²", f"{model.rsquared:.4f}")
                        c3.metric("Adjusted R²", f"{model.rsquared_adj:.4f}")
                        c4.metric("F-test p-value", p_value_text(model.f_pvalue))

                        coef = pd.DataFrame({
                            "Coefficient": model.params,
                            "Std. Error": model.bse,
                            "t-statistic": model.tvalues,
                            "p-value": model.pvalues,
                            "CI Lower": model.conf_int()[0],
                            "CI Upper": model.conf_int()[1],
                        })
                        st.dataframe(coef.round(6), use_container_width=True)

                        with st.expander("Full Statsmodels Regression Summary"):
                            st.text(model.summary().as_text())

                        fig, ax = plt.subplots(figsize=(9, 5))
                        sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax)
                        ax.axhline(0, linestyle="--")
                        ax.set_xlabel("Fitted values")
                        ax.set_ylabel("Residuals")
                        ax.set_title("OLS Residual Plot")
                        st.pyplot(fig, clear_figure=True, use_container_width=True)
                    except Exception as exc:
                        st.error(f"Regression could not be estimated with the selected variables: {exc}")

    else:
        # Logistic regression: require a binary target. We allow categorical or numeric binary targets.
        all_binary_candidates = []
        for c in sample_df.columns:
            non_null = sample_df[c].dropna()
            if len(non_null.unique()) == 2:
                all_binary_candidates.append(c)

        if not all_binary_candidates:
            st.warning("No binary outcome variable was detected in the sample.")
        else:
            target = st.selectbox("Binary dependent variable (Y)", all_binary_candidates)
            candidate_predictors = [c for c in sample_df.columns if c != target and (c in num_cols or c in cat_cols)]
            predictors = st.multiselect("Independent variables (X)", candidate_predictors, default=candidate_predictors[:3])

            if not predictors:
                st.info("Select at least one predictor.")
            else:
                model_df = sample_df[[target] + predictors].dropna().copy()
                y_raw = model_df[target]
                levels = list(pd.unique(y_raw))
                if len(levels) != 2:
                    st.warning("The selected target is no longer binary after removing missing values.")
                else:
                    # Map the first observed level to 0 and the second to 1; labels are shown to the user.
                    y = y_raw.map({levels[0]: 0, levels[1]: 1}).astype(float)
                    X = pd.get_dummies(model_df[predictors], drop_first=True, dtype=float)
                    X = X.apply(pd.to_numeric, errors="coerce")
                    X = sm.add_constant(X, has_constant="add")
                    valid = y.notna() & X.notna().all(axis=1)
                    y, X = y.loc[valid], X.loc[valid]
                    if y.nunique() < 2:
                        st.warning("The target has only one class in the usable observations.")
                    else:
                        try:
                            model = sm.Logit(y, X).fit(disp=False)
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Observations", f"{int(model.nobs):,}")
                            c2.metric("McFadden-style pseudo R²", f"{model.prsquared:.4f}")
                            c3.metric("LR-test p-value", p_value_text(model.llr_pvalue))

                            coef = pd.DataFrame({
                                "Coefficient": model.params,
                                "Odds Ratio": np.exp(model.params),
                                "Std. Error": model.bse,
                                "z-statistic": model.tvalues,
                                "p-value": model.pvalues,
                            })
                            st.dataframe(coef.round(6), use_container_width=True)
                            st.caption(f"Outcome coding: 0 = {levels[0]!r}; 1 = {levels[1]!r}")
                            with st.expander("Full Statsmodels Logistic Regression Summary"):
                                st.text(model.summary().as_text())
                        except Exception as exc:
                            st.error(f"Logistic regression could not be estimated: {exc}")

# -----------------------------------------------------------------------------
# PAGE 8: SAMPLE DATA
# -----------------------------------------------------------------------------
elif page == "📥 Sample Data":
    st.subheader("Reproducible Analytical Sample")
    st.write(
        f"The dashboard uses exactly {SAMPLE_SIZE:,} records selected from the uploaded dataset with "
        f"`random_state = {RANDOM_SEED}`. This is fixed for reproducibility."
    )
    st.dataframe(sample_df.head(100), use_container_width=True, height=600)

    st.download_button(
        "⬇️ Download the 2,500-record analytical sample",
        data=make_download_csv(sample_df),
        file_name="shopease_fbda_sample_2500.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "FBDA Project • Dynamic Analytical Dashboard with Python • "
    f"Group {GROUP_ID} • Fixed randomization seed {RANDOM_SEED} • Sample size {SAMPLE_SIZE:,}"
)
