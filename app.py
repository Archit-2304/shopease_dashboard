# Foundations of Big Data Analytics with Python (FBDA)
# FORE School of Management | PGDM-BDA
# Project: Dynamic Analytical Dashboard with Python
#
# Designed for the supplied ShopEase orders sample.
# The application uses Pandas, NumPy, Random, Matplotlib, Seaborn,
# SciPy, Statsmodels and Streamlit, as specified in the project brief.
#
# Run locally:
#   pip install streamlit pandas numpy matplotlib seaborn scipy statsmodels
#   streamlit run fbda_dashboard.py
#
# The app expects cleaned_orders.csv in the same folder by default.
# A CSV uploader is also provided so the app can be run on another machine.

from __future__ import annotations

import io
import os
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ShopEase | FBDA Analytical Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------------
# CONSTANTS / PROJECT METADATA
# -----------------------------------------------------------------------------
PROJECT_TITLE = "Dynamic Analytical Dashboard with Python"
COURSE = "Foundations of Big Data Analytics with Python (FBDA)"
EXPECTED_SAMPLE_SIZE = 2500
PROJECT_SEED = 66100102  # 066_100_102 with underscores removed
DEFAULT_FILE = "cleaned_orders.csv"

REQUIRED_COLUMNS = [
    "OrderID", "CustomerID", "OrderDate", "CustomerAge", "Gender", "City",
    "Category", "Product", "Quantity", "UnitPrice", "Discount",
    "PaymentMethod", "OrderStatus", "DeliveryDate", "Rating", "TotalAmount",
    "DiscountRate", "CalculatedAmount", "AmountMismatch", "InvalidRow",
    "Revenue", "OrderMonth",
]

NUMERIC_COLUMNS = [
    "CustomerAge", "Quantity", "UnitPrice", "Rating", "TotalAmount",
    "DiscountRate", "CalculatedAmount", "Revenue",
]

CATEGORICAL_COLUMNS = [
    "Gender", "City", "Category", "Product", "PaymentMethod", "OrderStatus",
]

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def money(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if abs(x) >= 10_000_000:
        return f"₹{x/10_000_000:.2f} Cr"
    if abs(x) >= 100_000:
        return f"₹{x/100_000:.2f} L"
    return f"₹{x:,.0f}"


def pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:.1f}%"


def safe_div(a: float, b: float) -> float:
    return np.nan if b == 0 else a / b


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def normalise_discount(series: pd.Series) -> pd.Series:
    """Convert mixed discount representations such as 0.10 and 10% to decimal."""
    s = series.astype("string").str.strip()
    result = pd.to_numeric(s.str.rstrip("%"), errors="coerce")
    percent_mask = s.str.endswith("%", na=False)
    result.loc[percent_mask] = result.loc[percent_mask] / 100
    # Values greater than 1 are interpreted as percentages.
    result.loc[(result > 1) & result.notna()] = result.loc[(result > 1) & result.notna()] / 100
    return result.astype(float)


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise dates, discount, numeric columns and create safe analytical fields."""
    df = clean_column_names(raw)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            "The uploaded CSV is missing required columns: " + ", ".join(missing_required)
        )

    # Preserve the supplied fields but make them analytically consistent.
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    df["OrderMonth"] = pd.to_datetime(df["OrderMonth"], errors="coerce")

    for c in [c for c in NUMERIC_COLUMNS if c != "DiscountRate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["DiscountRate"] = normalise_discount(df["Discount"])

    # Recalculate amount from core commercial fields for validation.
    df["RecomputedAmount"] = (
        df["Quantity"] * df["UnitPrice"] * (1 - df["DiscountRate"])
    )
    df["CalculatedMismatchCheck"] = ~np.isclose(
        df["TotalAmount"], df["RecomputedAmount"], rtol=1e-8, atol=1e-8, equal_nan=False
    )

    # Delivery duration is meaningful only when both dates exist.
    df["DeliveryDays"] = (
        (df["DeliveryDate"] - df["OrderDate"]).dt.total_seconds() / 86400
    )

    # Revenue is retained from the supplied dataset. For robustness, if it is
    # absent or completely unusable, fall back to TotalAmount for delivered
    # orders only. The supplied dataset contains Revenue, so this normally does
    # not alter the project data.
    if "Revenue" not in df or df["Revenue"].isna().all():
        df["Revenue"] = np.where(df["OrderStatus"].eq("Delivered"), df["TotalAmount"], 0.0)

    # Helpful derived dimensions.
    df["OrderYear"] = df["OrderDate"].dt.year
    df["OrderMonthName"] = df["OrderDate"].dt.strftime("%b")
    df["OrderWeekday"] = df["OrderDate"].dt.day_name()
    df["IsDelivered"] = df["OrderStatus"].eq("Delivered")
    df["IsCancelled"] = df["OrderStatus"].eq("Cancelled")
    df["IsPending"] = df["OrderStatus"].eq("Pending")
    df["HasRating"] = df["Rating"].notna()
    df["CustomerAgeBand"] = pd.cut(
        df["CustomerAge"],
        bins=[17, 25, 35, 45, 55, 65, 100],
        labels=["18–25", "26–35", "36–45", "46–55", "56–65", "66+"],
        include_lowest=True,
    )

    return df


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes))
    return prepare_data(raw)


@st.cache_data(show_spinner=False)
def load_default_file(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return prepare_data(raw)


def get_filtered_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    if filters["cities"]:
        out = out[out["City"].isin(filters["cities"])]
    if filters["categories"]:
        out = out[out["Category"].isin(filters["categories"])]
    if filters["statuses"]:
        out = out[out["OrderStatus"].isin(filters["statuses"])]
    if filters["genders"]:
        out = out[out["Gender"].isin(filters["genders"])]
    if filters["payments"]:
        out = out[out["PaymentMethod"].isin(filters["payments"])]
    if filters["products"]:
        out = out[out["Product"].isin(filters["products"])]
    if filters["date_range"]:
        if isinstance(filters["date_range"], (tuple, list)) and len(filters["date_range"]) == 2:
            start, end = filters["date_range"]
        else:
            start = end = filters["date_range"]
        out = out[(out["OrderDate"].dt.date >= start) & (out["OrderDate"].dt.date <= end)]
    return out


def confidence_interval_mean(series: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    mean = np.mean(x)
    se = stats.sem(x)
    margin = stats.t.ppf((1 + confidence) / 2, n - 1) * se
    return mean - margin, mean + margin


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUMERIC_COLUMNS:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "Variable": c,
            "Count": int(s.count()),
            "Minimum": s.min(),
            "25th Percentile": s.quantile(.25),
            "Median": s.median(),
            "Mean": s.mean(),
            "75th Percentile": s.quantile(.75),
            "Maximum": s.max(),
            "Range": s.max() - s.min(),
            "Std. Deviation": s.std(ddof=1),
            "Skewness": s.skew(),
            "Kurtosis": s.kurt(),
        })
    return pd.DataFrame(rows)


def category_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].astype("string").fillna("Missing")
    out = s.value_counts(dropna=False).rename_axis("Category").reset_index(name="Frequency")
    out["Relative Frequency"] = out["Frequency"] / len(df) if len(df) else np.nan
    out["Relative Frequency"] = out["Relative Frequency"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    return out


def plot_empty(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.axis("off")
    return fig


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("📊 ShopEase — Dynamic Analytical Dashboard")
st.caption(
    f"{COURSE}  |  {PROJECT_TITLE}  |  Fixed project randomization seed: {PROJECT_SEED:,}"
)

# -----------------------------------------------------------------------------
# DATA SOURCE / SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    st.markdown("**Data source**")

    uploaded_file = st.file_uploader(
        "Upload CSV (optional)",
        type=["csv"],
        help="Upload cleaned_orders.csv or another CSV with the required project columns.",
    )

    st.divider()
    st.subheader("Filters")

# Load data
try:
    if uploaded_file is not None:
        df = load_csv_bytes(uploaded_file.getvalue())
        source_label = uploaded_file.name
    elif os.path.exists(DEFAULT_FILE):
        df = load_default_file(DEFAULT_FILE)
        source_label = DEFAULT_FILE
    else:
        st.error(
            f"Could not find {DEFAULT_FILE}. Upload the cleaned dataset using the sidebar."
        )
        st.stop()
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
min_date = df["OrderDate"].min().date()
max_date = df["OrderDate"].max().date()

with st.sidebar:
    date_range = st.date_input(
        "Order date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    cities = st.multiselect("City", sorted(df["City"].dropna().unique()), default=[])
    categories = st.multiselect("Category", sorted(df["Category"].dropna().unique()), default=[])
    statuses = st.multiselect("Order status", sorted(df["OrderStatus"].dropna().unique()), default=[])
    genders = st.multiselect("Gender", sorted(df["Gender"].dropna().unique()), default=[])
    payments = st.multiselect(
        "Payment method",
        sorted(df["PaymentMethod"].dropna().unique()),
        default=[],
    )
    products = st.multiselect(
        "Product",
        sorted(df["Product"].dropna().unique()),
        default=[],
    )

    st.divider()
    st.caption(f"Source: {source_label}")
    st.caption(f"Loaded records: {len(df):,}")
    st.caption(f"Loaded variables: {df.shape[1]:,}")

filters = {
    "date_range": date_range,
    "cities": cities,
    "categories": categories,
    "statuses": statuses,
    "genders": genders,
    "payments": payments,
    "products": products,
}

fdf = get_filtered_data(df, filters)

# -----------------------------------------------------------------------------
# DATA QUALITY BANNER
# -----------------------------------------------------------------------------
with st.expander("🔎 Data Quality & Project Compliance", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records loaded", f"{len(df):,}", f"vs expected {EXPECTED_SAMPLE_SIZE:,}")
    c2.metric("Duplicate rows", f"{df.duplicated().sum():,}")
    c3.metric("InvalidRow flags", f"{int(df['InvalidRow'].sum()):,}")
    c4.metric("Amount mismatch flags", f"{int(df['AmountMismatch'].sum()):,}")

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        st.write("**Missing values by variable:**")
        st.dataframe(
            pd.DataFrame({
                "Missing Count": missing,
                "Missing %": (missing / len(df) * 100).round(2),
            }),
            use_container_width=True,
        )
    else:
        st.success("No missing values detected.")

    mismatch_rate = df["CalculatedMismatchCheck"].mean() * 100
    st.write(
        f"Independent amount validation: **{int(df['CalculatedMismatchCheck'].sum()):,}** records "
        f"fail the recalculated TotalAmount check ({mismatch_rate:.2f}%)."
    )
    st.info(
        "The uploaded cleaned file contains 2,401 records, whereas the project brief specifies a "
        "2,500-record random sample. The dashboard deliberately reports the actual loaded data "
        "rather than silently inventing or duplicating records. Verify the sampling/cleaning notebook "
        "before final submission if the official deliverable must contain exactly 2,500 rows."
    )

# -----------------------------------------------------------------------------
# KPI CALCULATIONS
# -----------------------------------------------------------------------------
orders = len(fdf)
revenue = fdf["Revenue"].sum()
order_value = fdf["TotalAmount"].mean() if orders else np.nan
avg_rating = fdf["Rating"].mean() if fdf["Rating"].notna().any() else np.nan
total_units = fdf["Quantity"].sum()
delivered = int(fdf["OrderStatus"].eq("Delivered").sum())
cancelled = int(fdf["OrderStatus"].eq("Cancelled").sum())
pending = int(fdf["OrderStatus"].eq("Pending").sum())
delivery_rate = safe_div(delivered, orders) * 100 if orders else np.nan
cancellation_rate = safe_div(cancelled, orders) * 100 if orders else np.nan
rated_orders = int(fdf["Rating"].notna().sum())

# -----------------------------------------------------------------------------
# EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
st.header("1. Executive Overview")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Orders", f"{orders:,}")
k2.metric("Revenue", money(revenue))
k3.metric("Avg. Order Value", money(order_value))
k4.metric("Units Sold", f"{total_units:,}")
k5.metric("Avg. Rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A")
k6.metric("Cancellation Rate", pct(cancellation_rate))

k7, k8, k9 = st.columns(3)
k7.metric("Delivery Rate", pct(delivery_rate))
k8.metric("Pending Orders", f"{pending:,}")
k9.metric("Rated Orders", f"{rated_orders:,}")

st.caption(
    "All KPIs respond to the sidebar filters. Revenue is the supplied Revenue field; "
    "TotalAmount represents order value after discount."
)

# -----------------------------------------------------------------------------
# SALES & REVENUE
# -----------------------------------------------------------------------------
st.header("2. Sales & Revenue Analytics")

left, right = st.columns(2)

with left:
    st.subheader("Revenue Trend")
    monthly = (
        fdf.dropna(subset=["OrderDate"])
        .assign(Month=lambda x: x["OrderDate"].dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Orders=("OrderID", "count"))
    )
    if len(monthly):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.lineplot(data=monthly, x="Month", y="Revenue", marker="o", ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue (₹)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.pyplot(plot_empty("No data for the selected filters."), clear_figure=True)

with right:
    st.subheader("Revenue by Category")
    cat_rev = (
        fdf.groupby("Category", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Orders=("OrderID", "count"))
        .sort_values("Revenue", ascending=False)
    )
    if len(cat_rev):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=cat_rev, x="Revenue", y="Category", ax=ax)
        ax.set_xlabel("Revenue (₹)")
        ax.set_ylabel("")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.pyplot(plot_empty("No data for the selected filters."), clear_figure=True)

left, right = st.columns(2)
with left:
    st.subheader("Revenue by City")
    city_rev = (
        fdf.groupby("City", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Orders=("OrderID", "count"))
        .sort_values("Revenue", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(city_rev):
        sns.barplot(data=city_rev, x="Revenue", y="City", ax=ax)
        ax.set_xlabel("Revenue (₹)")
        ax.set_ylabel("")
    else:
        ax.text(.5, .5, "No data", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Top Products by Revenue")
    prod_rev = (
        fdf.groupby("Product", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"))
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(prod_rev):
        sns.barplot(data=prod_rev, x="Revenue", y="Product", ax=ax)
        ax.set_xlabel("Revenue (₹)")
        ax.set_ylabel("")
    else:
        ax.text(.5, .5, "No data", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# -----------------------------------------------------------------------------
# CUSTOMER & PRODUCT ANALYSIS
# -----------------------------------------------------------------------------
st.header("3. Customer & Product Analytics")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Customer Age Distribution")
    age = fdf["CustomerAge"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(age):
        sns.histplot(age, bins=12, kde=True, ax=ax)
        ax.set_xlabel("Customer Age")
        ax.set_ylabel("Customers / Orders")
    else:
        ax.text(.5, .5, "No age data", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

with c2:
    st.subheader("Rating Distribution")
    rating = fdf["Rating"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(rating):
        sns.countplot(x=rating.astype(int), ax=ax)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Frequency")
    else:
        ax.text(.5, .5, "No rating data", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Revenue vs Unit Price")
    plot_df = fdf[["UnitPrice", "Revenue", "Quantity", "Category"]].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(plot_df):
        sns.scatterplot(data=plot_df, x="UnitPrice", y="Revenue", size="Quantity", alpha=.55, ax=ax)
        ax.set_xlabel("Unit Price (₹)")
        ax.set_ylabel("Revenue (₹)")
    else:
        ax.text(.5, .5, "No data", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

with c2:
    st.subheader("Order Value Distribution")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(fdf):
        sns.boxplot(y=fdf["TotalAmount"], ax=ax)
        ax.set_ylabel("Total Order Amount (₹)")
    else:
        ax.text(.5, .5, "No data", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# -----------------------------------------------------------------------------
# ORDER / OPERATIONS
# -----------------------------------------------------------------------------
st.header("4. Order & Operational Analytics")

c1, c2 = st.columns(2)
with c1:
    status_counts = fdf["OrderStatus"].value_counts().rename_axis("OrderStatus").reset_index(name="Count")
    st.subheader("Order Status")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(status_counts):
        sns.barplot(data=status_counts, x="OrderStatus", y="Count", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Orders")
    else:
        ax.text(.5, .5, "No data", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

with c2:
    st.subheader("Delivery Time Distribution")
    delivery = fdf.loc[fdf["DeliveryDays"].notna() & (fdf["DeliveryDays"] >= 0), "DeliveryDays"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(delivery):
        sns.histplot(delivery, bins=min(15, max(5, delivery.nunique())), kde=True, ax=ax)
        ax.set_xlabel("Delivery Time (days)")
        ax.set_ylabel("Orders")
    else:
        ax.text(.5, .5, "No completed delivery dates", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# -----------------------------------------------------------------------------
# CORRELATION / DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------------
st.header("5. Statistical Analysis")

with st.expander("Descriptive statistics — non-categorical variables", expanded=False):
    summary = make_summary_table(fdf)
    st.dataframe(summary.round(4), use_container_width=True)

with st.expander("Categorical frequency & relative frequency", expanded=False):
    chosen_cat = st.selectbox("Choose categorical variable", CATEGORICAL_COLUMNS, key="cat_summary")
    st.dataframe(category_summary(fdf, chosen_cat), use_container_width=True)

with st.expander("Correlation analysis — Pearson & Spearman", expanded=False):
    corr_cols = [c for c in NUMERIC_COLUMNS if c in fdf.columns]
    corr = fdf[corr_cols].corr(method="pearson")
    spear = fdf[corr_cols].corr(method="spearman")

    st.write("**Pearson correlation**")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.write("**Spearman correlation**")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(spear, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# -----------------------------------------------------------------------------
# INFERENTIAL ANALYSIS
# -----------------------------------------------------------------------------
st.header("6. Inferential Analysis")

# 95% CI for mean TotalAmount
ci_low, ci_high = confidence_interval_mean(fdf["TotalAmount"], 0.95)

ic1, ic2, ic3 = st.columns(3)
ic1.metric("Mean Order Value", money(fdf["TotalAmount"].mean()))
ic2.metric("95% CI — Lower", money(ci_low))
ic3.metric("95% CI — Upper", money(ci_high))

# Statistical test selection
with st.expander("A. Mean comparison — t-test / ANOVA", expanded=False):
    st.write("**Two-group t-test by Gender**")
    groups = [g["TotalAmount"].dropna() for _, g in fdf.groupby("Gender")]
    group_names = [name for name, _ in fdf.groupby("Gender")]
    if len(groups) == 2 and all(len(g) >= 2 for g in groups):
        t_stat, t_p = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
        st.write(pd.DataFrame({
            "Statistic": [t_stat],
            "p-value": [t_p],
            "Decision at α=0.05": ["Reject H0" if t_p < .05 else "Do not reject H0"],
        }).round(5))
        st.caption("H0: mean TotalAmount is equal across the two gender groups. Welch's t-test is used.")
    else:
        st.info("A two-group gender comparison could not be computed for the selected filters.")

    st.write("**One-way ANOVA — TotalAmount across Categories**")
    category_groups = [g["TotalAmount"].dropna() for _, g in fdf.groupby("Category")]
    category_groups = [g for g in category_groups if len(g) >= 2]
    if len(category_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*category_groups)
        st.write(pd.DataFrame({
            "F-statistic": [f_stat],
            "p-value": [p_val],
            "Decision at α=0.05": ["Reject H0" if p_val < .05 else "Do not reject H0"],
        }).round(5))
        st.caption("H0: all category means are equal.")
    else:
        st.info("At least two sufficiently populated categories are required for ANOVA.")

with st.expander("B. Test of variance — Levene", expanded=False):
    variance_groups = [g["TotalAmount"].dropna() for _, g in fdf.groupby("Category")]
    variance_groups = [g for g in variance_groups if len(g) >= 2]
    if len(variance_groups) >= 2:
        lev_stat, lev_p = stats.levene(*variance_groups, center="median")
        st.write(pd.DataFrame({
            "Levene statistic": [lev_stat],
            "p-value": [lev_p],
            "Decision at α=0.05": ["Reject H0" if lev_p < .05 else "Do not reject H0"],
        }).round(5))
        st.caption("H0: category groups have equal variance.")
    else:
        st.info("Insufficient groups for Levene's test.")

with st.expander("C. Normality — Shapiro-Wilk", expanded=False):
    x = fdf["TotalAmount"].dropna()
    # Shapiro-Wilk is most appropriate here because the filtered sample is well below 5,000.
    if 3 <= len(x) <= 5000:
        sh_stat, sh_p = stats.shapiro(x)
        st.write(pd.DataFrame({
            "Shapiro-Wilk statistic": [sh_stat],
            "p-value": [sh_p],
            "Decision at α=0.05": ["Reject H0 (not normal)" if sh_p < .05 else "Do not reject H0"],
        }).round(5))
        st.caption("H0: TotalAmount follows a normal distribution. Statistical significance does not by itself establish practical importance.")
    else:
        st.info("Shapiro-Wilk requires 3–5,000 observations for this implementation.")

with st.expander("D. Chi-square test of independence", expanded=False):
    chi_col1, chi_col2 = st.columns(2)
    with chi_col1:
        chi_a = st.selectbox("Variable A", CATEGORICAL_COLUMNS, index=1, key="chi_a")
    with chi_col2:
        chi_b = st.selectbox("Variable B", CATEGORICAL_COLUMNS, index=5, key="chi_b")

    if chi_a != chi_b:
        contingency = pd.crosstab(fdf[chi_a].fillna("Missing"), fdf[chi_b].fillna("Missing"))
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi_stat, chi_p, dof, expected = stats.chi2_contingency(contingency)
            st.write(pd.DataFrame({
                "Chi-square": [chi_stat],
                "Degrees of freedom": [dof],
                "p-value": [chi_p],
                "Decision at α=0.05": ["Reject H0" if chi_p < .05 else "Do not reject H0"],
            }).round(5))
            st.caption(f"H0: {chi_a} and {chi_b} are independent.")
            st.dataframe(contingency, use_container_width=True)
        else:
            st.info("Both categorical variables need at least two observed categories.")
    else:
        st.warning("Choose two different categorical variables.")

with st.expander("E. Non-parametric robustness check — Kruskal-Wallis", expanded=False):
    kw_groups = [g["TotalAmount"].dropna() for _, g in fdf.groupby("Category")]
    kw_groups = [g for g in kw_groups if len(g) >= 2]
    if len(kw_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*kw_groups)
        st.write(pd.DataFrame({
            "Kruskal-Wallis statistic": [kw_stat],
            "p-value": [kw_p],
            "Decision at α=0.05": ["Reject H0" if kw_p < .05 else "Do not reject H0"],
        }).round(5))
        st.caption("H0: the category groups have the same distribution of TotalAmount.")
    else:
        st.info("Insufficient groups for Kruskal-Wallis.")

# -----------------------------------------------------------------------------
# REGRESSION / MODELING
# -----------------------------------------------------------------------------
st.header("7. Regression & Predictive Analysis")
st.info(
    "Regression coefficients are associations, not proof of causality. In particular, "
    "Revenue/TotalAmount is mechanically related to Quantity, UnitPrice and DiscountRate, "
    "so those variables should not be interpreted as independent causal drivers."
)

reg_df = fdf[[
    "Revenue", "Quantity", "UnitPrice", "DiscountRate", "CustomerAge", "Gender", "Category", "City"
]].dropna().copy()

if len(reg_df) >= 30:
    try:
        model = smf.ols(
            "Revenue ~ Quantity + UnitPrice + DiscountRate + CustomerAge + C(Gender) + C(Category) + C(City)",
            data=reg_df,
        ).fit()

        m1, m2, m3 = st.columns(3)
        m1.metric("Observations", f"{int(model.nobs):,}")
        m2.metric("R²", f"{model.rsquared:.3f}")
        m3.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")

        coef = pd.DataFrame({
            "Coefficient": model.params,
            "Std. Error": model.bse,
            "t-statistic": model.tvalues,
            "p-value": model.pvalues,
        })
        coef["Significant at 5%"] = np.where(coef["p-value"] < .05, "Yes", "No")
        st.dataframe(coef.round(5), use_container_width=True)

        with st.expander("Full Statsmodels OLS summary", expanded=False):
            st.text(model.summary().as_text())
    except Exception as exc:
        st.warning(f"OLS model could not be estimated for the current filters: {exc}")
else:
    st.info("At least 30 complete observations are recommended for the regression section.")

# Logistic regression for cancellation
with st.expander("Logistic regression — probability of cancellation", expanded=False):
    log_df = fdf[[
        "IsCancelled", "Quantity", "UnitPrice", "DiscountRate", "CustomerAge", "Gender", "Category", "City"
    ]].dropna().copy()
    log_df["IsCancelled"] = log_df["IsCancelled"].astype(int)

    if len(log_df) >= 50 and log_df["IsCancelled"].nunique() == 2:
        try:
            log_model = smf.logit(
                "IsCancelled ~ Quantity + UnitPrice + DiscountRate + CustomerAge + C(Gender) + C(Category) + C(City)",
                data=log_df,
            ).fit(disp=False)
            odds = pd.DataFrame({
                "Log-Odds Coefficient": log_model.params,
                "Odds Ratio": np.exp(log_model.params),
                "p-value": log_model.pvalues,
            })
            odds["Significant at 5%"] = np.where(odds["p-value"] < .05, "Yes", "No")
            st.dataframe(odds.round(5), use_container_width=True)
            st.caption(
                "An odds ratio above 1 indicates higher modeled odds of cancellation for a one-unit increase in a numeric predictor, "
                "holding other included variables constant; categorical coefficients are relative to their reference category."
            )
        except Exception as exc:
            st.warning(f"Logistic regression could not be estimated: {exc}")
    else:
        st.info("Both cancellation outcomes and at least 50 usable observations are required.")

# -----------------------------------------------------------------------------
# MANAGERIAL INSIGHTS — DATA-DRIVEN, NOT HARDCODED
# -----------------------------------------------------------------------------
st.header("8. Automated Findings & Managerial Insight Prompts")

insights = []
if len(fdf):
    top_city = fdf.groupby("City")["Revenue"].sum().idxmax()
    top_cat = fdf.groupby("Category")["Revenue"].sum().idxmax()
    top_product = fdf.groupby("Product")["Revenue"].sum().idxmax()
    top_payment = fdf["PaymentMethod"].value_counts(dropna=True).idxmax()
    insights.append(f"Revenue concentration: **{top_city}** is the highest-revenue city within the selected data.")
    insights.append(f"Category performance: **{top_cat}** generates the highest revenue within the selected data.")
    insights.append(f"Product performance: **{top_product}** is the highest-revenue product within the selected data.")
    insights.append(f"Payment behaviour: **{top_payment}** is the most frequently observed non-missing payment method.")
    if cancellation_rate is not np.nan and pd.notna(cancellation_rate):
        insights.append(f"Order operations: the selected data has a **{cancellation_rate:.1f}% cancellation rate**.")

for i, insight in enumerate(insights, start=1):
    st.markdown(f"**{i}.** {insight}")

st.warning(
    "Managerial recommendations should be finalised after the team reviews the statistical evidence, "
    "business context and limitations. The automated text above intentionally describes findings rather "
    "than declaring a single strategic 'winner'."
)

# -----------------------------------------------------------------------------
# DATA DICTIONARY / RAW DATA
# -----------------------------------------------------------------------------
st.header("9. Data Dictionary & Detailed Data")

with st.expander("Data dictionary", expanded=False):
    dictionary = pd.DataFrame([
        ["OrderID", "Unique order identifier", "Integer", "Nominal / Index"],
        ["CustomerID", "Customer identifier", "String", "Nominal"],
        ["OrderDate", "Date and time of order", "DateTime", "Temporal"],
        ["CustomerAge", "Customer age in years", "Decimal", "Ratio"],
        ["Gender", "Customer gender", "String", "Nominal"],
        ["City", "Customer/order city", "String", "Nominal"],
        ["Category", "Product category", "String", "Nominal"],
        ["Product", "Product name", "String", "Nominal"],
        ["Quantity", "Units in the order", "Integer", "Ratio"],
        ["UnitPrice", "Price per unit", "Integer", "Ratio"],
        ["Discount", "Original discount representation", "String", "Nominal / source field"],
        ["PaymentMethod", "Payment method", "String", "Nominal"],
        ["OrderStatus", "Order status", "String", "Nominal"],
        ["DeliveryDate", "Delivery date and time", "DateTime", "Temporal"],
        ["Rating", "Customer rating", "Decimal", "Ordinal"],
        ["TotalAmount", "Order value after discount", "Decimal", "Ratio"],
        ["DiscountRate", "Discount converted to decimal rate", "Decimal", "Ratio"],
        ["CalculatedAmount", "Supplied calculated order amount", "Decimal", "Ratio"],
        ["AmountMismatch", "Supplied amount-validation flag", "Boolean", "Nominal"],
        ["InvalidRow", "Supplied invalid-row flag", "Boolean", "Nominal"],
        ["Revenue", "Revenue field supplied by the dataset", "Decimal", "Ratio"],
        ["OrderMonth", "Month derived/supplied for order", "Date", "Temporal"],
    ], columns=["Variable", "Definition", "Format", "Measurement Type"])
    st.dataframe(dictionary, use_container_width=True)

with st.expander("Filtered data preview", expanded=False):
    st.dataframe(fdf, use_container_width=True, height=450)

# -----------------------------------------------------------------------------
# DOWNLOADS
# -----------------------------------------------------------------------------
st.header("10. Export")

summary_csv = make_summary_table(fdf).to_csv(index=False).encode("utf-8")
filtered_csv = fdf.to_csv(index=False).encode("utf-8")

x1, x2 = st.columns(2)
with x1:
    st.download_button(
        "⬇️ Download descriptive statistics CSV",
        data=summary_csv,
        file_name="fbda_descriptive_statistics.csv",
        mime="text/csv",
        use_container_width=True,
    )
with x2:
    st.download_button(
        "⬇️ Download filtered analytical data CSV",
        data=filtered_csv,
        file_name="fbda_filtered_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "FBDA Project | FORE School of Management | Dynamic Analytical Dashboard with Python | "
    "Analytical results are sample-dependent and should be interpreted with the project's data and sampling limitations."
)
