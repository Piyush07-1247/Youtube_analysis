
# main.py
"""
Streamlit App — YouTube Content Analysis
---------------------------------------
• Upload your own `xlsx` dataset **or** use the bundled synthetic data.
• Explore category-level stats.
• Train Linear & Lasso regression models to predict views.
• Visualize 3‑cluster k‑means on engagement metrics.
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="YouTube Content Analysis", layout="wide")
st.title("📊 YouTube Content Analysis — Regression & Clustering")

# ---------- Data loader ----------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    return pd.read_excel("youtube_data.xlsx")

uploaded = st.sidebar.file_uploader("Upload an Excel dataset", type=["xlsx"])
df = load_data(uploaded)

st.sidebar.write("Dataset rows:", len(df))

# ---------- Feature engineering ----------
df["publish_date"] = pd.to_datetime(df["publish_date"])
df["days_since_publish"] = (datetime.now() - df["publish_date"]).dt.days

# ---------- Sidebar actions ----------
action = st.sidebar.selectbox(
    "Select analysis action",
    ("Show raw data", "Category statistics", "Train regression", "K-means clustering"),
)

# ---------- Show raw data ----------
if action == "Show raw data":
    st.subheader("Raw Dataset")
    st.dataframe(df.head(200))

# ---------- Category statistics ----------
elif action == "Category statistics":
    st.subheader("Average Views per Category")
    category_stats = (
        df.groupby("category")["views"]
        .agg(["mean", "sum", "count"])
        .sort_values("mean", ascending=False)
    )
    st.dataframe(category_stats)

    top_cat = category_stats.index[0]
    st.success(f"🏆 Most-watched on average: **{top_cat}**")

# ---------- Regression ----------
elif action == "Train regression":
    st.subheader("Regression Models — Predicting Views")

    X = df[["likes", "comment_count", "days_since_publish", "category"]]
    y = df["views"]

    numeric_features = ["likes", "comment_count", "days_since_publish"]
    categorical_features = ["category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    linreg_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )
    linreg_pipeline.fit(X_train, y_train)
    linreg_pred = linreg_pipeline.predict(X_test)
    linreg_rmse = np.sqrt(mean_squared_error(y_test, linreg_pred))

    # Lasso Regression
    lasso_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LassoCV(cv=5, random_state=42))]
    )
    lasso_pipeline.fit(X_train, y_train)
    lasso_pred = lasso_pipeline.predict(X_test)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

    st.write(f"**Linear Reg RMSE:** {linreg_rmse:,.0f} views")
    st.write(f"**Lasso Reg RMSE:** {lasso_rmse:,.0f} views")

# ---------- K-means ----------
else:
    st.subheader("K-means Clustering — Category Engagement")

    agg = (
        df.groupby("category")
        .agg({"views": "mean", "likes": "mean", "comment_count": "mean"})
        .reset_index()
    )

    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    agg["cluster"] = kmeans.fit_predict(agg[["views", "likes", "comment_count"]])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        agg["likes"],
        agg["views"],
        c=agg["cluster"],
        s=agg["comment_count"] / 50,
        alpha=0.7,
    )
    ax.set_xlabel("Average Likes")
    ax.set_ylabel("Average Views")
    ax.set_title("Category Clusters (3‑means)")
    for _, row in agg.iterrows():
        ax.text(row["likes"], row["views"], row["category"], fontsize=8)
    st.pyplot(fig)

    st.dataframe(agg)
