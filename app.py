import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Income Classifier", layout="wide")

st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-label { color: #a6adc8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #cdd6f4; font-size: 2rem; font-weight: 700; }
    .metric-value.good { color: #a6e3a1; }
    .metric-value.warn { color: #f9e2af; }
    </style>
""",
    unsafe_allow_html=True,
)


# ================================
# PREPROCESSING FUNCTION
# ================================
def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df.replace("?", np.nan, inplace=True)
    for col in ["workclass", "occupation", "native_country"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    df.drop(columns=["fnlwgt"], inplace=True, errors="ignore")
    df.drop(columns=["education"], inplace=True, errors="ignore")
    df["capital_net"] = df["capital_gain"] - df["capital_loss"]

    df["work_intensity"] = df["hours_per_week"] * df["education_num"]
    df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    return df


# ================================
# HEADER
# ================================
st.title("Income Classifier")
st.markdown(
    "Upload your **train** and **test** CSV files, configure the models, and explore results."
)
st.divider()

# ================================
# SIDEBAR — FILE UPLOAD + SETTINGS
# ================================
with st.sidebar:
    st.header("Configuration")

    train_file = st.file_uploader("Upload Train CSV", type="csv")
    test_file = st.file_uploader("Upload Test CSV", type="csv")

    st.subheader("Model Settings")
    lr_threshold = st.slider("Logistic Regression Threshold", 0.1, 0.9, 0.35, 0.05)
    dt_threshold = st.slider("Decision Tree Threshold", 0.1, 0.9, 0.4, 0.05)
    dt_max_depth = st.slider("Decision Tree Max Depth", 2, 20, 8)
    use_smote = st.checkbox("Apply SMOTE Balancing", value=True)

    run_btn = st.button("Run Pipeline", use_container_width=True, type="primary")

# ================================
# MAIN PIPELINE
# ================================
if not run_btn:
    st.info("Upload both CSV files and click **Run Pipeline** to start.")
    st.stop()

if train_file is None or test_file is None:
    st.error("Please upload both train and test CSV files.")
    st.stop()

# Load
with st.spinner("Loading data..."):
    train_raw = pd.read_csv(train_file)
    test_raw = pd.read_csv(test_file)
    train_raw.columns = train_raw.columns.str.strip()
    test_raw.columns = test_raw.columns.str.strip()

# ================================
# TAB LAYOUT
# ================================
tab_eda, tab_preprocess, tab_models, tab_compare = st.tabs(
    ["EDA", "Preprocessing", "Models", "Compare"]
)

# ================================
# TAB 1 — EDA
# ================================
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Train rows", f"{len(train_raw):,}")
    col2.metric("Test rows", f"{len(test_raw):,}")
    col3.metric("Features", str(train_raw.shape[1] - 1))

    st.markdown("#### Raw Data Preview")
    st.dataframe(train_raw.head(10), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        train_raw["Income"].value_counts().plot(
            kind="bar", ax=ax, color=["#89b4fa", "#a6e3a1"], edgecolor="none"
        )
        ax.set_title("Income Class Counts", fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown("#### Age Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(train_raw["age"], kde=True, ax=ax, color="#cba6f7", bins=30)
        ax.set_title("Age Distribution", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        train_raw.select_dtypes(include=np.number).corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        linewidths=0.5,
    )
    fig.tight_layout()
    st.pyplot(fig)

# ================================
# TAB 2 — PREPROCESSING
# ================================
with tab_preprocess:
    st.subheader("Preprocessing Steps")

    with st.spinner("Preprocessing..."):
        train = preprocess(train_raw)
        test = preprocess(test_raw)

    assert train["income"].isna().sum() == 0, "NaN in train income!"
    assert test["income"].isna().sum() == 0, "NaN in test income!"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Train — After Preprocessing**")
        st.dataframe(train.head(8), use_container_width=True)
    with c2:
        st.markdown("**Test — After Preprocessing**")
        st.dataframe(test.head(8), use_container_width=True)

    st.markdown("#### Missing Values After Preprocessing")
    missing = train.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values remaining in train set.")
    else:
        st.dataframe(missing.rename("Missing Count"))

    X_train = train.drop("income", axis=1)
    y_train = train["income"]
    X_test = test.drop("income", axis=1)
    y_test = test["income"]

    categorical = X_train.select_dtypes(include="object").columns.tolist()
    numerical = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    before_counts = np.bincount(y_train)
    # ===============
    # SMOTE
    # ===============
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
        after_counts = np.bincount(y_train)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Class Balance Before SMOTE**")
            st.bar_chart(
                pd.DataFrame({"Count": before_counts}, index=["<=50K", ">50K"])
            )
        with c2:
            st.markdown("**Class Balance After SMOTE**")
            st.bar_chart(pd.DataFrame({"Count": after_counts}, index=["<=50K", ">50K"]))
    else:
        st.markdown("**Class Balance (No SMOTE)**")
        st.bar_chart(pd.DataFrame({"Count": before_counts}, index=["<=50K", ">50K"]))

# ================================
# TRAIN MODELS (shared state via session)
# ================================
with st.spinner("Training models..."):
    log_reg = LogisticRegression(
        C=10,
        class_weight=None,
        max_iter=1000,
        solver="liblinear",
        random_state=42,
        penalty="l2",
    )
    log_reg.fit(X_train_proc, y_train)

    dec_tree = DecisionTreeClassifier(
        random_state=42,
        max_depth=15,
        min_samples_leaf=10,
        min_samples_split=15,
        criterion="gini",
        class_weight=None,
        ccp_alpha=0.0001,
    )
    dec_tree.fit(X_train_proc, y_train)


# ================================
# HELPER — evaluate one model
# ================================
def evaluate_model(model, X, y, name, threshold):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba > threshold).astype(int)

    metrics = {
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "Precision": round(precision_score(y, y_pred), 4),
        "Recall": round(recall_score(y, y_pred), 4),
        "F1 Score": round(f1_score(y, y_pred), 4),
        "ROC AUC": round(roc_auc_score(y, y_proba), 4),
    }

    cm = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_proba)

    return metrics, cm, fpr, tpr, y_proba


def show_model_results(model, X, y, name, threshold):
    metrics, cm, fpr, tpr, _ = evaluate_model(model, X, y, name, threshold)

    # Metric cards
    cols = st.columns(5)
    labels = list(metrics.keys())
    vals = list(metrics.values())
    for i, col in enumerate(cols):
        color = "good" if vals[i] >= 0.75 else "warn"
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{labels[i]}</div>
                <div class="metric-value {color}">{vals[i]:.4f}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["<=50K", ">50K"],
            yticklabels=["<=50K", ">50K"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown("**ROC Curve**")
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(fpr, tpr, color="#89b4fa", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    return metrics


# ================================
# TAB 3 — MODEL RESULTS
# ================================
with tab_models:
    st.subheader("Logistic Regression")
    st.caption(f"Threshold: {lr_threshold}")
    lr_metrics = show_model_results(
        log_reg, X_test_proc, y_test, "Logistic Regression", lr_threshold
    )

    st.divider()

    st.subheader("Decision Tree")
    st.caption(f"Threshold: {dt_threshold} · Max Depth: {dt_max_depth}")
    dt_metrics = show_model_results(
        dec_tree, X_test_proc, y_test, "Decision Tree", dt_threshold
    )

# ================================
# TAB 4 — SIDE-BY-SIDE COMPARE
# ================================
with tab_compare:
    st.subheader("Model Comparison")

    compare_df = pd.DataFrame(
        {
            "Logistic Regression": lr_metrics,
            "Decision Tree": dt_metrics,
        }
    )
    st.dataframe(
        compare_df.style.highlight_max(axis=1, color="#a6e3a1"),
        use_container_width=True,
    )

    st.markdown("#### Metric Bar Chart")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(lr_metrics))
    width = 0.35
    ax.bar(
        x - width / 2,
        list(lr_metrics.values()),
        width,
        label="Logistic Regression",
        color="#89b4fa",
    )
    ax.bar(
        x + width / 2,
        list(dt_metrics.values()),
        width,
        label="Decision Tree",
        color="#a6e3a1",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(lr_metrics.keys()))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Logistic Regression vs Decision Tree")
    fig.tight_layout()
    st.pyplot(fig)



    best = (
        "Logistic Regression"
        if lr_metrics["ROC AUC"] >= dt_metrics["ROC AUC"]
        else "Decision Tree"
    )
    st.success(f"**{best}** achieves the higher ROC AUC on the test set.")
