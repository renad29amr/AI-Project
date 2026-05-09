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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
)

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Income Classifier", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background: #1e1e2e; border: 1px solid #313244;
        border-radius: 10px; padding: 1rem 1.5rem; text-align: center;
    }
    .metric-label { color: #a6adc8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #cdd6f4; font-size: 2rem; font-weight: 700; }
    .metric-value.good { color: #a6e3a1; }
    .metric-value.warn { color: #f9e2af; }
    .predict-box {
        background: #1e1e2e; border: 2px solid #313244;
        border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1rem;
    }
    .predict-result-high { font-size: 2.5rem; font-weight: 800; color: #a6e3a1; }
    .predict-result-low  { font-size: 2.5rem; font-weight: 800; color: #f38ba8; }
    .predict-confidence  { font-size: 1rem; color: #a6adc8; margin-top: 0.5rem; }
    .predict-bar-wrap {
        background: #313244; border-radius: 999px; height: 14px;
        margin: 1rem auto; max-width: 400px; overflow: hidden;
    }
    .predict-bar-fill { height: 100%; border-radius: 999px; }
    </style>
""", unsafe_allow_html=True)


# ================================
# PREPROCESSING
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
    df.drop(columns=["native_country"], inplace=True, errors="ignore")
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
st.markdown("Upload your **train** and **test** CSV files, configure the models, and explore results.")
st.divider()

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("Configuration")
    train_file = st.file_uploader("Upload Train CSV", type="csv")
    test_file  = st.file_uploader("Upload Test CSV",  type="csv")

    st.subheader("Model Settings")
    lr_threshold = st.slider("Logistic Regression Threshold", 0.1, 0.9, 0.4, 0.05)
    dt_threshold = st.slider("Decision Tree Threshold",        0.1, 0.9, 0.4, 0.05)
    dt_max_depth = st.slider("Decision Tree Max Depth",  2, 20, 10)
    rf_threshold = st.slider("Random Forest Threshold",        0.1, 0.9, 0.6, 0.05)
    rf_n_estimators = st.slider("Random Forest Trees",  50, 500, 200, 50)
    rf_max_depth = st.slider("Random Forest Max Depth", 2, 20, 20)
    xgb_threshold = st.slider("XG boost Threshold", 0.1, 0.9, 0.5, 0.05)
    xgb_n_estimators = st.slider("XGBoost Rounds",      50, 500, 200, 50)
    xgb_lr = st.slider("XGBoost Learning Rate",         0.01, 0.3, 0.1, 0.01)
    xgb_max_depth = st.slider("XGBoost Max Depth",      2, 10, 5)

    st.markdown("---")
    # use_smote = st.checkbox("Apply SMOTE Balancing", value=True)
    run_btn   = st.button("Run Pipeline", use_container_width=True, type="primary")

# ================================
# SESSION STATE
# ================================
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if run_btn:
    if train_file is None or test_file is None:
        st.error("Please upload both train and test CSV files.")
        st.stop()

    with st.spinner("Loading, preprocessing and training all 4 models..."):
        train_raw = pd.read_csv(train_file)
        test_raw  = pd.read_csv(test_file)
        train_raw.columns = train_raw.columns.str.strip()
        test_raw.columns  = test_raw.columns.str.strip()

        train = preprocess(train_raw)
        test  = preprocess(test_raw)


        X_train = train.drop("income", axis=1)
        y_train = train["income"]
        X_test  = test.drop("income",  axis=1)
        y_test  = test["income"]

        categorical = X_train.select_dtypes(include="object").columns.tolist()
        numerical   = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

        col_prep = ColumnTransformer([
            ("num", StandardScaler(),                       numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        X_train_proc  = col_prep.fit_transform(X_train)
        X_test_proc   = col_prep.transform(X_test)
        # before_counts = np.bincount(y_train)

        # if use_smote:
        #     smote = SMOTE(random_state=42)
        #     X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
        # after_counts = np.bincount(y_train)

        # ── Logistic Regression ───────────────────────────────────────────
        log_reg = LogisticRegression(C=1, max_iter=1000, solver="liblinear",
                                     random_state=42, penalty="l2",class_weight=None)
        log_reg.fit(X_train_proc, y_train)

        # ── Decision Tree ─────────────────────────────────────────────────
        dec_tree = DecisionTreeClassifier(
            random_state=42, max_depth=dt_max_depth,
            min_samples_leaf=5, min_samples_split=2,
            criterion="entropy", ccp_alpha=0.0001)
        dec_tree.fit(X_train_proc, y_train)

        # ── Random Forest ─────────────────────────────────────────────────
        rand_forest = RandomForestClassifier(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth,
            min_samples_leaf=5, min_samples_split=5,
            random_state=42, n_jobs=-1,class_weight="balanced",max_features="sqrt")
        rand_forest.fit(X_train_proc, y_train)

        # ── XGBoost ───────────────────────────────────────────────────────
        xgb = XGBClassifier(
            n_estimators=xgb_n_estimators, learning_rate=xgb_lr,
            max_depth=xgb_max_depth,objective='binary:logistic')
        xgb.fit(X_train_proc, y_train)

        # ── Cache ─────────────────────────────────────────────────────────
        st.session_state.update({
            "train_raw": train_raw, "test_raw": test_raw,
            "train": train, "test": test,
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test,   "y_test": y_test,
            "X_train_proc": X_train_proc, "X_test_proc": X_test_proc,
            "col_prep": col_prep,
            # "before_counts": before_counts, "after_counts": after_counts,
            # "use_smote": use_smote,
            "log_reg": log_reg, "dec_tree": dec_tree,
            "rand_forest": rand_forest, "xgb": xgb,
            "pipeline_ready": True,
        })

if not st.session_state.pipeline_ready:
    st.info("Upload both CSV files and click **Run Pipeline** to start.")
    st.stop()

# ── Restore ───────────────────────────────────────────────────────────────
train_raw     = st.session_state.train_raw
test_raw      = st.session_state.test_raw
train         = st.session_state.train
test          = st.session_state.test
X_train       = st.session_state.X_train
y_train       = st.session_state.y_train
X_test        = st.session_state.X_test
y_test        = st.session_state.y_test
X_train_proc  = st.session_state.X_train_proc
X_test_proc   = st.session_state.X_test_proc
col_prep      = st.session_state.col_prep
# before_counts = st.session_state.before_counts
# after_counts  = st.session_state.after_counts
# use_smote     = st.session_state.use_smote
log_reg       = st.session_state.log_reg
dec_tree      = st.session_state.dec_tree
rand_forest   = st.session_state.rand_forest
xgb           = st.session_state.xgb

# ================================
# TABS
# ================================
tab_eda, tab_pre, tab_models, tab_compare, tab_predict = st.tabs(
    ["EDA", "Preprocessing", "Models", "Compare", "Predict"]
)

# ================================
# TAB 1 — EDA
# ================================
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train rows", f"{len(train_raw):,}")
    c2.metric("Test rows",  f"{len(test_raw):,}")
    c3.metric("Features",   str(train_raw.shape[1] - 1))

    st.markdown("#### Raw Data Preview")
    st.dataframe(train_raw.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        train_raw["Income"].value_counts().plot(kind="bar", ax=ax,
            color=["#89b4fa", "#a6e3a1"], edgecolor="none")
        ax.set_title("Income Class Counts", fontsize=11)
        ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
        fig.tight_layout(); st.pyplot(fig)

    with c2:
        st.markdown("#### Age Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(train_raw["age"], kde=True, ax=ax, color="#cba6f7", bins=30)
        ax.set_title("Age Distribution", fontsize=11)
        fig.tight_layout(); st.pyplot(fig)

    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(train_raw.select_dtypes(include=np.number).corr(),
                annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    fig.tight_layout(); st.pyplot(fig)

# ================================
# TAB 2 — PREPROCESSING
# ================================
with tab_pre:
    st.subheader("Preprocessing Steps")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Train — After Preprocessing**")
        st.dataframe(train.head(8), use_container_width=True)
    with c2:
        st.markdown("**Test — After Preprocessing**")
        st.dataframe(test.head(8), use_container_width=True)


    # if use_smote:
    #     c1, c2 = st.columns(2)
    #     with c1:
    #         st.markdown("**Class Balance Before SMOTE**")
    #         st.bar_chart(pd.DataFrame({"Count": before_counts}, index=["<=50K", ">50K"]))
    #     with c2:
    #         st.markdown("**Class Balance After SMOTE**")
    #         st.bar_chart(pd.DataFrame({"Count": after_counts},  index=["<=50K", ">50K"]))
    # else:
    #     st.markdown("**Class Balance**")
    #     st.bar_chart(pd.DataFrame({"Count": before_counts}, index=["<=50K", ">50K"]))



def evaluate_model(model, X, y, threshold):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba > threshold).astype(int)
    metrics = {
        "Accuracy":  round(accuracy_score(y, y_pred),  4),
        "Precision": round(precision_score(y, y_pred), 4),
        "Recall":    round(recall_score(y, y_pred),    4),
        "F1 Score":  round(f1_score(y, y_pred),        4),
        "ROC AUC":   round(roc_auc_score(y, y_proba),  4),
    }
    fpr, tpr, _ = roc_curve(y, y_proba)
    return metrics, confusion_matrix(y, y_pred), fpr, tpr


def show_model_results(model, X, y, threshold):
    metrics, cm, fpr, tpr = evaluate_model(model, X, y, threshold)

    cols = st.columns(5)
    for i, (label, val) in enumerate(metrics.items()):
        color = "good" if val >= 0.75 else "warn"
        cols[i].markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value {color}">{val:.4f}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout(); st.pyplot(fig)
    with c2:
        st.markdown("**ROC Curve**")
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(fpr, tpr, color="#89b4fa", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
        fig.tight_layout(); st.pyplot(fig)

    return metrics


# ================================
# TAB 3 — MODELS
# ================================
with tab_models:
    model_tabs = st.tabs([
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
    ])

    with model_tabs[0]:
        st.caption(f"Threshold: {lr_threshold}")
        lr_metrics = show_model_results(log_reg, X_test_proc, y_test, lr_threshold)

    with model_tabs[1]:
        st.caption(f"Threshold: {dt_threshold} · Max Depth: {dt_max_depth}")
        dt_metrics = show_model_results(dec_tree, X_test_proc, y_test, dt_threshold)

    with model_tabs[2]:
        st.caption(f"Threshold: {rf_threshold} · Trees: {rf_n_estimators} · Max Depth: {rf_max_depth}")
        rf_metrics = show_model_results(rand_forest, X_test_proc, y_test, rf_threshold)

        # Feature importance
        st.markdown("**Top 15 Feature Importances**")
        # try:
        #     ohe_cols  = col_prep.named_transformers_["cat"].get_feature_names_out(
        #         X_train.select_dtypes(include="object").columns.tolist())
        #     feat_names = numerical + list(ohe_cols)
        #     importances = rand_forest.feature_importances_
        #     top_idx = np.argsort(importances)[-15:][::-1]
        #     fig, ax = plt.subplots(figsize=(7, 4))
        #     ax.barh([feat_names[i] for i in top_idx][::-1],
        #             importances[top_idx][::-1], color="#cba6f7")
        #     ax.set_xlabel("Importance"); fig.tight_layout(); st.pyplot(fig)
        # except Exception:
        #     st.info("Feature importance chart unavailable.")

    with model_tabs[3]:
        st.caption(f"Threshold: {xgb_threshold} · Rounds: {xgb_n_estimators} · LR: {xgb_lr} · Depth: {xgb_max_depth}")
        xgb_metrics = show_model_results(xgb, X_test_proc, y_test, xgb_threshold)

        # # XGBoost feature importance
        # st.markdown("**Top 15 Feature Importances**")
        # try:
        #     ohe_cols  = col_prep.named_transformers_["cat"].get_feature_names_out(
        #         X_train.select_dtypes(include="object").columns.tolist())
        #     feat_names = numerical + list(ohe_cols)
        #     importances = xgb.feature_importances_
        #     top_idx = np.argsort(importances)[-15:][::-1]
        #     fig, ax = plt.subplots(figsize=(7, 4))
        #     ax.barh([feat_names[i] for i in top_idx][::-1],
        #             importances[top_idx][::-1], color="#89b4fa")
        #     ax.set_xlabel("Importance"); fig.tight_layout(); st.pyplot(fig)
        # except Exception:
        #     st.info("Feature importance chart unavailable.")


# ================================
# TAB 4 — COMPARE
# ================================
with tab_compare:
    st.subheader("All Models Comparison")

    compare_df = pd.DataFrame({
        "Logistic Regression": lr_metrics,
        "Decision Tree":       dt_metrics,
        "Random Forest":       rf_metrics,
        "XGBoost":             xgb_metrics,
    })
    st.dataframe(compare_df.style.highlight_max(axis=1, color="#a6e3a1"), use_container_width=True)

    st.markdown("#### Metric Bar Chart")
    all_metrics = {
        "Logistic Regression": lr_metrics,
        "Decision Tree":       dt_metrics,
        "Random Forest":       rf_metrics,
        "XGBoost":             xgb_metrics,
    }
    metric_names = list(lr_metrics.keys())
    colors = ["#89b4fa", "#a6e3a1", "#cba6f7", "#f9e2af"]
    x = np.arange(len(metric_names))
    w = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, mets) in enumerate(all_metrics.items()):
        ax.bar(x + (i - 1.5) * w, list(mets.values()), w, label=name, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1); ax.legend(loc="lower right")
    ax.set_title("Model Comparison — All Metrics")
    fig.tight_layout(); st.pyplot(fig)

    best = max(all_metrics, key=lambda m: all_metrics[m]["ROC AUC"])
    st.success(f"**{best}** achieves the highest ROC AUC on the test set.")

    # ROC curves overlay
    st.markdown("#### ROC Curves — All Models")
    model_map = {
        "Logistic Regression": (log_reg, "#89b4fa"),
        "Decision Tree":       (dec_tree, "#a6e3a1"),
        "Random Forest":       (rand_forest, "#cba6f7"),
        "XGBoost":             (xgb, "#f9e2af"),
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (model, color) in model_map.items():
        y_proba = model.predict_proba(X_test_proc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Comparison"); ax.legend()
    fig.tight_layout(); st.pyplot(fig)


# ================================
# TAB 5 — PREDICT
# ================================
with tab_predict:
    st.subheader(" Predict Income for a New Person")
    st.markdown("Fill in the details and click **Predict** to get a real-time estimate from all models.")

    def opts(col):
        return sorted(train_raw[col].dropna().str.strip().unique().tolist())

    st.markdown("#### Personal Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        age  = st.number_input("Age", min_value=17, max_value=90, value=35)
        sex  = st.selectbox("Sex", ["Male", "Female"])
        race = st.selectbox("Race", opts("race"))
    with c2:
        education_num  = st.slider("Education Level (years)", 1, 16, 10,
                                   help="1=No schooling · 9=HS grad · 13=Bachelor's · 16=Doctorate")
        marital_status = st.selectbox("Marital Status", opts("marital-status"))
        relationship   = st.selectbox("Relationship",   opts("relationship"))
    with c3:
        native_country = st.selectbox("Native Country", opts("native-country"))

    st.markdown("#### Work & Finances")
    c1, c2, c3 = st.columns(3)
    with c1:
        workclass  = st.selectbox("Workclass",  opts("workclass"))
        occupation = st.selectbox("Occupation", opts("occupation"))
    with c2:
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    with c3:
        capital_gain = st.number_input("Capital Gain ($)",  min_value=0, max_value=100000, value=0, step=500)
        capital_loss = st.number_input("Capital Loss ($)",  min_value=0, max_value=100000, value=0, step=500)

    st.markdown("#### Model Selection")
    predict_model = st.radio(
        "Which model(s) to use?",
        ["All 4 Models", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
        horizontal=True,
    )

    predict_btn = st.button("Predict", type="primary", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([{
            "age": age, "workclass": workclass, "fnlwgt": 0,
            "education": "Bachelors", "education-num": education_num,
            "marital-status": marital_status, "occupation": occupation,
            "relationship": relationship, "race": race, "sex": sex,
            "capital-gain": capital_gain, "capital-loss": capital_loss,
            "hours-per-week": hours_per_week, "native-country": native_country,
            "Income": "<=50K",
        }])

        input_clean = preprocess(input_df).drop("income", axis=1)
        for col in X_train.columns:
            if col not in input_clean.columns:
                input_clean[col] = 0
        input_clean       = input_clean[X_train.columns]
        input_transformed = col_prep.transform(input_clean)

        def render_prediction(model_name, model, threshold, color):
            prob  = model.predict_proba(input_transformed)[0][1]
            pred  = int(prob > threshold)
            label = ">50K" if pred == 1 else "<=50K"
            css   = "predict-result-high" if pred == 1 else "predict-result-low"
            bar_c = "#a6e3a1" if pred == 1 else "#f38ba8"
            emoji = "💚" if pred == 1 else "🔴"
            pct   = int(prob * 100)
            st.markdown(
                f'<div class="predict-box">'
                f'<div style="color:{color};font-size:0.9rem;font-weight:600;margin-bottom:0.5rem;">{model_name}</div>'
                f'<div style="color:#6c7086;font-size:0.75rem;margin-bottom:1rem;">threshold: {threshold}</div>'
                f'<div class="{css}">{emoji} {label}</div>'
                f'<div class="predict-confidence">Confidence: {prob:.1%}</div>'
                f'<div class="predict-bar-wrap"><div class="predict-bar-fill" style="width:{pct}%;background:{bar_c};"></div></div>'
                f'<div style="color:#6c7086;font-size:0.75rem;">P(&gt;50K) = {prob:.4f}</div>'
                f'</div>', unsafe_allow_html=True)

        st.divider()

        models_to_show = {
            "Logistic Regression": (log_reg,     lr_threshold,  "#89b4fa"),
            "Decision Tree":       (dec_tree,     dt_threshold,  "#a6e3a1"),
            "Random Forest":       (rand_forest,  rf_threshold,  "#cba6f7"),
            "XGBoost":             (xgb,          xgb_threshold, "#f9e2af"),
        }

        if predict_model == "All 4 Models":
            c1, c2 = st.columns(2)
            items = list(models_to_show.items())
            for i, (name, (model, thresh, color)) in enumerate(items):
                with (c1 if i % 2 == 0 else c2):
                    render_prediction(name, model, thresh, color)
        else:
            model, thresh, color = models_to_show[predict_model]
            render_prediction(predict_model, model, thresh, color)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from imblearn.over_sampling import SMOTE
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score,
#     f1_score, confusion_matrix, roc_auc_score, roc_curve,
# )

# # ================================
# # PAGE CONFIG
# # ================================
# st.set_page_config(page_title="Income Classifier", layout="wide")

# st.markdown("""
#     <style>
#     .metric-card {
#         background: #1e1e2e; border: 1px solid #313244;
#         border-radius: 10px; padding: 1rem 1.5rem; text-align: center;
#     }
#     .metric-label { color: #a6adc8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
#     .metric-value { color: #cdd6f4; font-size: 2rem; font-weight: 700; }
#     .metric-value.good { color: #a6e3a1; }
#     .metric-value.warn { color: #f9e2af; }
#     .predict-box {
#         background: #1e1e2e; border: 2px solid #313244;
#         border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1rem;
#     }
#     .predict-result-high { font-size: 2.5rem; font-weight: 800; color: #a6e3a1; }
#     .predict-result-low  { font-size: 2.5rem; font-weight: 800; color: #f38ba8; }
#     .predict-confidence  { font-size: 1rem; color: #a6adc8; margin-top: 0.5rem; }
#     .predict-bar-wrap {
#         background: #313244; border-radius: 999px; height: 14px;
#         margin: 1rem auto; max-width: 400px; overflow: hidden;
#     }
#     .predict-bar-fill { height: 100%; border-radius: 999px; }
#     </style>
# """, unsafe_allow_html=True)


# # ================================
# # PREPROCESSING
# # ================================
# def preprocess(df):
#     df = df.copy()
#     df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")
#     for col in df.select_dtypes(include="object").columns:
#         df[col] = df[col].str.strip()
#     df.replace("?", np.nan, inplace=True)
#     for col in ["workclass", "occupation", "native_country"]:
#         if col in df.columns:
#             df[col] = df[col].fillna(df[col].mode()[0])
#     df.drop(columns=["fnlwgt"], inplace=True, errors="ignore")
#     df.drop(columns=["education"], inplace=True, errors="ignore")
#     df.drop(columns=["native_country"], inplace=True, errors="ignore")
#     df["capital_net"] = df["capital_gain"] - df["capital_loss"]
#     df["work_intensity"] = df["hours_per_week"] * df["education_num"]
#     df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
#     df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
#     df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
#     return df


# # ================================
# # HEADER
# # ================================
# st.title("Income Classifier")
# st.markdown("Upload your **train** and **test** CSV files, configure the models, and explore results.")
# st.divider()

# # ================================
# # SIDEBAR
# # ================================
# with st.sidebar:
#     st.header("Configuration")
#     train_file = st.file_uploader("Upload Train CSV", type="csv")
#     test_file  = st.file_uploader("Upload Test CSV",  type="csv")

#     st.subheader("Model Settings")
#     lr_threshold = st.slider("Logistic Regression Threshold", 0.1, 0.9, 0.4, 0.05)
#     dt_threshold = st.slider("Decision Tree Threshold",        0.1, 0.9, 0.4, 0.05)
#     dt_max_depth = st.slider("Decision Tree Max Depth",  2, 20, 10)
#     rf_threshold = st.slider("Random Forest Threshold",        0.1, 0.9, 0.6, 0.05)
#     rf_n_estimators = st.slider("Random Forest Trees",  50, 500, 200, 50)
#     rf_max_depth = st.slider("Random Forest Max Depth", 2, 20, 20)
#     xgb_threshold = st.slider("XG boost Threshold", 0.1, 0.9, 0.5, 0.05)
#     xgb_n_estimators = st.slider("XGBoost Rounds",      50, 500, 200, 50)
#     xgb_lr = st.slider("XGBoost Learning Rate",         0.01, 0.3, 0.1, 0.01)
#     xgb_max_depth = st.slider("XGBoost Max Depth",      2, 10, 5)

#     st.markdown("---")
#     # use_smote = st.checkbox("Apply SMOTE Balancing", value=True)
#     run_btn   = st.button("Run Pipeline", use_container_width=True, type="primary")

# # ================================
# # SESSION STATE
# # ================================
# if "pipeline_ready" not in st.session_state:
#     st.session_state.pipeline_ready = False

# if run_btn:
#     if train_file is None or test_file is None:
#         st.error("Please upload both train and test CSV files.")
#         st.stop()

#     with st.spinner("Loading, preprocessing and training all 4 models..."):
#         train_raw = pd.read_csv(train_file)
#         test_raw  = pd.read_csv(test_file)
#         train_raw.columns = train_raw.columns.str.strip()
#         test_raw.columns  = test_raw.columns.str.strip()

#         train = preprocess(train_raw)
#         test  = preprocess(test_raw)


#         X_train = train.drop("income", axis=1)
#         y_train = train["income"]
#         X_test  = test.drop("income",  axis=1)
#         y_test  = test["income"]

#         categorical = X_train.select_dtypes(include="object").columns.tolist()
#         numerical   = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

#         col_prep = ColumnTransformer([
#             ("num", StandardScaler(),                       numerical),
#             ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
#         ])
#         X_train_proc  = col_prep.fit_transform(X_train)
#         X_test_proc   = col_prep.transform(X_test)
#         # before_counts = np.bincount(y_train)

#         # if use_smote:
#         #     smote = SMOTE(random_state=42)
#         #     X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
#         # after_counts = np.bincount(y_train)

#         # ── Logistic Regression ───────────────────────────────────────────
#         log_reg = LogisticRegression(C=1, max_iter=1000, solver="liblinear",
#                                      random_state=42, penalty="l2",class_weight=None)
#         log_reg.fit(X_train_proc, y_train)

#         # ── Decision Tree ─────────────────────────────────────────────────
#         dec_tree = DecisionTreeClassifier(
#             random_state=42, max_depth=dt_max_depth,
#             min_samples_leaf=5, min_samples_split=2,
#             criterion="entropy", ccp_alpha=0.0001)
#         dec_tree.fit(X_train_proc, y_train)

#         # ── Random Forest ─────────────────────────────────────────────────
#         rand_forest = RandomForestClassifier(
#             n_estimators=rf_n_estimators, max_depth=rf_max_depth,
#             min_samples_leaf=5, min_samples_split=5,
#             random_state=42, n_jobs=-1,class_weight="balanced",max_features="sqrt")
#         rand_forest.fit(X_train_proc, y_train)

#         # ── XGBoost ───────────────────────────────────────────────────────
#         xgb = XGBClassifier(
#             n_estimators=xgb_n_estimators, learning_rate=xgb_lr,
#             max_depth=xgb_max_depth,objective='binary:logistic')
#         xgb.fit(X_train_proc, y_train)

#         # ── Cache ─────────────────────────────────────────────────────────
#         st.session_state.update({
#             "train_raw": train_raw, "test_raw": test_raw,
#             "train": train, "test": test,
#             "X_train": X_train, "y_train": y_train,
#             "X_test": X_test,   "y_test": y_test,
#             "X_train_proc": X_train_proc, "X_test_proc": X_test_proc,
#             "col_prep": col_prep,
#             # "before_counts": before_counts, "after_counts": after_counts,
#             # "use_smote": use_smote,
#             "log_reg": log_reg, "dec_tree": dec_tree,
#             "rand_forest": rand_forest, "xgb": xgb,
#             "pipeline_ready": True,
#         })

# if not st.session_state.pipeline_ready:
#     st.info("Upload both CSV files and click **Run Pipeline** to start.")
#     st.stop()

# # ── Restore ───────────────────────────────────────────────────────────────
# train_raw     = st.session_state.train_raw
# test_raw      = st.session_state.test_raw
# train         = st.session_state.train
# test          = st.session_state.test
# X_train       = st.session_state.X_train
# y_train       = st.session_state.y_train
# X_test        = st.session_state.X_test
# y_test        = st.session_state.y_test
# X_train_proc  = st.session_state.X_train_proc
# X_test_proc   = st.session_state.X_test_proc
# col_prep      = st.session_state.col_prep
# # before_counts = st.session_state.before_counts
# # after_counts  = st.session_state.after_counts
# # use_smote     = st.session_state.use_smote
# log_reg       = st.session_state.log_reg
# dec_tree      = st.session_state.dec_tree
# rand_forest   = st.session_state.rand_forest
# xgb           = st.session_state.xgb

# # ================================
# # TABS
# # ================================
# tab_eda, tab_pre, tab_models, tab_compare, tab_predict = st.tabs(
#     ["EDA", "Preprocessing", "Models", "Compare", "Predict"]
# )

# # ================================
# # TAB 1 — EDA
# # ================================
# with tab_eda:
#     st.subheader("Exploratory Data Analysis")
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Train rows", f"{len(train_raw):,}")
#     c2.metric("Test rows",  f"{len(test_raw):,}")
#     c3.metric("Features",   str(train_raw.shape[1] - 1))

#     st.markdown("#### Raw Data Preview")
#     st.dataframe(train_raw.head(10), use_container_width=True)

#     c1, c2 = st.columns(2)
#     with c1:
#         st.markdown("#### Target Distribution")
#         fig, ax = plt.subplots(figsize=(5, 3))
#         train_raw["Income"].value_counts().plot(kind="bar", ax=ax,
#             color=["#89b4fa", "#a6e3a1"], edgecolor="none")
#         ax.set_title("Income Class Counts", fontsize=11)
#         ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
#         fig.tight_layout(); st.pyplot(fig)

#     with c2:
#         st.markdown("#### Age Distribution")
#         fig, ax = plt.subplots(figsize=(5, 3))
#         sns.histplot(train_raw["age"], kde=True, ax=ax, color="#cba6f7", bins=30)
#         ax.set_title("Age Distribution", fontsize=11)
#         fig.tight_layout(); st.pyplot(fig)

#     st.markdown("#### Correlation Heatmap")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(train_raw.select_dtypes(include=np.number).corr(),
#                 annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
#     fig.tight_layout(); st.pyplot(fig)

# # ================================
# # TAB 2 — PREPROCESSING
# # ================================
# with tab_pre:
#     st.subheader("Preprocessing Steps")

#     c1, c2 = st.columns(2)
#     with c1:
#         st.markdown("**Train — After Preprocessing**")
#         st.dataframe(train.head(8), use_container_width=True)
#     with c2:
#         st.markdown("**Test — After Preprocessing**")
#         st.dataframe(test.head(8), use_container_width=True)


#     # if use_smote:
#     #     c1, c2 = st.columns(2)
#     #     with c1:
#     #         st.markdown("**Class Balance Before SMOTE**")
#     #         st.bar_chart(pd.DataFrame({"Count": before_counts}, index=["<=50K", ">50K"]))
#     #     with c2:
#     #         st.markdown("**Class Balance After SMOTE**")
#     #         st.bar_chart(pd.DataFrame({"Count": after_counts},  index=["<=50K", ">50K"]))
#     # else:
#     #     st.markdown("**Class Balance**")
#     #     st.bar_chart(pd.DataFrame({"Count": before_counts}, index=["<=50K", ">50K"]))



# def evaluate_model(model, X, y, threshold):
#     y_proba = model.predict_proba(X)[:, 1]
#     y_pred  = (y_proba > threshold).astype(int)
#     metrics = {
#         "Accuracy":  round(accuracy_score(y, y_pred),  4),
#         "Precision": round(precision_score(y, y_pred), 4),
#         "Recall":    round(recall_score(y, y_pred),    4),
#         "F1 Score":  round(f1_score(y, y_pred),        4),
#         "ROC AUC":   round(roc_auc_score(y, y_proba),  4),
#     }
#     fpr, tpr, _ = roc_curve(y, y_proba)
#     return metrics, confusion_matrix(y, y_pred), fpr, tpr


# def show_model_results(model, X, y, threshold):
#     metrics, cm, fpr, tpr = evaluate_model(model, X, y, threshold)

#     cols = st.columns(5)
#     for i, (label, val) in enumerate(metrics.items()):
#         color = "good" if val >= 0.75 else "warn"
#         cols[i].markdown(
#             f'<div class="metric-card">'
#             f'<div class="metric-label">{label}</div>'
#             f'<div class="metric-value {color}">{val:.4f}</div>'
#             f'</div>', unsafe_allow_html=True)

#     st.markdown("")
#     c1, c2 = st.columns(2)
#     with c1:
#         st.markdown("**Confusion Matrix**")
#         fig, ax = plt.subplots(figsize=(4, 3))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
#                     xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
#         ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
#         fig.tight_layout(); st.pyplot(fig)
#     with c2:
#         st.markdown("**ROC Curve**")
#         auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
#         fig, ax = plt.subplots(figsize=(4, 3))
#         ax.plot(fpr, tpr, color="#89b4fa", lw=2, label=f"AUC = {auc:.3f}")
#         ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
#         ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
#         fig.tight_layout(); st.pyplot(fig)

#     return metrics


# # ================================
# # TAB 3 — MODELS
# # ================================
# with tab_models:
#     model_tabs = st.tabs([
#         "Logistic Regression",
#         "Decision Tree",
#         "Random Forest",
#         "XGBoost",
#     ])

#     with model_tabs[0]:
#         st.caption(f"Threshold: {lr_threshold}")
#         lr_metrics = show_model_results(log_reg, X_test_proc, y_test, lr_threshold)

#     with model_tabs[1]:
#         st.caption(f"Threshold: {dt_threshold} · Max Depth: {dt_max_depth}")
#         dt_metrics = show_model_results(dec_tree, X_test_proc, y_test, dt_threshold)

#     with model_tabs[2]:
#         st.caption(f"Threshold: {rf_threshold} · Trees: {rf_n_estimators} · Max Depth: {rf_max_depth}")
#         rf_metrics = show_model_results(rand_forest, X_test_proc, y_test, rf_threshold)

#         # Feature importance
#         st.markdown("**Top 15 Feature Importances**")
#         # try:
#         #     ohe_cols  = col_prep.named_transformers_["cat"].get_feature_names_out(
#         #         X_train.select_dtypes(include="object").columns.tolist())
#         #     feat_names = numerical + list(ohe_cols)
#         #     importances = rand_forest.feature_importances_
#         #     top_idx = np.argsort(importances)[-15:][::-1]
#         #     fig, ax = plt.subplots(figsize=(7, 4))
#         #     ax.barh([feat_names[i] for i in top_idx][::-1],
#         #             importances[top_idx][::-1], color="#cba6f7")
#         #     ax.set_xlabel("Importance"); fig.tight_layout(); st.pyplot(fig)
#         # except Exception:
#         #     st.info("Feature importance chart unavailable.")

#     with model_tabs[3]:
#         st.caption(f"Threshold: {xgb_threshold} · Rounds: {xgb_n_estimators} · LR: {xgb_lr} · Depth: {xgb_max_depth}")
#         xgb_metrics = show_model_results(xgb, X_test_proc, y_test, xgb_threshold)

#         # # XGBoost feature importance
#         # st.markdown("**Top 15 Feature Importances**")
#         # try:
#         #     ohe_cols  = col_prep.named_transformers_["cat"].get_feature_names_out(
#         #         X_train.select_dtypes(include="object").columns.tolist())
#         #     feat_names = numerical + list(ohe_cols)
#         #     importances = xgb.feature_importances_
#         #     top_idx = np.argsort(importances)[-15:][::-1]
#         #     fig, ax = plt.subplots(figsize=(7, 4))
#         #     ax.barh([feat_names[i] for i in top_idx][::-1],
#         #             importances[top_idx][::-1], color="#89b4fa")
#         #     ax.set_xlabel("Importance"); fig.tight_layout(); st.pyplot(fig)
#         # except Exception:
#         #     st.info("Feature importance chart unavailable.")


# # ================================
# # TAB 4 — COMPARE
# # ================================
# with tab_compare:
#     st.subheader("All Models Comparison")

#     compare_df = pd.DataFrame({
#         "Logistic Regression": lr_metrics,
#         "Decision Tree":       dt_metrics,
#         "Random Forest":       rf_metrics,
#         "XGBoost":             xgb_metrics,
#     })
#     st.dataframe(compare_df.style.highlight_max(axis=1, color="#a6e3a1"), use_container_width=True)

#     st.markdown("#### Metric Bar Chart")
#     all_metrics = {
#         "Logistic Regression": lr_metrics,
#         "Decision Tree":       dt_metrics,
#         "Random Forest":       rf_metrics,
#         "XGBoost":             xgb_metrics,
#     }
#     metric_names = list(lr_metrics.keys())
#     colors = ["#89b4fa", "#a6e3a1", "#cba6f7", "#f9e2af"]
#     x = np.arange(len(metric_names))
#     w = 0.2

#     fig, ax = plt.subplots(figsize=(11, 5))
#     for i, (name, mets) in enumerate(all_metrics.items()):
#         ax.bar(x + (i - 1.5) * w, list(mets.values()), w, label=name, color=colors[i])
#     ax.set_xticks(x); ax.set_xticklabels(metric_names)
#     ax.set_ylim(0, 1.1); ax.legend(loc="lower right")
#     ax.set_title("Model Comparison — All Metrics")
#     fig.tight_layout(); st.pyplot(fig)

#     best = max(all_metrics, key=lambda m: all_metrics[m]["ROC AUC"])
#     st.success(f"**{best}** achieves the highest ROC AUC on the test set.")

#     # ROC curves overlay
#     st.markdown("#### ROC Curves — All Models")
#     model_map = {
#         "Logistic Regression": (log_reg, "#89b4fa"),
#         "Decision Tree":       (dec_tree, "#a6e3a1"),
#         "Random Forest":       (rand_forest, "#cba6f7"),
#         "XGBoost":             (xgb, "#f9e2af"),
#     }
#     fig, ax = plt.subplots(figsize=(7, 5))
#     for name, (model, color) in model_map.items():
#         y_proba = model.predict_proba(X_test_proc)[:, 1]
#         fpr, tpr, _ = roc_curve(y_test, y_proba)
#         auc = roc_auc_score(y_test, y_proba)
#         ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")
#     ax.plot([0, 1], [0, 1], "--", color="gray")
#     ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
#     ax.set_title("ROC Curves Comparison"); ax.legend()
#     fig.tight_layout(); st.pyplot(fig)


# # ================================
# # TAB 5 — PREDICT
# # ================================
# # with tab_predict:
# #     st.subheader(" Predict Income for a New Person")
# #     st.markdown("Fill in the details and click **Predict** to get a real-time estimate from all models.")

# #     def opts(col):
# #         return sorted(train_raw[col].dropna().str.strip().unique().tolist())

# #     st.markdown("#### Personal Information")
# #     c1, c2, c3 = st.columns(3)
# #     with c1:
# #         age  = st.number_input("Age", min_value=17, max_value=90, value=35)
# #         sex  = st.selectbox("Sex", ["Male", "Female"])
# #         race = st.selectbox("Race", opts("race"))
# #     with c2:
# #         education_num  = st.slider("Education Level (years)", 1, 16, 10,
# #                                    help="1=No schooling · 9=HS grad · 13=Bachelor's · 16=Doctorate")
# #         marital_status = st.selectbox("Marital Status", opts("marital-status"))
# #         relationship   = st.selectbox("Relationship",   opts("relationship"))
# #     with c3:
# #         native_country = st.selectbox("Native Country", opts("native-country"))

# #     st.markdown("#### Work & Finances")
# #     c1, c2, c3 = st.columns(3)
# #     with c1:
# #         workclass  = st.selectbox("Workclass",  opts("workclass"))
# #         occupation = st.selectbox("Occupation", opts("occupation"))
# #     with c2:
# #         hours_per_week = st.slider("Hours per Week", 1, 99, 40)
# #     with c3:
# #         capital_gain = st.number_input("Capital Gain ($)",  min_value=0, max_value=100000, value=0, step=500)
# #         capital_loss = st.number_input("Capital Loss ($)",  min_value=0, max_value=100000, value=0, step=500)

# #     st.markdown("#### Model Selection")
# #     predict_model = st.radio(
# #         "Which model(s) to use?",
# #         ["All 4 Models", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
# #         horizontal=True,
# #     )

# #     predict_btn = st.button("Predict", type="primary", use_container_width=True)

# #     if predict_btn:
# #         input_df = pd.DataFrame([{
# #             "age": age, "workclass": workclass, "fnlwgt": 0,
# #             "education": "Bachelors", "education-num": education_num,
# #             "marital-status": marital_status, "occupation": occupation,
# #             "relationship": relationship, "race": race, "sex": sex,
# #             "capital-gain": capital_gain, "capital-loss": capital_loss,
# #             "hours-per-week": hours_per_week, "native-country": native_country,
# #             "Income": "<=50K",
# #         }])

# #         input_clean = preprocess(input_df).drop("income", axis=1)
# #         for col in X_train.columns:
# #             if col not in input_clean.columns:
# #                 input_clean[col] = 0
# #         input_clean       = input_clean[X_train.columns]
# #         input_transformed = col_prep.transform(input_clean)

# #         def render_prediction(model_name, model, threshold, color):
# #             prob  = model.predict_proba(input_transformed)[0][1]
# #             pred  = int(prob > threshold)
# #             label = ">50K" if pred == 1 else "<=50K"
# #             css   = "predict-result-high" if pred == 1 else "predict-result-low"
# #             bar_c = "#a6e3a1" if pred == 1 else "#f38ba8"
# #             emoji = "💚" if pred == 1 else "🔴"
# #             pct   = int(prob * 100)
# #             st.markdown(
# #                 f'<div class="predict-box">'
# #                 f'<div style="color:{color};font-size:0.9rem;font-weight:600;margin-bottom:0.5rem;">{model_name}</div>'
# #                 f'<div style="color:#6c7086;font-size:0.75rem;margin-bottom:1rem;">threshold: {threshold}</div>'
# #                 f'<div class="{css}">{emoji} {label}</div>'
# #                 f'<div class="predict-confidence">Confidence: {prob:.1%}</div>'
# #                 f'<div class="predict-bar-wrap"><div class="predict-bar-fill" style="width:{pct}%;background:{bar_c};"></div></div>'
# #                 f'<div style="color:#6c7086;font-size:0.75rem;">P(&gt;50K) = {prob:.4f}</div>'
# #                 f'</div>', unsafe_allow_html=True)

# #         st.divider()

# #         models_to_show = {
# #             "Logistic Regression": (log_reg,     lr_threshold,  "#89b4fa"),
# #             "Decision Tree":       (dec_tree,     dt_threshold,  "#a6e3a1"),
# #             "Random Forest":       (rand_forest,  rf_threshold,  "#cba6f7"),
# #             "XGBoost":             (xgb,          xgb_threshold, "#f9e2af"),
# #         }

# #         if predict_model == "All 4 Models":
# #             c1, c2 = st.columns(2)
# #             items = list(models_to_show.items())
# #             for i, (name, (model, thresh, color)) in enumerate(items):
# #                 with (c1 if i % 2 == 0 else c2):
# #                     render_prediction(name, model, thresh, color)
# #         else:
# #             model, thresh, color = models_to_show[predict_model]
# #             render_prediction(predict_model, model, thresh, color)
