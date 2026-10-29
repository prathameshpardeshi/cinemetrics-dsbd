"""
CineMetrics // DSBD — Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from backend import (
    load_and_clean_data,
    train_nlp_model,
    train_regression_model,
    run_mapreduce_wordcount,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMetrics // DSBD", layout="wide")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#0b0d10;--card:#14181d;--border:#262b33;--amber:#e5a93c;
--text:#c9cdd4;--muted:#6b7280;--white:#f0f1f3;--teal:#2dd4bf;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]
{background:var(--bg)!important;color:var(--text)!important;font-family:'Inter',sans-serif!important;}
[data-testid="stSidebar"]{background:#0f1115!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
h1,h2,h3,h4{color:var(--white)!important;font-family:'Inter',sans-serif!important;font-weight:700!important;}
header[data-testid="stHeader"]{display:none!important;}
footer{display:none!important;}
#MainMenu{display:none!important;}
div.block-container{padding-top:1.5rem!important;}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:10px;
padding:20px 24px;text-align:left;}
.stat-card .label{color:var(--muted);font-size:.72rem;text-transform:uppercase;
letter-spacing:.6px;margin-bottom:6px;}
.stat-card .value{color:var(--amber);font-family:'JetBrains Mono',monospace;
font-size:1.55rem;font-weight:600;line-height:1.2;}
.stat-card .sub{color:var(--muted);font-size:.73rem;margin-top:4px;}
.accent-bar{width:56px;height:3px;background:var(--amber);border-radius:2px;margin:-4px 0 18px 0;}
[data-testid="stDataFrame"],[data-testid="stTable"],.stDataFrame
{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
thead tr th{background:#1a1f26!important;color:var(--amber)!important;
font-family:'JetBrains Mono',monospace!important;font-size:.76rem!important;}
tbody tr td{color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;font-size:.8rem!important;}
details{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def stat_card(label, value, subtext=""):
    sub_html = f'<div class="sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f'<div class="stat-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )

PX_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c9cdd4", size=12),
    title_font=dict(color="#f0f1f3", size=14),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#262b33", zerolinecolor="#262b33"),
    yaxis=dict(gridcolor="#262b33", zerolinecolor="#262b33"),
    margin=dict(l=40, r=30, t=50, b=40),
)
AMBER = "#e5a93c"
TEAL = "#2dd4bf"


# ─── Cached data ─────────────────────────────────────────────────────────────
@st.cache_resource
def cached_data():
    raw, clean, audit = load_and_clean_data()
    nlp = train_nlp_model(clean)
    reg = train_regression_model(clean)
    mr = run_mapreduce_wordcount(clean)
    return raw, clean, audit, nlp, reg, mr

raw_df, clean_df, audit, nlp_res, reg_res, mr_res = cached_data()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### CineMetrics // DSBD")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
    view = st.radio(
        "NAV",
        [
            "01 / Pipeline & Data Wrangling",
            "02 / Exploratory Statistical Inference",
            "03 / NLP Sentiment & Naive Bayes",
            "04 / Box Office Regression & Simulator",
            "05 / MapReduce WordCount Engine",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Raw {raw_df.shape[0]} rows  /  Clean {clean_df.shape[0]} rows  /  {raw_df.shape[1]} cols")


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 01 — Pipeline & Data Wrangling
# ═════════════════════════════════════════════════════════════════════════════
def view_01():
    st.markdown("## 01 / Pipeline & Data Wrangling")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    # ── KPI cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Total Records", f"{raw_df.shape[0]:,}", f"{raw_df.shape[1]} features")
    with c2:
        total_nulls = sum(audit["nulls_fixed"].values())
        stat_card("Missing Values Imputed", str(total_nulls), "Median imputation")
    with c3:
        total_out = sum(audit["outliers_capped"].values())
        stat_card("IQR Outliers Capped", str(total_out), "Q1-1.5*IQR to Q3+1.5*IQR")
    with c4:
        post_skew_budget = audit["post_skew"].get("Budget_Cr", 0)
        stat_card("Post-Clean Budget Skew", f"{post_skew_budget:.4f}", "Budget_Cr skewness")

    st.markdown("")

    # ── Side-by-side boxplots: Pre vs Post ───────────────────────────────
    st.markdown("### Outlier Capping  /  Pre vs Post IQR")
    for col in ["Budget_Cr", "Box_Office_Est_Cr"]:
        capped_n = audit["outliers_capped"][col]
        st.markdown(f"**{col}** — {capped_n} outliers capped")
        left, right = st.columns(2)

        with left:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=raw_df[col].dropna(), name="Before", marker_color=AMBER,
                boxmean=True, line=dict(color=AMBER),
            ))
            fig.update_layout(title=f"{col} — Before IQR Capping", **PX_LAYOUT, height=370)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=clean_df[col].dropna(), name="After", marker_color=TEAL,
                boxmean=True, line=dict(color=TEAL),
            ))
            fig.update_layout(title=f"{col} — After IQR Capping", **PX_LAYOUT, height=370)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Audit table ──────────────────────────────────────────────────────
    st.markdown("### Column Audit Report")
    num_cols = raw_df.select_dtypes("number").columns
    rows = []
    for col in raw_df.columns:
        pre_null = int(audit["null_counts"].get(col, 0))
        post_null = int(clean_df[col].isnull().sum())
        pre_sk = round(audit["pre_skew"].get(col, np.nan), 4) if col in num_cols else None
        post_sk = round(audit["post_skew"].get(col, np.nan), 4) if col in num_cols else None
        rows.append({
            "Column": col,
            "Dtype": str(raw_df[col].dtype),
            "Nulls (raw)": pre_null,
            "Nulls (clean)": post_null,
            "Skew (raw)": pre_sk,
            "Skew (clean)": post_sk,
        })
    audit_df = pd.DataFrame(rows)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

    # ── Toggleable preview ───────────────────────────────────────────────
    with st.expander("Raw DataFrame Preview (first 25 rows)"):
        st.dataframe(raw_df.head(25), use_container_width=True, hide_index=True)
    with st.expander("Clean DataFrame Preview (first 25 rows)"):
        st.dataframe(clean_df.head(25), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 02 — Exploratory Statistical Inference
# ═════════════════════════════════════════════════════════════════════════════
def view_02():
    st.markdown("## 02 / Exploratory Statistical Inference")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    num_cols = ["Budget_Cr", "Marketing_Spend_Cr", "Runtime_Min",
                "Critic_Score", "Box_Office_Est_Cr"]

    # ── Grouped stats by Genre ───────────────────────────────────────────
    st.markdown("### Genre-wise Statistical Summary")
    sel_col = st.selectbox("Select metric column", num_cols, index=0)

    grouped = clean_df.groupby("Genre")[sel_col].agg(
        Mean="mean", Median="median", Std="std", Variance="var", Skewness="skew",
    ).round(3)
    st.dataframe(grouped, use_container_width=True)

    st.markdown("---")

    # ── Two-column: Heatmap + Scatter ────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("### Pearson Correlation Heatmap")
        corr = clean_df[num_cols].corr().round(3)
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=[[0, "#14181d"], [0.5, "#c98b1d"], [1, AMBER]],
            aspect="auto",
        )
        fig_heat.update_layout(
            **PX_LAYOUT, height=500,
            coloraxis_colorbar=dict(title="r", tickfont=dict(color="#c9cdd4")),
        )
        fig_heat.update_traces(textfont=dict(family="JetBrains Mono", size=11))
        st.plotly_chart(fig_heat, use_container_width=True)

    with right:
        st.markdown("### Budget vs Box Office by Genre")
        fig_sc = px.scatter(
            clean_df,
            x="Budget_Cr",
            y="Box_Office_Est_Cr",
            color="Genre",
            size="Critic_Score",
            color_discrete_sequence=[AMBER, TEAL, "#3b82f6", "#ef4444", "#a855f7"],
            opacity=0.72,
            hover_data=["Title", "Release_Platform"],
        )
        fig_sc.update_layout(
            **PX_LAYOUT, height=500,
            xaxis_title="Budget (Cr)",
            yaxis_title="Box Office Est. (Cr)",
        )
        fig_sc.update_traces(marker=dict(line=dict(width=0.4, color="#262b33")))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Full descriptive stats ───────────────────────────────────────────
    with st.expander("Full Descriptive Statistics (Clean Data)"):
        st.dataframe(clean_df[num_cols].describe().T.round(2), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Placeholder views
# ═════════════════════════════════════════════════════════════════════════════
def view_03():
    st.markdown("## 03 / NLP Sentiment & Naive Bayes")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    # ── KPI cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Model Accuracy", f"{nlp_res['accuracy']:.4f}", "MultinomialNB")
    with c2:
        stat_card("Precision", f"{nlp_res['precision']:.4f}", "Positive class")
    with c3:
        stat_card("Recall", f"{nlp_res['recall']:.4f}", "Positive class")
    with c4:
        stat_card("F1-Score", f"{nlp_res['f1']:.4f}", "Harmonic mean")

    st.markdown("")

    # ── Confusion matrix + Top words ─────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("### Confusion Matrix")
        cm = nlp_res["confusion_matrix"]
        labels = ["Positive", "Negative"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f"Pred {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=18, family="JetBrains Mono", color="#0b0d10"),
            colorscale=[[0, "#1a1f26"], [1, AMBER]],
            showscale=False,
        ))
        fig_cm.update_layout(**PX_LAYOUT, height=400,
                             xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

    with right:
        st.markdown("### Top Predictive TF-IDF Terms")
        pos_words = nlp_res["top_positive_words"][:10]
        neg_words = nlp_res["top_negative_words"][:10]
        word_df = pd.DataFrame({
            "Word": pos_words + neg_words,
            "Class": ["Positive"] * len(pos_words) + ["Negative"] * len(neg_words),
            "Rank": list(range(len(pos_words), 0, -1)) + list(range(len(neg_words), 0, -1)),
        })
        fig_w = px.bar(
            word_df, x="Rank", y="Word", color="Class",
            orientation="h",
            color_discrete_map={"Positive": TEAL, "Negative": "#ef4444"},
            barmode="group",
        )
        layout_w = dict(PX_LAYOUT)
        layout_w["legend"] = dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)")
        fig_w.update_layout(**layout_w, height=400,
                            xaxis_title="Importance Rank", yaxis_title="")
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("---")

    # ── Pipeline Inspector ───────────────────────────────────────────────
    st.markdown("### Pipeline Inspector")
    movie_options = clean_df["Title"].tolist()
    selected_title = st.selectbox("Select a movie to inspect", movie_options, index=0)
    row = clean_df[clean_df["Title"] == selected_title].iloc[0]

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Raw Review**")
        st.markdown(
            f'<div class="stat-card" style="font-size:.88rem;line-height:1.6;">{row["User_Review_Text"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'**Ground Truth Sentiment:** `{row["Sentiment_Label"]}`')

    with col_b:
        import re as _re
        tokens = _re.findall(r"\b[a-zA-Z]{3,}\b", row["User_Review_Text"].lower())
        st.markdown("**Cleaned Tokens**")
        st.code(" | ".join(tokens[:30]), language=None)

        vec = nlp_res["vectorizer"]
        tfidf_vec = vec.transform([row["User_Review_Text"]])
        feature_names = vec.get_feature_names_out()
        scores = tfidf_vec.toarray().flatten()
        top_idx = scores.argsort()[-10:][::-1]
        tfidf_table = pd.DataFrame({
            "Token": feature_names[top_idx],
            "TF-IDF Weight": scores[top_idx].round(4),
        })
        st.markdown("**Top TF-IDF Weights**")
        st.dataframe(tfidf_table, use_container_width=True, hide_index=True)

        pred = nlp_res["model"].predict(tfidf_vec)[0]
        st.markdown(f'**NB Prediction:** `{pred}`')


def view_04():
    st.markdown("## 04 / Box Office Regression & Simulator")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    # ── KPI cards ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("R2 Score", f"{reg_res['r2']:.4f}", "Coefficient of determination")
    with c2:
        stat_card("RMSE", f"{reg_res['rmse']:.2f} Cr", "Root mean squared error")
    with c3:
        stat_card("MAE", f"{reg_res['mae']:.2f} Cr", "Mean absolute error")

    st.markdown("")

    # ── Actual vs Predicted scatter ──────────────────────────────────────
    st.markdown("### Actual vs Predicted Box Office")
    res_df = reg_res["results_df"]
    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=res_df["Actual"], y=res_df["Predicted"],
        mode="markers",
        marker=dict(color=AMBER, size=6, opacity=0.7,
                    line=dict(width=0.4, color="#262b33")),
        name="Test samples",
    ))
    min_val = min(res_df["Actual"].min(), res_df["Predicted"].min())
    max_val = max(res_df["Actual"].max(), res_df["Predicted"].max())
    fig_ap.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color=TEAL, width=2, dash="dash"),
        name="Ideal (y=x)",
    ))
    fig_ap.update_layout(
        **PX_LAYOUT, height=480,
        xaxis_title="Actual Box Office (Cr)",
        yaxis_title="Predicted Box Office (Cr)",
    )
    st.plotly_chart(fig_ap, use_container_width=True)

    # ── Coefficients ─────────────────────────────────────────────────────
    with st.expander("Model Coefficients"):
        coef_df = pd.DataFrame({
            "Feature": list(reg_res["coefficients"].keys()),
            "Coefficient": [round(v, 4) for v in reg_res["coefficients"].values()],
        })
        coef_df.loc[len(coef_df)] = ["Intercept", round(reg_res["intercept"], 4)]
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Greenlight Simulator ─────────────────────────────────────────────
    st.markdown("### Greenlight Simulator")
    st.caption("Simulate a hypothetical release to predict box-office performance and audience sentiment.")

    with st.form("greenlight_form"):
        f1, f2 = st.columns(2)
        with f1:
            sim_budget = st.slider("Budget (Cr)", 8.0, 300.0, 80.0, 1.0)
            sim_marketing = st.slider("Marketing Spend (Cr)", 2.0, 80.0, 20.0, 1.0)
            sim_runtime = st.slider("Runtime (Min)", 95, 185, 140, 1)
        with f2:
            sim_critic = st.slider("Critic Score", 1.5, 9.8, 6.5, 0.1)
            sim_review = st.text_area(
                "Custom Audience Review",
                value="The screenplay was gripping and the climax was outstanding.",
                height=120,
            )

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        # Regression prediction
        import pandas as _pd
        sim_input = _pd.DataFrame([{
            "Budget_Cr": sim_budget,
            "Marketing_Spend_Cr": sim_marketing,
            "Runtime_Min": sim_runtime,
            "Critic_Score": sim_critic,
            "Sentiment_Flag": 1,  # placeholder, will overwrite below
        }])

        # NLP prediction on review
        review_vec = nlp_res["vectorizer"].transform([sim_review])
        sent_pred = nlp_res["model"].predict(review_vec)[0]
        sent_proba = nlp_res["model"].predict_proba(review_vec).max()
        sim_input["Sentiment_Flag"] = 1 if sent_pred == "Positive" else 0

        predicted_bo = reg_res["model"].predict(sim_input)[0]
        predicted_bo = max(predicted_bo, 0.0)
        roi = ((predicted_bo - sim_budget - sim_marketing) / (sim_budget + sim_marketing)) * 100

        st.markdown("")
        r1, r2, r3 = st.columns(3)
        with r1:
            stat_card("Predicted Box Office", f"{predicted_bo:.2f} Cr", "LinearRegression estimate")
        with r2:
            roi_sub = "Profitable" if roi > 0 else "Below breakeven"
            stat_card("Estimated ROI", f"{roi:.1f}%", roi_sub)
        with r3:
            stat_card("Sentiment Prediction", sent_pred, f"{sent_proba:.1%} confidence")


def view_05():
    st.markdown("## 05 / MapReduce WordCount Engine")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    # ── Genre filter ─────────────────────────────────────────────────────
    genre_options = ["All"] + sorted(clean_df["Genre"].unique().tolist())
    genre_pick = st.selectbox("Filter by Genre", genre_options, index=0)

    mr = run_mapreduce_wordcount(clean_df, genre_filter=genre_pick)

    # ── Three-column phases ──────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### MAP Phase")
        st.caption("Tokenize reviews into (word, 1) key-value pairs")
        map_str = "\n".join([f'("{w}", {c})' for w, c in mr["map_sample"]])
        st.code(map_str, language=None)
        st.markdown(f'**Total mapped tokens:** `{mr["total_tokens"]:,}`')

    with col2:
        st.markdown("### SHUFFLE & SORT")
        st.caption("Group by unique keys before reduction")
        stat_card("Total Mapped Tokens", f'{mr["total_tokens"]:,}', "All (word, 1) pairs")
        st.markdown("")
        stat_card("Unique Shuffled Keys", f'{mr["shuffle_unique_keys"]:,}', "Distinct words after grouping")

    with col3:
        st.markdown("### REDUCE Phase")
        st.caption("Aggregate counts per unique word")
        st.dataframe(mr["reduce_df"], use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Horizontal bar chart of reduced frequencies ──────────────────────
    st.markdown("### Top 15 Word Frequencies (Reduced Output)")
    rdf = mr["reduce_df"].sort_values("Count", ascending=True)
    fig_mr = go.Figure()
    fig_mr.add_trace(go.Bar(
        x=rdf["Count"], y=rdf["Word"],
        orientation="h",
        marker_color=AMBER,
        text=rdf["Count"],
        textposition="outside",
        textfont=dict(color="#c9cdd4", family="JetBrains Mono", size=11),
    ))
    fig_mr.update_layout(
        **PX_LAYOUT, height=460,
        xaxis_title="Frequency",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig_mr, use_container_width=True)


# ─── Router ──────────────────────────────────────────────────────────────────
{"01 / Pipeline & Data Wrangling": view_01,
 "02 / Exploratory Statistical Inference": view_02,
 "03 / NLP Sentiment & Naive Bayes": view_03,
 "04 / Box Office Regression & Simulator": view_04,
 "05 / MapReduce WordCount Engine": view_05,
}[view]()
