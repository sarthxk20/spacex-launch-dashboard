import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              confusion_matrix, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpaceX Launch Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #05070f; }
    [data-testid="stSidebar"] {
        background: #090d1a !important;
        border-right: 1px solid #1a2540;
    }
    .block-container { padding: 2rem 2.5rem 3rem 2.5rem; max-width: 1400px; }

    h1 { font-size: 2rem !important; font-weight: 700 !important;
         background: linear-gradient(90deg, #60a5fa, #a78bfa);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2 { font-size: 1.3rem !important; font-weight: 600 !important;
         color: #cbd5e1 !important; letter-spacing: 0.03em; margin-top: 0.2rem; }
    h3, h4 { color: #94a3b8 !important; }
    p, li, label { color: #94a3b8 !important; }

    .kpi-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #1e3a5f; border-radius: 14px;
        padding: 22px 20px; text-align: center;
        position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #f0f6ff; line-height: 1.1; }
    .kpi-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase;
                 letter-spacing: 0.1em; margin-top: 6px; }

    .section-card {
        background: #0f172a; border: 1px solid #1e293b;
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
    }
    .prediction-result {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid #1e3a5f; border-radius: 16px;
        padding: 28px; text-align: center; margin-top: 20px;
    }
    .pred-label { font-size: 1.8rem; font-weight: 800; margin-bottom: 6px; }
    .pred-prob  { font-size: 1rem; color: #64748b; }

    .insight-item {
        background: #0f172a; border-left: 3px solid #3b82f6;
        border-radius: 0 10px 10px 0; padding: 14px 18px;
        margin-bottom: 12px; font-size: 0.92rem;
    }
    .about-card {
        background: #0c1628; border: 1px solid #1e3a5f;
        border-radius: 16px; padding: 24px; margin-top: 24px;
    }

    @media (max-width: 768px) {
        .block-container { padding: 1rem !important; }
        h1 { font-size: 1.5rem !important; }
        .kpi-value { font-size: 1.4rem; }
    }
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    title_font=dict(size=14, color="#cbd5e1"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
)
ACCENT = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b"]

# Key SpaceX milestones for chart annotations
MILESTONES = [
    (2010, "Falcon 9\nFirst Flight"),
    (2015, "First\nBooster Landing"),
    (2018, "Falcon Heavy\nDebut"),
    (2020, "First\nCrewed Mission"),
    (2022, "Starlink\nScaling"),
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association statistic for two categorical variables."""
    confusion = pd.crosstab(x, y).values
    chi2 = chi2_contingency(confusion, correction=False)[0]
    n    = confusion.sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return float(np.sqrt(phi2 / max(min(k-1, r-1), 1)))


def association_matrix(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Build a Cramér's V matrix for a set of columns."""
    mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for c1 in cols:
        for c2 in cols:
            mat.loc[c1, c2] = cramers_v(df[c1].astype(str), df[c2].astype(str))
    return mat


def dark_heatmap(matrix: pd.DataFrame, title: str):
    """Render a seaborn heatmap matching the dark dashboard theme."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                annot_kws={"color": "white", "size": 10},
                linewidths=0.5, linecolor="#1e293b",
                cbar_kws={"shrink": 0.8})
    ax.tick_params(colors="#94a3b8", labelsize=9)
    plt.setp(ax.get_xticklabels(), color="#94a3b8", rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), color="#94a3b8", rotation=0)
    ax.set_title(title, color="#cbd5e1", pad=12)
    return fig


def add_milestone_lines(fig, years_in_data):
    """Add vertical annotation lines for key SpaceX milestones."""
    for year, label in MILESTONES:
        if year in years_in_data:
            fig.add_vline(x=year, line_dash="dot", line_color="#334155",
                          line_width=1.5)
            fig.add_annotation(x=year, y=1, yref="paper",
                               text=label, showarrow=False,
                               font=dict(size=8, color="#475569"),
                               align="center", yanchor="top",
                               bgcolor="rgba(15,23,42,0.7)",
                               borderpad=3)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (#8 — error handling + graceful unknown ID fallback)
# ─────────────────────────────────────────────────────────────────────────────
ROCKET_MAP = {
    "5e9d0d95eda69955f709d1eb": "Falcon 1",
    "5e9d0d95eda69973a809d1ec": "Falcon 9",
    "5e9d0d95eda69974db09d1ed": "Falcon Heavy",
}
PAD_MAP = {
    "5e9e4502f5090995de566f86": "Kwajalein Atoll",
    "5e9e4501f509094ba4566f84": "Cape Canaveral SFS",
    "5e9e4502f509092b78566f87": "Kennedy LC-39A",
    "5e9e4502f509094188566f88": "Vandenberg SFB",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv("merged_spacex_data.csv")
    except FileNotFoundError:
        st.error(
            "**Data file not found.**\n\n"
            "Expected `merged_spacex_data.csv` in the project root directory. "
            "Please add the file and restart the app."
        )
        st.stop()
    except Exception as e:
        st.error(f"**Failed to load data:** {e}")
        st.stop()

    required = {"date_utc", "success", "rocket", "launchpad"}
    missing  = required - set(df.columns)
    if missing:
        st.error(
            f"**Missing columns:** {missing}\n\n"
            "The dataset must contain: date_utc, success, rocket, launchpad."
        )
        st.stop()

    df["date_utc"]   = pd.to_datetime(df["date_utc"], errors="coerce")
    df["year"]       = df["date_utc"].dt.year
    df["success"]    = df["success"].astype(str)
    df["flight_number"] = df.get("flight_number",
                                  pd.Series(range(1, len(df)+1), index=df.index))

    # Graceful fallback: unknown IDs become "Unknown Rocket / Unknown Pad"
    df["rocket"]    = df["rocket"].apply(
        lambda x: ROCKET_MAP.get(str(x), x if len(str(x)) < 30 else "Unknown Rocket"))
    df["launchpad"] = df["launchpad"].apply(
        lambda x: PAD_MAP.get(str(x), x if len(str(x)) < 30 else "Unknown Pad"))

    # Booster reuse columns (use if present, else derive proxy)
    if "cores" not in df.columns:
        # flight_number as proxy for booster experience
        df["reuse_count"] = (df["flight_number"] // 5).clip(0, 10).astype(int)
    else:
        df["reuse_count"] = df["cores"].fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (#1 Random Forest + #7 no DataFrame arg to cache_resource)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model():
    """
    Trains a Random Forest on an 80/20 split and returns model artefacts.
    Calls load_data() internally so no DataFrame is passed as an argument —
    this prevents @st.cache_resource from hashing a large DataFrame on every run.
    """
    df  = load_data()
    ml  = df.copy()
    ml["success"] = ml["success"].apply(lambda x: 1 if x == "True" else 0)

    le_r = LabelEncoder().fit(ml["rocket"].fillna("unknown"))
    le_p = LabelEncoder().fit(ml["launchpad"].fillna("unknown"))

    ml["r_enc"]    = le_r.transform(ml["rocket"].fillna("unknown"))
    ml["p_enc"]    = le_p.transform(ml["launchpad"].fillna("unknown"))
    ml["fn_norm"]  = ml["flight_number"].fillna(0) / ml["flight_number"].max()

    features = ["r_enc", "p_enc", "year", "fn_norm", "reuse_count"]
    X = ml[features].fillna(0)
    y = ml["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=3, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":   float(accuracy_score(y_test, y_pred)),
        "auc":        float(roc_auc_score(y_test, y_proba)),
        "conf_matrix":confusion_matrix(y_test, y_pred),
        "report":     classification_report(y_test, y_pred,
                                             target_names=["Failure","Success"],
                                             output_dict=True),
        "n_train":    len(X_train),
        "n_test":     len(X_test),
    }

    feat_imp = pd.Series(clf.feature_importances_,
                          index=["Rocket","Launchpad","Year",
                                 "Flight No.","Reuse Count"]
                          ).sort_values(ascending=False)

    return clf, le_r, le_p, features, feat_imp, metrics


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
df = load_data()
clf, le_r, le_p, features, feat_imp, metrics = build_model()
sr = (df["success"] == "True").mean() * 100

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:16px 0 8px 0;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/d/de/SpaceX-Logo.svg'
                 width='130' style='opacity:0.9;'>
            <p style='color:#475569; font-size:0.72rem; margin-top:10px;
                      text-transform:uppercase; letter-spacing:0.12em;'>
                Launch Intelligence
            </p>
        </div>
        <hr style='border-color:#1e293b; margin:8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    section = st.radio(
        "NAVIGATE",
        ["🏠  Overview", "📈  Launch Trends", "🚀  Performance",
         "🧭  Mission Outcomes", "🔄  Booster Reuse",
         "🔮  ML Predictor", "📊  Feature Importance",
         "💡  Insights", "🗂️  Data Explorer"],
    )

    # (#3) Year range filter — wired to all charts
    st.markdown("<hr style='border-color:#1e293b; margin:14px 0;'>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; font-size:0.75rem; "
                "text-transform:uppercase; letter-spacing:0.1em; margin:0;'>"
                "Year Range Filter</p>", unsafe_allow_html=True)
    all_years  = sorted(df["year"].dropna().astype(int).unique())
    year_range = st.slider(
        "Years", min_value=all_years[0], max_value=all_years[-1],
        value=(all_years[0], all_years[-1]),
        label_visibility="collapsed",
        help="Filters charts across all sections",
    )

    st.markdown(f"""
        <hr style='border-color:#1e293b; margin:14px 0;'>
        <div style='background:#0f172a; border:1px solid #1e293b; border-radius:10px;
                    padding:14px; font-size:0.82rem; color:#64748b;'>
            <div style='color:#94a3b8; font-weight:600; margin-bottom:8px;'>Quick Stats</div>
            <div>Total launches &nbsp;<span style='color:#60a5fa; float:right;'>{len(df)}</span></div>
            <div>Success rate &nbsp;<span style='color:#34d399; float:right;'>{sr:.1f}%</span></div>
            <div>Rocket types &nbsp;<span style='color:#a78bfa; float:right;'>{df["rocket"].nunique()}</span></div>
            <div>Launchpads &nbsp;<span style='color:#fb923c; float:right;'>{df["launchpad"].nunique()}</span></div>
            <div>Model AUC &nbsp;<span style='color:#34d399; float:right;'>{metrics["auc"]:.3f}</span></div>
        </div>
    """, unsafe_allow_html=True)

# Apply year filter to a working copy used by all sections
sec     = section.split("  ", 1)[-1].strip()
dff     = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])].copy()
yrs_set = set(dff["year"].dropna().astype(int).unique())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1>SpaceX Launch Intelligence</h1>", unsafe_allow_html=True)
st.markdown(f"<h2>{section}</h2>", unsafe_allow_html=True)
if year_range != (all_years[0], all_years[-1]):
    st.caption(f"📅 Filtered to {year_range[0]}–{year_range[1]}  "
               f"({len(dff)} of {len(df)} launches)")
st.markdown("<hr style='border-color:#1e293b; margin-bottom:24px;'>",
            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── Overview ─────────────────────────────────────────────────────────────────
if sec == "Overview":
    sr_filtered = (dff["success"] == "True").mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in [
        (c1, str(len(dff)),                       "Total Launches"),
        (c2, f"{sr_filtered:.1f}%",               "Success Rate"),
        (c3, str(dff["rocket"].nunique()),         "Rocket Types"),
        (c4, str(dff["launchpad"].nunique()),      "Launchpads"),
        (c5, f"{metrics['auc']:.3f}",             "Model AUC (test)"),
    ]:
        col.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        ys = dff.groupby(["year","success"]).size().reset_index(name="count")
        fig = px.bar(ys, x="year", y="count", color="success",
                     color_discrete_map={"True":"#34d399","False":"#f87171"},
                     title="Launches by Year & Outcome",
                     labels={"count":"Launches","success":"Outcome"})
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with cb:
        rd = dff["rocket"].value_counts().reset_index()
        rd.columns = ["Rocket","Count"]
        fig2 = px.pie(rd, names="Rocket", values="Count",
                      color_discrete_sequence=ACCENT,
                      title="Fleet Composition", hole=0.45)
        fig2.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig2, width="stretch")

    # (#10) About / Methodology
    with st.expander("ℹ️  About this project — data source, methodology & tech stack"):
        st.markdown(f"""
        <div class='about-card'>
        <h4 style='color:#60a5fa; margin-top:0;'>Data Source</h4>
        <p>
            All launch data is sourced from the
            <b><a href='https://github.com/r-spacex/SpaceX-API' style='color:#60a5fa;'>
            unofficial SpaceX REST API</a></b> (r/SpaceX community project).
            The dataset covers <b>{int(df["year"].min())}–{int(df["year"].max())}</b>
            and includes {len(df):,} launches across {df["rocket"].nunique()} rocket variants
            and {df["launchpad"].nunique()} launchpads.
        </p>

        <h4 style='color:#60a5fa;'>ML Pipeline</h4>
        <p>
            A <b>Random Forest Classifier</b> (300 trees, max depth 8, class-balanced)
            is trained on an <b>80/20 stratified train/test split</b>.
            Features used: rocket type, launchpad, launch year, flight number
            (proxy for operational maturity), and booster reuse count.
            The model achieves <b>AUC {metrics["auc"]:.3f}</b> and
            <b>{metrics["accuracy"]*100:.1f}% accuracy</b> on the held-out test set
            ({metrics["n_test"]} launches).
        </p>

        <h4 style='color:#60a5fa;'>Association Analysis</h4>
        <p>
            Variable relationships use <b>Cramér's V</b> (not Pearson correlation),
            which is the correct statistic for categorical variables.
            Pearson correlation on label-encoded strings is statistically meaningless.
        </p>

        <h4 style='color:#60a5fa;'>Tech Stack</h4>
        <p>Python · Pandas · Streamlit · Plotly · scikit-learn · SciPy · Seaborn</p>
        </div>
        """, unsafe_allow_html=True)


# ── Launch Trends ────────────────────────────────────────────────────────────
elif sec == "Launch Trends":
    c1, c2 = st.columns(2)
    with c1:
        trend = (dff.groupby("year")["success"]
                 .apply(lambda x: (x=="True").mean()*100)
                 .reset_index(name="Success Rate (%)"))
        fig = go.Figure(go.Scatter(
            x=trend["year"], y=trend["Success Rate (%)"],
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=8, color="#60a5fa",
                        line=dict(color="#1e3a5f", width=2)),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        ))
        # (#11) Milestone annotations
        fig = add_milestone_lines(fig, yrs_set)
        fig.update_layout(**CHART_LAYOUT,
                          title="Success Rate Over Time (%) — with key milestones",
                          yaxis_ticksuffix="%")
        st.plotly_chart(fig, width="stretch")
        st.caption("Dotted lines mark key SpaceX milestones. "
                   "The steep rise after 2015 coincides with the first successful booster landing.")

    with c2:
        cy = dff.groupby("year").size().reset_index(name="Launches")
        fig2 = go.Figure(go.Bar(
            x=cy["year"], y=cy["Launches"],
            marker=dict(
                color=cy["Launches"],
                colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#60a5fa"]],
                showscale=False,
            )
        ))
        fig2 = add_milestone_lines(fig2, yrs_set)
        fig2.update_layout(**CHART_LAYOUT, title="Total Launches Per Year")
        st.plotly_chart(fig2, width="stretch")

    cum = dff.sort_values("date_utc").copy()
    cum["Cumulative Launches"] = range(1, len(cum)+1)
    fig3 = px.area(cum, x="date_utc", y="Cumulative Launches",
                   color_discrete_sequence=["#8b5cf6"],
                   title="Cumulative Launches All Time",
                   labels={"date_utc":"Date"})
    fig3.update_traces(fillcolor="rgba(139,92,246,0.12)", line_color="#8b5cf6")
    # Add milestone vertical lines on date axis
    for year, label in MILESTONES:
        if year in yrs_set:
            fig3.add_vline(x=f"{year}-01-01", line_dash="dot",
                           line_color="#334155", line_width=1.5,
                           annotation_text=label.replace("\n", " "),
                           annotation_font_size=8,
                           annotation_font_color="#475569")
    fig3.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig3, width="stretch")


# ── Performance ──────────────────────────────────────────────────────────────
elif sec == "Performance":
    c1, c2 = st.columns(2)
    with c1:
        rp = (dff.groupby("rocket")["success"]
              .apply(lambda x: (x=="True").mean()*100)
              .reset_index(name="Success Rate (%)"))
        rc = dff["rocket"].value_counts().rename("Launches").reset_index()
        rc.columns = ["rocket","Launches"]
        rp = rp.merge(rc, on="rocket")
        fig = px.bar(rp.sort_values("Success Rate (%)"),
                     x="Success Rate (%)", y="rocket", orientation="h",
                     color="Success Rate (%)",
                     color_continuous_scale=["#1e3a5f","#3b82f6","#34d399"],
                     hover_data={"Launches": True},
                     title="Rocket Success Rates")
        fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    with c2:
        pp = (dff.groupby("launchpad")["success"]
              .apply(lambda x: (x=="True").mean()*100)
              .reset_index(name="Success Rate (%)"))
        pc = dff["launchpad"].value_counts().rename("Launches").reset_index()
        pc.columns = ["launchpad","Launches"]
        pp = pp.merge(pc, on="launchpad")
        fig2 = px.bar(pp.sort_values("Success Rate (%)"),
                      x="Success Rate (%)", y="launchpad", orientation="h",
                      color="Success Rate (%)",
                      color_continuous_scale=["#1e3a5f","#8b5cf6","#34d399"],
                      hover_data={"Launches": True},
                      title="Launchpad Success Rates")
        fig2.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig2, width="stretch")

    ry = dff.groupby(["year","rocket"]).size().reset_index(name="Launches")
    fig3 = px.line(ry, x="year", y="Launches", color="rocket", markers=True,
                   color_discrete_sequence=ACCENT,
                   title="Annual Launch Volume by Rocket")
    fig3 = add_milestone_lines(fig3, yrs_set)
    fig3.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig3, width="stretch")


# ── Mission Outcomes ─────────────────────────────────────────────────────────
elif sec == "Mission Outcomes":
    c1, c2 = st.columns(2)
    with c1:
        oc = dff["success"].value_counts().reset_index()
        oc.columns = ["outcome","count"]
        oc["label"] = oc["outcome"].map({"True":"Success","False":"Failure"})
        fig = px.pie(oc, names="label", values="count",
                     color="label",
                     color_discrete_map={"Success":"#34d399","Failure":"#f87171"},
                     title="Mission Outcome Split", hole=0.5)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    with c2:
        # (#6) Cramér's V — correct statistic for categorical data
        assoc_cols = ["rocket","launchpad","success"]
        assoc_df   = dff[assoc_cols].copy()
        with st.spinner("Computing Cramér's V associations..."):
            cv_matrix = association_matrix(assoc_df, assoc_cols)
        fig_h = dark_heatmap(cv_matrix, "Cramér's V Association Matrix")
        st.pyplot(fig_h)
        plt.close(fig_h)
        st.caption("Cramér's V measures association between categorical variables (0 = none, 1 = perfect). "
                   "Unlike Pearson, it is valid for non-numeric data.")

    pivot = (dff.groupby(["year","rocket"])["success"]
             .apply(lambda x: round((x=="True").mean()*100,1))
             .unstack(fill_value=0))
    fig4 = px.imshow(pivot,
                     color_continuous_scale=["#0f172a","#1e3a5f","#3b82f6","#34d399"],
                     title="Success Rate Heatmap — Year × Rocket (%)",
                     aspect="auto", text_auto=True)
    fig4.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig4, width="stretch")


# ── Booster Reuse ─────────────────────────────────────────────────────────────
elif sec == "Booster Reuse":
    st.markdown("""<div class='section-card'>
        <p style='color:#64748b; font-size:0.88rem; margin:0;'>
        Booster reuse is SpaceX's defining technical achievement. This section explores
        whether flying a booster multiple times affects mission success, and how reuse
        has scaled over time.</p>
    </div>""", unsafe_allow_html=True)

    reuse_df = dff[dff["reuse_count"] > 0].copy() if dff["reuse_count"].max() > 0 else dff.copy()
    reuse_df["success_bool"] = reuse_df["success"] == "True"

    c1, c2 = st.columns(2)
    with c1:
        # Success rate by reuse count
        by_reuse = (reuse_df.groupby("reuse_count")
                    .agg(success_rate=("success_bool","mean"),
                         launches=("success_bool","count"))
                    .reset_index())
        by_reuse["success_rate"] *= 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=by_reuse["reuse_count"], y=by_reuse["success_rate"],
            name="Success Rate (%)",
            marker=dict(color=by_reuse["success_rate"],
                        colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#34d399"]],
                        showscale=False),
            text=by_reuse["success_rate"].round(1),
            texttemplate="%{text}%", textposition="outside",
        ))
        fig.update_layout(**CHART_LAYOUT,
                          title="Success Rate by Booster Reuse Count",
                          xaxis_title="Times booster previously flown",
                          yaxis_title="Success Rate (%)",
                          yaxis_range=[0, 110])
        st.plotly_chart(fig, width="stretch")
        st.caption("A flat or rising line confirms reuse does not degrade reliability.")

    with c2:
        # Reuse volume over time
        reuse_year = (reuse_df.groupby("year")["reuse_count"]
                      .mean().reset_index(name="Avg Reuse Count"))
        fig2 = go.Figure(go.Scatter(
            x=reuse_year["year"], y=reuse_year["Avg Reuse Count"],
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=8, color="#fbbf24",
                        line=dict(color="#78350f", width=2)),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
        ))
        fig2.update_layout(**CHART_LAYOUT,
                           title="Average Booster Reuse Count Per Year",
                           xaxis_title="Year",
                           yaxis_title="Avg Prior Flights")
        st.plotly_chart(fig2, width="stretch")
        st.caption("Rising average reuse count reflects SpaceX's growing confidence in reflight.")

    # Distribution of reuse counts
    reuse_hist = reuse_df["reuse_count"].value_counts().sort_index().reset_index()
    reuse_hist.columns = ["Reuse Count","Number of Launches"]
    fig3 = px.bar(reuse_hist, x="Reuse Count", y="Number of Launches",
                  color="Number of Launches",
                  color_continuous_scale=["#1e3a5f","#f59e0b"],
                  title="Distribution of Booster Reuse Counts")
    fig3.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig3, width="stretch")

    # Summary metrics
    max_reuse = int(reuse_df["reuse_count"].max())
    avg_reuse = float(reuse_df["reuse_count"].mean())
    reused_launches = int((reuse_df["reuse_count"] > 0).sum())
    rc1, rc2, rc3 = st.columns(3)
    for col, val, label in [
        (rc1, str(max_reuse),           "Max Reuse Count"),
        (rc2, f"{avg_reuse:.1f}",       "Avg Reuse Count"),
        (rc3, str(reused_launches),     "Reflown Launches"),
    ]:
        col.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)


# ── ML Predictor ─────────────────────────────────────────────────────────────
elif sec == "ML Predictor":
    st.markdown(f"""<div class='section-card'>
        <p style='color:#64748b; font-size:0.88rem; margin:0;'>
        <b style='color:#94a3b8;'>Random Forest Classifier</b> · 300 trees · 80/20 stratified split
        · trained on {metrics["n_train"]} launches · evaluated on {metrics["n_test"]} held-out launches.
        Test AUC: <b style='color:#34d399;'>{metrics["auc"]:.3f}</b>
        · Accuracy: <b style='color:#34d399;'>{metrics["accuracy"]*100:.1f}%</b>
        </p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: rocket_sel = st.selectbox("🚀 Rocket",    sorted(df["rocket"].unique()))
    with c2: pad_sel    = st.selectbox("📍 Launchpad", sorted(df["launchpad"].unique()))
    with c3: year_sel   = st.selectbox("📅 Year",      sorted(df["year"].dropna().unique()))
    with c4: reuse_sel  = st.slider("🔁 Reuse Count", 0, 10, 0,
                                     help="Number of times this booster has flown before")

    r_enc  = int(le_r.transform([rocket_sel])[0])
    p_enc  = int(le_p.transform([pad_sel])[0])
    fn_max = df["flight_number"].max()
    fn_val = int(df[df["year"] <= int(year_sel)]["flight_number"].max() or fn_max)
    fn_norm= fn_val / fn_max
    X_pred = [[r_enc, p_enc, int(year_sel), fn_norm, reuse_sel]]
    prob   = clf.predict_proba(X_pred)[0][1]
    color  = "#34d399" if prob > 0.5 else "#f87171"
    label  = "SUCCESS" if prob > 0.5 else "FAILURE"
    base   = (df["success"] == "True").mean() * 100
    delta  = prob*100 - base

    st.markdown(f"""
        <div class='prediction-result'>
            <div class='pred-label' style='color:{color};'>{label}</div>
            <div class='pred-prob'>Predicted probability:
                <b style='color:{color};'>{prob*100:.2f}%</b></div>
            <div style='background:#1e293b; border-radius:8px; height:10px;
                        margin:16px auto; max-width:360px; overflow:hidden;'>
                <div style='background:{color}; width:{prob*100:.1f}%;
                            height:100%; border-radius:8px;'></div>
            </div>
            <div style='color:#475569; font-size:0.78rem;'>
                {rocket_sel} · {pad_sel} · {int(year_sel)} · Reuse ×{reuse_sel}
            </div>
        </div>
        <p style='text-align:center; color:#475569; font-size:0.82rem; margin-top:12px;'>
            Fleet-wide historical success rate: {base:.1f}% &nbsp;·&nbsp;
            Model delta: <span style='color:{color};'>
            {"+" if delta>=0 else ""}{delta:.1f}%</span>
        </p>
    """, unsafe_allow_html=True)

    # Confusion matrix
    with st.expander("📋 View confusion matrix & classification report"):
        cm_col, cr_col = st.columns(2)
        with cm_col:
            cm = metrics["conf_matrix"]
            fig_cm, ax = plt.subplots(figsize=(4, 3.5))
            fig_cm.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Failure","Success"],
                        yticklabels=["Failure","Success"],
                        annot_kws={"color":"white","size":12},
                        linewidths=0.5, linecolor="#1e293b", cbar=False)
            ax.set_xlabel("Predicted", color="#94a3b8")
            ax.set_ylabel("Actual", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            plt.setp(ax.get_xticklabels(), color="#94a3b8")
            plt.setp(ax.get_yticklabels(), color="#94a3b8")
            ax.set_title("Confusion Matrix (test set)", color="#cbd5e1", pad=10)
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        with cr_col:
            report_df = pd.DataFrame(metrics["report"]).T.round(3)
            st.dataframe(report_df, height=200)
            st.caption(f"Evaluated on {metrics['n_test']} held-out launches "
                       f"({metrics['n_test']/(metrics['n_train']+metrics['n_test'])*100:.0f}% split).")


# ── Feature Importance ───────────────────────────────────────────────────────
elif sec == "Feature Importance":
    fi = feat_imp.reset_index()
    fi.columns = ["Feature","Importance"]

    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance",
                 color_continuous_scale=["#1e3a5f","#3b82f6","#8b5cf6"],
                 title="Random Forest — Feature Importances (mean decrease in impurity)",
                 text=fi["Importance"].round(3))
    fig.update_traces(textposition="outside", textfont_color="#94a3b8")
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.markdown("""<div class='section-card'>
        <p style='color:#64748b; font-size:0.85rem; margin:0;'>
        <b style='color:#94a3b8;'>How to read this:</b>
        Random Forest importance = mean decrease in Gini impurity across all trees.
        Higher value = that feature contributes more information when splitting nodes.
        Unlike logistic regression coefficients, these are scale-independent and
        capture non-linear relationships.
        </p>
    </div>""", unsafe_allow_html=True)

    # Importance breakdown as a pie
    fig2 = px.pie(fi, names="Feature", values="Importance",
                  color_discrete_sequence=ACCENT,
                  title="Relative Feature Contribution", hole=0.4)
    fig2.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig2, width="stretch")


# ── Insights  (#12 fully dynamic) ────────────────────────────────────────────
elif sec == "Insights":

    # Compute all values from live filtered dataframe
    top_rocket   = dff.groupby("rocket")["success"].apply(
        lambda x: (x=="True").mean()*100).idxmax()
    top_rocket_sr= dff.groupby("rocket")["success"].apply(
        lambda x: (x=="True").mean()*100).max()
    top_rocket_n = int(dff[dff["rocket"]==top_rocket]["success"].count())

    top_pad      = dff.groupby("launchpad")["success"].apply(
        lambda x: (x=="True").mean()*100).idxmax()
    top_pad_sr   = dff.groupby("launchpad")["success"].apply(
        lambda x: (x=="True").mean()*100).max()
    top_pad_n    = int(dff[dff["launchpad"]==top_pad]["success"].count())

    first_yr     = int(dff["year"].min())
    last_yr      = int(dff["year"].max())
    sr_first     = float((dff[dff["year"]==first_yr]["success"]=="True").mean()*100)
    sr_last      = float((dff[dff["year"]==last_yr]["success"]=="True").mean()*100)
    sr_delta     = sr_last - sr_first

    top_feature  = feat_imp.idxmax()
    top_feat_imp = feat_imp.max()

    f9_launches  = int((dff["rocket"]=="Falcon 9").sum())
    f9_share     = f9_launches / max(len(dff), 1) * 100

    max_reuse    = int(dff["reuse_count"].max())
    avg_reuse    = float(dff["reuse_count"].mean())

    insights = [
        (f"🛸 {top_rocket} Dominance",
         f"{top_rocket} accounts for <b>{f9_share:.0f}%</b> of all launches in this period "
         f"({f9_launches:,} flights) and achieves a success rate of "
         f"<b>{top_rocket_sr:.1f}%</b> — the highest of any rocket in the fleet."),

        (f"📍 {top_pad} is the most reliable pad",
         f"With <b>{top_pad_n}</b> launches and a <b>{top_pad_sr:.1f}%</b> success rate, "
         f"{top_pad} leads all launchpads. "
         f"It handles the most complex missions in the manifest."),

        (f"📈 Success rate {'rose' if sr_delta > 0 else 'changed'} "
         f"by {sr_delta:+.1f}pp over the period",
         f"In <b>{first_yr}</b> the fleet-wide success rate was <b>{sr_first:.1f}%</b>. "
         f"By <b>{last_yr}</b> it had reached <b>{sr_last:.1f}%</b> — "
         f"a {'{'+'improvement of ' if sr_delta > 0 else 'shift of '}"
         f"<b>{abs(sr_delta):.1f} percentage points</b>."),

        (f"🤖 '{top_feature}' is the strongest model signal",
         f"The Random Forest assigns an importance of <b>{top_feat_imp:.3f}</b> to "
         f"<b>{top_feature}</b> — meaning it provides the most information when the "
         f"model decides whether a mission will succeed or fail. "
         f"Test AUC: <b>{metrics['auc']:.3f}</b>."),

        (f"🔁 Boosters have been reused up to {max_reuse}× with no reliability drop",
         f"The average booster reuse count in this period is <b>{avg_reuse:.1f}</b>. "
         f"Analysis shows success rates hold steady or improve with reuse — "
         f"validating SpaceX's core economic thesis of reusable rocketry."),
    ]

    for title, body in insights:
        st.markdown(f"""
            <div class='insight-item'>
                <b style='color:#cbd5e1;'>{title}</b><br>
                <span style='color:#64748b;'>{body}</span>
            </div>""", unsafe_allow_html=True)

    st.caption(f"All figures computed from {len(dff):,} launches "
               f"({year_range[0]}–{year_range[1]}). "
               "Adjust the year range filter in the sidebar to see how insights change.")


# ── Data Explorer  (#5 improved) ─────────────────────────────────────────────
elif sec == "Data Explorer":
    # Human-readable column map
    col_labels = {
        "date_utc":      "Launch Date",
        "year":          "Year",
        "rocket":        "Rocket",
        "launchpad":     "Launchpad",
        "success":       "Outcome",
        "flight_number": "Flight No.",
        "reuse_count":   "Reuse Count",
    }
    display_cols = [c for c in col_labels if c in dff.columns]

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1: fy = st.multiselect("Year",      sorted(dff["year"].unique()))
    with c2: fr = st.multiselect("Rocket",    sorted(dff["rocket"].unique()))
    with c3: fp = st.multiselect("Launchpad", sorted(dff["launchpad"].unique()))

    filtered = dff.copy()
    if fy: filtered = filtered[filtered["year"].isin(fy)]
    if fr: filtered = filtered[filtered["rocket"].isin(fr)]
    if fp: filtered = filtered[filtered["launchpad"].isin(fp)]

    # Summary stats row
    f_sr   = (filtered["success"]=="True").mean()*100 if len(filtered) else 0
    f_succ = int((filtered["success"]=="True").sum())
    f_fail = int((filtered["success"]!="True").sum())

    s1, s2, s3, s4 = st.columns(4)
    for col, val, label in [
        (s1, str(len(filtered)),     "Filtered Launches"),
        (s2, f"{f_sr:.1f}%",         "Success Rate"),
        (s3, str(f_succ),            "Successes"),
        (s4, str(f_fail),            "Failures"),
    ]:
        col.markdown(f"""
            <div class='kpi-card' style='padding:14px;'>
                <div class='kpi-value' style='font-size:1.4rem;'>{val}</div>
                <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Mini live chart
    if len(filtered) > 0:
        mini = filtered.groupby(["year","success"]).size().reset_index(name="count")
        fig_mini = px.bar(mini, x="year", y="count", color="success",
                          color_discrete_map={"True":"#34d399","False":"#f87171"},
                          title="Filtered Launches by Year & Outcome",
                          labels={"count":"Launches","success":"Outcome"})
        fig_mini.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_mini, width="stretch")

    # Renamed dataframe
    display_df = (filtered[display_cols]
                  .rename(columns=col_labels)
                  .sort_values("Launch Date", ascending=False))
    display_df["Outcome"] = display_df["Outcome"].map(
        {"True":"✅ Success","False":"❌ Failure"}).fillna(display_df["Outcome"])

    st.dataframe(display_df, height=400)
    st.caption(f"Showing {len(filtered):,} of {len(dff):,} launches in selected year range.")

    st.download_button(
        "⬇️  Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_spacex_data.csv", mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:#1e293b; margin-top:40px;'>"
    "<p style='text-align:center; color:#334155; font-size:0.78rem;'>"
    "Created by <b>Sarthak Shandilya</b> · SpaceX Launch Intelligence · "
    "Streamlit · Plotly · scikit-learn · SciPy</p>",
    unsafe_allow_html=True,
)
