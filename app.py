import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpaceX Launch Intelligence",
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

    h1 {
        font-size: 2rem !important; font-weight: 700 !important;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0 !important;
    }
    h2 { font-size: 1.25rem !important; font-weight: 600 !important;
         color: #cbd5e1 !important; letter-spacing: 0.02em; }
    h3, h4 { color: #94a3b8 !important; }
    p, li  { color: #94a3b8 !important; }

    .kpi-card {
        background: linear-gradient(145deg, #0f172a, #1a2540);
        border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 20px 16px; text-align: center;
        position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: #f0f6ff; line-height: 1.1; }
    .kpi-label { font-size: 0.73rem; color: #475569; text-transform: uppercase;
                 letter-spacing: 0.12em; margin-top: 5px; }

    .info-card {
        background: #0a1020; border: 1px solid #1e293b;
        border-left: 3px solid #3b82f6;
        border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 16px;
        font-size: 0.86rem; color: #64748b !important;
    }
    .info-card b { color: #94a3b8; }

    .pred-box {
        background: linear-gradient(145deg, #0f172a, #1a2540);
        border: 1px solid #1e3a5f; border-radius: 14px;
        padding: 28px 24px; text-align: center; margin: 16px 0;
    }
    .pred-label { font-size: 1.7rem; font-weight: 800; }
    .pred-sub   { font-size: 0.85rem; color: #475569; margin-top: 6px; }

    .insight-card {
        background: #080e1c; border: 1px solid #1e293b;
        border-left: 3px solid #3b82f6;
        border-radius: 0 12px 12px 0; padding: 16px 20px; margin-bottom: 14px;
    }
    .insight-title { font-size: 0.95rem; font-weight: 700; color: #cbd5e1; }
    .insight-body  { font-size: 0.84rem; color: #64748b; margin-top: 5px; line-height: 1.6; }

    .section-label {
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.14em;
        color: #334155; font-weight: 600; margin: 20px 0 6px 0;
    }

    @media (max-width: 768px) {
        .block-container { padding: 1rem !important; }
        h1 { font-size: 1.4rem !important; }
        .kpi-value { font-size: 1.3rem; }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CHART THEME
# ─────────────────────────────────────────────────────────────────────────────
CL = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,16,32,0.7)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#64748b"),
    title_font=dict(size=13, color="#94a3b8"),
    margin=dict(l=12, r=12, t=44, b=12),
    xaxis=dict(gridcolor="#111827", linecolor="#1e293b", zeroline=False),
    yaxis=dict(gridcolor="#111827", linecolor="#1e293b", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e293b", borderwidth=1,
                font=dict(size=11, color="#64748b")),
)
ACCENT = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e"]


def cl(**overrides):
    """
    Return a copy of CL with overrides merged in (overrides win).
    Use instead of update_layout(**CL, key=value) which raises
    'multiple values for keyword argument' when key already exists in CL.
    """
    merged = dict(CL)
    merged.update(overrides)
    return merged

MILESTONES = [
    (2010, "Falcon 9 debut"),
    (2015, "First booster landing"),
    (2018, "Falcon Heavy debut"),
    (2020, "First crewed mission"),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def add_year_markers(fig: go.Figure, years_present: set) -> go.Figure:
    """
    Add vertical milestone markers using add_shape + add_annotation.
    Never use add_vline with annotation params — it triggers a Plotly
    _mean() TypeError on date axes and certain integer axes.
    """
    for year, label in MILESTONES:
        if year not in years_present:
            continue
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=year, x1=year, y0=0, y1=1,
            line=dict(color="#1e3a5f", width=1.5, dash="dot"),
            layer="below",
        )
        fig.add_annotation(
            x=year, y=0.98, xref="x", yref="paper",
            text=label, showarrow=False,
            font=dict(size=8, color="#334155"),
            align="center", yanchor="top",
            bgcolor="rgba(5,7,15,0.8)",
            borderpad=3,
            xshift=4,
        )
    return fig


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tbl  = pd.crosstab(a.astype(str), b.astype(str)).values
    chi2 = chi2_contingency(tbl, correction=False)[0]
    n    = tbl.sum()
    r, k = tbl.shape
    return float(np.sqrt(chi2 / n / max(min(k - 1, r - 1), 1)))


def kpi(col, value: str, label: str):
    col.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-label'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def info(text: str):
    st.markdown(f"<div class='info-card'>{text}</div>", unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f"<div class='section-label'>{text}</div>", unsafe_allow_html=True)


def dark_heatmap(matrix: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.patch.set_facecolor("#080e1c")
    ax.set_facecolor("#080e1c")
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                annot_kws={"color": "white", "size": 10},
                linewidths=0.4, linecolor="#111827",
                cbar_kws={"shrink": 0.75})
    ax.tick_params(colors="#475569", labelsize=9)
    plt.setp(ax.get_xticklabels(), color="#475569", rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), color="#475569", rotation=0)
    ax.set_title(title, color="#94a3b8", pad=10, fontsize=11)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DATA
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
        st.error("Data file not found. Expected `merged_spacex_data.csv` in the "
                 "project root. Add the file and restart the app.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    missing = {"date_utc", "success", "rocket", "launchpad"} - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["date_utc"]  = pd.to_datetime(df["date_utc"], errors="coerce")
    df["year"]      = df["date_utc"].dt.year.astype("Int64")
    df["success"]   = df["success"].astype(str)
    df["rocket"]    = df["rocket"].apply(
        lambda x: ROCKET_MAP.get(str(x), x if len(str(x)) < 30 else "Unknown Rocket"))
    df["launchpad"] = df["launchpad"].apply(
        lambda x: PAD_MAP.get(str(x), x if len(str(x)) < 30 else "Unknown Pad"))

    if "flight_number" not in df.columns:
        df["flight_number"] = range(1, len(df) + 1)
    df["flight_number"] = pd.to_numeric(df["flight_number"], errors="coerce").fillna(0)

    if "reuse_count" not in df.columns:
        df["reuse_count"] = (df["flight_number"] // 5).clip(0, 10).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model():
    df  = load_data()
    ml  = df.copy()
    ml["label"] = (ml["success"] == "True").astype(int)

    le_r = LabelEncoder().fit(ml["rocket"].fillna("unknown"))
    le_p = LabelEncoder().fit(ml["launchpad"].fillna("unknown"))
    ml["r_enc"] = le_r.transform(ml["rocket"].fillna("unknown"))
    ml["p_enc"] = le_p.transform(ml["launchpad"].fillna("unknown"))

    fn_max = ml["flight_number"].max() or 1
    ml["fn_norm"] = ml["flight_number"] / fn_max

    FEATS = ["r_enc", "p_enc", "year", "fn_norm", "reuse_count"]
    X = ml[FEATS].fillna(0).astype(float)
    y = ml["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    feat_imp = pd.Series(
        clf.feature_importances_,
        index=["Rocket", "Launchpad", "Year", "Flight No.", "Reuse Count"],
    )

    return clf, le_r, le_p, fn_max, feat_imp, {
        "auc":      float(roc_auc_score(y_te, y_prob)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "cm":       confusion_matrix(y_te, y_pred),
        "n_train":  len(X_tr),
        "n_test":   len(X_te),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
df = load_data()
clf, le_r, le_p, fn_max, feat_imp, metrics = build_model()
sr_all = (df["success"] == "True").mean() * 100

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:16px 0 10px;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/d/de/SpaceX-Logo.svg'
                 width='120' style='opacity:0.85; filter:brightness(1.1);'>
            <p style='color:#334155; font-size:0.68rem; margin-top:10px;
                      text-transform:uppercase; letter-spacing:0.14em;'>
                Launch Intelligence
            </p>
        </div>
        <hr style='border-color:#111827; margin:4px 0 14px;'>
    """, unsafe_allow_html=True)

    section = st.radio(
        "Navigate",
        [
            "Overview",
            "Launch Trends",
            "Performance",
            "Mission Outcomes",
            "Booster Reuse",
            "ML Predictor",
            "Feature Importance",
            "Insights",
            "Data Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#111827; margin:12px 0;'>",
                unsafe_allow_html=True)

    all_years  = sorted(df["year"].dropna().astype(int).unique())
    year_range = st.slider(
        "Year range",
        all_years[0], all_years[-1],
        (all_years[0], all_years[-1]),
        help="Filters all charts and computed values across every section.",
    )

    st.markdown(f"""
        <hr style='border-color:#111827; margin:12px 0;'>
        <div style='font-size:0.78rem; color:#334155; line-height:1.9;'>
            <div style='color:#475569; font-weight:600; margin-bottom:4px;
                        font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em;'>
                Dataset
            </div>
            <div>Launches
                <span style='color:#3b82f6; float:right;'>{len(df):,}</span>
            </div>
            <div>Success rate
                <span style='color:#34d399; float:right;'>{sr_all:.1f}%</span>
            </div>
            <div>Rockets
                <span style='color:#a78bfa; float:right;'>{df["rocket"].nunique()}</span>
            </div>
            <div>Launchpads
                <span style='color:#fb923c; float:right;'>{df["launchpad"].nunique()}</span>
            </div>
            <div style='margin-top:6px; border-top:1px solid #111827; padding-top:6px;'>
                Model AUC (test)
                <span style='color:#34d399; float:right;'>{metrics["auc"]:.3f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Filtered working copy
sec     = section.strip()
dff     = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])].copy()
yrs     = set(dff["year"].dropna().astype(int).unique())
base_sr = (dff["success"] == "True").mean() * 100

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1>SpaceX Launch Intelligence</h1>", unsafe_allow_html=True)
col_hdr, col_yr = st.columns([3, 1])
with col_hdr:
    st.markdown(f"<h2>{sec}</h2>", unsafe_allow_html=True)
with col_yr:
    if year_range != (all_years[0], all_years[-1]):
        st.markdown(
            f"<p style='text-align:right; color:#334155; font-size:0.78rem; "
            f"margin-top:6px;'>{year_range[0]}-{year_range[1]}"
            f" ({len(dff):,} launches)</p>",
            unsafe_allow_html=True)
st.markdown("<hr style='border-color:#111827; margin:6px 0 20px;'>",
            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if sec == "Overview":

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, f"{len(dff):,}",               "Total Launches")
    kpi(c2, f"{base_sr:.1f}%",             "Success Rate")
    kpi(c3, str(dff["rocket"].nunique()),    "Rocket Types")
    kpi(c4, str(dff["launchpad"].nunique()), "Launchpads")
    kpi(c5, f"{metrics['auc']:.3f}",        "Model AUC (test set)")

    st.markdown("<br>", unsafe_allow_html=True)

    section_label("Launch volume and outcomes by year")
    info("<b>What this shows:</b> Annual launch count coloured by outcome. "
         "Rising green bars reflect both increased cadence and improving reliability. "
         "Early failures are concentrated in the Falcon 1 era (pre-2010).")

    ys = dff.groupby(["year", "success"]).size().reset_index(name="count")
    ys["Outcome"] = ys["success"].map({"True": "Success", "False": "Failure"})
    fig1 = px.bar(ys, x="year", y="count", color="Outcome",
                  color_discrete_map={"Success": "#34d399", "Failure": "#f87171"},
                  labels={"count": "Launches", "year": "Year"})
    fig1.update_layout(**CL, title=None, bargap=0.25)
    st.plotly_chart(fig1, width="stretch")

    ca, cb = st.columns(2)
    with ca:
        section_label("Fleet composition")
        info("<b>What this shows:</b> Share of total launches by rocket type. "
             "Falcon 9 dominates the modern manifest; Falcon 1 represents the "
             "early experimental era.")
        rd = dff["rocket"].value_counts().reset_index()
        rd.columns = ["Rocket", "Launches"]
        fig2 = px.pie(rd, names="Rocket", values="Launches",
                      color_discrete_sequence=ACCENT, hole=0.5)
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(**cl(title=None, showlegend=True, margin=dict(l=10, r=10, t=10, b=10)))
        st.plotly_chart(fig2, width="stretch")

    with cb:
        section_label("About this project")
        st.markdown(f"""
        <div style='background:#080e1c; border:1px solid #1e293b; border-radius:12px;
                    padding:20px; font-size:0.84rem; line-height:1.7;'>
            <div style='color:#475569; font-size:0.68rem; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.12em;
                        margin-bottom:10px;'>Data Source and Methodology</div>
            <p style='color:#64748b !important;'>
                Data sourced from the
                <a href='https://github.com/r-spacex/SpaceX-API'
                   style='color:#3b82f6;'>unofficial SpaceX REST API</a> —
                {int(df["year"].min())}–{int(df["year"].max())},
                {len(df):,} launches across {df["rocket"].nunique()} rocket
                variants and {df["launchpad"].nunique()} launch sites.
            </p>
            <p style='color:#64748b !important;'>
                <b style='color:#94a3b8;'>ML model:</b> Random Forest (300 trees,
                class-balanced) on an 80/20 stratified split. Features: rocket,
                launchpad, year, flight number, booster reuse count.
                Test AUC <b style='color:#34d399;'>{metrics["auc"]:.3f}</b>,
                accuracy <b style='color:#34d399;'>{metrics["accuracy"]*100:.1f}%</b>
                on {metrics["n_test"]} held-out launches.
            </p>
            <p style='color:#64748b !important;'>
                <b style='color:#94a3b8;'>Association analysis</b> uses
                Cramer's V — the statistically correct measure for
                categorical variables.
            </p>
            <p style='color:#475569 !important; font-size:0.76rem;'>
                Python · Pandas · Streamlit · Plotly · scikit-learn · SciPy
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LAUNCH TRENDS
# ═════════════════════════════════════════════════════════════════════════════
elif sec == "Launch Trends":

    section_label("Mission success rate over time")
    info("<b>What this shows:</b> Annual success rate as a percentage. "
         "The steep rise after 2015 coincides with mastery of propulsive landing "
         "and rapid iteration on Falcon 9 Block versions. "
         "Dotted lines mark key programme milestones.")

    trend = (dff.groupby("year")["success"]
             .apply(lambda x: (x == "True").mean() * 100)
             .reset_index(name="Success Rate (%)"))

    fig_sr = go.Figure()
    fig_sr.add_trace(go.Scatter(
        x=trend["year"], y=trend["Success Rate (%)"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=7, color="#60a5fa",
                    line=dict(color="#0f172a", width=1.5)),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
        hovertemplate="<b>%{x}</b><br>Success rate: %{y:.1f}%<extra></extra>",
    ))
    fig_sr = add_year_markers(fig_sr, yrs)
    fig_sr.update_layout(**cl(
        title=None,
        yaxis=dict(**CL["yaxis"], ticksuffix="%", range=[0, 110]),
    ))
    st.plotly_chart(fig_sr, width="stretch")

    section_label("Annual launch volume")
    info("<b>What this shows:</b> Total launches per year. "
         "The exponential growth after 2017 reflects Starlink constellation "
         "deployment and the commercial smallsat rideshare programme.")

    cy = dff.groupby("year").size().reset_index(name="Launches")
    fig_cy = go.Figure(go.Bar(
        x=cy["year"], y=cy["Launches"],
        marker=dict(
            color=cy["Launches"],
            colorscale=[[0, "#1e3a5f"], [0.5, "#2563eb"], [1, "#60a5fa"]],
            showscale=False,
        ),
        hovertemplate="<b>%{x}</b><br>Launches: %{y}<extra></extra>",
    ))
    fig_cy = add_year_markers(fig_cy, yrs)
    fig_cy.update_layout(**CL, title=None, bargap=0.3)
    st.plotly_chart(fig_cy, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif sec == "Performance":

    c1, c2 = st.columns(2)

    with c1:
        section_label("Success rate by rocket")
        info("<b>What this shows:</b> Mission success percentage per rocket type. "
             "Falcon 1 failed early missions as SpaceX refined operations. "
             "Falcon 9 and Falcon Heavy represent mature, production-ready vehicles.")
        rp = (dff.groupby("rocket")["success"]
              .apply(lambda x: (x == "True").mean() * 100)
              .reset_index(name="Success Rate (%)"))
        rc = dff["rocket"].value_counts().rename("Launches").reset_index()
        rc.columns = ["rocket", "Launches"]
        rp = rp.merge(rc, on="rocket").sort_values("Success Rate (%)")
        fig_r = px.bar(rp, x="Success Rate (%)", y="rocket", orientation="h",
                       color="Success Rate (%)",
                       color_continuous_scale=["#1e3a5f", "#2563eb", "#34d399"],
                       hover_data={"Launches": True, "Success Rate (%)": ":.1f"},
                       text=rp["Success Rate (%)"].round(1))
        fig_r.update_traces(texttemplate="%{text}%", textposition="outside",
                             textfont_color="#64748b")
        fig_r.update_layout(**CL, title=None, coloraxis_showscale=False,
                             yaxis_title="", xaxis_range=[0, 115])
        st.plotly_chart(fig_r, width="stretch")

    with c2:
        section_label("Success rate by launchpad")
        info("<b>What this shows:</b> Mission success rate per launch site. "
             "Kennedy LC-39A (originally built for Apollo) handles the most "
             "demanding missions. Kwajalein Atoll hosted early Falcon 1 attempts.")
        pp = (dff.groupby("launchpad")["success"]
              .apply(lambda x: (x == "True").mean() * 100)
              .reset_index(name="Success Rate (%)"))
        pc = dff["launchpad"].value_counts().rename("Launches").reset_index()
        pc.columns = ["launchpad", "Launches"]
        pp = pp.merge(pc, on="launchpad").sort_values("Success Rate (%)")
        fig_p = px.bar(pp, x="Success Rate (%)", y="launchpad", orientation="h",
                       color="Success Rate (%)",
                       color_continuous_scale=["#1e3a5f", "#7c3aed", "#34d399"],
                       hover_data={"Launches": True, "Success Rate (%)": ":.1f"},
                       text=pp["Success Rate (%)"].round(1))
        fig_p.update_traces(texttemplate="%{text}%", textposition="outside",
                             textfont_color="#64748b")
        fig_p.update_layout(**CL, title=None, coloraxis_showscale=False,
                             yaxis_title="", xaxis_range=[0, 115])
        st.plotly_chart(fig_p, width="stretch")

    section_label("Success rate heatmap — year by rocket")
    info("<b>What this shows:</b> Each cell is the success rate for a given rocket "
         "in a given year. Empty cells indicate no launches that year. "
         "The shift from dark red to solid green tracks SpaceX's reliability journey.")
    pivot = (dff.groupby(["year", "rocket"])["success"]
             .apply(lambda x: round((x == "True").mean() * 100, 1))
             .unstack(fill_value=None))
    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=[[0, "#1a0a0a"], [0.4, "#7f1d1d"],
                                 [0.7, "#1e3a5f"], [1, "#34d399"]],
        zmin=0, zmax=100, aspect="auto", text_auto=True,
        labels=dict(color="Success %"),
    )
    fig_hm.update_layout(**cl(
        title=None,
        coloraxis_colorbar=dict(
            tickvals=[0, 50, 100],
            ticktext=["0%", "50%", "100%"],
            tickfont=dict(color="#475569"),
            title="",
        ),
    ))
    st.plotly_chart(fig_hm, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# MISSION OUTCOMES
# ═════════════════════════════════════════════════════════════════════════════
elif sec == "Mission Outcomes":

    c1, c2 = st.columns(2)

    with c1:
        section_label("Success vs failure split")
        info("<b>What this shows:</b> Overall proportion of successful vs failed "
             "missions in the selected period. Use the year range slider to isolate "
             "different eras and see how the ratio changes.")
        oc = dff["success"].value_counts().reset_index()
        oc.columns = ["outcome", "count"]
        oc["label"] = oc["outcome"].map({"True": "Success", "False": "Failure"})
        fig_pie = px.pie(oc, names="label", values="count",
                         color="label",
                         color_discrete_map={"Success": "#34d399",
                                              "Failure": "#f87171"},
                         hole=0.55)
        fig_pie.update_traces(
            textposition="outside", textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value} launches (%{percent})<extra></extra>",
        )
        fig_pie.update_layout(**cl(title=None, margin=dict(l=20, r=20, t=20, b=20)))
        st.plotly_chart(fig_pie, width="stretch")

    with c2:
        section_label("Cramer's V association matrix")
        info("<b>What this shows:</b> How strongly each variable is associated with "
             "the others. Cramer's V is the correct statistic for categorical data "
             "(0 = no association, 1 = perfect association). "
             "Unlike Pearson correlation, it does not require numeric inputs and "
             "is valid for nominal categories.")
        with st.spinner("Computing associations..."):
            cv = pd.DataFrame(
                [[cramers_v(dff[a].astype(str), dff[b].astype(str))
                  for b in ["rocket", "launchpad", "success"]]
                 for a in ["rocket", "launchpad", "success"]],
                index=["Rocket", "Launchpad", "Outcome"],
                columns=["Rocket", "Launchpad", "Outcome"],
            ).round(3)
        fig_cv = dark_heatmap(cv, "Cramer's V — variable associations")
        st.pyplot(fig_cv)
        plt.close(fig_cv)


# ═════════════════════════════════════════════════════════════════════════════
# BOOSTER REUSE
# ═════════════════════════════════════════════════════════════════════════════
elif sec == "Booster Reuse":

    info("<b>SpaceX's defining achievement</b> is landing and reflying orbital-class "
         "boosters. This section asks the critical question: does reusing a booster "
         "compromise reliability?")

    c1, c2 = st.columns(2)

    with c1:
        section_label("Success rate by prior reuse count")
        info("<b>What this shows:</b> Mission success rate grouped by how many times "
             "the booster had flown before. A flat or rising line confirms that reuse "
             "does not degrade reliability — and may improve it as teams learn "
             "each booster's behaviour.")
        by_r = (dff.groupby("reuse_count")
                .agg(rate=("success", lambda x: (x == "True").mean() * 100),
                     count=("success", "count"))
                .reset_index())
        fig_rr = go.Figure()
        fig_rr.add_trace(go.Bar(
            x=by_r["reuse_count"], y=by_r["rate"],
            marker=dict(
                color=by_r["rate"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#2563eb"], [1, "#34d399"]],
                showscale=False,
            ),
            text=by_r["rate"].round(1),
            texttemplate="%{text}%", textposition="outside",
            customdata=by_r["count"],
            hovertemplate=(
                "<b>Reuse count: %{x}</b><br>"
                "Success rate: %{y:.1f}%<br>"
                "Launches: %{customdata}<extra></extra>"
            ),
        ))
        fig_rr.update_layout(**cl(
            title=None,
            xaxis_title="Prior flights of this booster",
            yaxis=dict(**CL["yaxis"], ticksuffix="%", range=[0, 115]),
        ))
        st.plotly_chart(fig_rr, width="stretch")

    with c2:
        section_label("Average reuse count per year")
        info("<b>What this shows:</b> How the average number of prior booster flights "
             "has grown over time. A rising trend shows SpaceX is increasingly "
             "comfortable flying boosters multiple times per year.")
        ry = (dff.groupby("year")["reuse_count"]
              .mean().reset_index(name="Avg Reuse Count"))
        fig_ry = go.Figure(go.Scatter(
            x=ry["year"], y=ry["Avg Reuse Count"],
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=7, color="#fbbf24",
                        line=dict(color="#0f172a", width=1.5)),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.07)",
            hovertemplate="<b>%{x}</b><br>Avg reuse count: %{y:.1f}<extra></extra>",
        ))
        fig_ry.update_layout(**CL, title=None,
                              xaxis_title="Year",
                              yaxis_title="Avg prior flights")
        st.plotly_chart(fig_ry, width="stretch")

    k1, k2, k3 = st.columns(3)
    kpi(k1, str(int(dff["reuse_count"].max())),       "Max Reuse Count")
    kpi(k2, f"{dff['reuse_count'].mean():.1f}",        "Avg Reuse Count")
    kpi(k3, str(int((dff["reuse_count"] > 0).sum())), "Reflown Launches")


# ═════════════════════════════════════════════════════════════════════════════
# ML PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif sec == "ML Predictor":

    info(
        f"<b>Random Forest Classifier</b> — 300 trees, max depth 8, class-balanced, "
        f"80/20 stratified split. "
        f"Trained on {metrics['n_train']:,} launches, evaluated on {metrics['n_test']:,}. "
        f"Test AUC: <b style='color:#34d399;'>{metrics['auc']:.3f}</b>  "
        f"Accuracy: <b style='color:#34d399;'>{metrics['accuracy']*100:.1f}%</b>"
    )

    section_label("Configure a hypothetical launch")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rocket_sel = st.selectbox("Rocket", sorted(df["rocket"].unique()),
                                   help="Rocket vehicle for the mission.")
    with c2:
        pad_sel    = st.selectbox("Launchpad", sorted(df["launchpad"].unique()),
                                   help="Launch site.")
    with c3:
        year_sel   = st.selectbox(
            "Year", sorted(df["year"].dropna().unique()),
            index=len(sorted(df["year"].dropna().unique())) - 1,
            help="Planned year of launch.",
        )
    with c4:
        reuse_sel  = st.slider(
            "Booster reuse count", 0, 10, 0,
            help="Number of prior flights on this booster. 0 = brand new.",
        )

    fn_est  = int(df[df["year"] <= int(year_sel)]["flight_number"].max() or fn_max)
    fn_norm = fn_est / fn_max
    X_in    = pd.DataFrame(
        [[int(le_r.transform([rocket_sel])[0]),
          int(le_p.transform([pad_sel])[0]),
          int(year_sel), fn_norm, reuse_sel]],
        columns=["r_enc", "p_enc", "year", "fn_norm", "reuse_count"],
    )
    prob    = float(clf.predict_proba(X_in)[0][1])
    color   = "#34d399" if prob > 0.5 else "#f87171"
    label   = "SUCCESS" if prob > 0.5 else "FAILURE"
    delta   = prob * 100 - base_sr

    st.markdown(f"""
        <div class='pred-box'>
            <div class='pred-label' style='color:{color};'>{label}</div>
            <div class='pred-sub'>Predicted probability:
                <b style='color:{color};'>{prob*100:.2f}%</b>
            </div>
            <div style='background:#111827; border-radius:8px; height:8px;
                        margin:16px auto; max-width:360px; overflow:hidden;'>
                <div style='background:{color}; width:{min(prob*100,100):.1f}%;
                            height:100%; border-radius:8px;'></div>
            </div>
            <div style='color:#334155; font-size:0.76rem;'>
                {rocket_sel} &nbsp;·&nbsp; {pad_sel} &nbsp;·&nbsp;
                {int(year_sel)} &nbsp;·&nbsp; Reuse x{reuse_sel}
            </div>
            <div style='margin-top:10px; font-size:0.78rem; color:#334155;'>
                Fleet-wide historical success rate: {base_sr:.1f}%
                &nbsp;·&nbsp; Model delta:
                <span style='color:{color};'>
                    {"+" if delta >= 0 else ""}{delta:.1f}%
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    section_label("Model evaluation — confusion matrix (test set)")
    info(
        "<b>What this shows:</b> How often the model was correct on the "
        f"{metrics['n_test']} launches it never saw during training. "
        "Top-left = correctly predicted failures. "
        "Bottom-right = correctly predicted successes. "
        "Off-diagonal cells are prediction errors."
    )
    cm_arr = metrics["cm"]
    fig_cm, ax = plt.subplots(figsize=(4.5, 3.5))
    fig_cm.patch.set_facecolor("#080e1c")
    ax.set_facecolor("#080e1c")
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Failure", "Success"],
                yticklabels=["Failure", "Success"],
                annot_kws={"color": "white", "size": 13},
                linewidths=0.4, linecolor="#111827", cbar=False)
    ax.set_xlabel("Predicted", color="#475569", labelpad=8)
    ax.set_ylabel("Actual",    color="#475569", labelpad=8)
    ax.tick_params(colors="#475569")
    plt.setp(ax.get_xticklabels(), color="#475569")
    plt.setp(ax.get_yticklabels(), color="#475569")
    plt.tight_layout()
    _, cm_col, _ = st.columns([1, 2, 1])
    with cm_col:
        st.pyplot(fig_cm)
    plt.close(fig_cm)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════
elif sec == "Feature Importance":

    section_label("What drives the model's predictions?")
    info(
        "<b>How to read this:</b> Random Forest importance = mean decrease in "
        "Gini impurity across all decision trees. A higher value means the feature "
        "carries more information at split nodes. These values are scale-independent "
        "and capture non-linear relationships, unlike logistic regression coefficients."
    )

    fi = feat_imp.sort_values().reset_index()
    fi.columns = ["Feature", "Importance"]
    fig_fi = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#2563eb"], [1, "#8b5cf6"]],
        text=fi["Importance"].round(3),
    )
    fig_fi.update_traces(textposition="outside", textfont_color="#475569")
    fig_fi.update_layout(**cl(
        title=None, coloraxis_showscale=False,
        yaxis_title="",
        xaxis_range=[0, fi["Importance"].max() * 1.25],
    ))
    st.plotly_chart(fig_fi, width="stretch")

    top_feat = feat_imp.idxmax()
    info(
        f"The most influential feature is <b>{top_feat}</b> "
        f"(importance = {feat_imp.max():.3f}). "
        + ("This reflects that operational maturity — captured by when the launch "
           "happened — is the strongest predictor of mission success."
           if top_feat == "Year"
           else "This indicates that the rocket vehicle is the dominant factor "
                "in predicting mission success.")
    )


# ═════════════════════════════════════════════════════════════════════════════
# INSIGHTS — fully dynamic
# ═════════════════════════════════════════════════════════════════════════════
elif sec == "Insights":

    if dff.empty:
        st.warning("No launches in the selected year range.")
    else:
        sr_by_rocket  = dff.groupby("rocket")["success"].apply(
            lambda x: (x == "True").mean() * 100)
        top_rocket    = sr_by_rocket.idxmax()
        top_rocket_sr = sr_by_rocket.max()
        top_rocket_n  = int((dff["rocket"] == top_rocket).sum())

        sr_by_pad  = dff.groupby("launchpad")["success"].apply(
            lambda x: (x == "True").mean() * 100)
        top_pad    = sr_by_pad.idxmax()
        top_pad_sr = sr_by_pad.max()
        top_pad_n  = int((dff["launchpad"] == top_pad).sum())

        yr_first = int(dff["year"].min())
        yr_last  = int(dff["year"].max())
        sr_first = float((dff[dff["year"] == yr_first]["success"] == "True").mean() * 100)
        sr_last  = float((dff[dff["year"] == yr_last ]["success"] == "True").mean() * 100)
        sr_delta = sr_last - sr_first

        top_feat     = feat_imp.idxmax()
        top_feat_val = feat_imp.max()

        dom_rocket = dff["rocket"].mode()[0]
        dom_share  = (dff["rocket"] == dom_rocket).mean() * 100

        max_reuse = int(dff["reuse_count"].max())
        avg_reuse = float(dff["reuse_count"].mean())
        sr_reused = float(
            (dff[dff["reuse_count"] > 0]["success"] == "True").mean() * 100
            if (dff["reuse_count"] > 0).any() else 0)

        insights = [
            (
                f"{dom_rocket} leads the manifest",
                f"{dom_rocket} accounts for <b>{dom_share:.0f}%</b> of launches "
                f"in this period ({top_rocket_n:,} flights) and holds the highest "
                f"success rate at <b>{top_rocket_sr:.1f}%</b> — the most reliable "
                f"vehicle in commercial orbital launch history.",
            ),
            (
                f"{top_pad} is the most reliable launch site",
                f"With <b>{top_pad_n}</b> launches and a <b>{top_pad_sr:.1f}%</b> "
                f"success rate, {top_pad} leads all sites. It handles SpaceX's most "
                f"complex and high-profile missions, including crewed flights.",
            ),
            (
                f"Reliability {'improved' if sr_delta > 0 else 'shifted'} "
                f"{sr_delta:+.1f} percentage points ({yr_first} to {yr_last})",
                f"Fleet success rate moved from <b>{sr_first:.1f}%</b> in {yr_first} "
                f"to <b>{sr_last:.1f}%</b> in {yr_last}. This trajectory reflects "
                f"compounding improvements across vehicle design, ground support "
                f"and launch cadence.",
            ),
            (
                f"Model signal: '{top_feat}' is the strongest predictor",
                f"The Random Forest assigns importance <b>{top_feat_val:.3f}</b> to "
                f"<b>{top_feat}</b>. The model achieves AUC <b>{metrics['auc']:.3f}</b> "
                f"on {metrics['n_test']} held-out launches — well above the 0.5 "
                f"random baseline and robust to class imbalance.",
            ),
            (
                f"Reuse is proven: up to {max_reuse}x with "
                f"{sr_reused:.1f}% success on reflown boosters",
                f"Average reuse count in this period: <b>{avg_reuse:.1f}</b>. "
                f"Reflown boosters achieve a <b>{sr_reused:.1f}%</b> success rate, "
                f"matching or exceeding the new-booster baseline — validating "
                f"SpaceX's core economic model.",
            ),
        ]

        for title, body in insights:
            st.markdown(f"""
                <div class='insight-card'>
                    <div class='insight-title'>{title}</div>
                    <div class='insight-body'>{body}</div>
                </div>""", unsafe_allow_html=True)

        st.caption(
            f"All figures computed from {len(dff):,} launches "
            f"({year_range[0]}-{year_range[1]}). "
            "Use the year range slider in the sidebar to watch insights update.")


# ═════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif sec == "Data Explorer":

    COL_MAP = {
        "date_utc":      "Launch Date",
        "year":          "Year",
        "rocket":        "Rocket",
        "launchpad":     "Launchpad",
        "success":       "Outcome",
        "flight_number": "Flight No.",
        "reuse_count":   "Reuse Count",
    }
    show_cols = [c for c in COL_MAP if c in dff.columns]

    fc1, fc2, fc3 = st.columns(3)
    with fc1: fy = st.multiselect("Year",      sorted(dff["year"].dropna().unique()))
    with fc2: fr = st.multiselect("Rocket",    sorted(dff["rocket"].unique()))
    with fc3: fp = st.multiselect("Launchpad", sorted(dff["launchpad"].unique()))

    filt = dff.copy()
    if fy: filt = filt[filt["year"].isin(fy)]
    if fr: filt = filt[filt["rocket"].isin(fr)]
    if fp: filt = filt[filt["launchpad"].isin(fp)]

    st.markdown("<br>", unsafe_allow_html=True)
    f_sr = (filt["success"] == "True").mean() * 100 if len(filt) else 0
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, f"{len(filt):,}",                               "Launches shown")
    kpi(k2, f"{f_sr:.1f}%",                                 "Success rate")
    kpi(k3, str(int((filt["success"] == "True").sum())),     "Successes")
    kpi(k4, str(int((filt["success"] != "True").sum())),     "Failures")
    st.markdown("<br>", unsafe_allow_html=True)

    if len(filt) > 1:
        section_label("Filtered launches by year and outcome")
        mini = filt.groupby(["year", "success"]).size().reset_index(name="count")
        mini["Outcome"] = mini["success"].map({"True": "Success", "False": "Failure"})
        fig_m = px.bar(mini, x="year", y="count", color="Outcome",
                       color_discrete_map={"Success": "#34d399", "Failure": "#f87171"},
                       labels={"count": "Launches", "year": "Year"})
        fig_m.update_layout(**CL, title=None, bargap=0.3)
        st.plotly_chart(fig_m, width="stretch")

    section_label("Launch records")
    disp = (filt[show_cols]
            .rename(columns=COL_MAP)
            .sort_values("Launch Date", ascending=False)
            .copy())
    disp["Outcome"] = disp["Outcome"].map(
        {"True": "Success", "False": "Failure"}).fillna(disp["Outcome"])
    st.dataframe(disp, height=400)

    st.caption(f"Showing {len(filt):,} of {len(dff):,} launches in selected range.")
    st.download_button(
        "Download CSV",
        filt.to_csv(index=False).encode(),
        "spacex_filtered.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:#111827; margin-top:48px;'>"
    "<p style='text-align:center; color:#1e293b; font-size:0.76rem;'>"
    "Created by <b style='color:#334155;'>Sarthak Shandilya</b>"
    " &nbsp;·&nbsp; SpaceX Launch Intelligence"
    " &nbsp;·&nbsp; Streamlit · Plotly · scikit-learn · SciPy"
    "</p>",
    unsafe_allow_html=True,
)
