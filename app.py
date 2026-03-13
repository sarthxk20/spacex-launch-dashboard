import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("merged_spacex_data.csv")
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    df["year"]    = df["date_utc"].dt.year
    df["success"] = df["success"].astype(str)
    rocket_map = {
        "5e9d0d95eda69955f709d1eb": "Falcon 1",
        "5e9d0d95eda69973a809d1ec": "Falcon 9",
        "5e9d0d95eda69974db09d1ed": "Falcon Heavy",
    }
    pad_map = {
        "5e9e4502f5090995de566f86": "Kwajalein Atoll",
        "5e9e4501f509094ba4566f84": "Cape Canaveral SFS",
        "5e9e4502f509092b78566f87": "Kennedy LC-39A",
        "5e9e4502f509094188566f88": "Vandenberg SFB",
    }
    df["rocket"]    = df["rocket"].apply(lambda x: rocket_map.get(x, x))
    df["launchpad"] = df["launchpad"].apply(lambda x: pad_map.get(x, x))
    return df


@st.cache_resource
def build_model(df: pd.DataFrame):
    ml = df.copy()
    ml["success"] = ml["success"].apply(lambda x: 1 if x == "True" else 0)
    le_r = LabelEncoder().fit(ml["rocket"].fillna("unknown"))
    le_p = LabelEncoder().fit(ml["launchpad"].fillna("unknown"))
    ml["r_enc"] = le_r.transform(ml["rocket"].fillna("unknown"))
    ml["p_enc"] = le_p.transform(ml["launchpad"].fillna("unknown"))
    X = ml[["r_enc", "p_enc", "year"]].fillna(0)
    m = LogisticRegression(max_iter=1000)
    m.fit(X, ml["success"])
    return m, le_r, le_p


df           = load_data()
model, le_r, le_p = build_model(df)
sr           = (df["success"] == "True").mean() * 100

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  — navigation via radio; conditional rendering replaces JS scrolling
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
         "🧭  Mission Outcomes", "🔮  ML Predictor",
         "📊  Feature Importance", "💡  Insights", "🗂️  Data Explorer"],
    )

    st.markdown(f"""
        <hr style='border-color:#1e293b; margin:16px 0;'>
        <div style='background:#0f172a; border:1px solid #1e293b; border-radius:10px;
                    padding:14px; font-size:0.82rem; color:#64748b;'>
            <div style='color:#94a3b8; font-weight:600; margin-bottom:8px;'>Quick Stats</div>
            <div>Total launches &nbsp;<span style='color:#60a5fa; float:right;'>{len(df)}</span></div>
            <div>Success rate &nbsp;<span style='color:#34d399; float:right;'>{sr:.1f}%</span></div>
            <div>Rocket types &nbsp;<span style='color:#a78bfa; float:right;'>{df["rocket"].nunique()}</span></div>
            <div>Launchpads &nbsp;<span style='color:#fb923c; float:right;'>{df["launchpad"].nunique()}</span></div>
        </div>
    """, unsafe_allow_html=True)

# Strip emoji prefix → clean section name for matching
sec = section.split("  ", 1)[-1].strip()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1>SpaceX Launch Intelligence</h1>", unsafe_allow_html=True)
st.markdown(f"<h2>{section}</h2>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#1e293b; margin-bottom:24px;'>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTIONS
# ═════════════════════════════════════════════════════════════════════════════

if sec == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, str(len(df)),                  "Total Launches"),
        (c2, f"{sr:.1f}%",                  "Success Rate"),
        (c3, str(df["rocket"].nunique()),    "Rocket Types"),
        (c4, str(df["launchpad"].nunique()), "Launchpads"),
    ]:
        col.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        ys = df.groupby(["year","success"]).size().reset_index(name="count")
        fig = px.bar(ys, x="year", y="count", color="success",
                     color_discrete_map={"True":"#34d399","False":"#f87171"},
                     title="Launches by Year & Outcome",
                     labels={"count":"Launches","success":"Outcome"})
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with cb:
        rd = df["rocket"].value_counts().reset_index()
        rd.columns = ["rocket","count"]
        fig2 = px.pie(rd, names="rocket", values="count",
                      color_discrete_sequence=ACCENT, title="Fleet Composition", hole=0.45)
        fig2.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig2, width="stretch")


elif sec == "Launch Trends":
    c1, c2 = st.columns(2)
    with c1:
        trend = df.groupby("year")["success"].apply(
            lambda x: (x=="True").mean()*100).reset_index()
        fig = go.Figure(go.Scatter(
            x=trend["year"], y=trend["success"], mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=8, color="#60a5fa", line=dict(color="#1e3a5f", width=2)),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
        fig.update_layout(**CHART_LAYOUT, title="Success Rate Over Time (%)",
                          yaxis_ticksuffix="%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        cy = df.groupby("year").size().reset_index(name="Launches")
        fig2 = px.bar(cy, x="year", y="Launches", color="Launches",
                      color_continuous_scale=["#1e3a5f","#3b82f6","#60a5fa"],
                      title="Total Launches Per Year")
        fig2.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch")

    cum = df.sort_values("date_utc").copy()
    cum["cumulative"] = range(1, len(cum)+1)
    fig3 = px.area(cum, x="date_utc", y="cumulative",
                   color_discrete_sequence=["#8b5cf6"],
                   title="Cumulative Launches All Time",
                   labels={"date_utc":"Date","cumulative":"Total Launches"})
    fig3.update_traces(fillcolor="rgba(139,92,246,0.12)", line_color="#8b5cf6")
    fig3.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig3, width="stretch")


elif sec == "Performance":
    c1, c2 = st.columns(2)
    with c1:
        rp = (df.groupby("rocket")["success"]
              .apply(lambda x: (x=="True").mean()*100)
              .reset_index().rename(columns={"success":"Success Rate (%)"}))
        fig = px.bar(rp.sort_values("Success Rate (%)"),
                     x="Success Rate (%)", y="rocket", orientation="h",
                     color="Success Rate (%)",
                     color_continuous_scale=["#1e3a5f","#3b82f6","#34d399"],
                     title="Rocket Success Rates")
        fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        pp = (df.groupby("launchpad")["success"]
              .apply(lambda x: (x=="True").mean()*100)
              .reset_index().rename(columns={"success":"Success Rate (%)"}))
        fig2 = px.bar(pp.sort_values("Success Rate (%)"),
                      x="Success Rate (%)", y="launchpad", orientation="h",
                      color="Success Rate (%)",
                      color_continuous_scale=["#1e3a5f","#8b5cf6","#34d399"],
                      title="Launchpad Success Rates")
        fig2.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig2, width="stretch")

    ry = df.groupby(["year","rocket"]).size().reset_index(name="Launches")
    fig3 = px.line(ry, x="year", y="Launches", color="rocket", markers=True,
                   color_discrete_sequence=ACCENT, title="Annual Launch Volume by Rocket")
    fig3.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig3, width="stretch")


elif sec == "Mission Outcomes":
    c1, c2 = st.columns(2)
    with c1:
        oc = df["success"].value_counts().reset_index()
        oc.columns = ["outcome","count"]
        oc["label"] = oc["outcome"].map({"True":"Success","False":"Failure"})
        fig = px.pie(oc, names="label", values="count",
                     color="label",
                     color_discrete_map={"Success":"#34d399","Failure":"#f87171"},
                     title="Mission Outcome Split", hole=0.5)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c2:
        cd = df.copy()
        cd["success"]  = cd["success"].apply(lambda x: 1 if x=="True" else 0)
        cd["r_enc"]    = le_r.transform(cd["rocket"].fillna("unknown"))
        cd["p_enc"]    = le_p.transform(cd["launchpad"].fillna("unknown"))
        corr = cd[["r_enc","p_enc","year","success"]].corr()
        f, ax = plt.subplots(figsize=(6,4))
        f.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    annot_kws={"color":"white","size":10},
                    linewidths=0.5, linecolor="#1e293b", cbar_kws={"shrink":0.8})
        ax.tick_params(colors="#94a3b8", labelsize=9)
        plt.setp(ax.get_xticklabels(), color="#94a3b8")
        plt.setp(ax.get_yticklabels(), color="#94a3b8")
        ax.set_title("Feature Correlation Matrix", color="#cbd5e1", pad=12)
        st.pyplot(f); plt.close(f)

    pivot = (df.groupby(["year","rocket"])["success"]
             .apply(lambda x: round((x=="True").mean()*100,1))
             .unstack(fill_value=0))
    fig4 = px.imshow(pivot,
                     color_continuous_scale=["#0f172a","#1e3a5f","#3b82f6","#34d399"],
                     title="Success Rate Heatmap — Year × Rocket (%)",
                     aspect="auto", text_auto=True)
    fig4.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig4, width="stretch")


elif sec == "ML Predictor":
    st.markdown("""<div class='section-card'>
        <p style='color:#64748b; font-size:0.88rem; margin:0;'>
        Logistic Regression trained on all historical launches.
        Select a configuration to predict mission success probability.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: rocket_sel = st.selectbox("🚀 Rocket",    sorted(df["rocket"].unique()))
    with c2: pad_sel    = st.selectbox("📍 Launchpad", sorted(df["launchpad"].unique()))
    with c3: year_sel   = st.selectbox("📅 Year",      sorted(df["year"].dropna().unique()))

    r_enc = int(le_r.transform([rocket_sel])[0])
    p_enc = int(le_p.transform([pad_sel])[0])
    prob  = model.predict_proba([[r_enc, p_enc, int(year_sel)]])[0][1]
    color = "#34d399" if prob > 0.5 else "#f87171"
    label = "SUCCESS" if prob > 0.5 else "FAILURE"
    base  = (df["success"] == "True").mean() * 100
    delta = prob*100 - base

    st.markdown(f"""
        <div class='prediction-result'>
            <div class='pred-label' style='color:{color};'>{label}</div>
            <div class='pred-prob'>Predicted probability: <b style='color:{color};'>{prob*100:.2f}%</b></div>
            <div style='background:#1e293b; border-radius:8px; height:10px;
                        margin:16px auto; max-width:320px; overflow:hidden;'>
                <div style='background:{color}; width:{prob*100:.1f}%; height:100%; border-radius:8px;'></div>
            </div>
            <div style='color:#475569; font-size:0.78rem;'>{rocket_sel} · {pad_sel} · {int(year_sel)}</div>
        </div>
        <p style='text-align:center; color:#475569; font-size:0.82rem; margin-top:12px;'>
            Fleet-wide historical success rate: {base:.1f}% &nbsp;·&nbsp;
            Model delta: <span style='color:{color};'>{"+" if delta>=0 else ""}{delta:.1f}%</span>
        </p>
    """, unsafe_allow_html=True)


elif sec == "Feature Importance":
    imp = pd.DataFrame({
        "Feature":    ["Rocket Type","Launchpad","Launch Year"],
        "Importance": np.abs(model.coef_[0]),
    }).sort_values("Importance")
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 color="Importance",
                 color_continuous_scale=["#1e3a5f","#3b82f6","#8b5cf6"],
                 title="Logistic Regression — Absolute Coefficient Magnitude",
                 text=imp["Importance"].round(3))
    fig.update_traces(textposition="outside", textfont_color="#94a3b8")
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.markdown("""<div class='section-card'>
        <p style='color:#64748b; font-size:0.85rem; margin:0;'>
        <b style='color:#94a3b8;'>How to read this:</b> Each bar shows how strongly
        that feature pulls the predicted probability. Higher = more influence.</p>
    </div>""", unsafe_allow_html=True)


elif sec == "Insights":
    for title, body in [
        ("🛸 Falcon 9 Dominance",
         "Falcon 9 accounts for the majority of launches and consistently achieves "
         "the highest success rate, underpinning the entire commercial manifest."),
        ("📍 Kennedy LC-39A is the flagship pad",
         "Originally built for Apollo, LC-39A records the highest launch success "
         "rate of any SpaceX launchpad."),
        ("📈 Reliability compounds over time",
         "Success rate has climbed steeply year-over-year as iterative improvements "
         "to vehicle design and ground support compound."),
        ("🤖 Launch year is the strongest model signal",
         "The logistic regression assigns the highest weight to launch year, "
         "reflecting that operational maturity is the best predictor of success."),
        ("⚠️ Falcon 1 era context",
         "Early Falcon 1 launches skew the overall failure rate. Post-2013 data "
         "shows a dramatically higher fleet-wide success rate."),
    ]:
        st.markdown(f"""
            <div class='insight-item'>
                <b style='color:#cbd5e1;'>{title}</b><br>
                <span style='color:#64748b;'>{body}</span>
            </div>""", unsafe_allow_html=True)


elif sec == "Data Explorer":
    c1, c2, c3 = st.columns(3)
    with c1: fy = st.multiselect("Year",      sorted(df["year"].unique()))
    with c2: fr = st.multiselect("Rocket",    sorted(df["rocket"].unique()))
    with c3: fp = st.multiselect("Launchpad", sorted(df["launchpad"].unique()))

    filtered = df.copy()
    if fy: filtered = filtered[filtered["year"].isin(fy)]
    if fr: filtered = filtered[filtered["rocket"].isin(fr)]
    if fp: filtered = filtered[filtered["launchpad"].isin(fp)]

    st.markdown(f"<p style='color:#475569; font-size:0.82rem;'>"
                f"Showing <b style='color:#60a5fa;'>{len(filtered)}</b> of {len(df)} launches</p>",
                unsafe_allow_html=True)
    st.dataframe(filtered.sort_values("date_utc", ascending=False),
                 height=420)
    st.download_button("⬇️  Download filtered CSV",
                       filtered.to_csv(index=False).encode("utf-8"),
                       "filtered_spacex_data.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:#1e293b; margin-top:40px;'>"
    "<p style='text-align:center; color:#334155; font-size:0.78rem;'>"
    "Created by <b>Sarthak Shandilya</b> · SpaceX Launch Intelligence · "
    "Streamlit + Plotly + scikit-learn</p>",
    unsafe_allow_html=True,
)
