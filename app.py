import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# -------------------------------------------
# PAGE SETUP
# -------------------------------------------
st.set_page_config(page_title="🚀 SpaceX Launch Dashboard", layout="wide")


# -------------------------------------------
# GLOBAL CLEAN DARK THEME + MOBILE CSS
# -------------------------------------------
st.markdown("""
    <style>

        body, .main {
            background-color: #0e1117 !important;
            color: #e6eef0 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
        }

        h1, h2, h3, h4 {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        .metric-card, .chart-card, .summary-box {
            background-color: #111827 !important;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 25px;
        }

        .sidebar-box {
            background-color: #111827 !important;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            text-align: center;
        }

        hr {
            border: 1px solid rgba(255,255,255,0.1);
        }

        html {
            scroll-behavior: smooth !important;
        }

        /* MOBILE OPTIMIZATION */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            h1 { font-size: 26px !important; }
            h2 { font-size: 20px !important; }
            h3 { font-size: 16px !important; }
            p, li { font-size: 14px !important; }

            .metric-card, .chart-card {
                padding: 14px !important;
            }

            [data-testid="stSidebar"] {
                width: 100% !important;
                max-width: 100% !important;
                position: relative !important;
            }

            .js-plotly-plot {
                max-width: 100% !important;
                height: auto !important;
            }

            .stDataFrame {
                overflow-x: scroll !important;
            }
        }

    </style>
""", unsafe_allow_html=True)


# -------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------
st.sidebar.title("Navigate")

section = st.sidebar.radio(
    "Jump to section:",
    [
        "Overview",
        "Launch Trends",
        "Performance",
        "Mission Outcomes",
        "ML Predictor",
        "Feature Importance",
        "Insights",
        "Data Explorer"
    ]
)

slug_map = {
    "Overview": "overview",
    "Launch Trends": "launch-trends",
    "Performance": "performance",
    "Mission Outcomes": "mission-outcomes",
    "ML Predictor": "ml-predictor",
    "Feature Importance": "feature-importance",
    "Insights": "insights",
    "Data Explorer": "data-explorer"
}

selected_slug = slug_map[section]

components.html(f"""
    <script>
        const el = window.parent.document.getElementById("{selected_slug}");
        if (el) {{
            el.scrollIntoView({{behavior: "smooth", block: "start"}});
        }}
    </script>
""", height=0)


st.sidebar.markdown("""
    <div class='sidebar-box'>
        <img src="https://upload.wikimedia.org/wikipedia/commons/d/de/SpaceX-Logo.svg" width="150">
        <h4>SpaceX Dashboard</h4>
        <p style='font-size:13px;color:#b9c7c6;'>Interactive analytics powered by Streamlit</p>
    </div>
""", unsafe_allow_html=True)


# -------------------------------------------
# LOAD DATA
# -------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("merged_spacex_data.csv")
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    df["year"] = df["date_utc"].dt.year
    df["success"] = df["success"].astype(str)

    rocket_map = {
        "5e9d0d95eda69955f709d1eb": "Falcon 1",
        "5e9d0d95eda69973a809d1ec": "Falcon 9",
        "5e9d0d95eda69974db09d1ed": "Falcon Heavy"
    }
    pad_map = {
        "5e9e4502f5090995de566f86": "Kwajalein Atoll",
        "5e9e4501f509094ba4566f84": "Cape Canaveral SFS",
        "5e9e4502f509092b78566f87": "Kennedy LC-39A",
        "5e9e4502f509094188566f88": "Vandenberg SFB"
    }

    df["rocket"]    = df["rocket"].apply(lambda x: rocket_map.get(x, x))
    df["launchpad"] = df["launchpad"].apply(lambda x: pad_map.get(x, x))
    return df


df = load_data()


# -------------------------------------------
# BUILD & CACHE ML MODEL
# Fit encoders once here so training and prediction
# always use the exact same category ordering.
# -------------------------------------------
@st.cache_resource
def build_model(df: pd.DataFrame):
    ml = df.copy()
    ml["success"] = ml["success"].apply(lambda x: 1 if x == "True" else 0)

    le_rocket = LabelEncoder().fit(ml["rocket"].fillna("unknown"))
    le_pad    = LabelEncoder().fit(ml["launchpad"].fillna("unknown"))

    ml["rocket_enc"]    = le_rocket.transform(ml["rocket"].fillna("unknown"))
    ml["launchpad_enc"] = le_pad.transform(ml["launchpad"].fillna("unknown"))

    X = ml[["rocket_enc", "launchpad_enc", "year"]].fillna(0)
    y = ml["success"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, le_rocket, le_pad, X.columns.tolist()


model, le_rocket, le_pad, feature_names = build_model(df)


# ============================================================
# 1️⃣ OVERVIEW
# ============================================================
st.markdown("<h1 id='overview'>🚀 SpaceX Launch Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

total_launches  = len(df)
success_count   = (df["success"] == "True").sum()
success_rate    = success_count / total_launches * 100 if total_launches else 0
rocket_count    = df["rocket"].nunique()
launchpad_count = df["launchpad"].nunique()
first_year      = int(df["year"].min())
last_year       = int(df["year"].max())

st.markdown(f"""
<div class='summary-box'>
<h2>Mission Overview ({first_year} – {last_year})</h2>
<p><b>{total_launches}</b> launches — <b>{success_rate:.1f}%</b> success rate.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='metric-card'><h4>Total Launches</h4><h2>{total_launches}</h2></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card'><h4>Success Rate</h4><h2>{success_rate:.1f}%</h2></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card'><h4>Rockets</h4><h2>{rocket_count}</h2></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card'><h4>Launchpads</h4><h2>{launchpad_count}</h2></div>", unsafe_allow_html=True)


# ============================================================
# 2️⃣ LAUNCH TRENDS
# ============================================================
st.markdown("<h2 id='launch-trends'>📈 Launch Trends</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    trend = df.groupby("year")["success"].apply(lambda x: (x == "True").mean() * 100).reset_index()
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig1 = px.line(trend, x="year", y="success", markers=True)
    fig1.update_layout(template="plotly_dark", title="Success Rate Over Time")
    st.plotly_chart(fig1, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    count_year = df.groupby("year").size().reset_index(name="Launches")
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig2 = px.bar(count_year, x="year", y="Launches")
    fig2.update_layout(template="plotly_dark", title="Launches Per Year")
    st.plotly_chart(fig2, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 3️⃣ PERFORMANCE
# ============================================================
st.markdown("<h2 id='performance'>🚀 Rocket & Launchpad Performance</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    rocket_perf = df.groupby("rocket")["success"].apply(lambda x: (x == "True").mean() * 100).reset_index()
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig3 = px.bar(rocket_perf, x="rocket", y="success")
    fig3.update_layout(template="plotly_dark", title="Rocket Success Rates")
    st.plotly_chart(fig3, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    pad_perf = df.groupby("launchpad")["success"].apply(lambda x: (x == "True").mean() * 100).reset_index()
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig4 = px.bar(pad_perf, x="launchpad", y="success")
    fig4.update_layout(template="plotly_dark", title="Launchpad Success Rates")
    st.plotly_chart(fig4, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 4️⃣ MISSION OUTCOMES
# ============================================================
st.markdown("<h2 id='mission-outcomes'>🧭 Mission Outcomes</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    outcome = df["success"].value_counts().reset_index()
    outcome.columns = ["success", "count"]

    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig5 = px.pie(outcome, names="success", values="count")
    fig5.update_layout(template="plotly_dark", title="Success vs Failure")
    st.plotly_chart(fig5, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Build correlation df from the same encoded data the model uses
    corr_df = df.copy()
    corr_df["success"]      = corr_df["success"].apply(lambda x: 1 if x == "True" else 0)
    corr_df["rocket_enc"]   = le_rocket.transform(corr_df["rocket"].fillna("unknown"))
    corr_df["launchpad_enc"]= le_pad.transform(corr_df["launchpad"].fillna("unknown"))
    corr = corr_df[["rocket_enc", "launchpad_enc", "year", "success"]].corr()

    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    # Match the dark background of the rest of the dashboard
    fig6, ax = plt.subplots(figsize=(6, 4))
    fig6.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    sns.heatmap(
        corr, annot=True, cmap="cool", ax=ax,
        annot_kws={"color": "white"},
        linewidths=0.5, linecolor="#1f2937",
    )
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), color="white")
    plt.setp(ax.get_yticklabels(), color="white")
    st.pyplot(fig6)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 5️⃣ ML PREDICTOR
# ============================================================
st.markdown("<h2 id='ml-predictor'>🔮 Launch Success Predictor</h2>", unsafe_allow_html=True)

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    rocket_sel = st.selectbox("Rocket", df["rocket"].unique())
with col2:
    pad_sel = st.selectbox("Launchpad", df["launchpad"].unique())
with col3:
    year_sel = st.selectbox("Year", sorted(df["year"].dropna().unique()))

# Use the same fitted encoders the model was trained with
r_enc = int(le_rocket.transform([rocket_sel])[0])
p_enc = int(le_pad.transform([pad_sel])[0])
y_val = int(year_sel)   # cast numpy int → Python int

prob  = model.predict_proba([[r_enc, p_enc, y_val]])[0][1]
label = "SUCCESS ✅" if prob > 0.5 else "FAILURE ❌"

st.markdown(
    f"<h3 style='text-align:center;'>Prediction: {label}<br>Probability: {prob*100:.2f}%</h3>",
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 6️⃣ FEATURE IMPORTANCE
# ============================================================
st.markdown("<h2 id='feature-importance'>📊 Feature Importance</h2>", unsafe_allow_html=True)

importance = pd.DataFrame({
    "Feature":    ["Rocket", "Launchpad", "Year"],
    "Importance": np.abs(model.coef_[0])
})

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
fig_imp = px.bar(importance, x="Feature", y="Importance")
fig_imp.update_layout(template="plotly_dark", title="Feature Impact on Prediction")
st.plotly_chart(fig_imp, width="stretch")
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 7️⃣ INSIGHTS
# ============================================================
st.markdown("<h2 id='insights'>💡 Insights</h2>", unsafe_allow_html=True)

st.markdown("""
- **Falcon 9** shows the highest reliability.  
- **Kennedy LC-39A** excels as the most successful launchpad.  
- Success rate has increased significantly over the years.  
- Launch success is strongly influenced by rocket + launchpad + year.  
""")


# ============================================================
# 8️⃣ DATA EXPLORER
# ============================================================
st.markdown("<h2 id='data-explorer'>🧮 Data Explorer</h2>", unsafe_allow_html=True)

# Define filtered before the expander so the download button always has it
filtered = df.copy()

with st.expander("🔍 Filter Data"):
    years   = sorted(df["year"].unique())
    rockets = sorted(df["rocket"].unique())
    pads    = sorted(df["launchpad"].unique())

    c1, c2, c3 = st.columns(3)
    with c1:
        fy = st.multiselect("Year", years)
    with c2:
        fr = st.multiselect("Rocket", rockets)
    with c3:
        fp = st.multiselect("Launchpad", pads)

    if fy: filtered = filtered[filtered["year"].isin(fy)]
    if fr: filtered = filtered[filtered["rocket"].isin(fr)]
    if fp: filtered = filtered[filtered["launchpad"].isin(fp)]

    st.dataframe(filtered)

st.download_button(
    "Download CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    "filtered_spacex_data.csv"
)


# ============================================================
# FOOTER
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>Created by <b>Sarthak Shandilya</b></p>",
    unsafe_allow_html=True
)
