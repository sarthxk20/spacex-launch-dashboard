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
# SIDEBAR NAVIGATION (Option B)
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

# GUARANTEED WORKING SCROLLING
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
# LOAD DATA & MAP REAL ROCKET + LAUNCHPAD NAMES
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

    df["rocket"] = df["rocket"].apply(lambda x: rocket_map.get(x, x))
    df["launchpad"] = df["launchpad"].apply(lambda x: pad_map.get(x, x))

    return df


df = load_data()


# ============================================================
# 1️⃣ OVERVIEW
# ============================================================
st.markdown("<h1 id='overview'>🚀 SpaceX Launch Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

total_launches = len(df)
success_count = (df["success"] == "True").sum()
success_rate = success_count / total_launches * 100 if total_launches else 0
rocket_count = df["rocket"].nunique()
launchpad_count = df["launchpad"].nunique()
first_year = int(df["year"].min())
last_year = int(df["year"].max())

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
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    count_year = df.groupby("year").size().reset_index(name="Launches")
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig2 = px.bar(count_year, x="year", y="Launches")
    fig2.update_layout(template="plotly_dark", title="Launches Per Year")
    st.plotly_chart(fig2, use_container_width=True)
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
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    pad_perf = df.groupby("launchpad")["success"].apply(lambda x: (x == "True").mean() * 100).reset_index()
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig4 = px.bar(pad_perf, x="launchpad", y="success")
    fig4.update_layout(template="plotly_dark", title="Launchpad Success Rates")
    st.plotly_chart(fig4, use_container_width=True)
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
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    ml_df = df.copy()

    for c in ["rocket", "launchpad"]:
        ml_df[c] = LabelEncoder().fit_transform(ml_df[c].fillna("unknown"))

    ml_df["success"] = ml_df["success"].apply(lambda x: 1 if x == "True" else 0)

    # FIX: only numeric columns allowed
    corr = ml_df.select_dtypes(include=[np.number]).corr()

    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig6, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="cool")
    st.pyplot(fig6)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 5️⃣ ML PREDICTOR
# ============================================================
st.markdown("<h2 id='ml-predictor'>🔮 Launch Success Predictor</h2>", unsafe_allow_html=True)

ml = df.copy()
for c in ["rocket", "launchpad"]:
    ml[c] = LabelEncoder().fit_transform(ml[c].fillna("unknown"))
ml["success"] = ml["success"].apply(lambda x: 1 if x == "True" else 0)

X = ml[["rocket", "launchpad", "year"]].fillna(0)
y = ml["success"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    rocket_sel = st.selectbox("Rocket", df["rocket"].unique())
with col2:
    pad_sel = st.selectbox("Launchpad", df["launchpad"].unique())
with col3:
    year_sel = st.selectbox("Year", sorted(df["year"].dropna().unique()))

r_enc = LabelEncoder().fit(df["rocket"]).transform([rocket_sel])[0]
p_enc = LabelEncoder().fit(df["launchpad"]).transform([pad_sel])[0]

prob = model.predict_proba([[r_enc, p_enc, year_sel]])[0][1]
label = "SUCCESS" if prob > 0.5 else "FAILURE"

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
    "Feature": X.columns,
    "Importance": np.abs(model.coef_[0])
})

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
fig_imp = px.bar(importance, x="Feature", y="Importance")
fig_imp.update_layout(template="plotly_dark", title="Feature Impact")
st.plotly_chart(fig_imp, use_container_width=True)
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

with st.expander("🔍 Filter Data"):
    years = sorted(df["year"].unique())
    rockets = sorted(df["rocket"].unique())
    pads = sorted(df["launchpad"].unique())

    c1, c2, c3 = st.columns(3)
    with c1:
        fy = st.multiselect("Year", years)
    with c2:
        fr = st.multiselect("Rocket", rockets)
    with c3:
        fp = st.multiselect("Launchpad", pads)

    filtered = df.copy()
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
