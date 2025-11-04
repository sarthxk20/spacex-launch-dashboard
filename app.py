import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Page setup ---
st.set_page_config(page_title="🚀 SpaceX Launch Dashboard", layout="wide")

# --- Center the Sidebar Logo ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] img {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: SpaceX Branding Card ---
st.sidebar.markdown("""
    <div style="
        background: linear-gradient(145deg, #0f172a, #1e293b);
        padding: 22px;
        border-radius: 15px;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.2);
        text-align: center;
        color: white;
        margin-bottom: 20px;
    ">
        <img src="https://upload.wikimedia.org/wikipedia/commons/d/de/SpaceX-Logo.svg" width="180" style="margin-bottom:15px;">
        <h3 style='margin-bottom: 10px; color: #00FFFF;'>🚀 SpaceX Dashboard</h3>
        <hr style='border: 1px solid #00FFFF; margin: 10px 0;'>
        <h4 style='margin-bottom: 8px;'>ℹ️ About This Dashboard</h4>
        <p style='font-size: 14px; line-height: 1.6;'>
            This interactive dashboard visualizes <b>SpaceX launch data</b> —
            exploring mission success trends, rocket & launchpad performance,
            and predictive insights using machine learning. <br><br>
            🔹 Built with <b>Streamlit, Plotly & Sklearn</b><br>
            🔹 Data: SpaceX Launch Archive<br>
            🔹 Author: <b>Sarthak Shandilya</b>
        </p>
    </div>
""", unsafe_allow_html=True)


# --- Custom CSS ---
st.markdown("""
    <style>
        body { background-color: #0e1117; color: white; }
        .main { background-color: #0e1117; }
        .glow { color: #00FFFF; text-shadow: 0 0 20px #00FFFF; font-weight: bold; }
        .metric-card {
            background: linear-gradient(135deg, #111827, #1f2937);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
            text-align: center;
        }
        .chart-card {
            background-color: #111827;
            padding: 20px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.1);
        }
        .summary-box {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.15);
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar Hover Glow Effect ---
st.markdown("""
    <style>
        /* Apply hover glow to the sidebar card */
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] > div {
            transition: all 0.3s ease-in-out;
        }
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] > div:hover {
            box-shadow: 0 0 40px rgba(0, 255, 255, 0.4);
            transform: scale(1.01);
        }
    </style>
""", unsafe_allow_html=True)


# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("merged_spacex_data.csv")
    df['date_utc'] = pd.to_datetime(df['date_utc'], errors='coerce')
    df['year'] = df['date_utc'].dt.year
    df['success'] = df['success'].astype(str)
    return df

df = load_data()

# --- Title ---
st.markdown("<h1 class='glow' style='text-align:center;'>🚀 SpaceX Launch Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Mission Overview ---
total_launches = len(df)
success_count = (df['success'] == "True").sum()
success_rate = (success_count / total_launches * 100) if total_launches > 0 else 0
unique_rockets = df['rocket'].nunique()
unique_launchpads = df['launchpad'].nunique()
first_year, latest_year = int(df['year'].min()), int(df['year'].max())
most_successful_rocket = df.groupby("rocket")["success"].apply(lambda x: (x == "True").mean()).idxmax()
most_active_year = df['year'].value_counts().idxmax()

st.markdown(f"""
<div class='summary-box'>
    <h2 class='glow'>🧭 SpaceX Mission Overview ({first_year} – {latest_year})</h2>
    <p>
        Since <b>{first_year}</b>, SpaceX has launched <b>{total_launches}</b> missions — achieving a stellar <b>{success_rate:.1f}%</b> success rate.
        Over this period, the company has continuously improved its rockets, reusability, and mission cadence.
    </p>
    <p>
        With <b>{unique_rockets}</b> rocket types launched from <b>{unique_launchpads}</b> global launchpads, SpaceX’s journey highlights
        technological excellence and operational efficiency. The <b>{most_successful_rocket}</b> remains its most dependable rocket.
    </p>
    <p>
        The year <b>{most_active_year}</b> recorded the highest launch activity, marking SpaceX’s rapid scaling toward consistent orbital success.
    </p>
</div>
""", unsafe_allow_html=True)

# --- KPI Section ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><h4>🚀 Total Launches</h4><h2>{total_launches}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h4>✅ Success Rate</h4><h2>{success_rate:.2f}%</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h4>🛰️ Rockets Used</h4><h2>{unique_rockets}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><h4>🌍 Launchpads</h4><h2>{unique_launchpads}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# === Launch Trends ===
st.markdown("<h2 class='glow'>📈 Launch Trends Over the Years</h2>", unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    success_trend = (
        df.groupby("year")["success"]
        .apply(lambda x: (x == "True").mean() * 100)
        .reset_index(name="Success Rate (%)")
    )
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig1 = px.line(success_trend, x="year", y="Success Rate (%)", markers=True, color_discrete_sequence=["#00FFFF"])
    fig1.update_layout(template="plotly_dark", title="Success Rate Over Time", title_x=0.5)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    launches_per_year = df.groupby("year").size().reset_index(name="Launches")
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig2 = px.bar(launches_per_year, x="year", y="Launches", color="Launches", color_continuous_scale="tealgrn")
    fig2.update_layout(template="plotly_dark", title="Total Launches Per Year", title_x=0.5)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# === Rocket & Launchpad Performance ===
st.markdown("<h2 class='glow'>🚀 Rocket & Launchpad Performance</h2>", unsafe_allow_html=True)
col_c, col_d = st.columns(2)

with col_c:
    rocket_perf = (
        df.groupby("rocket")["success"]
        .apply(lambda x: (x == "True").mean() * 100)
        .reset_index(name="Success Rate (%)")
        .sort_values(by="Success Rate (%)", ascending=False)
    )
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig3 = px.bar(rocket_perf, x="rocket", y="Success Rate (%)",
                  color="Success Rate (%)", color_continuous_scale="Plotly3")
    fig3.update_layout(template="plotly_dark", title="Rocket Success Comparison", title_x=0.5)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_d:
    launchpad_success = (
        df.groupby("launchpad")["success"]
        .apply(lambda x: (x == "True").mean() * 100)
        .reset_index(name="Success Rate (%)")
    )
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig4 = px.bar(launchpad_success, x="launchpad", y="Success Rate (%)",
                  color="Success Rate (%)", color_continuous_scale="Plasma")
    fig4.update_layout(template="plotly_dark", title="Launchpad Success Comparison", title_x=0.5)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# === Mission Outcomes & Correlation ===
st.markdown("<h2 class='glow'>🧭 Mission Outcomes & Feature Correlation</h2>", unsafe_allow_html=True)
col_e, col_f = st.columns(2)

with col_e:
    outcome_counts = df["success"].value_counts().reset_index()
    outcome_counts.columns = ["Outcome", "Count"]
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig5 = px.pie(outcome_counts, names="Outcome", values="Count",
                  color_discrete_sequence=["#00FF7F", "#FF4500"])
    fig5.update_traces(textinfo="percent+label")
    fig5.update_layout(template="plotly_dark", title="Mission Outcome Distribution", title_x=0.5)
    st.plotly_chart(fig5, use_container_width=True)
    # --- Subheading under the pie chart ---
    st.markdown("<p style='text-align:center; color:#00FFFF; font-size:16px;'>✔️ This chart compares successful and failed SpaceX missions across all recorded launches.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_f:
    ml_df = df.copy()
    for col in ['rocket', 'launchpad']:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col])
    ml_df['success'] = ml_df['success'].apply(lambda x: 1 if x == "True" else 0)
    corr = ml_df.corr(numeric_only=True)
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='cool', ax=ax)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Machine Learning Predictor ---
st.markdown("---")
st.markdown("<h2 class='glow'>🔮 Launch Success Predictor</h2>", unsafe_allow_html=True)

ml_df = df.copy()
for col in ['rocket', 'launchpad']:
    le = LabelEncoder()
    ml_df[col] = le.fit_transform(ml_df[col])
ml_df = ml_df.dropna(subset=['success'])
ml_df['success'] = ml_df['success'].apply(lambda x: 1 if x == "True" else 0)

X = ml_df[['rocket', 'launchpad', 'year']]
y = ml_df['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
st.markdown("<h4 class='glow'>🧠 Predict Launch Success</h4>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
rockets = df['rocket'].unique().tolist()
launchpads = df['launchpad'].unique().tolist()
years = sorted(df['year'].dropna().unique().astype(int))
with col1:
    rocket_choice = st.selectbox("Select Rocket", rockets)
with col2:
    launchpad_choice = st.selectbox("Select Launchpad", launchpads)
with col3:
    year_choice = st.selectbox("Select Year", years)
rocket_encoded = LabelEncoder().fit(df['rocket']).transform([rocket_choice])[0]
launchpad_encoded = LabelEncoder().fit(df['launchpad']).transform([launchpad_choice])[0]
pred_input = pd.DataFrame([[rocket_encoded, launchpad_encoded, year_choice]], columns=['rocket', 'launchpad', 'year'])
prob = model.predict_proba(pred_input)[0][1]
prediction = "✅ Likely to Succeed" if prob > 0.5 else "🚨 Risky Launch"
st.markdown(f"""
<div style='text-align:center; font-size:20px;'>
    <b>Predicted Outcome:</b> {prediction}<br><br>
    <b>Success Probability:</b> {prob*100:.2f}%
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)



# --- Feature Importance Visualization ---
st.markdown("<h4 class='glow'>📊 Feature Importance</h4>", unsafe_allow_html=True)

import numpy as np

# Get absolute importance values from logistic regression coefficients
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.coef_[0])
}).sort_values(by='Importance', ascending=False)

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
fig_imp = px.bar(
    importance,
    x='Feature',
    y='Importance',
    color='Importance',
    color_continuous_scale='turbo',
    title='Most Influential Features for Launch Success'
)
fig_imp.update_layout(template='plotly_dark', title_x=0.5)
st.plotly_chart(fig_imp, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)



accuracy = model.score(X_test, y_test) * 100
st.metric("Model Accuracy", f"{accuracy:.2f}%")


# --- Key Insights ---
st.markdown("---")
st.markdown("<h2 class='glow'>💡 Key Insights</h2>", unsafe_allow_html=True)
st.markdown("""
- 🚀 **Falcon 9** dominates as the most reliable and frequently launched rocket.  
- 🌍 Launches concentrate around a few efficient launchpads, optimizing reusability.  
- 📈 Success rates have shown a strong upward trend with continuous improvements.  
- 🔮 Predictive modeling confirms modern rockets yield higher success probabilities.  
- 🧭 SpaceX’s evolution showcases innovation, precision, and unmatched reliability in orbital missions.  
""")

# --- Data Explorer ---
st.markdown("---")
st.markdown("<h2 class='glow'>🧮 Data Explorer</h2>", unsafe_allow_html=True)
with st.expander("🔍 Explore the Dataset"):
    years = sorted(df['year'].dropna().unique())
    rockets = sorted(df['rocket'].dropna().unique())
    launchpads = sorted(df['launchpad'].dropna().unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        year_filter = st.multiselect("Filter by Year", years)
    with col2:
        rocket_filter = st.multiselect("Filter by Rocket", rockets)
    with col3:
        launchpad_filter = st.multiselect("Filter by Launchpad", launchpads)
    filtered_df = df.copy()
    if year_filter:
        filtered_df = filtered_df[filtered_df['year'].isin(year_filter)]
    if rocket_filter:
        filtered_df = filtered_df[filtered_df['rocket'].isin(rocket_filter)]
    if launchpad_filter:
        filtered_df = filtered_df[filtered_df['launchpad'].isin(launchpad_filter)]
    st.dataframe(filtered_df)

with st.expander("🚀 Mission Details"):
    st.dataframe(df[["name", "date_utc", "rocket", "launchpad", "success", "details"]])

# --- Download Section ---
st.markdown("---")
st.markdown("<h2 class='glow'>📥 Download Data</h2>", unsafe_allow_html=True)
st.download_button(
    "📂 Download Filtered Dataset (CSV)",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_spacex_data.csv'
)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Created by <b>Sarthak Shandilya</b> | Powered by Streamlit & Plotly 🚀</p>", unsafe_allow_html=True)