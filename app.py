# ============================================================
# CROP YIELD PREDICTOR - Streamlit Web App
# Author  : Sharaft Hussain
# M.Sc. (Hons.) Agronomy — MNS University of Agriculture Multan
# NAVTTC Advance Python Programming A+
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Predictor | Sharaft Hussain",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stToolbar"] {display: none !important;}
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
.stDeployButton {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}
[data-testid="stSidebar"] {display: block !important; min-width: 300px !important; width: 300px !important;}
[data-testid="stSidebarCollapsedControl"] {display: none !important;}
section[data-testid="stSidebar"] > div {width: 300px !important;}
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    .main { background: linear-gradient(135deg, #F0FDF8, #F0F9FF); }

    .header-box {
        background: linear-gradient(135deg, #0F4C3A, #1A6B5F);
        padding: 36px 28px;
        border-radius: 18px;
        margin-bottom: 28px;
        color: white;
    }
    .header-box h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 8px; color: white; }
    .header-box p  { font-size: 1rem; color: #A7F3D0; margin: 0; }

    .result-box {
        background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
        border: 2px solid #34D399;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-yield { font-size: 4rem; font-weight: 800; color: #065F46; }
    .result-unit  { font-size: 1.1rem; color: #6B7280; }

    .tag {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.2);
        color: #D1FAE5;
        font-size: 0.75rem;
        padding: 4px 14px;
        border-radius: 99px;
        margin: 4px 4px 0 0;
    }
    .insight-box {
        background: #F0FDF4;
        border-left: 4px solid #1A6B5F;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 16px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0F4C3A, #1A6B5F) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        font-family: 'Outfit', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Train ML Model (cached) ──────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 300
    crops_enc = np.random.choice([0, 1, 2, 3, 4, 5], n)
    crop_names = {0: 'Wheat', 1: 'Maize', 2: 'Canola', 3: 'Sugarcane', 4: 'Cotton', 5: 'Rice'}
    base_map   = {0: 3.5,     1: 5.0,     2: 2.0,      3: 60.0,        4: 2.5,      5: 4.0}

    df = pd.DataFrame({
        'crop':        crops_enc,
        'rainfall':    np.random.uniform(150, 650, n),
        'temperature': np.random.uniform(18, 40, n),
        'nitrogen':    np.random.uniform(50, 200, n),
        'phosphorus':  np.random.uniform(25, 100, n),
        'irrigation':  np.random.randint(2, 9, n),
        'soil_ph':     np.random.uniform(6.0, 9.0, n),
        'density':     np.random.uniform(70, 160, n),
    })

    def calc_yield(row):
        base = base_map[row['crop']]
        y = (base
             + 0.003 * row['rainfall']
             + 0.01  * row['nitrogen']
             + 0.005 * row['phosphorus']
             + 0.15  * row['irrigation']
             - 0.04  * abs(row['temperature'] - 25)
             - 0.10  * abs(row['soil_ph'] - 7.2)
             + np.random.normal(0, 0.25))
        return max(round(y, 2), 0.5)

    df['yield'] = df.apply(calc_yield, axis=1)

    features = ['crop','rainfall','temperature','nitrogen','phosphorus','irrigation','soil_ph','density']
    X, y = df[features], df['yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    r2  = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, round(r2, 3), round(mae, 3)

model, r2, mae = train_model()

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🌾 Crop Yield Predictor</h1>
  <p>Enter your field conditions to predict expected crop yield using Machine Learning — Pakistan Agriculture</p>
  <br>
  <span class="tag">M.Sc. Agronomy</span>
  <span class="tag">NAVTTC Python A+</span>
  <span class="tag">MNS University Multan</span>
  <span class="tag">Random Forest ML</span>
</div>
""", unsafe_allow_html=True)

# ── Model Stats ───────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("🤖 Model Accuracy (R²)", f"{r2 * 100:.1f}%")
c2.metric("📉 Mean Abs. Error", f"{mae} ton/ha")
c3.metric("🌱 Crops Supported", "6 (Wheat, Maize, Canola, Sugarcane, Cotton, Rice)")

st.divider()

# ── Sidebar Inputs ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Field Parameters")
    st.markdown("Adjust the sliders below:")

    crop_name = st.selectbox("🌾 Crop Type", ["Wheat", "Maize", "Canola", "Sugarcane", "Cotton", "Rice"])
    crop_enc  = {"Wheat": 0, "Maize": 1, "Canola": 2, "Sugarcane": 3, "Cotton": 4, "Rice": 5}[crop_name]

    st.markdown("---")
    rainfall    = st.slider("🌧️ Rainfall (mm)",        100, 700, 350, 10)
    temperature = st.slider("🌡️ Temperature (°C)",      15,  45,  25,  1)
    nitrogen    = st.slider("🧪 Nitrogen (kg/ha)",       30, 200, 120,  5)
    phosphorus  = st.slider("⚗️ Phosphorus (kg/ha)",     15, 100,  60,  5)
    irrigation  = st.slider("💧 Irrigation (times)",      1,  10,   5,  1)
    soil_ph     = st.slider("🪨 Soil pH",               5.5, 9.0, 7.2, 0.1)
    density     = st.slider("🌱 Sowing Density (kg/ha)", 60, 180, 120,  5)

    st.markdown("---")
    predict_btn = st.button("🔬 PREDICT YIELD")

    st.markdown("---")
    st.markdown("**👨‍🎓 Developed by**")
    st.markdown("**Sharaft Hussain**")
    st.markdown("M.Sc. Agronomy | NAVTTC Python A+")
    st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/sharaft-hussain-saroya)")

# ── Prediction ────────────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([{
        'crop': crop_enc, 'rainfall': rainfall, 'temperature': temperature,
        'nitrogen': nitrogen, 'phosphorus': phosphorus,
        'irrigation': irrigation, 'soil_ph': soil_ph, 'density': density
    }])

    predicted_yield = round(float(model.predict(input_data)[0]), 2)
    base = {"Wheat": 3.5, "Maize": 5.0, "Canola": 2.0, "Sugarcane": 60.0, "Cotton": 2.5, "Rice": 4.0}[crop_name]
    pct  = (predicted_yield / (base * 1.6)) * 100

    if pct >= 85: rating, stars, rcolor = "Excellent 🌟", "⭐⭐⭐⭐⭐", "#22C55E"
    elif pct >= 70: rating, stars, rcolor = "Good 👍",    "⭐⭐⭐⭐",   "#84CC16"
    elif pct >= 55: rating, stars, rcolor = "Average 😐", "⭐⭐⭐",     "#EAB308"
    elif pct >= 40: rating, stars, rcolor = "Below Avg ⚠️","⭐⭐",     "#F97316"
    else:           rating, stars, rcolor = "Poor ❌",    "⭐",        "#EF4444"

    # Result
    col1, col2 = st.columns([1, 1])

    with col1:
        emoji = {"Wheat":"🌾","Maize":"🌽","Canola":"🌼","Sugarcane":"🎋","Cotton":"🌿","Rice":"🍚"}[crop_name]
        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:3rem">{emoji}</div>
            <div style="font-size:0.9rem;color:#6B7280;letter-spacing:2px;font-weight:600;text-transform:uppercase">
                Predicted Yield — {crop_name}
            </div>
            <div class="result-yield">{predicted_yield}</div>
            <div class="result-unit">ton / hectare</div>
            <br>
            <div style="font-size:1.3rem">{stars}</div>
            <div style="font-weight:700;color:{rcolor};font-size:1.1rem">{rating}</div>
        </div>
        """, unsafe_allow_html=True)

        # Insights
        st.markdown('<div class="insight-box"><b>💡 Agronomic Insights</b><br>', unsafe_allow_html=True)
        insights = []
        if temperature > 32:
            insights.append(f"⚠️ High temperature ({temperature}°C) — use heat-tolerant varieties")
        if soil_ph > 8.0:
            insights.append(f"⚠️ Alkaline soil (pH {soil_ph}) — apply gypsum")
        if nitrogen < 80:
            insights.append(f"⚠️ Low nitrogen ({nitrogen} kg/ha) — increase fertilizer")
        if rainfall < 200:
            insights.append(f"⚠️ Low rainfall — increase irrigation frequency")
        if not insights:
            insights.append("✅ Field conditions are well-optimized!")
        insights.append(f"✅ Optimal soil pH for {crop_name}: 6.8–7.5 | Your pH: {soil_ph}")
        for i in insights:
            st.markdown(f"- {i}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Radar Chart
        categories  = ['Rainfall', 'Nitrogen', 'Irrigation', 'Phosphorus', 'Temp Opt', 'Soil pH']
        values = [
            min((rainfall / 650) * 100, 100),
            min((nitrogen / 200) * 100, 100),
            min((irrigation / 9)  * 100, 100),
            min((phosphorus / 100)* 100, 100),
            max(0, 100 - abs(temperature - 25) * 4),
            max(0, 100 - abs(soil_ph - 7.2) * 20),
        ]
        values_plot = values + [values[0]]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.fill(angles, values_plot, color='#1A6B5F', alpha=0.25)
        ax.plot(angles, values_plot, color='#1A6B5F', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('Field Performance Radar', fontweight='bold', color='#1A6B5F', pad=20)
        ax.grid(color='#E5E7EB', linewidth=0.8)
        st.pyplot(fig)

    st.divider()

    # Feature Importance Bar Chart
    st.markdown("### 📊 What Affects Yield the Most?")
    features    = ['crop','rainfall','temperature','nitrogen','phosphorus','irrigation','soil_ph','density']
    feat_labels = ['Crop Type','Rainfall','Temperature','Nitrogen','Phosphorus','Irrigation','Soil pH','Density']
    importance  = pd.Series(model.feature_importances_, index=feat_labels).sort_values()

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    colors = ['#0F4C3A' if v == importance.max() else '#1A6B5F' if v > importance.mean() else '#A7F3D0'
              for v in importance.values]
    importance.plot(kind='barh', ax=ax2, color=colors, edgecolor='white')
    ax2.set_title('Feature Importance — Random Forest Model', fontweight='bold', color='#1A6B5F')
    ax2.set_xlabel('Importance Score')
    ax2.spines[['top','right']].set_visible(False)
    st.pyplot(fig2)

else:
    # Welcome screen
    st.markdown("### 👈 Set your field parameters in the sidebar and click **PREDICT YIELD**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🌾 **Wheat**\nBase Yield: 3.5 ton/ha\nBest Temp: 20–25°C")
    with col2:
        st.info("🌽 **Maize**\nBase Yield: 5.0 ton/ha\nBest Temp: 22–28°C")
    with col3:
        st.info("🌼 **Canola**\nBase Yield: 2.0 ton/ha\nBest Temp: 15–20°C")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.info("🎋 **Sugarcane**\nBase Yield: 60 ton/ha\nBest Temp: 27–35°C")
    with col5:
        st.info("🌿 **Cotton**\nBase Yield: 2.5 ton/ha\nBest Temp: 25–35°C")
    with col6:
        st.info("🍚 **Rice**\nBase Yield: 4.0 ton/ha\nBest Temp: 22–30°C")

    st.markdown("---")
    st.markdown("**👨‍🎓 About This Project**")
    st.markdown("""
    This app was built by **Sharaft Hussain**, M.Sc. (Hons.) Agronomy graduate from
    MNS University of Agriculture Multan, combining agronomic expertise with
    Python programming skills (NAVTTC Advance Python A+) to create a practical
    Machine Learning tool for Pakistani farmers and researchers.
    """)
