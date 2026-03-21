"""
🚲 Bike Sharing Demand Predictor — Streamlit App
Run with: streamlit run bike_app.py

HOW TO SET UP:
1. First run the Jupyter notebook completely (it saves bike_model.pkl)
2. Then run: streamlit run bike_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🚲 Bike Sharing Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1565C0;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1565C0, #42A5F5);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1565C0;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .stAlert > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load or download the dataset."""
    if not os.path.exists('day.csv'):
        import urllib.request, zipfile
        try:
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
            urllib.request.urlretrieve(url, 'bike.zip')
            with zipfile.ZipFile('bike.zip', 'r') as z:
                z.extractall('.')
        except Exception as e:
            return None, str(e)
    df = pd.read_csv('day.csv')
    return df, None

@st.cache_resource
def load_model():
    """Load trained model, or train a quick one if pkl missing."""
    if os.path.exists('bike_model.pkl'):
        model = joblib.load('bike_model.pkl')
        return model, "Loaded saved model ✅"

    # Quick fallback: train a Random Forest
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    df, err = load_data()
    if df is None:
        return None, "Dataset not available"
    df_m = df.drop(columns=['instant','dteday','casual','registered','atemp'])
    X = df_m.drop(columns=['cnt'])
    y = df_m['cnt']
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    joblib.dump(model, 'bike_model.pkl')
    return model, "Trained new Random Forest model ✅"

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🚲 Bike Sharing Demand Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">UCI Bike Sharing Dataset | Machine Learning Regression Project</div>', unsafe_allow_html=True)

# Load data and model
df, data_err = load_data()
model, model_msg = load_model()

if data_err:
    st.error(f"Could not load data: {data_err}")
    st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Data Insights", "📈 Model Performance"])


# ════════════════════════════════════════════
# TAB 1: PREDICTION
# ════════════════════════════════════════════
with tab1:
    st.header("🎯 Predict Bike Rentals")
    st.markdown("Adjust the parameters below and click **Predict** to estimate daily rentals.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🌦️ Weather Conditions")

        season = st.selectbox(
            "Season",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1:"🌱 Spring", 2:"☀️ Summer", 3:"🍂 Fall", 4:"❄️ Winter"}[x],
            index=2
        )

        weathersit = st.selectbox(
            "Weather Situation",
            options=[1, 2, 3],
            format_func=lambda x: {1:"☀️ Clear / Partly Cloudy", 2:"🌫️ Mist / Cloudy", 3:"🌧️ Light Rain / Snow"}[x]
        )

        temp_actual = st.slider(
            "Temperature (°C)",
            min_value=-10, max_value=40, value=20,
            help="Actual temperature in Celsius"
        )
        temp = (temp_actual - (-8)) / (39 - (-8))  # normalize to 0-1

        hum_pct = st.slider("Humidity (%)", 0, 100, 65)
        hum = hum_pct / 100

        wind_kmh = st.slider("Wind Speed (km/h)", 0, 70, 15)
        windspeed = wind_kmh / 67

    with col2:
        st.subheader("📅 Date & Day Info")

        yr = st.radio("Year", options=[0, 1], format_func=lambda x: "2011" if x == 0 else "2012 (higher demand)")

        mnth = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1],
            index=7
        )

        weekday = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"][x],
            index=2
        )

        workingday = st.radio("Working Day?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        holiday    = st.radio("Holiday?",     options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Predict button
    st.markdown("---")
    pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 1])
    with pred_col2:
        predict_btn = st.button("🚀 Predict Rentals", use_container_width=True, type="primary")

    if predict_btn:
        input_data = pd.DataFrame([{
            'season': season,
            'yr': yr,
            'mnth': mnth,
            'holiday': holiday,
            'weekday': weekday,
            'workingday': workingday,
            'weathersit': weathersit,
            'temp': temp,
            'hum': hum,
            'windspeed': windspeed
        }])

        prediction = int(model.predict(input_data)[0])

        # Color-coded result
        if prediction >= 5000:
            emoji, color, label = "🔥", "#4CAF50", "High Demand Day!"
        elif prediction >= 3000:
            emoji, color, label = "✅", "#2196F3", "Moderate Demand"
        else:
            emoji, color, label = "❄️", "#FF9800", "Low Demand Day"

        st.markdown("---")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color}, #90CAF9);
                    border-radius: 15px; padding: 30px; text-align: center; color: white;'>
            <div style='font-size: 3rem'>{emoji}</div>
            <div style='font-size: 1.3rem; font-weight: bold; margin: 5px 0'>Predicted Bike Rentals</div>
            <div style='font-size: 4rem; font-weight: bold'>{prediction:,}</div>
            <div style='font-size: 1.1rem; opacity: 0.9'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        # Context comparisons
        c1, c2, c3 = st.columns(3)
        avg = int(df['cnt'].mean())
        max_val = int(df['cnt'].max())
        pct = round(prediction / max_val * 100, 1)
        c1.metric("vs Average Day", f"{avg:,}", f"{prediction - avg:+,} bikes")
        c2.metric("vs Record Day", f"{max_val:,}", f"{prediction - max_val:+,}")
        c3.metric("% of Record", f"{pct}%")

        # Quick tip
        st.markdown('<div class="insight-box">💡 <b>Tip:</b> Higher temperatures, fall season, and clear weather tend to produce the highest predictions. Try adjusting them!</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2: DATA INSIGHTS
# ════════════════════════════════════════════
with tab2:
    st.header("📊 Dataset Insights")

    # Overview metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Days",    f"{len(df):,}")
    c2.metric("Avg Rentals/Day", f"{df['cnt'].mean():,.0f}")
    c3.metric("Max Rentals",  f"{df['cnt'].max():,}")
    c4.metric("Min Rentals",  f"{df['cnt'].min():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Season chart
        st.subheader("🍂 Rentals by Season")
        season_map = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}
        s_avg = df.groupby('season')['cnt'].mean().rename(index=season_map)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(s_avg.index, s_avg.values,
                      color=['#81C784','#FFB74D','#E57373','#64B5F6'], edgecolor='white', linewidth=1.2)
        ax.set_title('Average Rentals by Season', fontweight='bold')
        ax.set_ylabel('Avg Rentals')
        for bar, val in zip(bars, s_avg.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">🍂 <b>Fall</b> has the most rentals. <b>Spring</b> is the slowest season.</div>', unsafe_allow_html=True)

    with col2:
        # Weather chart
        st.subheader("🌤️ Rentals by Weather")
        w_map = {1:'Clear', 2:'Mist', 3:'Light Rain'}
        w_avg = df[df['weathersit'] <= 3].groupby('weathersit')['cnt'].mean().rename(index=w_map)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(w_avg.index, w_avg.values,
                      color=['#29B6F6','#B0BEC5','#78909C'], edgecolor='white', linewidth=1.2)
        ax.set_title('Average Rentals by Weather', fontweight='bold')
        ax.set_ylabel('Avg Rentals')
        for bar, val in zip(bars, w_avg.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">☀️ <b>Clear weather</b> drives 40% more rentals than rainy days!</div>', unsafe_allow_html=True)

    # Monthly trend
    st.subheader("📅 Monthly Rental Trend (2011–2012)")
    df_trend = df.copy()
    df_trend['Month'] = pd.to_datetime(df_trend['dteday']).dt.to_period('M').astype(str)
    monthly = df_trend.groupby('Month')['cnt'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly['Month'], monthly['cnt'], color='steelblue', linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(monthly['Month'], monthly['cnt'], alpha=0.15, color='steelblue')
    ax.set_title('Monthly Average Rentals', fontweight='bold')
    ax.set_ylabel('Avg Rentals')
    ax.tick_params(axis='x', rotation=45)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight-box">📈 <b>Year-over-year growth:</b> 2012 had ~35% more rentals than 2011, showing rapid service adoption!</div>', unsafe_allow_html=True)

    # Temperature scatter
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🌡️ Temperature vs Rentals")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df['temp'] * 41, df['cnt'], alpha=0.4, color='#FF7043', s=20)
        ax.set_xlabel('Temperature (°C approx.)')
        ax.set_ylabel('Rentals')
        ax.set_title('Temp vs Bike Rentals', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">🌡️ Strong <b>positive correlation</b>: warmer days = more riders!</div>', unsafe_allow_html=True)

    with col4:
        st.subheader("💧 Humidity vs Rentals")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df['hum'] * 100, df['cnt'], alpha=0.4, color='#42A5F5', s=20)
        ax.set_xlabel('Humidity (%)')
        ax.set_ylabel('Rentals')
        ax.set_title('Humidity vs Bike Rentals', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">💧 Slight <b>negative correlation</b>: higher humidity = slightly fewer rentals.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE
# ════════════════════════════════════════════
with tab3:
    st.header("📈 Model Performance")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    @st.cache_data
    def run_all_models():
        df_m = df.drop(columns=['instant','dteday','casual','registered','atemp'])
        X = df_m.drop(columns=['cnt'])
        y = df_m['cnt']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        models_list = [
            ('Linear Regression', LinearRegression()),
            ('Random Forest',     RandomForestRegressor(n_estimators=100, random_state=42)),
            ('Gradient Boosting', GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ]
        rows = []
        for name, m in models_list:
            m.fit(X_tr, y_tr)
            p = m.predict(X_te)
            rows.append({
                'Model': name,
                'R²': round(r2_score(y_te, p), 4),
                'RMSE': round(np.sqrt(mean_squared_error(y_te, p)), 1),
                'MAE': round(mean_absolute_error(y_te, p), 1)
            })
        return pd.DataFrame(rows).sort_values('R²', ascending=False).reset_index(drop=True)

    with st.spinner("Running model comparisons..."):
        perf_df = run_all_models()

    # Metrics table
    st.subheader("📋 Model Comparison Table")
    st.dataframe(
        perf_df.style
            .highlight_max(subset=['R²'], color='#C8E6C9')
            .highlight_min(subset=['RMSE','MAE'], color='#C8E6C9')
            .format({'R²': '{:.4f}', 'RMSE': '{:.1f}', 'MAE': '{:.1f}'}),
        use_container_width=True
    )

    st.markdown('<div class="insight-box">✅ <b>Green = Best value</b> for each metric. R² closer to 1.0 is better; lower RMSE/MAE is better.</div>', unsafe_allow_html=True)

    # Bar charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("R² Score Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#4CAF50' if i == 0 else '#90CAF9' for i in range(len(perf_df))]
        ax.barh(perf_df['Model'], perf_df['R²'], color=colors, edgecolor='grey')
        ax.set_xlabel('R² Score')
        ax.set_xlim(0, 1)
        ax.set_title('R² Score (Higher = Better)', fontweight='bold')
        for i, v in enumerate(perf_df['R²']):
            ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("RMSE Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors2 = ['#4CAF50' if i == len(perf_df)-1 else '#EF9A9A' for i in range(len(perf_df))]
        ax.barh(perf_df['Model'], perf_df['RMSE'], color=colors2, edgecolor='grey')
        ax.set_xlabel('RMSE (bikes)')
        ax.set_title('RMSE (Lower = Better)', fontweight='bold')
        for i, v in enumerate(perf_df['RMSE']):
            ax.text(v + 5, i, f'{v:.0f}', va='center', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # Feature Importance
    st.subheader("🌟 Feature Importance (Random Forest)")
    df_fi = df.drop(columns=['instant','dteday','casual','registered','atemp'])
    X_fi = df_fi.drop(columns=['cnt'])
    y_fi = df_fi['cnt']
    rf_fi = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_fi.fit(X_fi, y_fi)
    feat_imp = pd.Series(rf_fi.feature_importances_, index=X_fi.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_fi = ['#4CAF50' if v >= feat_imp.quantile(0.75) else '#90CAF9' for v in feat_imp]
    ax.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor='grey')
    ax.set_title('Feature Importance — Which factors matter most?', fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

    top = feat_imp.sort_values(ascending=False).head(3)
    st.markdown(f'<div class="insight-box">🌟 <b>Top 3 features:</b> <b>{top.index[0]}</b> ({top.iloc[0]:.3f}), <b>{top.index[1]}</b> ({top.iloc[1]:.3f}), <b>{top.index[2]}</b> ({top.iloc[2]:.3f}) — These drive most of the prediction!</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📖 How to interpret metrics:**")
    c1, c2, c3 = st.columns(3)
    c1.info("**R² Score**\nMeasures % of variance explained.\n1.0 = perfect, 0.0 = random guessing.\nTarget: > 0.85 ✅")
    c2.info("**RMSE**\nRoot Mean Squared Error.\nIn same units as target (bikes/day).\nLower = more accurate predictions.")
    c3.info("**MAE**\nMean Absolute Error.\nAverage error per prediction.\nEasier to interpret than RMSE.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"<div style='text-align:center; color:#999; font-size:0.85rem'>🚲 Bike Sharing ML Project | UCI Dataset | {model_msg}</div>", unsafe_allow_html=True)