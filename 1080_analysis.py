
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="1080 Sprint Analysis", page_icon="⚡", layout="wide")

# Styling
st.markdown("""
    <style>
    .metric-box {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 padding: 20px; border-radius: 10px; color: white; text-align: center;}
    .success-box {background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px;}
    .warning-box {background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 5px;}
    .info-box {background-color: #d1ecf1; border-left: 5px solid #17a2b8; padding: 15px; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Load and process sprint data"""
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        
        # Parse dates
        # Parse dates - try multiple formats
        df['SessionTime_clean'] = df['SessionTime'].astype(str).str.split(' -').str[0]
        df['SessionDateTime'] = pd.to_datetime(df['SessionTime_clean'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        
        if df['SessionDateTime'].isna().all():
            df['SessionDateTime'] = pd.to_datetime(df['SessionTime'], errors='coerce')
        
        # Ensure column is datetime type before using .dt accessor
        if not pd.api.types.is_datetime64_any_dtype(df['SessionDateTime']):
            df['SessionDateTime'] = pd.to_datetime(df['SessionDateTime'], errors='coerce')
        
        # Check if we have any valid dates
        if df['SessionDateTime'].isna().all():
            st.error("Could not parse any dates from 'SessionTime' column. Check date format.")
            return None
        
        df['SessionDate'] = df['SessionDateTime'].dt.date
        df = df[df['SessionDate'].notna() & df['TopSpeed'].notna()].copy()
        
        # Calculate velocity decrements
        for athlete in df['Client'].unique():
            mask = df['Client'] == athlete
            min_load = df[mask]['Concentric Load [kg]'].min()
            unloaded = df[mask & (df['Concentric Load [kg]'] == min_load)]
            
            if len(unloaded) > 0:
                unloaded_speeds = unloaded.groupby('SessionDate')['TopSpeed'].max()
                for date in df[mask]['SessionDate'].unique():
                    if date in unloaded_speeds.index:
                        date_mask = mask & (df['SessionDate'] == date)
                        df.loc[date_mask, 'Velocity Decrement (%)'] = (
                            (unloaded_speeds[date] - df.loc[date_mask, 'TopSpeed']) / 
                            unloaded_speeds[date] * 100
                        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def calculate_stats(data1, data2):
    """Calculate t-test and effect size"""
    if len(data1) > 1 and len(data2) > 1:
        _, p_value = stats.ttest_ind(data1, data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std_pooled = np.sqrt(((len(data1)-1)*np.std(data1, ddof=1)**2 + 
                              (len(data2)-1)*np.std(data2, ddof=1)**2) / 
                             (len(data1)+len(data2)-2))
        effect_size = (mean2 - mean1) / std_pooled if std_pooled > 0 else 0
        return p_value, effect_size
    return None, None

def get_session_stats(df, athlete, date):
    """Get comprehensive stats for athlete on specific date"""
    data = df[(df['Client'] == athlete) & (df['SessionDate'] == date)]
    if len(data) == 0:
        return None
    
    return {
        'date': date,
        'max_speed': data['TopSpeed'].max(),
        'avg_speed': data['TopSpeed'].mean(),
        'speeds': data['TopSpeed'].values,
        'best_accel': data['0-5m Time [s]'].min(),
        'avg_accel': data['0-5m Time [s]'].mean(),
        'accel_times': data['0-5m Time [s]'].dropna().values,
        'max_acceleration': data['MaxAcceleration'].max(),
        'consistency': data['TopSpeed'].std() / data['TopSpeed'].mean() * 100,
        'sprints': len(data)
    }

# ============================================================================
# CLUSTERING & SIMILARITY
# ============================================================================

def cluster_athletes(df):
    """Cluster athletes by performance"""
    metrics = []
    for athlete in df['Client'].unique():
        data = df[df['Client'] == athlete]
        loads = data.groupby('Concentric Load [kg]')['TopSpeed'].max().sort_index()
        retention = (loads.iloc[-1] / loads.iloc[0] * 100) if len(loads) > 1 else 100
        
        metrics.append({
            'Athlete': athlete,
            'Max Speed': data['TopSpeed'].max(),
            'Avg Speed': data['TopSpeed'].mean(),
            'Best Accel': data['0-5m Time [s]'].min(),
            'Consistency': data['TopSpeed'].std(),
            'Retention': retention
        })
    
    df_metrics = pd.DataFrame(metrics)
    features = ['Max Speed', 'Avg Speed', 'Best Accel', 'Consistency', 'Retention']
    X = StandardScaler().fit_transform(df_metrics[features].fillna(df_metrics[features].mean()))
    
    n_clusters = min(3, len(metrics))
    df_metrics['Cluster'] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    df_metrics['PCA1'], df_metrics['PCA2'] = coords[:, 0], coords[:, 1]
    
    return df_metrics

def find_similar(df, target_athlete, top_n=3):
    """Find similar athletes"""
    athletes = df['Client'].unique()
    target_data = df[df['Client'] == target_athlete]
    target_metrics = [
        target_data['TopSpeed'].max(),
        target_data['TopSpeed'].mean(),
        target_data['0-5m Time [s]'].min()
    ]
    
    similarities = []
    for athlete in athletes:
        if athlete == target_athlete:
            continue
        data = df[df['Client'] == athlete]
        metrics = [data['TopSpeed'].max(), data['TopSpeed'].mean(), data['0-5m Time [s]'].min()]
        
        distance = np.sqrt(sum((t - m)**2 for t, m in zip(target_metrics, metrics))) / np.sqrt(len(target_metrics))
        similarity = 100 / (1 + distance)

        similarities.append({
            'Athlete': athlete,
            'Similarity': similarity,
            'Max Speed': metrics[0],
            'Best Accel': metrics[2]
        })
    
    return pd.DataFrame(similarities).sort_values('Similarity', ascending=False).head(top_n)

# ============================================================================
# SOCCER-SPECIFIC ANALYTICS
# ============================================================================

def find_optimal_power_zone(df, athlete):
    """Find load that produces maximum power output"""
    athlete_data = df[df['Client'] == athlete]
    
    # Get AVERAGE values at each load
    load_power = athlete_data.groupby('Concentric Load [kg]').agg({
        'TopSpeed': 'mean',
        'MaxAcceleration': 'mean'
    }).reset_index()
    
    # Calculate power: Mass × Acceleration × Velocity
    weight = athlete_data['Client Weight [kg]'].iloc[0]
    load_power['Estimated Power'] = (weight + load_power['Concentric Load [kg]']) * load_power['MaxAcceleration'] * load_power['TopSpeed']
    
    optimal_load = load_power.loc[load_power['Estimated Power'].idxmax(), 'Concentric Load [kg]']
    max_power = load_power['Estimated Power'].max()
    
    # FIXED: Define training zones with safety checks
    loads_sorted = np.sort(load_power['Concentric Load [kg]'].values)
    min_load = loads_sorted[0]
    max_load = loads_sorted[-1]
    load_range = max_load - min_load
    
    # Speed zone: lightest third OR up to optimal*0.7 (whichever is smaller)
    speed_upper = min(min_load + load_range/3, optimal_load * 0.7)
    speed_zone = (min_load, max(speed_upper, min_load + 1))  # At least 1kg range
    
    # Power zone: centered on optimal, ±20%, clamped to tested loads
    power_lower = max(optimal_load * 0.8, min_load)
    power_upper = min(optimal_load * 1.2, max_load)
    power_zone = (power_lower, power_upper)
    
    # Strength zone: heavier than power zone
    strength_lower = min(optimal_load * 1.3, max_load - 5)  # Leave 5kg at top
    if strength_lower < max_load:
        strength_zone = (strength_lower, max_load)
    else:
        strength_zone = (max_load * 0.9, max_load)  # Use top 10% if no room
    
    return {
        'optimal_load': optimal_load,
        'max_power': max_power,
        'speed_zone': speed_zone,
        'power_zone': power_zone,
        'strength_zone': strength_zone,
        'load_power_data': load_power
    }

def analyze_asymmetry(df, athlete):
    """Analyze left/right asymmetries"""
    athlete_data = df[(df['Client'] == athlete) & (df['Side'].notna())]
    
    if len(athlete_data) == 0 or athlete_data['Side'].nunique() < 2:
        return None
    
    left_data = athlete_data[athlete_data['Side'] == 'Left']
    right_data = athlete_data[athlete_data['Side'] == 'Right']
    
    if len(left_data) == 0 or len(right_data) == 0:
        return None
    
    left_speed = left_data['TopSpeed'].max()
    right_speed = right_data['TopSpeed'].max()
    left_accel = left_data['MaxAcceleration'].max()
    right_accel = right_data['MaxAcceleration'].max()
    
    speed_asym = abs(left_speed - right_speed) / max(left_speed, right_speed) * 100
    accel_asym = abs(left_accel - right_accel) / max(left_accel, right_accel) * 100
    
    risk_level = "Low" if speed_asym < 5 else "Moderate" if speed_asym < 10 else "High"
    
    return {
        'left_speed': left_speed,
        'right_speed': right_speed,
        'left_accel': left_accel,
        'right_accel': right_accel,
        'speed_asymmetry': speed_asym,
        'accel_asymmetry': accel_asym,
        'risk_level': risk_level
    }

def calculate_acceleration_dominance(df, athlete):
    """Determine if athlete is acceleration or top-speed dominant"""
    athlete_data = df[df['Client'] == athlete]
    
    best_accel = athlete_data['0-5m Time [s]'].min()
    max_speed = athlete_data['TopSpeed'].max()
    
    # FIXED: Use velocity-to-acceleration-time ratio
    # Higher ratio = speed-dominant (high top speed, slower acceleration)
    # Lower ratio = accel-dominant (quick start, lower top speed)
    velocity_accel_ratio = max_speed / best_accel
    
    # Calculate contributions based on ratio
    # Normalized to percentage for display purposes
    # Using ratio thresholds: <7.0 = accel, >8.5 = speed, between = balanced
    if velocity_accel_ratio > 8.5:
        profile = "Top Speed Dominant"
        speed_contribution = 60
        accel_contribution = 40
    elif velocity_accel_ratio < 7.0:
        profile = "Acceleration Dominant"
        speed_contribution = 40
        accel_contribution = 60
    else:
        profile = "Balanced"
        speed_contribution = 50
        accel_contribution = 50
    
    return {
        'best_accel': best_accel,
        'max_speed': max_speed,
        'velocity_accel_ratio': velocity_accel_ratio,
        'accel_contribution': accel_contribution,
        'speed_contribution': speed_contribution,
        'profile': profile
    }

def analyze_eccentric_strength(df, athlete):
    """Analyze eccentric loading capacity"""
    athlete_data = df[df['Client'] == athlete]
    
    if 'Eccentric Load [kg]' not in df.columns or athlete_data['Eccentric Load [kg]'].isna().all():
        return None
    
    # ADDED: Data quality check - need minimum sessions with eccentric data
    ecc_sessions = athlete_data['Eccentric Load [kg]'].notna().sum()
    if ecc_sessions < 5:
        return None  # Not enough data for reliable assessment
    
    avg_concentric = athlete_data['Concentric Load [kg]'].mean()
    avg_eccentric = athlete_data['Eccentric Load [kg]'].mean()
    
    ecc_con_ratio = avg_eccentric / avg_concentric if avg_concentric > 0 else 0
    
    # Check if can handle higher eccentric loads
    max_ecc = athlete_data['Eccentric Load [kg]'].max()
    max_con = athlete_data['Concentric Load [kg]'].max()
    
    # FIXED: Better thresholds based on strength training literature
    # Typical ecc:con ratio is 1.2-1.3 (can handle 20-30% more eccentric load)
    if ecc_con_ratio > 1.4:
        capacity = "Excellent"
    elif ecc_con_ratio > 1.2:
        capacity = "Good"
    elif ecc_con_ratio >= 1.0:
        capacity = "Developing"
    else:
        capacity = "Needs Development"
    
    return {
        'avg_concentric': avg_concentric,
        'avg_eccentric': avg_eccentric,
        'ratio': ecc_con_ratio,
        'max_eccentric': max_ecc,
        'max_concentric': max_con,
        'capacity': capacity,
        'sessions_with_ecc': ecc_sessions
    }

def calculate_power_endurance(df, athlete, date):
    """Calculate power maintenance across a session"""
    session_data = df[(df['Client'] == athlete) & (df['SessionDate'] == date)]
    session_data = session_data.sort_values('RepTime')
    
    if len(session_data) < 6:
        return None
    
    first_sprints = session_data.head(3)['TopSpeed'].mean()
    last_sprints = session_data.tail(3)['TopSpeed'].mean()
    
    decrement = ((first_sprints - last_sprints) / first_sprints) * 100
    
    endurance_score = "Excellent" if decrement < 3 else "Good" if decrement < 5 else "Moderate" if decrement < 8 else "Poor"
    
    return {
        'first_sprints_avg': first_sprints,
        'last_sprints_avg': last_sprints,
        'decrement': decrement,
        'endurance_score': endurance_score
    }

def calculate_injury_risk(df, athlete):
    """Comprehensive injury risk assessment"""
    risk_factors = []
    risk_score = 0
    
    # Check asymmetry
    asym = analyze_asymmetry(df, athlete)
    if asym:
        if asym['risk_level'] == "High":
            risk_factors.append(f"⚠️ High asymmetry: {asym['speed_asymmetry']:.1f}%")
            risk_score += 35  # CHANGED: 30 → 35
        elif asym['risk_level'] == "Moderate":
            risk_factors.append(f"⚡ Moderate asymmetry: {asym['speed_asymmetry']:.1f}%")
            risk_score += 20  # CHANGED: 15 → 20
    
    # FIXED: Check recent performance drop (last 3 sessions vs previous 3 sessions)
    athlete_data = df[df['Client'] == athlete]
    dates = sorted(athlete_data['SessionDate'].unique())
    
    if len(dates) >= 6:  # CHANGED: need at least 6 sessions for proper comparison
        # Recent: last 3 sessions
        recent_sessions = dates[-3:]
        recent_max = max([
            athlete_data[athlete_data['SessionDate'] == d]['TopSpeed'].max() 
            for d in recent_sessions
        ])
        
        # Baseline: previous 3 sessions (sessions 4-6 from end)
        baseline_sessions = dates[-6:-3]
        baseline_max = max([
            athlete_data[athlete_data['SessionDate'] == d]['TopSpeed'].max() 
            for d in baseline_sessions
        ])
        
        drop = ((baseline_max - recent_max) / baseline_max) * 100
        if drop > 5:
            risk_factors.append(f"📉 Recent speed drop: {drop:.1f}%")
            risk_score += 35  # CHANGED: 25 → 35
    
    # Check consistency
    cv = (athlete_data['TopSpeed'].std() / athlete_data['TopSpeed'].mean()) * 100
    if cv > 10:
        risk_factors.append(f"📊 High variability: {cv:.1f}% CV")
        risk_score += 30  # CHANGED: 20 → 30
    
    # Overall risk (thresholds adjusted for 0-100 scale)
    if risk_score >= 50:
        risk_level = "🔴 High Risk"
        recommendation = "Reduce training load, focus on corrective exercises"
    elif risk_score >= 25:
        risk_level = "🟡 Moderate Risk"
        recommendation = "Monitor closely, consider lighter loads"
    else:
        risk_level = "🟢 Low Risk"
        recommendation = "Continue current training program"
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'recommendation': recommendation
    }

def predict_performance(df, athlete):
    """Performance trend predictor using EWMA (Exponential Weighted Moving Average)"""
    athlete_data = df[df['Client'] == athlete].copy()
    dates = sorted(athlete_data['SessionDate'].unique())
    
    if len(dates) < 3:
        return None
    
    # Prepare data
    athlete_data['Days'] = (athlete_data['SessionDate'] - dates[0]).apply(lambda x: x.days)
    
    daily_max = athlete_data.groupby('Days').agg({
        'TopSpeed': 'max',
        '0-5m Time [s]': 'min',
        'MaxAcceleration': 'max'
    }).reset_index()
    
    if len(daily_max) < 3:
        return None
    
    # FIXED: Use EWMA for trend calculation (recent data weighted more)
    # Alpha = 0.3 means recent sessions get 30% weight, older sessions decay exponentially
    speeds = daily_max['TopSpeed'].values
    days = daily_max['Days'].values
    
    # Calculate EWMA
    alpha = 0.3  # Weighting factor (higher = more weight to recent)
    ewma = np.zeros(len(speeds))
    ewma[0] = speeds[0]
    
    for i in range(1, len(speeds)):
        ewma[i] = alpha * speeds[i] + (1 - alpha) * ewma[i-1]
    
    # Calculate trend from EWMA (slope of last few points)
    if len(ewma) >= 5:
        # Use last 5 points for trend
        recent_days = days[-5:]
        recent_ewma = ewma[-5:]
    else:
        # Use all points if less than 5
        recent_days = days
        recent_ewma = ewma
    
    # Calculate trend rate (m/s per day)
    X_trend = recent_days.reshape(-1, 1)
    y_trend = recent_ewma
    
    trend_model = LinearRegression()
    trend_model.fit(X_trend, y_trend)
    trend_rate = trend_model.coef_[0]
    
    # Current performance (EWMA smoothed)
    current_speed = ewma[-1]
    current_max = speeds[-1]  # Actual last session max
    
    # FIXED: Predict with dampening and caps
    days_ahead = [7, 14, 30]
    predictions = []
    
    for days_forward in days_ahead:
        # Linear prediction from EWMA
        raw_prediction = current_speed + (trend_rate * days_forward)
        
        # CRITICAL: Cap predictions at ±5% from current max
        max_increase = current_max * 1.05  # Can't improve more than 5%
        min_decrease = current_max * 0.95  # Can't decline more than 5%
        
        # Apply dampening for longer predictions (predictions get more conservative)
        dampening_factor = 1.0 - (days_forward / 100)  # Reduces confidence over time
        damped_prediction = current_speed + (raw_prediction - current_speed) * dampening_factor
        
        # Cap the prediction
        capped_prediction = np.clip(damped_prediction, min_decrease, max_increase)
        predictions.append(capped_prediction)
    
    # FIXED: Better trend classification based on rate and consistency
    # Calculate consistency (lower std = more reliable trend)
    speed_std = np.std(speeds[-5:]) if len(speeds) >= 5 else np.std(speeds)
    consistency = "high" if speed_std < 0.1 else "moderate" if speed_std < 0.2 else "low"
    
    # Trend direction with better thresholds
    # 0.001 m/s per day = 0.03 m/s per month = meaningful change
    if trend_rate > 0.002:
        trend_direction = "📈 Improving"
        confidence = "High" if consistency == "high" else "Moderate"
    elif trend_rate < -0.002:
        trend_direction = "📉 Declining"
        confidence = "High" if consistency == "high" else "Moderate"
    else:
        trend_direction = "➡️ Stable"
        confidence = "High" if consistency == "high" else "Low"
    
    return {
        'current_speed': current_max,  # Actual last session
        'smoothed_speed': current_speed,  # EWMA smoothed
        'predicted_7d': predictions[0],
        'predicted_14d': predictions[1],
        'predicted_30d': predictions[2],
        'trend': trend_direction,
        'trend_rate': trend_rate,
        'confidence': confidence,
        'consistency': consistency
    }

# ============================================================================
# MAIN APP
# ============================================================================

st.title("⚡ 1080 Sprint Performance Analysis")

# Sidebar - File Upload Only
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload 1080 Sprint CSV", type=['csv'])
    
    if uploaded_file:
        with st.spinner("Loading..."):
            st.session_state.data = load_data(uploaded_file)
        
        if st.session_state.data is not None:
            st.success(f"✅ {len(st.session_state.data)} sprints loaded")
            st.metric("Athletes", st.session_state.data['Client'].nunique())
            st.metric("Training Days", st.session_state.data['SessionDate'].nunique())

# Main Content
if st.session_state.data is None:
    st.info("👈 Upload your 1080 Sprint CSV file to begin")
    st.markdown("""
    ### Features:
    - **🔬 Clustering & Similarity**: Find athlete groups and training partners
    - **📈 Performance Tracking**: Monitor all key metrics over time
    - **👥 Team Trends**: Analyze team-wide improvements
    - **🎯 Individual Progression**: Track from baseline with statistics
    """)
    st.stop()

df = st.session_state.data
athletes = sorted(df['Client'].unique())
dates = sorted(df['SessionDate'].unique())

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Clustering & Similarity",
    "📈 Player Performance",
    "👥 Team Trends",
    "🎯 Individual Progression",
    "⚽ Soccer Analytics"
])

# ============================================================================
# TAB 1: CLUSTERING & SIMILARITY
# ============================================================================

with tab1:
    st.header("🔬 Athlete Clustering & Similarity Analysis")
    
    cluster_df = cluster_athletes(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Performance Clusters")
        fig = px.scatter(
            cluster_df, x='PCA1', y='PCA2', color='Cluster', text='Athlete',
            title='Athlete Groups (PCA Visualization)', height=500
        )
        fig.update_traces(textposition='top center', marker=dict(size=15))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cluster Summary")
        for cluster in cluster_df['Cluster'].unique():
            with st.expander(f"Cluster {cluster}"):
                cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
                st.write("**Athletes:**", ", ".join(cluster_data['Athlete'].tolist()))
                st.metric("Avg Max Speed", f"{cluster_data['Max Speed'].mean():.2f} m/s")
                st.metric("Avg Retention", f"{cluster_data['Retention'].mean():.1f}%")
    
    st.markdown("---")
    
    # Similarity Analysis
    st.subheader("👥 Find Similar Athletes")
    selected = st.selectbox("Select Athlete", athletes, key='similarity')
    
    col1, col2 = st.columns(2)
    
    with col1:
        similar = find_similar(df, selected)
        st.dataframe(similar, use_container_width=True)
        st.caption("Most similar athletes - ideal training partners")
    
    with col2:
        athlete_data = df[df['Client'] == selected]
        st.metric("Max Speed", f"{athlete_data['TopSpeed'].max():.2f} m/s")
        st.metric("Best 0-5m", f"{athlete_data['0-5m Time [s]'].min():.3f} s")
        st.metric("Training Sessions", athlete_data['SessionDate'].nunique())
        
        # Performance Profile
        loads = athlete_data.groupby('Concentric Load [kg]')['TopSpeed'].max().sort_index()
        if len(loads) > 1:
            retention = (loads.iloc[-1] / loads.iloc[0]) * 100
            if retention > 65:
                st.success("**Profile:** Force-Dominant 💪\n\nFocus: Max velocity work")
            elif retention > 50:
                st.info("**Profile:** Balanced ⚖️\n\nFocus: Mixed training")
            else:
                st.warning("**Profile:** Velocity-Dominant 🏃\n\nFocus: Strength work")

# ============================================================================
# TAB 2: PLAYER PERFORMANCE
# ============================================================================

with tab2:
    st.header("📈 Player Performance Over Time")
    
    selected_athlete = st.selectbox("Select Athlete", athletes, key='performance')
    athlete_data = df[df['Client'] == selected_athlete]
    athlete_dates = sorted(athlete_data['SessionDate'].unique())
    
    # Summary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Max Speed", f"{athlete_data['TopSpeed'].max():.2f} m/s")
    with col2:
        st.metric("Avg Speed", f"{athlete_data['TopSpeed'].mean():.2f} m/s")
    with col3:
        st.metric("Best 0-5m", f"{athlete_data['0-5m Time [s]'].min():.3f} s")
    with col4:
        cv = (athlete_data['TopSpeed'].std() / athlete_data['TopSpeed'].mean()) * 100
        st.metric("Consistency", f"{cv:.1f}% CV")
    with col5:
        st.metric("Sessions", len(athlete_dates))
    
    st.markdown("---")
    
    # Metrics over time
    progress = athlete_data.groupby('SessionDate').agg({
        'TopSpeed': 'max',
        '0-5m Time [s]': 'min',
        'MaxAcceleration': 'max'
    }).reset_index().sort_values('SessionDate')
    
    # Multi-metric chart
    metric_choice = st.selectbox("Select Metric", 
                                  ["Max Speed", "0-5m Time", "Max Acceleration"],
                                  key='metric')
    
    metric_map = {
        "Max Speed": ('TopSpeed', 'Speed (m/s)', False),
        "0-5m Time": ('0-5m Time [s]', 'Time (s)', True),
        "Max Acceleration": ('MaxAcceleration', 'Acceleration (m/s²)', False)
    }
    
    col, ylabel, invert = metric_map[metric_choice]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=progress['SessionDate'], y=progress[col],
        mode='lines+markers', name=metric_choice,
        line=dict(color='#3b82f6', width=3), marker=dict(size=10)
    ))
    
    # Trend line
    x_num = np.arange(len(progress))
    z = np.polyfit(x_num, progress[col].values, 1)
    fig.add_trace(go.Scatter(
        x=progress['SessionDate'], y=np.poly1d(z)(x_num),
        mode='lines', name='Trend',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        xaxis_title='Date', yaxis_title=ylabel,
        height=400, template='plotly_white', hovermode='x unified'
    )
    if invert:
        fig.update_yaxes(autorange='reversed')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Load-Velocity Profile
    st.markdown("---")
    st.subheader("Load-Velocity Profile")
    
    lv_data = athlete_data.groupby('Concentric Load [kg]')['TopSpeed'].max().reset_index()
    lv_data = lv_data.sort_values('Concentric Load [kg]')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lv_data['Concentric Load [kg]'], y=lv_data['TopSpeed'],
        mode='lines+markers', line=dict(color='#10b981', width=3), marker=dict(size=10)
    ))
    
    fig.update_layout(
        xaxis_title='Load (kg)', yaxis_title='Max Velocity (m/s)',
        height=350, template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: TEAM TRENDS
# ============================================================================

# ============================================================================
# TAB 3: TEAM TRENDS
# ============================================================================

with tab3:
    st.header("👥 Team-Wide Trend Analysis")
    
    # FIXED: Use date ranges instead of exact dates
    st.subheader("Select Time Periods to Compare")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Baseline Period**")
        baseline_start = st.selectbox("From", sorted(dates, reverse=True), key='baseline_start', index=len(dates)-1)
        baseline_end = st.selectbox("To", sorted(dates, reverse=True), key='baseline_end', index=max(0, len(dates)-3))
    
    with col2:
        st.markdown("**Recent Period**")
        recent_start = st.selectbox("From", sorted(dates, reverse=True), key='recent_start', index=min(2, len(dates)-1))
        recent_end = st.selectbox("To", sorted(dates, reverse=True), key='recent_end', index=0)
    
    # Validate date ranges
    if baseline_start >= baseline_end:
        st.warning("⚠️ Baseline: 'From' date must be before 'To' date")
        st.stop()
    
    if recent_start >= recent_end:
        st.warning("⚠️ Recent: 'From' date must be before 'To' date")
        st.stop()
    
    if baseline_end >= recent_start:
        st.warning("⚠️ Baseline period must end before Recent period starts")
        st.stop()
    
    st.markdown("---")
    
    # FIXED: Analyze team progress using date ranges
    team_results = []
    for athlete in athletes:
        athlete_data = df[df['Client'] == athlete]
        
        # Get baseline period data
        baseline_mask = (athlete_data['SessionDate'] >= baseline_start) & (athlete_data['SessionDate'] <= baseline_end)
        baseline_data = athlete_data[baseline_mask]
        
        # Get recent period data
        recent_mask = (athlete_data['SessionDate'] >= recent_start) & (athlete_data['SessionDate'] <= recent_end)
        recent_data = athlete_data[recent_mask]
        
        # Only include athletes who have data in BOTH periods
        if len(baseline_data) > 0 and len(recent_data) > 0:
            # Get best performance in each period
            baseline_max = baseline_data['TopSpeed'].max()
            recent_max = recent_data['TopSpeed'].max()
            
            # Get all speeds for statistical testing
            baseline_speeds = baseline_data['TopSpeed'].values
            recent_speeds = recent_data['TopSpeed'].values
            
            # Calculate change
            speed_change = recent_max - baseline_max
            speed_change_pct = (speed_change / baseline_max) * 100
            
            # Statistical test
            p_val, effect = calculate_stats(baseline_speeds, recent_speeds)
            
            team_results.append({
                'Athlete': athlete,
                'Baseline Sessions': len(baseline_data),
                'Recent Sessions': len(recent_data),
                'Baseline Max': baseline_max,
                'Recent Max': recent_max,
                'Speed Change': speed_change,
                'Speed Change (%)': speed_change_pct,
                'P-Value': p_val,
                'Effect Size': effect,
                'Improved': speed_change > 0
            })
    
    team_df = pd.DataFrame(team_results)
    
    if len(team_df) == 0:
        st.warning("⚠️ No athletes have data in both time periods. Try different date ranges.")
        st.stop()
    
    # Summary
    improved = team_df['Improved'].sum()
    total = len(team_df)
    significant = (team_df['P-Value'] < 0.05).sum() if team_df['P-Value'].notna().any() else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Athletes Analyzed", f"{total}")
    with col2:
        st.metric("Athletes Improved", f"{improved}/{total}")
    with col3:
        st.metric("Improvement Rate", f"{(improved/total*100):.0f}%")
    with col4:
        st.metric("Statistically Significant", f"{significant}/{total}")
    
    st.markdown("---")
    
    # Period summary
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Baseline:** {baseline_start} to {baseline_end}")
        st.metric("Avg Team Max Speed", f"{team_df['Baseline Max'].mean():.2f} m/s")
    with col2:
        st.info(f"**Recent:** {recent_start} to {recent_end}")
        st.metric("Avg Team Max Speed", f"{team_df['Recent Max'].mean():.2f} m/s")
    
    st.markdown("---")
    
    # Insight
    if improved / total >= 0.75:
        st.markdown('<div class="success-box">✅ <b>Excellent Team Progress!</b> ' +
                   f'{improved}/{total} athletes improved. Training is highly effective.</div>',
                   unsafe_allow_html=True)
    elif improved / total >= 0.5:
        st.markdown('<div class="info-box">📊 <b>Good Progress:</b> ' +
                   f'{improved}/{total} athletes improved. Continue monitoring.</div>',
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ <b>Review Program:</b> ' +
                   f'Only {improved}/{total} improved. Consider adjustments.</div>',
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualization
    fig = go.Figure()
    colors = ['#28a745' if x else '#dc3545' for x in team_df['Improved']]
    
    fig.add_trace(go.Bar(
        x=team_df['Athlete'], y=team_df['Speed Change (%)'],
        marker_color=colors,
        text=team_df['Speed Change (%)'].apply(lambda x: f"{x:+.1f}%"),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Speed Change by Athlete',
        xaxis_title='Athlete', yaxis_title='Change (%)',
        height=400, template='plotly_white', showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Results")
    display_df = team_df.copy()
    display_df['Baseline Max'] = display_df['Baseline Max'].apply(lambda x: f"{x:.2f}")
    display_df['Recent Max'] = display_df['Recent Max'].apply(lambda x: f"{x:.2f}")
    display_df['Speed Change'] = display_df['Speed Change'].apply(lambda x: f"{x:+.3f} m/s")
    display_df['Speed Change (%)'] = display_df['Speed Change (%)'].apply(lambda x: f"{x:+.1f}%")
    display_df['P-Value'] = display_df['P-Value'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['Status'] = display_df['Improved'].apply(lambda x: "🟢 Improved" if x else "🔴 Declined")
    
    st.dataframe(display_df[['Athlete', 'Baseline Sessions', 'Recent Sessions', 
                              'Baseline Max', 'Recent Max', 'Speed Change', 
                              'Speed Change (%)', 'P-Value', 'Status']], 
                 use_container_width=True)

# ============================================================================
# TAB 4: INDIVIDUAL PROGRESSION
# ============================================================================

with tab4:
    st.header("🎯 Individual Progression from Baseline")
    
    selected_athlete = st.selectbox("Select Athlete", athletes, key='progression')
    athlete_data = df[df['Client'] == selected_athlete]
    athlete_dates = sorted(athlete_data['SessionDate'].unique())
    
    if len(athlete_dates) < 2:
        st.warning(f"⚠️ {selected_athlete} needs at least 2 sessions")
        st.stop()
    
    baseline = athlete_dates[0]
    baseline_stats = get_session_stats(df, selected_athlete, baseline)
    
    # Header info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Baseline:** {baseline}")
    with col2:
        st.info(f"**Sessions:** {len(athlete_dates)}")
    with col3:
        st.info(f"**Period:** {(athlete_dates[-1] - baseline).days} days")
    
    st.markdown("---")
    
    # Calculate progression
    progression = []
    for date in athlete_dates[1:]:
        session_stats = get_session_stats(df, selected_athlete, date)
        if session_stats:
            speed_change = session_stats['max_speed'] - baseline_stats['max_speed']
            p_val, effect = calculate_stats(baseline_stats['speeds'], session_stats['speeds'])
            
            progression.append({
                'Date': date,
                'Days': (date - baseline).days,
                'Max Speed': session_stats['max_speed'],
                'Change': speed_change,
                'Change (%)': (speed_change / baseline_stats['max_speed']) * 100,
                'P-Value': p_val,
                'Effect': effect,
                'Improved': speed_change > 0
            })
    
    prog_df = pd.DataFrame(progression)
    
    # Summary
    improved = prog_df['Improved'].sum()
    total = len(prog_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Improved Sessions", f"{improved}/{total}")
    with col2:
        st.metric("Avg Improvement", f"{prog_df['Change (%)'].mean():+.1f}%")
    with col3:
        st.metric("Latest Speed", f"{prog_df.iloc[-1]['Max Speed']:.2f} m/s")
    with col4:
        st.metric("Total Gain", f"{prog_df.iloc[-1]['Change']:+.2f} m/s")
    
    st.markdown("---")
    
    # Insight
    if improved / total >= 0.75:
        st.markdown('<div class="success-box">🌟 <b>Excellent Progression!</b> ' +
                   f'{improved}/{total} sessions faster than baseline.</div>',
                   unsafe_allow_html=True)
    elif improved / total >= 0.5:
        st.markdown('<div class="info-box">📈 <b>Positive Progress:</b> ' +
                   f'{improved}/{total} sessions show improvement.</div>',
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ <b>Review Needed:</b> ' +
                   f'Only {improved}/{total} sessions improved.</div>',
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart
    fig = go.Figure()
    
    all_dates = [baseline] + list(prog_df['Date'])
    all_speeds = [baseline_stats['max_speed']] + list(prog_df['Max Speed'])
    colors = ['gray'] + ['green' if x else 'red' for x in prog_df['Improved']]
    
    fig.add_trace(go.Scatter(
        x=all_dates, y=all_speeds,
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
        name='Max Speed'
    ))
    
    # Trend
    x_num = np.arange(len(all_dates))
    z = np.polyfit(x_num, all_speeds, 1)
    fig.add_trace(go.Scatter(
        x=all_dates, y=np.poly1d(z)(x_num),
        mode='lines', name='Trend',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        xaxis_title='Date', yaxis_title='Max Speed (m/s)',
        height=400, template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⭐ Gray = Baseline | 🟢 Green = Improved | 🔴 Red = Declined")
    
    # Table
    st.subheader("Session Details")
    display_prog = prog_df.copy()
    display_prog['Date'] = display_prog['Date'].astype(str)
    display_prog['Max Speed'] = display_prog['Max Speed'].apply(lambda x: f"{x:.2f}")
    display_prog['Change'] = display_prog['Change'].apply(lambda x: f"{x:+.3f}")
    display_prog['Change (%)'] = display_prog['Change (%)'].apply(lambda x: f"{x:+.1f}%")
    display_prog['P-Value'] = display_prog['P-Value'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_prog['Status'] = display_prog['Improved'].apply(lambda x: "🟢" if x else "🔴")
    
    st.dataframe(display_prog[['Date', 'Days', 'Max Speed', 'Change', 'Change (%)', 'P-Value', 'Status']], 
                 use_container_width=True)
    
    # Download
    csv = prog_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Data", csv, 
                      f"{selected_athlete.replace(' ', '_')}_progression.csv", "text/csv")

# ============================================================================
# TAB 5: SOCCER ANALYTICS
# ============================================================================

with tab5:
    st.header("⚽ Soccer-Specific Performance Analytics")
    st.caption("Advanced metrics tailored for soccer performance and injury prevention")
    
    selected_athlete = st.selectbox("Select Player", athletes, key='soccer_athlete')
    athlete_data = df[df['Client'] == selected_athlete]
    
    # Create sub-tabs for organization
    soccer_tab1, soccer_tab2, soccer_tab3 = st.tabs([
        "🎯 Training Prescription",
        "🚨 Injury Risk & Asymmetry", 
        "📊 Performance Analysis"
    ])
    
    # ========================================================================
    # SOCCER TAB 1: TRAINING PRESCRIPTION
    # ========================================================================
    
    with soccer_tab1:
        st.subheader("⚡ Optimal Power Zone Analysis")
        
        power_analysis = find_optimal_power_zone(df, selected_athlete)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimal Load", f"{power_analysis['optimal_load']:.0f} kg")
            st.caption("Load producing maximum power")
        with col2:
            st.metric("Max Power Output", f"{power_analysis['max_power']:.0f} W")
            st.caption("Peak power capability")
        with col3:
            loads_trained = athlete_data['Concentric Load [kg]'].nunique()
            st.metric("Loads Tested", loads_trained)
        
        st.markdown("---")
        
        # Training zones visualization
        st.subheader("🎓 Personalized Training Zones")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("**🟢 Speed Zone**")
            st.markdown(f"**{power_analysis['speed_zone'][0]:.1f} - {power_analysis['speed_zone'][1]:.1f} kg**")
            st.markdown("**Focus:** Max velocity development")
            st.markdown("**Exercises:**")
            st.markdown("- Overspeed training")
            st.markdown("- Flying sprints")
            st.markdown("- Light sled pulls")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**🟡 Power Zone**")
            st.markdown(f"**{power_analysis['power_zone'][0]:.1f} - {power_analysis['power_zone'][1]:.1f} kg**")
            st.markdown("**Focus:** Maximum power output")
            st.markdown("**Exercises:**")
            st.markdown("- Resisted sprints")
            st.markdown("- Jump training")
            st.markdown("- Explosive starts")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**🔴 Strength Zone**")
            st.markdown(f"**{power_analysis['strength_zone'][0]:.1f} - {power_analysis['strength_zone'][1]:.1f} kg**")
            st.markdown("**Focus:** Force production")
            st.markdown("**Exercises:**")
            st.markdown("- Heavy sled pushes")
            st.markdown("- Hill sprints")
            st.markdown("- Strength training")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Power curve
        st.subheader("📈 Power-Load Relationship")
        fig = go.Figure()
        
        power_data = power_analysis['load_power_data']
        fig.add_trace(go.Scatter(
            x=power_data['Concentric Load [kg]'],
            y=power_data['Estimated Power'],
            mode='lines+markers',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=10),
            name='Power Output'
        ))
        
        # Highlight optimal
        optimal_idx = power_data['Estimated Power'].idxmax()
        fig.add_trace(go.Scatter(
            x=[power_data.loc[optimal_idx, 'Concentric Load [kg]']],
            y=[power_data.loc[optimal_idx, 'Estimated Power']],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Optimal Load'
        ))
        
        fig.update_layout(
            xaxis_title='Load (kg)',
            yaxis_title='Estimated Power (W)',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly prescription
        st.markdown("---")
        st.subheader("📅 Suggested Weekly Training Plan")
        
        plan = pd.DataFrame([
            {'Day': 'Monday', 'Zone': '🟡 Power', 'Load': f"{power_analysis['optimal_load']:.0f}kg", 
             'Sets × Reps': '6 × 3', 'Rest': '3-4 min'},
            {'Day': 'Wednesday', 'Zone': '🟢 Speed', 'Load': f"{power_analysis['speed_zone'][1]:.0f}kg", 
             'Sets × Reps': '4 × 4', 'Rest': '2-3 min'},
            {'Day': 'Friday', 'Zone': '🔴 Strength', 'Load': f"{power_analysis['strength_zone'][0]:.0f}kg", 
             'Sets × Reps': '5 × 3', 'Rest': '4-5 min'},
        ])
        
        st.dataframe(plan, use_container_width=True)
    
    # ========================================================================
    # SOCCER TAB 2: INJURY RISK & ASYMMETRY
    # ========================================================================
    
    with soccer_tab2:
        st.subheader("🚨 Comprehensive Injury Risk Assessment")
        
        risk = calculate_injury_risk(df, selected_athlete)
        
        # Risk score gauge
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"### {risk['risk_level']}")
            st.metric("Risk Score", f"{risk['risk_score']}/100")
            
            # Progress bar with color
            if risk['risk_score'] >= 50:
                st.error(f"Risk: {risk['risk_score']}%")
            elif risk['risk_score'] >= 25:
                st.warning(f"Risk: {risk['risk_score']}%")
            else:
                st.success(f"Risk: {risk['risk_score']}%")
        
        with col2:
            st.markdown("**Risk Factors Identified:**")
            if len(risk['risk_factors']) == 0:
                st.success("✅ No significant risk factors detected")
            else:
                for factor in risk['risk_factors']:
                    st.markdown(f"- {factor}")
            
            st.markdown(f"\n**Recommendation:** {risk['recommendation']}")
        
        st.markdown("---")
        
        # Asymmetry Analysis
        st.subheader("⚖️ Left/Right Asymmetry Analysis")
        
        asym = analyze_asymmetry(df, selected_athlete)
        
        if asym is None:
            st.info("ℹ️ Asymmetry data not available. Ensure sprints are tagged with Left/Right side.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Speed Asymmetry", f"{asym['speed_asymmetry']:.1f}%")
                st.caption(f"L: {asym['left_speed']:.2f} | R: {asym['right_speed']:.2f} m/s")
            
            with col2:
                st.metric("Accel Asymmetry", f"{asym['accel_asymmetry']:.1f}%")
                st.caption(f"L: {asym['left_accel']:.2f} | R: {asym['right_accel']:.2f} m/s²")
            
            with col3:
                if asym['risk_level'] == "High":
                    st.error(f"Risk Level: {asym['risk_level']}")
                elif asym['risk_level'] == "Moderate":
                    st.warning(f"Risk Level: {asym['risk_level']}")
                else:
                    st.success(f"Risk Level: {asym['risk_level']}")
            
            # Visual comparison
            fig = go.Figure()
            
            sides = ['Left', 'Right']
            speeds = [asym['left_speed'], asym['right_speed']]
            colors = ['#3b82f6', '#ef4444']
            
            fig.add_trace(go.Bar(
                x=sides, y=speeds,
                marker_color=colors,
                text=[f"{s:.2f}" for s in speeds],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Speed Comparison by Side',
                yaxis_title='Max Speed (m/s)',
                height=350,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if asym['speed_asymmetry'] > 10:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("**⚠️ Corrective Actions Required:**")
                st.markdown("- Unilateral strength training")
                st.markdown("- Single-leg plyometrics")
                st.markdown("- Address movement patterns")
                st.markdown("- Consider PT evaluation")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Eccentric Strength
        st.subheader("🔄 Eccentric Strength Capacity")
        
        ecc = analyze_eccentric_strength(df, selected_athlete)
        
        if ecc is None:
            st.info("ℹ️ Eccentric load data not available in this dataset.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Ecc:Con Ratio", f"{ecc['ratio']:.2f}")
                st.caption("Target: >1.2 for soccer")
            
            with col2:
                st.metric("Max Eccentric Load", f"{ecc['max_eccentric']:.1f} kg")
            
            with col3:
                if ecc['capacity'] == "Excellent":
                    st.success(f"Capacity: {ecc['capacity']}")
                elif ecc['capacity'] == "Good":
                    st.info(f"Capacity: {ecc['capacity']}")
                else:
                    st.warning(f"Capacity: {ecc['capacity']}")
            
            st.caption("**Why it matters:** Eccentric strength is critical for deceleration, change of direction, and injury prevention in soccer.")
    
    # ========================================================================
    # SOCCER TAB 3: PERFORMANCE ANALYSIS
    # ========================================================================
    
    with soccer_tab3:
        st.subheader("💪 Acceleration Dominance Profile")
        
        accel_profile = calculate_acceleration_dominance(df, selected_athlete)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best 0-5m Time", f"{accel_profile['best_accel']:.3f} s")
            st.caption("Acceleration capability")
        
        with col2:
            st.metric("Max Speed", f"{accel_profile['max_speed']:.2f} m/s")
            st.caption("Top-end velocity")
        
        with col3:
            st.metric("Profile Type", accel_profile['profile'])
        
        # Visual breakdown
        fig = go.Figure(data=[go.Pie(
            labels=['Acceleration Phase', 'Top Speed Phase'],
            values=[accel_profile['accel_contribution'], accel_profile['speed_contribution']],
            marker_colors=['#10b981', '#3b82f6'],
            hole=0.4
        )])
        
        fig.update_layout(
            title='Performance Contribution Breakdown',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Training recommendation
        if "Acceleration" in accel_profile['profile']:
            st.success("✅ Strong acceleration - maintain with power work, add max velocity training")
        elif "Top Speed" in accel_profile['profile']:
            st.info("📊 Good top speed - add more acceleration-focused training (0-20m sprints)")
        else:
            st.success("✅ Well-balanced profile - continue mixed training approach")
        
        st.markdown("---")
        
        # Power Endurance
        st.subheader("📊 Power Endurance Analysis")
        
        athlete_dates = sorted(athlete_data['SessionDate'].unique())
        if len(athlete_dates) > 0:
            latest_date = athlete_dates[-1]
            endurance = calculate_power_endurance(df, selected_athlete, latest_date)
            
            if endurance is None:
                st.info("ℹ️ Need at least 6 sprints in a session to analyze power endurance")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("First 3 Sprints Avg", f"{endurance['first_sprints_avg']:.2f} m/s")
                
                with col2:
                    st.metric("Last 3 Sprints Avg", f"{endurance['last_sprints_avg']:.2f} m/s")
                
                with col3:
                    st.metric("Sprint Decrement", f"{endurance['decrement']:.1f}%")
                
                # Endurance score
                if endurance['endurance_score'] == "Excellent":
                    st.success(f"**Power Endurance:** {endurance['endurance_score']} - Maintains quality throughout session")
                elif endurance['endurance_score'] == "Good":
                    st.info(f"**Power Endurance:** {endurance['endurance_score']} - Good maintenance")
                else:
                    st.warning(f"**Power Endurance:** {endurance['endurance_score']} - Consider conditioning work")
        
        st.markdown("---")
        
        # Performance Predictor
        st.subheader("🏆 Performance Trend Prediction")
        
        prediction = predict_performance(df, selected_athlete)
        
        if prediction is None:
            st.info("ℹ️ Need at least 3 training sessions to generate predictions")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Current Max Speed", f"{prediction['current_speed']:.2f} m/s")
                st.metric("7-Day Prediction", f"{prediction['predicted_7d']:.2f} m/s", 
                         f"{(prediction['predicted_7d'] - prediction['current_speed']):+.2f}")
                st.metric("14-Day Prediction", f"{prediction['predicted_14d']:.2f} m/s",
                         f"{(prediction['predicted_14d'] - prediction['current_speed']):+.2f}")
                st.metric("30-Day Prediction", f"{prediction['predicted_30d']:.2f} m/s",
                         f"{(prediction['predicted_30d'] - prediction['current_speed']):+.2f}")
            
            with col2:
                st.markdown(f"### {prediction['trend']}")
                st.markdown(f"**Trend Rate:** {prediction['trend_rate']:.4f} m/s per day")
                
                if "Improving" in prediction['trend']:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("**Positive trajectory!** Continue current training program. " +
                               "Monitor for plateau and adjust volume/intensity as needed.")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif "Declining" in prediction['trend']:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("**Performance declining.** Consider: reduced training load, " +
                               "recovery focus, or program adjustments.")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**Stable performance.** Maintain current approach or introduce " +
                               "new stimulus to drive adaptation.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.caption("*Predictions based on linear regression of training data. Actual results may vary based on training load, recovery, and other factors.")

# Footer
st.markdown("---")
st.caption("1080 Sprint Analysis | Statistical testing • Machine learning • Soccer-specific analytics")
