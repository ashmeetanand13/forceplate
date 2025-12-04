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

        # Standardize acceleration column name (handle both meters and yards)
        if '0-5yd Time [s]' in df.columns and '0-5m Time [s]' not in df.columns:
            df['0-5m Time [s]'] = df['0-5yd Time [s]']

        
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

# ============================================================================
# LOAD-VELOCITY PROFILE FUNCTIONS
# ============================================================================

def calculate_fv_profile(df, athlete, session_date):
    """
    Calculate Force-Velocity profile for a specific session.
    Returns F0, V0, Pmax derived from linear regression of Load vs Velocity.
    """
    session_data = df[(df['Client'] == athlete) & (df['SessionDate'] == session_date)]
    
    if len(session_data) < 2:
        return None
    
    # Get max velocity at each load for this session
    load_velocity = session_data.groupby('Concentric Load [kg]')['TopSpeed'].max().reset_index()
    load_velocity = load_velocity.sort_values('Concentric Load [kg]')
    
    if len(load_velocity) < 2:
        return None
    
    loads = load_velocity['Concentric Load [kg]'].values
    velocities = load_velocity['TopSpeed'].values
    
    # Get athlete weight
    weight = session_data['Client Weight [kg]'].iloc[0]
    
    # Linear regression: Velocity = V0 - (V0/F0) * Force
    # Where Force = (body_weight + load) * acceleration (simplified as load for relative comparison)
    # For F-V profiling, we use total system mass
    total_mass = weight + loads
    
    # Fit linear regression: V = V0 - slope * Load
    # slope = V0 / F0 (relative)
    X = loads.reshape(-1, 1)
    y = velocities
    
    model = LinearRegression()
    model.fit(X, y)
    
    # V0 = y-intercept (theoretical max velocity at 0 load)
    V0 = model.intercept_
    
    # Slope = -V0/F0, so F0 = -V0/slope
    slope = model.coef_[0]
    
    if slope >= 0:  # Invalid profile (velocity should decrease with load)
        return None
    
    # F0 in terms of load (kg) - theoretical max load at 0 velocity
    F0_load = -V0 / slope
    
    # Convert to Newtons: F0 = (body_weight + F0_load) * g
    F0_newtons = (weight + F0_load) * 9.81
    
    # Pmax = F0 * V0 / 4 (from F-V relationship)
    Pmax = (F0_newtons * V0) / 4
    
    # R-squared for quality assessment
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'date': session_date,
        'V0': V0,
        'F0_load': F0_load,
        'F0_newtons': F0_newtons,
        'Pmax': Pmax,
        'slope': slope,
        'r_squared': r_squared,
        'loads': loads,
        'velocities': velocities,
        'weight': weight,
        'n_points': len(loads)
    }


def get_sessions_with_fv_data(df, athlete, min_loads=2):
    """Get list of sessions that have enough load variation for F-V profiling."""
    athlete_data = df[df['Client'] == athlete]
    sessions = []
    
    for date in sorted(athlete_data['SessionDate'].unique()):
        session_data = athlete_data[athlete_data['SessionDate'] == date]
        n_loads = session_data['Concentric Load [kg]'].nunique()
        
        if n_loads >= min_loads:
            sessions.append({
                'date': date,
                'n_loads': n_loads,
                'n_sprints': len(session_data),
                'load_range': f"{session_data['Concentric Load [kg]'].min():.0f}-{session_data['Concentric Load [kg]'].max():.0f} kg"
            })
    
    return sessions


# ============================================================================
# ENHANCED ASYMMETRY FUNCTIONS
# ============================================================================

def analyze_asymmetry_detailed(df, athlete, session_date=None):
    """
    Detailed asymmetry analysis with load matching.
    If session_date is None, analyzes all data.
    """
    if session_date:
        athlete_data = df[(df['Client'] == athlete) & (df['SessionDate'] == session_date)]
    else:
        athlete_data = df[df['Client'] == athlete]
    
    # Filter to only rows with Side data
    athlete_data = athlete_data[athlete_data['Side'].isin(['Left', 'Right'])]
    
    if len(athlete_data) == 0:
        return None
    
    left_data = athlete_data[athlete_data['Side'] == 'Left']
    right_data = athlete_data[athlete_data['Side'] == 'Right']
    
    if len(left_data) == 0 or len(right_data) == 0:
        return None
    
    # Overall metrics
    results = {
        'left_count': len(left_data),
        'right_count': len(right_data),
        'metrics': {}
    }
    
    # Analyze multiple metrics
    metrics_to_analyze = [
        ('TopSpeed', 'Max Speed (m/s)', 'max'),
        ('MaxAcceleration', 'Max Acceleration (m/s²)', 'max'),
        ('0-5m Time [s]', '0-5m Time (s)', 'min')
    ]
    
    for col, label, agg_func in metrics_to_analyze:
        if col not in athlete_data.columns:
            continue
            
        if agg_func == 'max':
            left_val = left_data[col].max()
            right_val = right_data[col].max()
        else:
            left_val = left_data[col].min()
            right_val = right_data[col].min()
        
        if pd.isna(left_val) or pd.isna(right_val):
            continue
        
        avg_val = (left_val + right_val) / 2
        
        # Asymmetry Index: (Left - Right) / Average * 100
        # Positive = Left dominant, Negative = Right dominant
        asymmetry = ((left_val - right_val) / avg_val) * 100 if avg_val > 0 else 0
        
        # For time metrics, flip the sign (lower is better)
        if 'Time' in col:
            asymmetry = -asymmetry  # Now positive = Left faster (better)
        
        abs_asymmetry = abs(asymmetry)
        
        # Risk classification
        if abs_asymmetry < 5:
            risk = 'Low'
            color = '#10b981'  # Green
        elif abs_asymmetry < 10:
            risk = 'Moderate'
            color = '#f59e0b'  # Yellow
        else:
            risk = 'High'
            color = '#ef4444'  # Red
        
        dominant_side = 'Left' if asymmetry > 0 else 'Right' if asymmetry < 0 else 'Balanced'
        
        results['metrics'][col] = {
            'label': label,
            'left': left_val,
            'right': right_val,
            'asymmetry': asymmetry,
            'abs_asymmetry': abs_asymmetry,
            'risk': risk,
            'color': color,
            'dominant': dominant_side
        }
    
    return results


def get_asymmetry_trend(df, athlete):
    """Track asymmetry changes over time."""
    athlete_data = df[df['Client'] == athlete]
    dates = sorted(athlete_data['SessionDate'].unique())
    
    trend_data = []
    
    for date in dates:
        asym = analyze_asymmetry_detailed(df, athlete, date)
        if asym and 'TopSpeed' in asym['metrics']:
            trend_data.append({
                'date': date,
                'speed_asymmetry': asym['metrics']['TopSpeed']['abs_asymmetry'],
                'speed_dominant': asym['metrics']['TopSpeed']['dominant'],
                'accel_asymmetry': asym['metrics'].get('MaxAcceleration', {}).get('abs_asymmetry', np.nan),
                'left_speed': asym['metrics']['TopSpeed']['left'],
                'right_speed': asym['metrics']['TopSpeed']['right']
            })
    
    return pd.DataFrame(trend_data) if trend_data else None


def get_load_matched_asymmetry(df, athlete, session_date=None):
    """Compare L/R performance at matching loads."""
    if session_date:
        athlete_data = df[(df['Client'] == athlete) & (df['SessionDate'] == session_date)]
    else:
        athlete_data = df[df['Client'] == athlete]
    
    athlete_data = athlete_data[athlete_data['Side'].isin(['Left', 'Right'])]
    
    if len(athlete_data) == 0:
        return None
    
    # Find loads that have both L and R data
    left_loads = set(athlete_data[athlete_data['Side'] == 'Left']['Concentric Load [kg]'].unique())
    right_loads = set(athlete_data[athlete_data['Side'] == 'Right']['Concentric Load [kg]'].unique())
    common_loads = left_loads & right_loads
    
    if len(common_loads) == 0:
        return None
    
    results = []
    for load in sorted(common_loads):
        left = athlete_data[(athlete_data['Side'] == 'Left') & (athlete_data['Concentric Load [kg]'] == load)]
        right = athlete_data[(athlete_data['Side'] == 'Right') & (athlete_data['Concentric Load [kg]'] == load)]
        
        left_speed = left['TopSpeed'].max()
        right_speed = right['TopSpeed'].max()
        avg_speed = (left_speed + right_speed) / 2
        asymmetry = ((left_speed - right_speed) / avg_speed) * 100 if avg_speed > 0 else 0
        
        results.append({
            'load': load,
            'left_speed': left_speed,
            'right_speed': right_speed,
            'asymmetry': asymmetry,
            'abs_asymmetry': abs(asymmetry)
        })
    
    return pd.DataFrame(results)


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
    - **📈 Player Analysis**: Performance tracking, baseline progression & load-velocity profiles
    - **👥 Team Trends**: Analyze team-wide improvements
    - **⚽ Soccer Analytics**: Sport-specific insights and injury risk
    - **📊 Load-Velocity Profiles**: Compare F-V profiles across sessions
    - **⚖️ Asymmetry Analysis**: Left/Right imbalance tracking
    """)
    st.stop()

df = st.session_state.data
athletes = sorted(df['Client'].unique())
dates = sorted(df['SessionDate'].unique())

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔬 Clustering & Similarity",
    "📈 Player Analysis",
    "👥 Team Trends",
    "⚽ Soccer Analytics",
    "📊 Load-Velocity Profiles",
    "⚖️ Asymmetry Analysis"
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
# TAB 2: PLAYER ANALYSIS (Merged Performance + Progression)
# ============================================================================

with tab2:
    st.header("📈 Player Analysis")
    
    selected_athlete = st.selectbox("Select Athlete", athletes, key='player_analysis')
    athlete_data = df[df['Client'] == selected_athlete]
    athlete_dates = sorted(athlete_data['SessionDate'].unique())
    
    # Summary Metrics Row
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
    
    # Create sub-sections
    perf_tab1, perf_tab2, perf_tab3 = st.tabs([
        "📊 Performance Over Time",
        "🎯 Baseline Progression",
        "📉 Load-Velocity Profile"
    ])
    
    # ========================================================================
    # SUB-TAB 1: Performance Over Time
    # ========================================================================
    with perf_tab1:
        # Metrics over time
        progress = athlete_data.groupby('SessionDate').agg({
            'TopSpeed': 'max',
            '0-5m Time [s]': 'min',
            'MaxAcceleration': 'max'
        }).reset_index().sort_values('SessionDate')
        
        metric_choice = st.selectbox("Select Metric", 
                                      ["Max Speed", "0-5m Time", "Max Acceleration"],
                                      key='metric_perf')
        
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
        if len(progress) >= 2:
            x_num = np.arange(len(progress))
            valid_data = progress[col].dropna()
            if len(valid_data) >= 2:
                z = np.polyfit(x_num[:len(valid_data)], valid_data.values, 1)
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
        
        # Quick stats for this metric
        if len(progress) >= 2:
            first_val = progress[col].iloc[0]
            last_val = progress[col].iloc[-1]
            if invert:
                change_pct = ((first_val - last_val) / first_val) * 100
            else:
                change_pct = ((last_val - first_val) / first_val) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("First Session", f"{first_val:.2f}")
            with col2:
                st.metric("Latest Session", f"{last_val:.2f}")
            with col3:
                st.metric("Total Change", f"{change_pct:+.1f}%")
    
    # ========================================================================
    # SUB-TAB 2: Baseline Progression
    # ========================================================================
    with perf_tab2:
        if len(athlete_dates) < 2:
            st.warning(f"⚠️ {selected_athlete} needs at least 2 sessions for progression analysis")
        else:
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
            
            if len(progression) > 0:
                prog_df = pd.DataFrame(progression)
                
                # Summary metrics
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
                
                # Progression Chart
                all_dates = [baseline] + list(prog_df['Date'])
                all_speeds = [baseline_stats['max_speed']] + list(prog_df['Max Speed'])
                colors = ['gray'] + ['green' if x else 'red' for x in prog_df['Improved']]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=all_dates, y=all_speeds,
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
                    name='Max Speed'
                ))
                
                # Baseline reference line
                fig.add_hline(y=baseline_stats['max_speed'], line_dash="dash", 
                             line_color="gray", annotation_text="Baseline")
                
                fig.update_layout(
                    xaxis_title='Date', yaxis_title='Max Speed (m/s)',
                    height=400, template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("⭐ Gray = Baseline | 🟢 Green = Improved | 🔴 Red = Declined")
                
                # Detailed Table
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
                
                # Download button
                csv = prog_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Progression Data", csv, 
                                  f"{selected_athlete.replace(' ', '_')}_progression.csv", "text/csv")
    
    # ========================================================================
    # SUB-TAB 3: Load-Velocity Profile
    # ========================================================================
    with perf_tab3:
        lv_data = athlete_data.groupby('Concentric Load [kg]')['TopSpeed'].max().reset_index()
        lv_data = lv_data.sort_values('Concentric Load [kg]')
        
        if len(lv_data) < 2:
            st.warning("Need at least 2 different loads for Load-Velocity profile")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=lv_data['Concentric Load [kg]'], y=lv_data['TopSpeed'],
                mode='lines+markers', 
                line=dict(color='#10b981', width=3), 
                marker=dict(size=12)
            ))
            
            # Add regression line
            X = lv_data['Concentric Load [kg]'].values.reshape(-1, 1)
            y = lv_data['TopSpeed'].values
            model = LinearRegression()
            model.fit(X, y)
            
            x_range = np.linspace(lv_data['Concentric Load [kg]'].min(), 
                                  lv_data['Concentric Load [kg]'].max(), 50)
            y_pred = model.predict(x_range.reshape(-1, 1))
            
            fig.add_trace(go.Scatter(
                x=x_range, y=y_pred,
                mode='lines', name='Linear Fit',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig.update_layout(
                xaxis_title='Load (kg)', yaxis_title='Max Velocity (m/s)',
                height=400, template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Quick metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Loads Tested", len(lv_data))
            with col2:
                st.metric("Speed at Min Load", f"{lv_data['TopSpeed'].iloc[0]:.2f} m/s")
            with col3:
                retention = (lv_data['TopSpeed'].iloc[-1] / lv_data['TopSpeed'].iloc[0]) * 100
                st.metric("Velocity Retention", f"{retention:.1f}%")
            
            # Profile interpretation
            if retention > 65:
                st.success("**Force-Dominant Profile** 💪 - Good velocity maintenance under load. Focus: Max velocity work")
            elif retention > 50:
                st.info("**Balanced Profile** ⚖️ - Continue mixed training approach")
            else:
                st.warning("**Velocity-Dominant Profile** 🏃 - Speed drops significantly with load. Focus: Strength work")

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
# TAB 4: SOCCER ANALYTICS
# ============================================================================

with tab4:
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

# ============================================================================
# TAB 5: LOAD-VELOCITY PROFILES
# ============================================================================

with tab5:
    st.header("📊 Multi-Session Load-Velocity Profile Comparison")
    
    st.markdown("""
    Compare Force-Velocity profiles across multiple training sessions to track neuromuscular adaptations.
    **Key Metrics:**
    - **V0**: Theoretical maximum velocity (speed capability)
    - **F0**: Theoretical maximum force (strength capability)  
    - **Pmax**: Maximum power output (F0 × V0 / 4)
    """)
    
    # Athlete selection
    selected_athlete_fv = st.selectbox("Select Athlete", athletes, key='fv_athlete')
    
    # Get sessions with F-V data
    fv_sessions = get_sessions_with_fv_data(df, selected_athlete_fv, min_loads=2)
    
    if len(fv_sessions) == 0:
        st.warning("⚠️ No sessions with multiple load conditions found for this athlete. Need at least 2 different loads per session for F-V profiling.")
    else:
        # Display available sessions
        sessions_df = pd.DataFrame(fv_sessions)
        st.markdown(f"**{len(fv_sessions)} sessions available** with load variation")
        
        # Session selector
        session_dates = [s['date'] for s in fv_sessions]
        selected_sessions = st.multiselect(
            "Select Sessions to Compare (2-4 recommended)",
            session_dates,
            default=session_dates[:min(3, len(session_dates))],
            format_func=lambda x: f"{x} ({[s['load_range'] for s in fv_sessions if s['date']==x][0]})",
            key='fv_sessions'
        )
        
        if len(selected_sessions) < 1:
            st.info("👆 Select at least one session to view its F-V profile")
        else:
            # Calculate profiles for selected sessions
            profiles = []
            for date in selected_sessions:
                profile = calculate_fv_profile(df, selected_athlete_fv, date)
                if profile:
                    profiles.append(profile)
            
            if len(profiles) == 0:
                st.error("Could not calculate F-V profiles for selected sessions. Need more load variation.")
            else:
                # Visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Load-Velocity Overlay")
                    
                    fig = go.Figure()
                    colors = px.colors.qualitative.Set2
                    
                    for i, profile in enumerate(profiles):
                        color = colors[i % len(colors)]
                        date_str = str(profile['date'])
                        
                        # Data points
                        fig.add_trace(go.Scatter(
                            x=profile['loads'],
                            y=profile['velocities'],
                            mode='markers',
                            name=f"{date_str} (data)",
                            marker=dict(size=12, color=color),
                            showlegend=True
                        ))
                        
                        # Regression line (extend to F0)
                        x_line = np.linspace(0, max(profile['F0_load'], profile['loads'].max()), 50)
                        y_line = profile['V0'] + profile['slope'] * x_line
                        y_line = np.clip(y_line, 0, None)  # Can't have negative velocity
                        
                        fig.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode='lines',
                            name=f"{date_str} (fit)",
                            line=dict(color=color, dash='dash', width=2),
                            showlegend=True
                        ))
                        
                        # Mark V0 and F0
                        fig.add_trace(go.Scatter(
                            x=[0], y=[profile['V0']],
                            mode='markers',
                            marker=dict(size=10, symbol='diamond', color=color),
                            name=f"V0: {profile['V0']:.2f}",
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        xaxis_title='Load (kg)',
                        yaxis_title='Velocity (m/s)',
                        height=500,
                        template='plotly_white',
                        hovermode='closest',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Profile Metrics")
                    
                    for i, profile in enumerate(profiles):
                        with st.expander(f"📅 {profile['date']}", expanded=i==0):
                            st.metric("V0 (Max Velocity)", f"{profile['V0']:.2f} m/s")
                            st.metric("F0 (Max Force)", f"{profile['F0_newtons']:.0f} N")
                            st.metric("Pmax (Power)", f"{profile['Pmax']:.0f} W")
                            st.metric("R² (Fit Quality)", f"{profile['r_squared']:.3f}")
                            st.caption(f"Based on {profile['n_points']} load conditions")
                
                # Delta comparison (if multiple sessions)
                if len(profiles) >= 2:
                    st.markdown("---")
                    st.subheader("📈 Changes Over Time")
                    
                    first = profiles[0]
                    last = profiles[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        v0_change = ((last['V0'] - first['V0']) / first['V0']) * 100
                        st.metric("V0 Change", f"{v0_change:+.1f}%", 
                                 delta=f"{last['V0'] - first['V0']:+.2f} m/s")
                    
                    with col2:
                        f0_change = ((last['F0_newtons'] - first['F0_newtons']) / first['F0_newtons']) * 100
                        st.metric("F0 Change", f"{f0_change:+.1f}%",
                                 delta=f"{last['F0_newtons'] - first['F0_newtons']:+.0f} N")
                    
                    with col3:
                        pmax_change = ((last['Pmax'] - first['Pmax']) / first['Pmax']) * 100
                        st.metric("Pmax Change", f"{pmax_change:+.1f}%",
                                 delta=f"{last['Pmax'] - first['Pmax']:+.0f} W")
                    
                    with col4:
                        # Profile shift interpretation
                        if v0_change > 2 and f0_change > 2:
                            shift = "↗️ Overall Improvement"
                        elif v0_change > 2:
                            shift = "🏃 Velocity Shift"
                        elif f0_change > 2:
                            shift = "💪 Force Shift"
                        elif v0_change < -2 or f0_change < -2:
                            shift = "⚠️ Declining"
                        else:
                            shift = "➡️ Stable"
                        st.metric("Profile Shift", shift)
                    
                    # Interpretation
                    st.markdown("---")
                    if v0_change > 3:
                        st.success("✅ **Velocity capability improved** - Sprint-specific adaptations occurring")
                    if f0_change > 3:
                        st.success("✅ **Force capability improved** - Strength gains transferring to sprinting")
                    if pmax_change > 5:
                        st.success("✅ **Power output increased** - Both force and velocity contributing")
                    if v0_change < -3 or f0_change < -3:
                        st.warning("⚠️ **Performance declining** - Consider recovery, deload, or program adjustment")

# ============================================================================
# TAB 6: ASYMMETRY ANALYSIS
# ============================================================================

with tab6:
    st.header("⚖️ Left/Right Asymmetry Analysis")
    
    st.markdown("""
    Analyze bilateral differences in sprint performance. Asymmetries >10% are associated with increased injury risk.
    
    **Asymmetry Index Formula:** `(Left - Right) / Average × 100`
    - Positive = Left side dominant
    - Negative = Right side dominant
    """)
    
    # Check if Side data exists
    if 'Side' not in df.columns or df['Side'].isna().all():
        st.warning("⚠️ No Left/Right side data available in this dataset. Ensure your 1080 export includes the 'Side' column.")
        st.info("To enable asymmetry analysis, run sprint tests with the 1080 configured to record Left and Right sides separately.")
    else:
        # Get athletes with L/R data
        athletes_with_lr = []
        for athlete in athletes:
            athlete_data = df[(df['Client'] == athlete) & (df['Side'].isin(['Left', 'Right']))]
            if athlete_data['Side'].nunique() >= 2:
                athletes_with_lr.append(athlete)
        
        if len(athletes_with_lr) == 0:
            st.warning("⚠️ No athletes have both Left and Right data. Need sprints recorded on both sides.")
        else:
            st.success(f"✅ {len(athletes_with_lr)} athletes have Left/Right data available")
            
            selected_athlete_asym = st.selectbox("Select Athlete", athletes_with_lr, key='asym_athlete')
            
            # Overall asymmetry analysis
            st.markdown("---")
            st.subheader("📊 Overall Asymmetry Summary")
            
            asym_detail = analyze_asymmetry_detailed(df, selected_athlete_asym)
            
            if asym_detail is None:
                st.error("Could not calculate asymmetry for this athlete.")
            else:
                st.caption(f"Based on {asym_detail['left_count']} Left and {asym_detail['right_count']} Right sprints")
                
                # Metrics cards
                cols = st.columns(len(asym_detail['metrics']))
                
                for i, (metric_key, metric_data) in enumerate(asym_detail['metrics'].items()):
                    with cols[i]:
                        st.markdown(f"**{metric_data['label']}**")
                        
                        # Color-coded asymmetry display
                        asym_color = metric_data['color']
                        st.markdown(f"""
                        <div style="background-color: {asym_color}20; border-left: 4px solid {asym_color}; 
                                    padding: 10px; border-radius: 5px; margin: 5px 0;">
                            <span style="font-size: 24px; font-weight: bold; color: {asym_color};">
                                {metric_data['abs_asymmetry']:.1f}%
                            </span>
                            <br>
                            <span style="font-size: 12px;">Risk: {metric_data['risk']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"Left: {metric_data['left']:.2f} | Right: {metric_data['right']:.2f}")
                        st.caption(f"Dominant: {metric_data['dominant']}")
                
                # Visual comparison
                st.markdown("---")
                st.subheader("🔄 Side-by-Side Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Speed comparison bar chart
                    if 'TopSpeed' in asym_detail['metrics']:
                        speed_data = asym_detail['metrics']['TopSpeed']
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Left', 'Right'],
                                y=[speed_data['left'], speed_data['right']],
                                marker_color=['#3b82f6', '#ef4444'],
                                text=[f"{speed_data['left']:.2f}", f"{speed_data['right']:.2f}"],
                                textposition='outside'
                            )
                        ])
                        
                        fig.update_layout(
                            title='Max Speed by Side',
                            yaxis_title='Speed (m/s)',
                            height=350,
                            template='plotly_white',
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Acceleration comparison
                    if 'MaxAcceleration' in asym_detail['metrics']:
                        accel_data = asym_detail['metrics']['MaxAcceleration']
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Left', 'Right'],
                                y=[accel_data['left'], accel_data['right']],
                                marker_color=['#3b82f6', '#ef4444'],
                                text=[f"{accel_data['left']:.2f}", f"{accel_data['right']:.2f}"],
                                textposition='outside'
                            )
                        ])
                        
                        fig.update_layout(
                            title='Max Acceleration by Side',
                            yaxis_title='Acceleration (m/s²)',
                            height=350,
                            template='plotly_white',
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Load-matched asymmetry
                st.markdown("---")
                st.subheader("🎯 Load-Matched Asymmetry")
                
                load_matched = get_load_matched_asymmetry(df, selected_athlete_asym)
                
                if load_matched is None or len(load_matched) == 0:
                    st.info("ℹ️ No matching loads found between Left and Right sides. Need identical load conditions on both sides.")
                else:
                    st.markdown("Comparing Left vs Right at the same resistance levels:")
                    
                    # Add risk classification
                    def classify_risk(val):
                        if val < 5:
                            return '🟢 Low'
                        elif val < 10:
                            return '🟡 Moderate'
                        else:
                            return '🔴 High'
                    
                    load_matched['Risk'] = load_matched['abs_asymmetry'].apply(classify_risk)
                    
                    display_cols = ['load', 'left_speed', 'right_speed', 'asymmetry', 'Risk']
                    display_df = load_matched[display_cols].copy()
                    display_df.columns = ['Load (kg)', 'Left Speed', 'Right Speed', 'Asymmetry %', 'Risk']
                    display_df['Left Speed'] = display_df['Left Speed'].round(2)
                    display_df['Right Speed'] = display_df['Right Speed'].round(2)
                    display_df['Asymmetry %'] = display_df['Asymmetry %'].round(1)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Asymmetry by load chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=load_matched['load'],
                        y=load_matched['asymmetry'],
                        marker_color=[
                            '#10b981' if abs(a) < 5 else '#f59e0b' if abs(a) < 10 else '#ef4444'
                            for a in load_matched['asymmetry']
                        ],
                        text=[f"{a:.1f}%" for a in load_matched['asymmetry']],
                        textposition='outside'
                    ))
                    
                    # Add threshold lines
                    fig.add_hline(y=5, line_dash="dash", line_color="#f59e0b", 
                                 annotation_text="5% threshold")
                    fig.add_hline(y=-5, line_dash="dash", line_color="#f59e0b")
                    fig.add_hline(y=10, line_dash="dash", line_color="#ef4444",
                                 annotation_text="10% threshold")
                    fig.add_hline(y=-10, line_dash="dash", line_color="#ef4444")
                    
                    fig.update_layout(
                        title='Asymmetry by Load',
                        xaxis_title='Load (kg)',
                        yaxis_title='Asymmetry % (+ = Left dominant)',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Asymmetry trend over time
                st.markdown("---")
                st.subheader("📈 Asymmetry Trend Over Time")
                
                trend = get_asymmetry_trend(df, selected_athlete_asym)
                
                if trend is None or len(trend) < 2:
                    st.info("ℹ️ Need at least 2 sessions with Left/Right data to show trend.")
                else:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=trend['date'],
                        y=trend['speed_asymmetry'],
                        mode='lines+markers',
                        name='Speed Asymmetry',
                        line=dict(color='#3b82f6', width=3),
                        marker=dict(size=10)
                    ))
                    
                    if 'accel_asymmetry' in trend.columns:
                        fig.add_trace(go.Scatter(
                            x=trend['date'],
                            y=trend['accel_asymmetry'],
                            mode='lines+markers',
                            name='Acceleration Asymmetry',
                            line=dict(color='#10b981', width=3),
                            marker=dict(size=10)
                        ))
                    
                    # Threshold zones
                    fig.add_hrect(y0=0, y1=5, fillcolor="#10b981", opacity=0.1, 
                                 annotation_text="Low Risk", annotation_position="right")
                    fig.add_hrect(y0=5, y1=10, fillcolor="#f59e0b", opacity=0.1,
                                 annotation_text="Moderate", annotation_position="right")
                    fig.add_hrect(y0=10, y1=trend['speed_asymmetry'].max() + 5, 
                                 fillcolor="#ef4444", opacity=0.1,
                                 annotation_text="High Risk", annotation_position="right")
                    
                    fig.update_layout(
                        xaxis_title='Date',
                        yaxis_title='Asymmetry %',
                        height=400,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend interpretation
                    if len(trend) >= 3:
                        first_val = trend['speed_asymmetry'].iloc[0]
                        last_val = trend['speed_asymmetry'].iloc[-1]
                        change = last_val - first_val
                        
                        if change < -2:
                            st.success(f"✅ **Asymmetry improving** - Decreased by {abs(change):.1f}% since first session")
                        elif change > 2:
                            st.warning(f"⚠️ **Asymmetry worsening** - Increased by {change:.1f}% since first session")
                        else:
                            st.info(f"➡️ **Asymmetry stable** - Change of {change:+.1f}%")
                
                # Recommendations
                st.markdown("---")
                st.subheader("📋 Recommendations")
                
                max_asym = max([m['abs_asymmetry'] for m in asym_detail['metrics'].values()])
                
                if max_asym > 10:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("**⚠️ High Asymmetry Detected - Action Required:**")
                    st.markdown("- Prioritize unilateral strength training (single-leg exercises)")
                    st.markdown("- Include single-leg plyometrics in warm-up")
                    st.markdown("- Consider physiotherapy evaluation")
                    st.markdown("- Monitor weekly until below 10%")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif max_asym > 5:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**⚡ Moderate Asymmetry - Monitor:**")
                    st.markdown("- Include unilateral exercises 2x per week")
                    st.markdown("- Address any movement pattern issues")
                    st.markdown("- Re-test in 2-3 weeks")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("**✅ Low Asymmetry - Good Balance:**")
                    st.markdown("- Continue current training program")
                    st.markdown("- Maintain bilateral strength work")
                    st.markdown("- Re-test monthly to monitor")
                    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("1080 Sprint Analysis | Statistical testing • Machine learning • Soccer-specific analytics")
