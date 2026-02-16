"""
Force Plate Analysis Streamlit Application
==========================================
A comprehensive, robust interface for analyzing force plate jump data
Handles both wide and long format CSV files
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import io
from scipy import stats

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Force Plate Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# DATA LOADING AND FORMAT HANDLING FUNCTIONS
# ==========================================

def detect_csv_format(df):
    """
    Detect whether the CSV is in wide or long format
    
    Returns:
        str: 'wide' or 'long' format type
    """
    # Check for long format indicators
    long_format_columns = {'TARGET_VARIABLE', 'RESULT', 'ATHLETE_NAME', 'TEST_DATE'}
    long_format_match = len(long_format_columns & set(df.columns))
    
    # Quick return if strong long format match
    if long_format_match >= 3:
        return 'long'
    
    # Check for wide format indicators
    # Wide format typically has many columns with units in brackets
    columns_with_brackets = sum(1 for col in df.columns if '[' in col and ']' in col)
    
    # Check for common wide format column names (from various force plate systems)
    wide_format_id_columns = {'Name', 'TestId', 'Date', 'Time', 'Segment', 'Position', 
                              'Type', 'Excluded', 'Tags', 'System Weight', 'ExternalId'}
    wide_format_match = len(wide_format_id_columns & set(df.columns))
    
    # Check for force plate metric column names (without brackets)
    force_plate_metrics = {'Jump Height', 'Braking RFD', 'Stiffness', 'Peak Braking Force',
                          'Peak Propulsive Force', 'Flight Time', 'Takeoff Velocity',
                          'Peak Velocity', 'RSI', 'mRSI', 'Countermovement Depth',
                          'Jump Momentum', 'Braking Phase', 'Propulsive Phase'}
    metric_match = len(force_plate_metrics & set(df.columns))
    
    # Wide format decision - more flexible detection
    if columns_with_brackets > 5 or 'Name' in df.columns or wide_format_match >= 3 or metric_match >= 3:
        return 'wide'
    
    # Fallback: check numeric columns ratio
    # Wide format usually has many numeric columns with fewer rows
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 10 and len(df) < 1000:
        return 'wide'
    
    return 'long'

def standardize_column_names(df, format_type):
    """
    Standardize column names based on format type
    """
    if format_type == 'wide':
        # Rename common columns for wide format
        # Supports multiple naming conventions from different force plate systems
        column_mapping = {
            # Standard naming
            'Name': 'ATHLETE_NAME',
            'ExternalId': 'ATHLETE_ID',
            'Test Type': 'TEST_TYPE',
            'Date': 'TEST_DATE',
            'Time': 'TEST_TIME',
            'BW [KG]': 'BODY_WEIGHT_KG',
            'Additional Load [kg]': 'ADDITIONAL_LOAD_KG',
            # Alternative naming (e.g., from Hawkin Dynamics, ForceDecks, etc.)
            'TestId': 'TEST_ID',
            'Segment': 'SEGMENT',
            'Position': 'POSITION',
            'Type': 'TEST_TYPE',
            'Excluded': 'EXCLUDED',
            'Tags': 'TAGS',
            'System Weight': 'BODY_WEIGHT_KG',
        }
        # Only rename columns that actually exist to avoid unnecessary operations
        existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        if existing_mappings:
            return df.rename(columns=existing_mappings)
    
    return df

def extract_metric_name_and_unit(column_name):
    """
    Extract clean metric name and unit from column name
    Example: 'Jump Height (Imp-Mom) [cm]' -> ('Jump Height (Imp-Mom)', 'cm')
    """
    # Find the last occurrence of [ and ]
    start_bracket = column_name.rfind('[')
    end_bracket = column_name.rfind(']')
    
    # Check if valid brackets exist and are in correct order
    if start_bracket > 0 and start_bracket < end_bracket:
        unit = column_name[start_bracket + 1:end_bracket]
        metric_name = column_name[:start_bracket].rstrip()  # Use rstrip() instead of strip()
        return metric_name, unit
    
    # No valid unit found
    return column_name, ''

def convert_wide_to_long(df_wide):
    """
    Convert wide format dataframe to long format
    """
    # Standardize column names first
    df_wide = standardize_column_names(df_wide, 'wide')
    
    # Common ID columns in wide format (supports multiple naming conventions)
    potential_id_cols = {'ATHLETE_NAME', 'ATHLETE_ID', 'TEST_TYPE', 'TEST_DATE', 
                        'TEST_TIME', 'BODY_WEIGHT_KG', 'ADDITIONAL_LOAD_KG',
                        'Reps', 'Tags', 'TAGS',
                        # Additional ID columns from various force plate systems
                        'TEST_ID', 'SEGMENT', 'POSITION', 'EXCLUDED',
                        'TestId', 'Date', 'Time', 'Name', 'Segment', 'Position', 
                        'Type', 'Excluded', 'System Weight'}
    
    # Keywords that indicate metric columns (expanded for comprehensive coverage)
    metric_keywords = {
        # Movement phases & mechanics
        'force', 'power', 'velocity', 'time', 'height', 'impulse', 
        'rfd', 'depth', 'duration', 'peak', 'mean', 'avg', 'average',
        'concentric', 'eccentric', 'jump', 'rsi', 'takeoff', 'landing', 
        'contraction', 'flight', 'braking', 'countermovement', 'propulsive',
        # Additional metrics from force plate systems
        'stiffness', 'displacement', 'relative', 'momentum', 'unweighting',
        'phase', 'net', 'positive', 'ratio', 'index', 'stabilization',
        'performance', 'left', 'right', 'l|r', 'asymmetry', 'asym',
        # mRSI and modified metrics
        'mrsi', 'modified'
    }
    
    # Separate ID and metric columns
    id_columns = list(potential_id_cols & set(df_wide.columns))
    metric_columns = []
    
    for col in df_wide.columns:
        if col in id_columns:
            continue
            
        # Check if it's likely a metric column
        # Method 1: Has brackets (units) or special characters
        if '[' in col or '/' in col or '%' in col or '|' in col:
            metric_columns.append(col)
        else:
            col_lower = col.lower()
            # Method 2: Contains metric keywords
            if any(keyword in col_lower for keyword in metric_keywords):
                metric_columns.append(col)
    
    # If no ID columns identified, use index
    if not id_columns:
        df_wide = df_wide.assign(ROW_ID=df_wide.index)
        id_columns = ['ROW_ID']
    
    # Ensure we have ATHLETE_NAME
    if 'ATHLETE_NAME' not in id_columns:
        df_wide = df_wide.assign(ATHLETE_NAME=[f"Athlete_{i}" for i in range(len(df_wide))])
        id_columns.append('ATHLETE_NAME')
    
    # Melt the dataframe to long format
    df_long = pd.melt(
        df_wide,
        id_vars=id_columns,
        value_vars=metric_columns,
        var_name='METRIC_FULL',
        value_name='RESULT'
    )
    
    # Extract metric name and unit more efficiently
    df_long[['TARGET_VARIABLE', 'UNITS']] = pd.DataFrame(
        [extract_metric_name_and_unit(x) for x in df_long['METRIC_FULL']], 
        index=df_long.index
    )
    
    # Drop the temporary column
    df_long = df_long.drop('METRIC_FULL', axis=1)
    
    # Add missing columns with vectorized operations
    df_long = df_long.assign(
        TEST_ID=range(len(df_long)),
        LIMB='Trial',
        TRIAL_LIMB='Bilateral',
        RESULT_TYPE='Measured',
        DESCRIPTION=df_long['TARGET_VARIABLE']
    )
    
    # Ensure TEST_DATE is in datetime format
    if 'TEST_DATE' in df_long.columns:
        df_long['TEST_DATE'] = pd.to_datetime(df_long['TEST_DATE'], errors='coerce')
    
    # Convert RESULT to numeric and drop NaN in one operation
    df_long['RESULT'] = pd.to_numeric(df_long['RESULT'], errors='coerce')
    df_long = df_long.dropna(subset=['RESULT'])
    
    return df_long

def clean_metric_names(df):
    """
    Standardize metric names for consistency
    """
    if 'TARGET_VARIABLE' not in df.columns:
        return df
    
    # Strip spaces first (do once)
    df['TARGET_VARIABLE'] = df['TARGET_VARIABLE'].str.strip()
    
    # Combine all replacements into a single operation
    replacements = {
        'Peak Power / BM': 'Peak Power Per BM',
        'Flight Time:Contraction Time': 'Flight Time To Contraction Time Ratio',
        'RSI-modified': 'RSI Modified',
        '% (Asym)': 'Asymmetry',
        'Imp-Mom': 'Impulse Momentum',
        'RFD': 'Rate of Force Development'
    }
    
    # Use regex=False and single replace call with dictionary
    df['TARGET_VARIABLE'] = df['TARGET_VARIABLE'].replace(replacements, regex=False)
    
    return df

@st.cache_data
def load_and_clean_data(uploaded_file):
    """
    Robust data loader that handles both wide and long format CSVs
    """
    try:
        # Read the CSV
        df_raw = pd.read_csv(uploaded_file, low_memory=False)
        
        # Detect format
        format_type = detect_csv_format(df_raw)
        st.sidebar.info(f"📊 Detected format: **{format_type.upper()}** format")
        
        # Convert to long format if needed (work in place when possible)
        if format_type == 'wide':
            df = convert_wide_to_long(df_raw)
            st.sidebar.success("✅ Converted to long format")
        else:
            df = df_raw  # No copy needed here
        
        # Clean metric names in place
        df = clean_metric_names(df)
        
        # Check required columns once
        required_columns = {'ATHLETE_NAME', 'TARGET_VARIABLE', 'RESULT', 'TEST_DATE', 'LIMB'}
        existing_columns = set(df.columns)
        missing_columns = required_columns - existing_columns
        
        if missing_columns:
            st.sidebar.warning(f"⚠️ Missing columns: {list(missing_columns)}")
            # Add missing LIMB column if needed
            if 'LIMB' not in existing_columns:
                df['LIMB'] = 'Trial'
        
        # Filter for Trial limb only (bilateral measurements) - avoid copy if already filtered
        if 'LIMB' in df.columns and not df['LIMB'].eq('Trial').all():
            df = df[df['LIMB'] == 'Trial']
        
        # Convert dates and add temporal columns in one operation
        if 'TEST_DATE' in df.columns:
            df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE'], errors='coerce')
            # Add all temporal columns at once
            df = df.assign(
                YEAR=df['TEST_DATE'].dt.year,
                QUARTER=df['TEST_DATE'].dt.quarter,
                MONTH=df['TEST_DATE'].dt.month
            )
        else:
            # Create dummy date and temporal columns at once
            current_time = pd.Timestamp.now()
            df = df.assign(
                TEST_DATE=current_time,
                YEAR=current_time.year,
                QUARTER=1,
                MONTH=1
            )
        
        # Convert RESULT to numeric (already done in convert_wide_to_long for wide format)
        if format_type != 'wide':
            df['RESULT'] = pd.to_numeric(df['RESULT'], errors='coerce')
        
        # Replace infinite values with NaN in one operation
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Show conversion summary
        if format_type == 'wide':
            st.sidebar.markdown("**📋 Conversion Summary:**")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Original Columns", len(df_raw.columns))
                st.metric("Metrics Extracted", df['TARGET_VARIABLE'].nunique())
            with col2:
                st.metric("Athletes", df['ATHLETE_NAME'].nunique())
                st.metric("Data Points", len(df))
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please check your CSV format")
        return None
# ==========================================
# ANALYSIS HELPER FUNCTIONS
# ==========================================

def get_jump_types(df):
    """Extract unique jump types from the data"""
    # Check for explicit jump type columns first
    if 'TEST_TYPE' in df.columns:
        return sorted(df['TEST_TYPE'].dropna().unique())
    elif 'JUMP_TYPE' in df.columns:
        return sorted(df['JUMP_TYPE'].dropna().unique())
    
    # Try to infer from TARGET_VARIABLE
    jump_indicators = {'CMJ', 'SJ', 'DJ', 'JUMP'}
    jump_types = set()
    
    # Use vectorized string operations for efficiency
    if 'TARGET_VARIABLE' in df.columns:
        target_vars = df['TARGET_VARIABLE'].dropna().unique()
        for var in target_vars:
            var_upper = var.upper()
            for indicator in jump_indicators:
                if indicator in var_upper:
                    jump_types.add(indicator)
                    break  # No need to check other indicators for this variable
    
    return sorted(jump_types) if jump_types else ['All Jumps']

def filter_data_by_jump(df, jump_type):
    """Filter data for specific jump type"""
    if jump_type == 'All Jumps':
        return df
    
    # Try different column names - return first match found
    if 'TEST_TYPE' in df.columns:
        return df[df['TEST_TYPE'] == jump_type]
    
    if 'JUMP_TYPE' in df.columns:
        return df[df['JUMP_TYPE'] == jump_type]
    
    # Filter by TARGET_VARIABLE containing jump type
    # Use case=False for case-insensitive matching
    if 'TARGET_VARIABLE' in df.columns:
        return df[df['TARGET_VARIABLE'].str.contains(jump_type, na=False, case=False)]
    
    # No filtering possible - return original
    return df

@st.cache_data(ttl=3600)
def analyze_metrics(df):
    """Analyze and categorize available metrics - returns ALL metrics with stats"""
    all_metrics = df['TARGET_VARIABLE'].unique()
    
    # Group by metric once
    grouped = df.groupby('TARGET_VARIABLE', observed=True)
    
    # Use aggregation instead of looping
    agg_results = grouped.agg({
        'RESULT': ['count', 
                   lambda x: pd.to_numeric(x, errors='coerce').dropna().mean(),
                   lambda x: pd.to_numeric(x, errors='coerce').dropna().std(),
                   lambda x: len(pd.to_numeric(x, errors='coerce').dropna())],
        'ATHLETE_NAME': 'nunique'
    })
    
    # Flatten multi-level columns
    agg_results.columns = ['total_count', 'mean', 'std', 'valid_count', 'num_athletes']
    agg_results = agg_results.reset_index()
    agg_results.columns = ['metric', 'total_count', 'mean', 'std', 'num_datapoints', 'num_athletes']
    
    # Calculate derived metrics
    agg_results['completeness_pct'] = (agg_results['num_datapoints'] / agg_results['total_count']) * 100
    agg_results['cv'] = agg_results['std'] / agg_results['mean'].replace(0, np.nan)
    agg_results['cv'] = agg_results['cv'].fillna(0)
    
    # Drop total_count
    metrics_df = agg_results.drop('total_count', axis=1).sort_values('completeness_pct', ascending=False)
    
    # Categorize metrics using vectorized string operations
    metric_upper = metrics_df['metric'].str.upper()
    
    # Create category dictionary - expanded for comprehensive force plate metrics
    categories = {
        'power': metrics_df.loc[metric_upper.str.contains('POWER', na=False), 'metric'].tolist(),
        'force': metrics_df.loc[metric_upper.str.contains('FORCE', na=False) & 
                                ~metric_upper.str.contains('RFD', na=False), 'metric'].tolist(),
        'velocity': metrics_df.loc[metric_upper.str.contains('VELOCITY|TAKEOFF', regex=True, na=False), 'metric'].tolist(),
        'jump': metrics_df.loc[metric_upper.str.contains('JUMP|HEIGHT|RSI|MRSI|MOMENTUM', regex=True, na=False), 'metric'].tolist(),
        'timing': metrics_df.loc[metric_upper.str.contains('TIME|DURATION|PHASE', regex=True, na=False), 'metric'].tolist(),
        'rfd': metrics_df.loc[metric_upper.str.contains('RFD', na=False), 'metric'].tolist(),
        # New categories for expanded metrics
        'impulse': metrics_df.loc[metric_upper.str.contains('IMPULSE', na=False), 'metric'].tolist(),
        'stiffness': metrics_df.loc[metric_upper.str.contains('STIFFNESS|DISPLACEMENT|DEPTH', regex=True, na=False), 'metric'].tolist(),
        'landing': metrics_df.loc[metric_upper.str.contains('LANDING|STABILIZATION', regex=True, na=False), 'metric'].tolist(),
        'asymmetry': metrics_df.loc[metric_upper.str.contains('L\\|R|LEFT|RIGHT|ASYM|INDEX', regex=True, na=False), 'metric'].tolist(),
        'braking': metrics_df.loc[metric_upper.str.contains('BRAKING', na=False) & 
                                  ~metric_upper.str.contains('RFD', na=False), 'metric'].tolist(),
        'propulsive': metrics_df.loc[metric_upper.str.contains('PROPULSIVE', na=False), 'metric'].tolist(),
    }
    
    # Get ALL available metrics (filter out low quality ones)
    all_available_metrics = metrics_df[
        (metrics_df['completeness_pct'] >= 50) &  # Lowered from 70 to include more
        (metrics_df['num_athletes'] >= 2)  # At least 2 athletes
    ]['metric'].tolist()
    
    return metrics_df, all_available_metrics, categories

@st.cache_data(ttl=3600)
def compute_kmo_bartlett(correlation_matrix, n_samples):
    """
    KMO (Kaiser-Meyer-Olkin) and Bartlett's test - validates whether PCA is appropriate.
    KMO > 0.6 = acceptable, > 0.8 = meritorious. Bartlett p < 0.05 = correlations exist.
    Reference: Kaiser 1970, used in Merrigan et al. 2022 for CMJ force-time PCA.
    """
    try:
        n_vars = correlation_matrix.shape[0]
        inv_corr = np.linalg.pinv(correlation_matrix)
        diag_inv = np.diag(1.0 / np.sqrt(np.diag(inv_corr) + 1e-10))
        partial_corr = -1.0 * (diag_inv @ inv_corr @ diag_inv)
        np.fill_diagonal(partial_corr, 1.0)
        corr_sq_sum = np.sum(correlation_matrix ** 2) - n_vars
        partial_sq_sum = np.sum(partial_corr ** 2) - n_vars
        kmo_overall = corr_sq_sum / (corr_sq_sum + partial_sq_sum + 1e-10)
        det = max(np.linalg.det(correlation_matrix), 1e-300)
        chi_sq = -((n_samples - 1) - (2 * n_vars + 5) / 6) * np.log(det)
        dof = n_vars * (n_vars - 1) / 2
        p_value = 1 - stats.chi2.cdf(chi_sq, dof)
        return {'kmo': round(kmo_overall, 3), 'bartlett_chi2': round(chi_sq, 1), 'bartlett_p': p_value, 'bartlett_dof': int(dof)}
    except Exception:
        return {'kmo': None, 'bartlett_chi2': None, 'bartlett_p': None, 'bartlett_dof': None}

def parallel_analysis_threshold(n_samples, n_features, n_iter=100, percentile=95):
    """
    Horn's Parallel Analysis: compares real eigenvalues against random data eigenvalues.
    More robust than Kaiser criterion per Goretzko et al. 2021.
    Returns threshold eigenvalues from random data.
    """
    random_eigs = np.zeros((n_iter, n_features))
    for i in range(n_iter):
        rand = np.random.normal(size=(n_samples, n_features))
        rand_corr = np.corrcoef(rand, rowvar=False)
        random_eigs[i] = sorted(np.linalg.eigvalsh(rand_corr), reverse=True)
    return np.percentile(random_eigs, percentile, axis=0)

def create_correlation_heatmap(athlete_metrics_df, max_metrics=30):
    """
    Correlation matrix heatmap - MUST be checked before PCA.
    PCA exploits correlations; if metrics are uncorrelated, PCA adds no value.
    Per Merrigan et al. 2022, CMJ metrics are often r > 0.85 correlated,
    making PCA ideal for dimension reduction.
    """
    if len(athlete_metrics_df.columns) > max_metrics:
        variances = athlete_metrics_df.var().sort_values(ascending=False)
        subset = athlete_metrics_df[variances.head(max_metrics).index.tolist()]
    else:
        subset = athlete_metrics_df
    corr = subset.corr()
    display_names = [m[:25] for m in corr.columns]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=display_names, y=display_names,
        colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="r"),
        hovertemplate='%{x}<br>%{y}<br>r = %{z:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title="Metric Correlation Matrix (verifies PCA suitability)",
        height=600 + len(display_names) * 5,
        xaxis={'tickangle': 45}, yaxis={'autorange': 'reversed'}
    )
    return fig, corr

def create_biplot(pca_model, pca_data, feature_names, athlete_names, cluster_labels=None, top_n=10):
    """
    PCA Biplot: scores (athletes as dots) + loadings (metrics as arrows) on same plot.
    Arrow direction = which PC the metric loads onto.
    Arrow length = strength of contribution.
    Angle between arrows = correlation (small = correlated, 90° = uncorrelated).
    This is the gold-standard PCA visualization in sports science papers
    (Merrigan et al. 2022, Keogh et al. 2024, Bird et al. 2022).
    """
    fig = go.Figure()
    loadings = pca_model.components_[:2].T
    score_scale = max(np.abs(pca_data[:, :2]).max(), 1e-10)
    loading_scale = max(np.abs(loadings).max(), 1e-10)
    scores_s = pca_data[:, :2] / score_scale
    loadings_s = loadings / loading_scale
    colors = px.colors.qualitative.Set1
    if cluster_labels is not None:
        for cl in np.unique(cluster_labels):
            mask = cluster_labels == cl
            fig.add_trace(go.Scatter(
                x=scores_s[mask, 0], y=scores_s[mask, 1],
                mode='markers+text',
                text=[athlete_names[i][:12] for i in range(len(athlete_names)) if mask[i]],
                textposition="top center", name=f'Cluster {cl+1}',
                marker=dict(size=14, color=colors[cl % len(colors)], line=dict(width=1, color='black'))
            ))
    else:
        fig.add_trace(go.Scatter(
            x=scores_s[:, 0], y=scores_s[:, 1], mode='markers+text',
            text=[n[:12] for n in athlete_names], textposition="top center",
            name='Athletes', marker=dict(size=14, color='steelblue')
        ))
    mag = np.sqrt(loadings_s[:, 0]**2 + loadings_s[:, 1]**2)
    top_idx = np.argsort(mag)[-top_n:]
    for idx in top_idx:
        fig.add_annotation(ax=0, ay=0, axref="x", ayref="y",
            x=loadings_s[idx, 0], y=loadings_s[idx, 1],
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='red', opacity=0.7)
        fig.add_annotation(x=loadings_s[idx, 0]*1.12, y=loadings_s[idx, 1]*1.12,
            text=feature_names[idx][:20], showarrow=False, font=dict(size=9, color='red'))
    v1, v2 = pca_model.explained_variance_ratio_[0], pca_model.explained_variance_ratio_[1]
    fig.update_layout(title="PCA Biplot (Athletes + Metric Loadings)",
        xaxis_title=f"PC1 ({v1:.1%})", yaxis_title=f"PC2 ({v2:.1%})",
        height=600, showlegend=True, plot_bgcolor='rgba(248,248,248,1)')
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
        line=dict(dash='dot', color='gray', width=1), showlegend=False, hoverinfo='skip'))
    return fig

@st.cache_data(ttl=3600)
def perform_pca_analysis(df, selected_metrics, selected_athletes):
    """Perform PCA analysis with diagnostics (KMO, Bartlett, Kaiser, parallel analysis)"""
    # Filter data
    df_filtered = df[df['ATHLETE_NAME'].isin(selected_athletes)]
    
    # Create athlete × metric matrix
    athlete_metrics = df_filtered[df_filtered['TARGET_VARIABLE'].isin(selected_metrics)].pivot_table(
        index='ATHLETE_NAME',
        columns='TARGET_VARIABLE',
        values='RESULT',
        aggfunc='mean'
    )
    
    # Drop all-NaN columns, fill remaining with median (more robust than mean for skewed data)
    athlete_metrics = athlete_metrics.dropna(axis=1, how='all')
    athlete_metrics_filled = athlete_metrics.fillna(athlete_metrics.median())
    
    # Drop zero-variance columns (can't standardize, no information)
    zero_var = athlete_metrics_filled.columns[athlete_metrics_filled.std() == 0]
    if len(zero_var) > 0:
        athlete_metrics_filled = athlete_metrics_filled.drop(columns=zero_var)
    
    # Standardize (z-score) - required when metrics have different units
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(athlete_metrics_filled)
    
    # Compute correlation matrix for diagnostics
    corr_matrix = np.corrcoef(scaled_data, rowvar=False)
    diagnostics = compute_kmo_bartlett(corr_matrix, len(athlete_metrics_filled))
    
    # Full PCA to get all eigenvalues for scree plot
    pca_full = PCA()
    pca_full.fit(scaled_data)
    eigenvalues = pca_full.explained_variance_
    
    # Kaiser criterion: keep components with eigenvalue > 1
    n_kaiser = max(1, int(np.sum(eigenvalues > 1.0)))
    
    # Parallel analysis (more robust than Kaiser per Goretzko et al. 2021)
    n_samples, n_features = scaled_data.shape
    try:
        pa_thresh = parallel_analysis_threshold(n_samples, n_features)
        n_parallel = max(1, int(np.sum(eigenvalues > pa_thresh)))
    except Exception:
        n_parallel = n_kaiser
    
    # Use more conservative estimate, minimum 2 for visualization
    n_components = max(2, min(n_kaiser, n_parallel, n_features, n_samples))
    
    # Refit with selected n_components
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    
    # FIXED: Feature importance weighted by explained variance ratio
    # Original bug: np.abs(pca.components_[:3]).mean(axis=0) equally weighted all 3 PCs
    # Fix: weight by variance explained so PC1 (e.g., 40%) matters more than PC3 (e.g., 8%)
    weights = pca.explained_variance_ratio_
    feature_importance = (np.abs(pca.components_) * weights[:, np.newaxis]).sum(axis=0)
    
    return {
        'athlete_metrics': athlete_metrics_filled,
        'pca_data': pca_data,
        'pca_model': pca,
        'pca_full': pca_full,
        'feature_importance': feature_importance,
        'feature_names': list(athlete_metrics_filled.columns),
        'scaled_data': scaled_data,
        'scaler': scaler,
        'correlation_matrix': corr_matrix,
        'diagnostics': diagnostics,
        'n_kaiser': n_kaiser,
        'n_parallel': n_parallel,
        'eigenvalues': eigenvalues
    }

def create_visualizations(pca_results, num_clusters=3):
    """Create visualization dashboard"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            'PCA Space - Athlete Similarity Map',
            'Top 10 Most Important Features',
            'K-means Clustering Analysis',
            'Variance Explained by Components'
        ],
        specs=[[{"type": "scatter"}],
               [{"type": "bar"}],
               [{"type": "scatter"}],
               [{"type": "bar"}]],
        vertical_spacing=0.1,
        row_heights=[0.35, 0.25, 0.35, 0.2]
    )
    
    pca_data = pca_results['pca_data']
    athlete_names = list(pca_results['athlete_metrics'].index)
    colors = px.colors.qualitative.Set1
    
    # 1. PCA Space - Individual athlete colors
    for i, athlete in enumerate(athlete_names):
        fig.add_trace(
            go.Scatter(
                x=[pca_data[i, 0]],
                y=[pca_data[i, 1]],
                mode='markers+text',
                name=athlete,
                text=[athlete[:15]],  # Truncated for display
                textposition="top center",
                marker=dict(color=colors[i % len(colors)], size=15),
                showlegend=True,
                legendgroup='athletes',
                hovertemplate='<b>%{fullData.name}</b><br>' +  # Full name in hover
                              'PC1: %{x:.2f}<br>' +
                              'PC2: %{y:.2f}<br>' +
                              '<extra></extra>'  # Removes secondary box
            ),
            row=1, col=1
        )
    
    # 2. Feature Importance
    feature_importance = pca_results['feature_importance']
    top_features_idx = np.argsort(feature_importance)[-10:]
    feature_names = pca_results['feature_names']
    
    fig.add_trace(
        go.Bar(
            x=feature_importance[top_features_idx],
            y=[feature_names[i][:40] for i in top_features_idx],
            orientation='h',
            marker_color='lightcoral',
            showlegend=False,
            text=np.round(feature_importance[top_features_idx], 3),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                          'Importance: %{x:.4f}<br>' +
                          '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 3. K-means Clustering - FIXED: cluster on PCA-reduced space
    # Original bug: inconsistently clustered on pca_data[:,:3] here but scaled_data in enhanced PCA
    n_dims = min(3, pca_data.shape[1])
    cluster_input = pca_data[:, :n_dims]
    
    actual_clusters = min(num_clusters, len(athlete_names))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(cluster_input)
    
    # Silhouette score: validates cluster quality (-1 to 1, higher = better separated)
    sil_score = None
    if actual_clusters >= 2 and len(athlete_names) > actual_clusters:
        sil_score = silhouette_score(cluster_input, clusters)

    cluster_colors = px.colors.qualitative.Set1
    
    # Store cluster info for custom legend
    cluster_info = []
    
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        cluster_athletes = np.array(athlete_names)[mask]
        cluster_color = cluster_colors[cluster % len(cluster_colors)]
        
        cluster_info.append({
            'cluster': cluster,
            'color': cluster_color,
            'count': len(cluster_athletes)
        })
        
        # Get full names and truncated display names
        display_names = [name[:12] for name in cluster_athletes]
        
        fig.add_trace(
            go.Scatter(
                x=pca_data[mask, 0],
                y=pca_data[mask, 1],
                mode='markers+text',
                name=f'Cluster {cluster + 1}',
                text=display_names,  # Truncated for display
                textposition="top center",
                marker=dict(
                    color=cluster_color, 
                    size=15,
                    line=dict(width=2, color='black')
                ),
                showlegend=False,
                legendgroup='clusters',
                customdata=cluster_athletes,  # Store full names
                hovertemplate='<b>%{customdata}</b><br>' +  # Full name from customdata
                              'Cluster: ' + str(cluster + 1) + '<br>' +
                              'PC1: %{x:.2f}<br>' +
                              'PC2: %{y:.2f}<br>' +
                              '<extra></extra>'
            ),
            row=3, col=1
        )
    
    # 4. Explained Variance - use full PCA for complete scree plot
    pca_for_scree = pca_results.get('pca_full', pca_results['pca_model'])
    explained_var = pca_for_scree.explained_variance_ratio_[:10]
    cumulative_var = np.cumsum(explained_var)
    
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(explained_var) + 1)),
            y=explained_var,
            marker_color='lightblue',
            name='Individual',
            showlegend=False,
            hovertemplate='PC%{x}<br>' +
                          'Variance: %{y:.1%}<br>' +
                          '<extra></extra>'
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_var) + 1)),
            y=cumulative_var,
            mode='lines+markers',
            marker_color='red',
            name='Cumulative',
            showlegend=False,
            hovertemplate='PC%{x}<br>' +
                          'Cumulative: %{y:.1%}<br>' +
                          '<extra></extra>'
        ),
        row=4, col=1
    )
    
    # Update layout with proper margins
    fig.update_layout(
        height=2500,
        title_text="<b>Force Plate Analysis - PCA & Clustering Dashboard</b>",
        title_font_size=20,
        showlegend=True,
        legend=dict(
            title="Athletes",
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01
        ),
        margin=dict(r=200),
        font=dict(size=12),
        hovermode='closest'  # Better hover behavior
    )
    
    # Add custom cluster legend as annotations near Row 3
    legend_y_start = 0.45
    legend_y_spacing = 0.03
    
    # Add title for cluster legend
    fig.add_annotation(
        text="<b>Clusters</b>",
        xref="paper", yref="paper",
        x=1.01, y=legend_y_start,
        showarrow=False,
        font=dict(size=14, color="black"),
        align="left",
        xanchor="left"
    )
    
    # Add cluster entries
    for i, info in enumerate(cluster_info):
        y_pos = legend_y_start - (i + 1) * legend_y_spacing
        
        # Add colored circle
        fig.add_annotation(
            text="●",
            xref="paper", yref="paper",
            x=1.01, y=y_pos,
            showarrow=False,
            font=dict(size=20, color=info['color']),
            xanchor="left"
        )
        
        # Add cluster label
        fig.add_annotation(
            text=f"Cluster {info['cluster'] + 1} ({info['count']})",
            xref="paper", yref="paper",
            x=1.035, y=y_pos,
            showarrow=False,
            font=dict(size=12, color="black"),
            align="left",
            xanchor="left"
        )
    
    # Update axes labels
    fig.update_xaxes(title_text="<b>Principal Component 1</b>", row=1, col=1, title_font_size=14)
    fig.update_yaxes(title_text="<b>Principal Component 2</b>", row=1, col=1, title_font_size=14)
    
    fig.update_xaxes(title_text="<b>Importance Score</b>", row=2, col=1, title_font_size=14)
    fig.update_yaxes(title_text="<b>Features</b>", row=2, col=1, title_font_size=14)
    
    fig.update_xaxes(title_text="<b>Principal Component 1</b>", row=3, col=1, title_font_size=14)
    fig.update_yaxes(title_text="<b>Principal Component 2</b>", row=3, col=1, title_font_size=14)
    
    fig.update_xaxes(title_text="<b>Component Number</b>", row=4, col=1, title_font_size=14)
    fig.update_yaxes(title_text="<b>Variance Explained</b>", row=4, col=1, title_font_size=14)
    
    return fig, clusters, sil_score

def get_metrics_by_pattern(patterns, available_metrics):
    """Get metrics that match any of the given patterns"""
    matching = []
    for metric in available_metrics:
        metric_upper = metric.upper()
        if any(pattern.upper() in metric_upper for pattern in patterns):
            matching.append(metric)
    return matching

def search_and_filter_metrics(all_metrics, search_term="", quick_filter="All"):
    """Search and filter metrics with quick presets"""
    # Quick filter presets
    quick_filters = {
        "All": [],
        "Power & Force": ["POWER", "FORCE"],
        "Speed & Velocity": ["VELOCITY", "SPEED", "TAKEOFF"],
        "Jump Performance": ["JUMP", "HEIGHT", "RSI", "FLIGHT"],
        "Timing": ["TIME", "DURATION", "PHASE"],
        "RFD": ["RFD", "RPD"],
        "Asymmetry": ["ASYM", "L|R", "LEFT", "RIGHT"]
    }

    filtered_metrics = all_metrics

    # Apply quick filter
    if quick_filter != "All" and quick_filter in quick_filters:
        patterns = quick_filters[quick_filter]
        filtered_metrics = get_metrics_by_pattern(patterns, filtered_metrics)

    # Apply search term
    if search_term:
        filtered_metrics = [
            m for m in filtered_metrics
            if search_term.upper() in m.upper()
        ]

    return filtered_metrics

def perform_enhanced_pca_analysis(df, enhanced_metrics, selected_athletes, n_clusters):
    """Perform robust enhanced PCA analysis with proper error handling"""
    try:
        # Prepare data for PCA
        athlete_data_dict = {}
        metric_coverage = {}
        
        # First pass: Check data coverage for each metric
        for metric in enhanced_metrics:
            athletes_with_data = []
            for athlete in selected_athletes:
                athlete_metric_data = df[
                    (df['ATHLETE_NAME'] == athlete) & 
                    (df['TARGET_VARIABLE'] == metric)
                ]
                if len(athlete_metric_data) > 0:
                    values = pd.to_numeric(athlete_metric_data['RESULT'], errors='coerce').dropna()
                    if len(values) > 0:
                        athletes_with_data.append(athlete)
                        if athlete not in athlete_data_dict:
                            athlete_data_dict[athlete] = {}
                        athlete_data_dict[athlete][metric] = values.mean()
            
            metric_coverage[metric] = len(athletes_with_data)
        
        # Keep metrics that have data for at least 50% of athletes
        min_coverage = max(3, len(selected_athletes) * 0.5)
        valid_metrics = [m for m, coverage in metric_coverage.items() if coverage >= min_coverage]
        
        if len(valid_metrics) < 3:
            return None, f"Insufficient data: Only {len(valid_metrics)} metrics have adequate coverage"
        
        # Build athlete matrix
        athlete_matrix = []
        athlete_names = []
        
        for athlete in selected_athletes:
            if athlete in athlete_data_dict:
                athlete_vector = []
                valid_for_athlete = True
                
                for metric in valid_metrics:
                    if metric in athlete_data_dict[athlete]:
                        athlete_vector.append(athlete_data_dict[athlete][metric])
                    else:
                        # Impute missing values with median from other athletes
                        metric_values = [
                            athlete_data_dict[other_athlete][metric] 
                            for other_athlete in athlete_data_dict 
                            if metric in athlete_data_dict[other_athlete]
                        ]
                        if len(metric_values) > 0:
                            athlete_vector.append(np.median(metric_values))
                        else:
                            valid_for_athlete = False
                            break
                
                if valid_for_athlete and len(athlete_vector) == len(valid_metrics):
                    athlete_matrix.append(athlete_vector)
                    athlete_names.append(athlete)
        
        if len(athlete_matrix) < 3:
            return None, f"Insufficient athletes: Only {len(athlete_matrix)} athletes have adequate data"
        
        # Convert to numpy and handle any remaining issues
        athlete_array = np.array(athlete_matrix)
        
        # Check for and handle any NaN or infinite values
        if np.any(np.isnan(athlete_array)) or np.any(np.isinf(athlete_array)):
            # Replace NaN/inf with column medians
            for col in range(athlete_array.shape[1]):
                col_data = athlete_array[:, col]
                valid_values = col_data[np.isfinite(col_data)]
                if len(valid_values) > 0:
                    median_val = np.median(valid_values)
                    athlete_array[~np.isfinite(athlete_array[:, col]), col] = median_val
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(athlete_array)
        
        # Apply PCA
        n_components = min(3, len(valid_metrics), len(athlete_matrix))
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(scaled_data)
        
        # FIXED: Cluster on PCA-reduced space, not raw scaled_data
        # Original bug: clustering in full feature space but visualizing in PCA space = misleading
        actual_clusters = min(n_clusters, len(athlete_matrix))
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_data)
        
        # Silhouette score for cluster quality validation
        sil_score = None
        if actual_clusters >= 2 and len(athlete_names) > actual_clusters:
            sil_score = silhouette_score(pca_data, cluster_labels)
        
        return {
            'athlete_array': athlete_array,
            'scaled_data': scaled_data,
            'pca_data': pca_data,
            'pca_model': pca,
            'cluster_labels': cluster_labels,
            'athlete_names': athlete_names,
            'valid_metrics': valid_metrics,
            'n_components': n_components,
            'n_clusters': actual_clusters,
            'scaler': scaler,
            'silhouette_score': sil_score
        }, None
        
    except Exception as e:
        return None, f"Error in PCA analysis: {str(e)}"

def analyze_trends(df, selected_athletes, selected_metrics):
    """Analyze performance trends over time with statistical significance"""
    df_trends = df[
        (df['ATHLETE_NAME'].isin(selected_athletes)) &
        (df['TARGET_VARIABLE'].isin(selected_metrics))
    ].copy()

    # Create proper date column for better plotting
    df_trends['DATE'] = pd.to_datetime(df_trends['TEST_DATE'])

    # Group by actual date (not period) for better plotting
    df_trends = df_trends.groupby(['ATHLETE_NAME', 'DATE', 'TARGET_VARIABLE'])['RESULT'].mean().reset_index()
    df_trends = df_trends.sort_values('DATE')

    # Calculate improvement rates with statistical significance
    improvement_data = []
    for athlete in selected_athletes:
        for metric in selected_metrics:
            athlete_metric = df_trends[
                (df_trends['ATHLETE_NAME'] == athlete) &
                (df_trends['TARGET_VARIABLE'] == metric)
            ].sort_values('DATE')

            if len(athlete_metric) >= 2:
                first_val = athlete_metric['RESULT'].iloc[0]
                last_val = athlete_metric['RESULT'].iloc[-1]

                if first_val != 0:
                    improvement = ((last_val - first_val) / abs(first_val)) * 100
                else:
                    improvement = 0

                # Statistical significance test (linear regression slope)
                significance = 'N/A'
                effect_size_d = 'N/A'
                if len(athlete_metric) >= 3:
                    x = np.arange(len(athlete_metric))
                    y = athlete_metric['RESULT'].values
                    try:
                        _, _, _, p_value, _ = stats.linregress(x, y)
                        if p_value < 0.001:
                            significance = '***'
                        elif p_value < 0.01:
                            significance = '**'
                        elif p_value < 0.05:
                            significance = '*'
                        else:
                            significance = 'ns'
                        
                        # Cohen's d: split into halves, compare means
                        mid = len(y) // 2
                        first_half, second_half = y[:mid], y[mid:]
                        if len(first_half) > 0 and len(second_half) > 0:
                            pooled_sd = np.sqrt((np.var(first_half, ddof=1) + np.var(second_half, ddof=1)) / 2)
                            if pooled_sd > 0:
                                effect_size_d = f"{(np.mean(second_half) - np.mean(first_half)) / pooled_sd:.2f}"
                    except:
                        significance = 'N/A'

                improvement_data.append({
                    'Athlete': athlete,
                    'Metric': metric,
                    'Improvement (%)': improvement,
                    'First Value': first_val,
                    'Last Value': last_val,
                    'Data Points': len(athlete_metric),
                    'Significance': significance,
                    'Effect Size (d)': effect_size_d
                })

    improvement_df = pd.DataFrame(improvement_data)

    # Create trend figure with 2 columns and up to 10 metrics
    num_metrics = min(10, len(selected_metrics))
    num_rows = (num_metrics + 1) // 2

    fig_trends = make_subplots(
        rows=num_rows,
        cols=2,
        subplot_titles=[f"{metric[:40]}..." if len(metric) > 40 else metric for metric in selected_metrics[:num_metrics]],
        vertical_spacing=0.08,
        horizontal_spacing=0.12
    )

    colors = px.colors.qualitative.Set1

    for i, metric in enumerate(selected_metrics[:num_metrics]):
        row = (i // 2) + 1
        col = (i % 2) + 1

        for j, athlete in enumerate(selected_athletes):
            athlete_data = df_trends[
                (df_trends['ATHLETE_NAME'] == athlete) &
                (df_trends['TARGET_VARIABLE'] == metric)
            ].sort_values('DATE')

            if len(athlete_data) > 0:
                fig_trends.add_trace(
                    go.Scatter(
                        x=athlete_data['DATE'],
                        y=athlete_data['RESULT'],
                        mode='lines+markers',
                        name=athlete[:15],
                        line=dict(color=colors[j % len(colors)], width=2),
                        marker=dict(size=7),
                        legendgroup=athlete[:15],
                        showlegend=(i == 0),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x|%b %Y}<br>' +
                                    'Value: %{y:.2f}<br>' +
                                    '<extra></extra>'
                    ),
                    row=row, col=col
                )

    # Update all x-axes and y-axes
    for i in range(1, num_rows + 1):
        for j in range(1, 3):
            fig_trends.update_xaxes(
                row=i, col=j,
                tickformat='%b %Y',
                tickmode='auto',
                nticks=8,
                tickangle=45,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title_font=dict(size=12)
            )
            fig_trends.update_yaxes(
                row=i, col=j,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title_font=dict(size=12)
            )

    chart_height = 400 * num_rows

    fig_trends.update_layout(
        height=chart_height,
        title_text="<b>Performance Trends Over Time</b>",
        title_font_size=20,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11)
    )

    return fig_trends, improvement_df

@st.cache_data(ttl=3600)
def create_data_quality_dashboard(df, selected_athletes, all_metrics):
    """Create comprehensive data quality visualization"""
    # Build completeness matrix
    completeness_matrix = []

    for athlete in selected_athletes:
        athlete_data = df[df['ATHLETE_NAME'] == athlete]
        row = {'Athlete': athlete}

        for metric in all_metrics:
            metric_data = athlete_data[athlete_data['TARGET_VARIABLE'] == metric]
            if len(metric_data) > 0:
                valid_count = metric_data['RESULT'].notna().sum()
                row[metric] = valid_count
            else:
                row[metric] = 0

        completeness_matrix.append(row)

    completeness_df = pd.DataFrame(completeness_matrix)
    completeness_df = completeness_df.set_index('Athlete')

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=completeness_df.values,
        x=[m[:30] for m in completeness_df.columns],
        y=completeness_df.index,
        colorscale='Viridis',
        colorbar=dict(title="Data Points"),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Data Points: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title="Data Completeness Matrix (Athletes × Metrics)",
        xaxis_title="Metrics",
        yaxis_title="Athletes",
        height=400 + len(selected_athletes) * 15,
        xaxis={'tickangle': 45}
    )

    # Calculate quality scores
    quality_scores = []
    for athlete in selected_athletes:
        athlete_data = df[df['ATHLETE_NAME'] == athlete]

        total_possible = len(all_metrics)
        metrics_with_data = athlete_data['TARGET_VARIABLE'].nunique()
        avg_data_points = len(athlete_data) / max(metrics_with_data, 1)

        quality_score = (metrics_with_data / total_possible) * 100

        quality_scores.append({
            'Athlete': athlete,
            'Metrics Coverage (%)': round(quality_score, 1),
            'Metrics with Data': metrics_with_data,
            'Avg Data Points/Metric': round(avg_data_points, 1),
            'Total Data Points': len(athlete_data)
        })

    quality_df = pd.DataFrame(quality_scores).sort_values('Metrics Coverage (%)', ascending=False)

    return fig, quality_df, completeness_df

def generate_cluster_insights(pca_results, cluster_labels, athlete_names):
    """Automatically generate cluster interpretations"""
    cluster_insights = {}

    # Get the original data (scaled)
    scaled_data = pca_results.get('athlete_metrics', None)
    if scaled_data is None:
        return cluster_insights

    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_athletes = [athlete_names[i] for i, mask in enumerate(cluster_mask) if mask]

        # Get cluster centroid
        cluster_data = scaled_data.iloc[cluster_mask]
        cluster_mean = cluster_data.mean()
        overall_mean = scaled_data.mean()
        overall_std = scaled_data.std()

        # Calculate z-scores
        z_scores = (cluster_mean - overall_mean) / (overall_std + 1e-8)

        # Identify top strengths (high positive z-scores)
        top_strengths = z_scores.nlargest(3)
        top_weaknesses = z_scores.nsmallest(3)

        # Generate descriptive name
        strength_names = []
        for metric in top_strengths.index[:2]:
            if 'POWER' in metric.upper():
                strength_names.append('Power')
            elif 'FORCE' in metric.upper():
                strength_names.append('Force')
            elif 'VELOCITY' in metric.upper() or 'SPEED' in metric.upper():
                strength_names.append('Speed')
            elif 'JUMP' in metric.upper() or 'HEIGHT' in metric.upper():
                strength_names.append('Jump')
            elif 'RFD' in metric.upper():
                strength_names.append('RFD')
            else:
                strength_names.append(metric.split()[0])

        # Remove duplicates
        strength_names = list(dict.fromkeys(strength_names))

        if strength_names:
            cluster_name = f"High {'/'.join(strength_names[:2])}"
        else:
            cluster_name = f"Cluster {cluster_id + 1}"

        cluster_insights[cluster_id] = {
            'name': cluster_name,
            'athletes': cluster_athletes,
            'size': len(cluster_athletes),
            'top_strengths': [(metric, float(z_scores[metric])) for metric in top_strengths.index],
            'top_weaknesses': [(metric, float(z_scores[metric])) for metric in top_weaknesses.index],
            'avg_performance': float(z_scores.mean())
        }

    return cluster_insights

def create_athlete_profile_card(df, athlete_name, selected_athletes):
    """Generate individual athlete profile with strengths/weaknesses"""
    athlete_data = df[df['ATHLETE_NAME'] == athlete_name]

    # Calculate athlete metrics
    athlete_metrics = athlete_data.groupby('TARGET_VARIABLE')['RESULT'].mean()

    # Calculate group averages for comparison
    group_data = df[df['ATHLETE_NAME'].isin(selected_athletes)]
    group_metrics = group_data.groupby('TARGET_VARIABLE')['RESULT'].agg(['mean', 'std'])

    # Calculate z-scores
    profile = []
    for metric in athlete_metrics.index:
        if metric in group_metrics.index:
            athlete_val = athlete_metrics[metric]
            group_mean = group_metrics.loc[metric, 'mean']
            group_std = group_metrics.loc[metric, 'std']

            if group_std > 0:
                z_score = (athlete_val - group_mean) / group_std
            else:
                z_score = 0

            profile.append({
                'Metric': metric,
                'Value': round(athlete_val, 2),
                'Group Avg': round(group_mean, 2),
                'Z-Score': round(z_score, 2),
                'Percentile': round(stats.norm.cdf(z_score) * 100, 1)
            })

    profile_df = pd.DataFrame(profile).sort_values('Z-Score', ascending=False)

    # Identify strengths and weaknesses
    strengths = profile_df.head(5)
    weaknesses = profile_df.tail(5)

    return profile_df, strengths, weaknesses

def compare_athletes(df, athlete1, athlete2, selected_metrics):
    """Head-to-head athlete comparison"""
    comparison_data = []

    for metric in selected_metrics:
        a1_data = df[(df['ATHLETE_NAME'] == athlete1) & (df['TARGET_VARIABLE'] == metric)]
        a2_data = df[(df['ATHLETE_NAME'] == athlete2) & (df['TARGET_VARIABLE'] == metric)]

        if len(a1_data) > 0 and len(a2_data) > 0:
            a1_val = a1_data['RESULT'].mean()
            a2_val = a2_data['RESULT'].mean()

            diff = a1_val - a2_val
            diff_pct = (diff / abs(a2_val)) * 100 if a2_val != 0 else 0

            comparison_data.append({
                'Metric': metric,
                athlete1: round(a1_val, 2),
                athlete2: round(a2_val, 2),
                'Difference': round(diff, 2),
                'Difference (%)': round(diff_pct, 1),
                'Winner': athlete1 if diff > 0 else athlete2
            })

    comparison_df = pd.DataFrame(comparison_data)

    # Create radar chart
    if len(comparison_df) > 0:
        # Normalize values for radar chart
        metrics = comparison_df['Metric'].head(10).tolist()
        a1_vals = comparison_df[athlete1].head(10).tolist()
        a2_vals = comparison_df[athlete2].head(10).tolist()

        # FIXED: Per-metric normalization (original used global min-max which distorts radar)
        a1_norm = []
        a2_norm = []
        for v1, v2 in zip(a1_vals, a2_vals):
            local_min = min(v1, v2)
            local_max = max(v1, v2)
            if local_max > local_min:
                a1_norm.append(20 + ((v1 - local_min) / (local_max - local_min)) * 80)
                a2_norm.append(20 + ((v2 - local_min) / (local_max - local_min)) * 80)
            else:
                a1_norm.append(60)
                a2_norm.append(60)

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=a1_norm + [a1_norm[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=athlete1,
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatterpolar(
            r=a2_norm + [a2_norm[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=athlete2,
            line=dict(color='red')
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title=f"{athlete1} vs {athlete2} - Performance Comparison"
        )

        return comparison_df, fig

    return comparison_df, None

def export_pca_results(pca_results, cluster_labels, athlete_names):
    """Export PCA results to downloadable format"""
    export_data = []

    pca_data = pca_results['pca_data']

    for i, athlete in enumerate(athlete_names):
        row = {
            'Athlete': athlete,
            'Cluster': int(cluster_labels[i]) + 1,
            'PC1': round(pca_data[i, 0], 4),
            'PC2': round(pca_data[i, 1], 4) if pca_data.shape[1] > 1 else 0,
            'PC3': round(pca_data[i, 2], 4) if pca_data.shape[1] > 2 else 0
        }
        export_data.append(row)

    export_df = pd.DataFrame(export_data)
    return export_df

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    st.title("🦘 Force Plate - Analysis System")
    st.markdown("### Analyze force plate data to track athlete performance and identify patterns")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose your force plate CSV file",
            type=['csv'],
            help="Upload the CSV file containing force plate measurements"
        )
        
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            
            # Load data
            with st.spinner("Loading and cleaning data..."):
                df = load_and_clean_data(uploaded_file)
            
            if df is not None:
                st.markdown("---")
                st.header("📊 Data Overview")
                st.metric("Total Records", f"{len(df):,}")
                st.metric("Total Athletes", df['ATHLETE_NAME'].nunique())
                st.metric("Date Range", f"{df['TEST_DATE'].min().strftime('%Y-%m-%d')} \
                to {df['TEST_DATE'].max().strftime('%Y-%m-%d')}")

                st.markdown("""
                        <style>
                        [data-testid="stMetricValue"] {
                            font-size: 1.2rem;
                        }
                        [data-testid="stMetricLabel"] {
                            font-size: 0.85rem;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                
                # Jump type selection
                st.markdown("---")
                st.header("🎯 Analysis Settings")
                
                jump_types = get_jump_types(df)
                selected_jump = st.selectbox(
                    "Select Jump Type",
                    options=jump_types,
                    help="Choose which type of jump to analyze"
                )
                
                # Filter data by jump type
                df_jump = filter_data_by_jump(df, selected_jump)
                
                # Athlete selection
                available_athletes = sorted(df_jump['ATHLETE_NAME'].unique())
                
                # Show athletes with most data
                athlete_counts = df_jump.groupby('ATHLETE_NAME').size().sort_values(ascending=False)
                top_athletes = athlete_counts.head(20).index.tolist()
                
                selected_athletes = st.multiselect(
                    "Select Athletes to Analyze",
                    options=available_athletes,
                    default=top_athletes[:20],
                    help="Choose athletes for comparison (recommended: 5-15 athletes)"
                )

                # Date range filter
                st.markdown("---")
                st.subheader("📅 Date Range Filter")

                min_date = df_jump['TEST_DATE'].min().date()
                max_date = df_jump['TEST_DATE'].max().date()

                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help="Filter data by test date range"
                )

                # Apply date filter to df_jump
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df_jump = df_jump[
                        (df_jump['TEST_DATE'].dt.date >= start_date) & 
                        (df_jump['TEST_DATE'].dt.date <= end_date)
                    ]
                    st.success(f"📊 Filtered to {len(df_jump):,} records between {start_date} and {end_date}")
                elif len(date_range) == 1:
                    st.warning("Please select both start and end dates")

                
                if len(selected_athletes) < 2:
                    st.warning("⚠️ Please select at least 2 athletes for analysis")
    
    # Main content area
    if uploaded_file is not None and df is not None and len(selected_athletes) >= 2:

        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Quality",
            "📈 Metrics Analysis",
            "🎯 PCA & Clustering",
            "📊 Performance Trends",
            "👤 Athlete Profiles",
            "📋 Summary Report"
        ])
        
        with tab1:
            st.header("Data Quality Dashboard")
            st.markdown("Understand your data completeness and quality before analysis")

            with st.spinner("Analyzing data quality..."):
                metrics_df, all_available_metrics, metric_categories = analyze_metrics(df_jump)

                # Limit metrics for visualization to prevent overwhelming display
                top_metrics_for_viz = all_available_metrics[:30]

                quality_fig, quality_df, completeness_matrix = create_data_quality_dashboard(
                    df_jump, selected_athletes, top_metrics_for_viz
                )

            # Display quality scores
            st.subheader("📊 Athlete Data Quality Scores")
            st.markdown("Higher scores indicate more complete data coverage across metrics")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.dataframe(quality_df, use_container_width=True, hide_index=True)

            with col2:
                st.metric("Average Coverage", f"{quality_df['Metrics Coverage (%)'].mean():.1f}%")
                st.metric("Best Coverage", f"{quality_df['Metrics Coverage (%)'].max():.1f}%")
                st.metric("Lowest Coverage", f"{quality_df['Metrics Coverage (%)'].min():.1f}%")

            # Display completeness heatmap
            st.subheader("🗺️ Data Completeness Heatmap")
            st.markdown(f"Showing top {len(top_metrics_for_viz)} metrics")
            st.plotly_chart(quality_fig, use_container_width=True)

            # Missing data insights
            st.subheader("💡 Data Insights")

            total_possible = len(selected_athletes) * len(all_available_metrics)
            total_actual = sum(quality_df['Total Data Points'])
            overall_completeness = (total_actual / (len(selected_athletes) * len(all_available_metrics))) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Completeness", f"{overall_completeness:.1f}%")
            with col2:
                st.metric("Total Metrics Available", len(all_available_metrics))
            with col3:
                st.metric("Total Data Points", total_actual)

            # Recommendations
            low_coverage_athletes = quality_df[quality_df['Metrics Coverage (%)'] < 50]
            if len(low_coverage_athletes) > 0:
                st.warning(f"⚠️ {len(low_coverage_athletes)} athletes have less than 50% metric coverage. Consider collecting more data for: {', '.join(low_coverage_athletes['Athlete'].tolist()[:3])}")
            else:
                st.success("✅ All athletes have good data coverage!")

        with tab2:
            st.header("Metrics Analysis")

            with st.spinner("Analyzing metrics..."):
                metrics_df, all_available_metrics, metric_categories = analyze_metrics(df_jump)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Metric Categories")
                category_counts = pd.DataFrame({
                    'Category': ['Power', 'Force', 'Velocity', 'Jump/Height', 'Timing', 'RFD'],
                    'Count': [
                        len(metric_categories['power']),
                        len(metric_categories['force']),
                        len(metric_categories['velocity']),
                        len(metric_categories['jump']),
                        len(metric_categories['timing']),
                        len(metric_categories['rfd'])
                    ]
                })
                
                fig_cat = px.bar(category_counts, x='Category', y='Count', 
                                 color='Category', title='Metrics by Category')
                st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Top Quality Metrics")
                top_metrics = metrics_df.head(10)[['metric', 'completeness_pct', 'num_athletes']]
                top_metrics.columns = ['Metric', 'Completeness %', '# Athletes']
                top_metrics['Completeness %'] = top_metrics['Completeness %'].round(1)
                st.dataframe(top_metrics, hide_index=True)
            
            st.subheader("🔍 Metric Search & Filter")

            # Search and quick filter
            col1, col2 = st.columns([2, 1])

            with col1:
                search_term = st.text_input(
                    "🔎 Search metrics",
                    placeholder="Type to search... (e.g., 'power', 'force', 'velocity')",
                    key="metric_search"
                )

            with col2:
                quick_filter = st.selectbox(
                    "Quick Filter",
                    options=["All", "Power & Force", "Speed & Velocity", "Jump Performance", "Timing", "RFD", "Asymmetry"],
                    key="quick_filter"
                )

            # Apply search and filter
            filtered_metrics = search_and_filter_metrics(all_available_metrics, search_term, quick_filter)

            st.info(f"Showing {len(filtered_metrics)} of {len(all_available_metrics)} total metrics")

            # Display filtered metrics
            if filtered_metrics:
                with st.expander(f"📊 View {len(filtered_metrics)} filtered metrics"):
                    # Group into columns for better display
                    num_cols = 3
                    cols = st.columns(num_cols)

                    for idx, metric in enumerate(filtered_metrics[:30]):  # Show first 30
                        col_idx = idx % num_cols
                        with cols[col_idx]:
                            st.write(f"• {metric[:45]}")

                    if len(filtered_metrics) > 30:
                        st.write(f"... and {len(filtered_metrics) - 30} more")
            else:
                st.warning("No metrics match your search criteria")

            # Show breakdown by category
            st.markdown("---")
            st.subheader("📂 Metrics by Category")
            with st.expander("View all metrics organized by category"):
                for category, metrics in metric_categories.items():
                    if metrics:  # Only show non-empty categories
                        st.markdown(f"**{category.title()}**: {len(metrics)} metrics")
                        for m in metrics[:5]:
                            st.write(f"  • {m}")
                        if len(metrics) > 5:
                            st.write(f"  • ... and {len(metrics) - 5} more")
        
        with tab3:
            st.header("PCA Analysis & Clustering")

            # Get metrics first
            with st.spinner("Analyzing metrics..."):
                metrics_df, all_available_metrics, metric_categories = analyze_metrics(df_jump)

            # ==========================================
            # SECTION 1: OVERALL PCA WITH ALL METRICS
            # ==========================================

            st.subheader("🎯 Overall PCA Analysis - All Available Metrics")
            st.markdown(f"*Using all {len(all_available_metrics)} available metrics from {selected_jump}*")

            if len(all_available_metrics) >= 3:
                # Add cluster selection control
                col1, col2 = st.columns([1, 2])
                with col1:
                    num_clusters_overall = st.slider(
                        "Number of Clusters (Overall)",
                        min_value=2,
                        max_value=min(8, len(selected_athletes)),
                        value=3,
                        help="Choose how many groups to create",
                        key="overall_clusters"
                    )

                with st.spinner("Performing Overall PCA analysis with all metrics..."):
                    pca_results_overall = perform_pca_analysis(df_jump, all_available_metrics, selected_athletes)
                    fig_pca_overall, clusters_overall, sil_score_overall = create_visualizations(pca_results_overall, num_clusters_overall)

                # === NEW: PCA Diagnostics Panel ===
                st.subheader("🔬 PCA Diagnostics")
                diag = pca_results_overall.get('diagnostics', {})
                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                with dcol1:
                    kmo = diag.get('kmo')
                    kmo_label = "N/A" if kmo is None else f"{kmo:.3f}"
                    kmo_quality = "" if kmo is None else (" ✅" if kmo > 0.7 else " ⚠️" if kmo > 0.5 else " ❌")
                    st.metric("KMO Score", kmo_label + kmo_quality)
                with dcol2:
                    bp = diag.get('bartlett_p')
                    bp_label = "N/A" if bp is None else (f"p < 0.001 ✅" if bp < 0.001 else f"p = {bp:.4f}")
                    st.metric("Bartlett's Test", bp_label)
                with dcol3:
                    st.metric("Kaiser Components", pca_results_overall.get('n_kaiser', 'N/A'))
                with dcol4:
                    st.metric("Parallel Analysis", pca_results_overall.get('n_parallel', 'N/A'))
                
                if sil_score_overall is not None:
                    qual = "Poor" if sil_score_overall < 0.25 else "Fair" if sil_score_overall < 0.5 else "Good" if sil_score_overall < 0.75 else "Excellent"
                    st.info(f"**Silhouette Score:** {sil_score_overall:.3f} ({qual}) — measures how well-separated the clusters are")
                
                # NEW: Correlation heatmap
                with st.expander("🔥 Correlation Matrix (verify PCA suitability)"):
                    st.markdown("*PCA works by exploiting correlations. Blocks of red/blue = correlated metrics that PCA can compress.*")
                    corr_fig, _ = create_correlation_heatmap(pca_results_overall['athlete_metrics'])
                    st.plotly_chart(corr_fig, use_container_width=True)
                
                # NEW: Biplot
                with st.expander("🎯 PCA Biplot (Athletes + Metric Arrows)"):
                    st.markdown("*Arrows show which metrics drive each PC direction. Arrow angle = correlation between metrics.*")
                    biplot_fig = create_biplot(
                        pca_results_overall['pca_model'],
                        pca_results_overall['pca_data'],
                        pca_results_overall['feature_names'],
                        list(pca_results_overall['athlete_metrics'].index),
                        cluster_labels=clusters_overall
                    )
                    st.plotly_chart(biplot_fig, use_container_width=True)

                st.plotly_chart(fig_pca_overall, use_container_width=True)

                # Generate and display cluster insights
                st.subheader("🎯 Cluster Insights (Overall Analysis)")
                athlete_names_overall = list(pca_results_overall['athlete_metrics'].index)
                cluster_insights_overall = generate_cluster_insights(
                    pca_results_overall, clusters_overall, athlete_names_overall
                )

                # Display insights for each cluster
                for cluster_id, insights in cluster_insights_overall.items():
                    with st.expander(f"📊 {insights['name']} - {insights['size']} athletes"):
                        st.markdown(f"**Athletes:** {', '.join(insights['athletes'])}")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**💪 Key Strengths:**")
                            for metric, z_score in insights['top_strengths'][:3]:
                                st.write(f"• {metric[:35]}: +{z_score:.1f}σ")

                        with col2:
                            st.markdown("**📈 Development Areas:**")
                            for metric, z_score in insights['top_weaknesses'][:3]:
                                st.write(f"• {metric[:35]}: {z_score:.1f}σ")

                # Export functionality
                st.markdown("---")
                st.subheader("📥 Export PCA Results")

                export_df_overall = export_pca_results(pca_results_overall, clusters_overall, athlete_names_overall)

                col1, col2 = st.columns(2)
                with col1:
                    csv_export = export_df_overall.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Overall PCA Results (CSV)",
                        data=csv_export,
                        file_name=f"pca_overall_{selected_jump}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            else:
                st.warning("⚠️ Not enough metrics available for Overall PCA analysis")

            # ==========================================
            # SECTION 2: ENHANCED PCA WITH CATEGORY SELECTION
            # ==========================================
            
            st.markdown("---")
            st.subheader("🎯 Enhanced PCA Analysis - Category-Based")
            st.markdown("*Select specific metric categories for targeted analysis*")
            
            # Get available metrics from the data
            available_metrics_in_data = df_jump['TARGET_VARIABLE'].unique().tolist()
            
            # Build categories dynamically based on available data
            METRIC_CATEGORIES = {}
            
            # All Performance Metrics (exclude technical ones)
            technical_patterns = ["RESULT_TYPE", "START_OF_MOVEMENT_THRESHOLD", "START_OF_INTEGRATION"]
            all_performance = [m for m in available_metrics_in_data if not any(tech in m for tech in technical_patterns)]
            METRIC_CATEGORIES["All Performance Metrics"] = all_performance
            
            # Force Production
            force_patterns = ["FORCE", "WEIGHT_RELATIVE", "BODYMASS_RELATIVE", "BM_REL"]
            force_metrics = get_metrics_by_pattern(force_patterns, available_metrics_in_data)
            force_metrics = [m for m in force_metrics if "RFD" not in m.upper() and "RPD" not in m.upper()]
            METRIC_CATEGORIES["Force Production"] = force_metrics
            
            # Power Output  
            power_patterns = ["POWER", "VELOCITY_AT_PEAK_POWER"]
            METRIC_CATEGORIES["Power Output"] = get_metrics_by_pattern(power_patterns, available_metrics_in_data)
            
            # Speed/Velocity
            velocity_patterns = ["VELOCITY", "ACCELERATION"]
            velocity_metrics = get_metrics_by_pattern(velocity_patterns, available_metrics_in_data)
            velocity_metrics = [m for m in velocity_metrics if "POWER" not in m.upper()]
            METRIC_CATEGORIES["Speed/Velocity"] = velocity_metrics
            
            # Jump Mechanics
            jump_patterns = ["JUMP_HEIGHT", "FLIGHT_TIME", "RSI", "STIFFNESS", "DISPLACEMENT"]
            METRIC_CATEGORIES["Jump Mechanics"] = get_metrics_by_pattern(jump_patterns, available_metrics_in_data)
            
            # Rate of Force Development
            rfd_patterns = ["RFD", "RPD"]
            METRIC_CATEGORIES["Rate of Force Development"] = get_metrics_by_pattern(rfd_patterns, available_metrics_in_data)
            
            # Impulse & Momentum
            impulse_patterns = ["IMPULSE", "MOMENTUM"]
            METRIC_CATEGORIES["Impulse & Momentum"] = get_metrics_by_pattern(impulse_patterns, available_metrics_in_data)
            
            # Timing & Phases
            timing_patterns = ["TIME", "DURATION", "PHASE", "BEGIN_", "CONTRACTION"]
            METRIC_CATEGORIES["Timing & Phases"] = get_metrics_by_pattern(timing_patterns, available_metrics_in_data)
            
            # Ratios & Performance Indices  
            ratio_patterns = ["RATIO", "RELATIVE", "INDEX", "_REL_", "MODIFIED"]
            ratio_metrics = get_metrics_by_pattern(ratio_patterns, available_metrics_in_data)
            excluded_patterns = ["FORCE", "POWER", "VELOCITY", "IMPULSE"]
            ratio_metrics = [m for m in ratio_metrics if not any(exc in m.upper() for exc in excluded_patterns)]
            METRIC_CATEGORIES["Ratios & Performance Indices"] = ratio_metrics
            
            # Remove empty categories
            METRIC_CATEGORIES = {k: v for k, v in METRIC_CATEGORIES.items() if len(v) > 0}
            
            # Display category summary
            with st.expander("📊 Available Categories Summary"):
                for category, metrics in METRIC_CATEGORIES.items():
                    st.write(f"**{category}**: {len(metrics)} metrics")
                    if len(metrics) <= 10:
                        for metric in metrics[:5]:
                            st.write(f"  • {metric}")
                        if len(metrics) > 5:
                            st.write(f"  • ... and {len(metrics) - 5} more")
            
            # Category selection with checkboxes
            st.markdown("**📋 Select Metric Categories:**")
            
            num_categories = len(METRIC_CATEGORIES)
            if num_categories <= 3:
                cols = st.columns(num_categories)
            else:
                cols = st.columns(3)
            
            selected_categories = []
            category_names = list(METRIC_CATEGORIES.keys())
            
            for i, category in enumerate(category_names):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if st.checkbox(
                        f"{category} ({len(METRIC_CATEGORIES[category])})", 
                        key=f"enhanced_{category.lower().replace(' ', '_').replace('&', 'and')}"
                    ):
                        selected_categories.append(category)
            
            if selected_categories:
                # Combine selected metrics from all categories
                enhanced_metrics = []
                for category in selected_categories:
                    enhanced_metrics.extend(METRIC_CATEGORIES[category])
                
                # Remove duplicates while preserving order
                enhanced_metrics = list(dict.fromkeys(enhanced_metrics))
                
                if len(enhanced_metrics) >= 3:
                    st.success(f"Selected {len(enhanced_metrics)} metrics from {len(selected_categories)} categories")
                    
                    # Enhanced controls
                    col1, col2 = st.columns(2)
                    with col1:
                        enhanced_clusters = st.slider(
                            "Number of Clusters (Enhanced)",
                            min_value=2,
                            max_value=min(8, len(selected_athletes)),
                            value=3,
                            key="enhanced_clusters"
                        )
                    
                    with col2:
                        visualization_type = st.selectbox(
                            "Visualization Type",
                            ["2D View", "3D View", "Both"],
                            key="enhanced_viz_type"
                        )
                    
                    # Perform Enhanced PCA Analysis
                    with st.spinner("Performing Enhanced PCA Analysis..."):
                        pca_result, error_msg = perform_enhanced_pca_analysis(
                            df_jump, enhanced_metrics, selected_athletes, enhanced_clusters
                        )
                        
                        if pca_result is None:
                            st.error(f"Enhanced PCA Analysis failed: {error_msg}")
                        else:
                            # Extract results
                            pca_data = pca_result['pca_data']
                            cluster_labels = pca_result['cluster_labels']
                            athlete_names = pca_result['athlete_names']
                            valid_metrics = pca_result['valid_metrics']
                            pca_model = pca_result['pca_model']
                            n_components = pca_result['n_components']
                            actual_clusters = pca_result['n_clusters']
                            
                            st.info(f"Using {len(valid_metrics)} metrics with {len(athlete_names)} athletes")
                            
                            # Show silhouette score if available
                            sil = pca_result.get('silhouette_score')
                            if sil is not None:
                                qual = "Poor" if sil < 0.25 else "Fair" if sil < 0.5 else "Good" if sil < 0.75 else "Excellent"
                                st.success(f"**Cluster Silhouette Score:** {sil:.3f} ({qual})")
                            
                            # Create visualizations
                            colors = px.colors.qualitative.Set1
                            
                            if visualization_type in ["2D View", "Both"]:
                                st.markdown("**2D PCA Visualization:**")
                                fig_2d = go.Figure()
                                
                                for cluster in range(actual_clusters):
                                    mask = cluster_labels == cluster
                                    fig_2d.add_trace(go.Scatter(
                                        x=pca_data[mask, 0],
                                        y=pca_data[mask, 1] if n_components > 1 else np.zeros(np.sum(mask)),
                                        mode='markers+text',
                                        text=[name[:10] for i, name in enumerate(athlete_names) if mask[i]],
                                        textposition="top center",
                                        name=f'Cluster {cluster + 1}',
                                        marker=dict(size=12, color=colors[cluster % len(colors)])
                                    ))
                                
                                fig_2d.update_layout(
                                    title=f"Enhanced PCA Analysis - {', '.join(selected_categories)}",
                                    xaxis_title=f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%} variance)",
                                    yaxis_title=f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%} variance)" if n_components > 1 else "PC2",
                                    height=500,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_2d, use_container_width=True)
                            
                            if visualization_type in ["3D View", "Both"] and n_components >= 3:
                                st.markdown("**3D PCA Visualization:**")
                                fig_3d = go.Figure()
                                
                                for cluster in range(actual_clusters):
                                    mask = cluster_labels == cluster
                                    fig_3d.add_trace(go.Scatter3d(
                                        x=pca_data[mask, 0],
                                        y=pca_data[mask, 1],
                                        z=pca_data[mask, 2],
                                        mode='markers+text',
                                        text=[name[:8] for i, name in enumerate(athlete_names) if mask[i]],
                                        name=f'Cluster {cluster + 1}',
                                        marker=dict(size=8, color=colors[cluster % len(colors)])
                                    ))
                                
                                fig_3d.update_layout(
                                    title=f"3D Enhanced PCA Analysis",
                                    scene=dict(
                                        xaxis_title=f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})",
                                        yaxis_title=f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})",
                                        zaxis_title=f"PC3 ({pca_model.explained_variance_ratio_[2]:.1%})"
                                    ),
                                    height=600
                                )
                                
                                st.plotly_chart(fig_3d, use_container_width=True)
                            
                            # Component Analysis
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📊 Variance Explained:**")
                                variance_df = pd.DataFrame({
                                    'Component': [f'PC{i+1}' for i in range(n_components)],
                                    'Variance Explained': [f"{var:.1%}" for var in pca_model.explained_variance_ratio_],
                                    'Cumulative': [f"{np.sum(pca_model.explained_variance_ratio_[:i+1]):.1%}" for i in range(n_components)]
                                })
                                st.dataframe(variance_df, use_container_width=True)
                            
                            with col2:
                                st.markdown("**🎯 Cluster Assignments:**")
                                cluster_df_enhanced = pd.DataFrame({
                                    'Athlete': athlete_names,
                                    'Cluster': [f"Cluster {c + 1}" for c in cluster_labels]
                                })
                                st.dataframe(cluster_df_enhanced, use_container_width=True)
                            
                            # Component Loadings Analysis
                            st.markdown("**🔍 Component Loadings (Top Contributing Metrics):**")
                            
                            # Create loading analysis for each component
                            for comp in range(min(3, n_components)):
                                with st.expander(f"PC{comp+1} - Top Contributing Metrics ({pca_model.explained_variance_ratio_[comp]:.1%} variance)"):
                                    loadings = pca_model.components_[comp]
                                    loading_df = pd.DataFrame({
                                        'Metric': valid_metrics,
                                        'Loading': loadings,
                                        'Abs_Loading': np.abs(loadings)
                                    }).sort_values('Abs_Loading', ascending=False).head(10)
                                    
                                    # Create loading plot
                                    fig_loading = go.Figure(go.Bar(
                                        x=loading_df['Loading'],
                                        y=loading_df['Metric'],
                                        orientation='h',
                                        marker_color=['red' if x < 0 else 'blue' for x in loading_df['Loading']]
                                    ))
                                    fig_loading.update_layout(
                                        title=f"PC{comp+1} Loadings",
                                        xaxis_title="Loading Value",
                                        height=400
                                    )
                                    st.plotly_chart(fig_loading, use_container_width=True)
                                    
                                    # Show interpretation
                                    st.write("**Component Interpretation:**")
                                    positive_metrics = loading_df[loading_df['Loading'] > 0.3]['Metric'].tolist()
                                    negative_metrics = loading_df[loading_df['Loading'] < -0.3]['Metric'].tolist()
                                    
                                    if positive_metrics:
                                        st.write(f"**High positive contributors:** {', '.join(positive_metrics[:3])}")
                                    if negative_metrics:
                                        st.write(f"**High negative contributors:** {', '.join(negative_metrics[:3])}")
                            
                            # Cluster Characteristics
                            st.markdown("**📈 Cluster Characteristics:**")
                            
                            for cluster in range(actual_clusters):
                                cluster_athletes = [athlete_names[i] for i, c in enumerate(cluster_labels) if c == cluster]
                                
                                with st.expander(f"Cluster {cluster + 1} - {len(cluster_athletes)} athletes"):
                                    st.write(f"**Athletes:** {', '.join(cluster_athletes)}")
                                    
                                    # Calculate cluster centroid in original space
                                    cluster_mask = cluster_labels == cluster
                                    if np.sum(cluster_mask) > 0:
                                        cluster_centroid = np.mean(pca_result['athlete_array'][cluster_mask], axis=0)
                                        
                                        # Find top characteristics
                                        centroid_df = pd.DataFrame({
                                            'Metric': valid_metrics,
                                            'Average_Value': cluster_centroid
                                        })
                                        
                                        # Standardize for comparison
                                        overall_means = np.mean(pca_result['athlete_array'], axis=0)
                                        overall_stds = np.std(pca_result['athlete_array'], axis=0)
                                        
                                        z_scores = (cluster_centroid - overall_means) / (overall_stds + 1e-8)
                                        centroid_df['Z_Score'] = z_scores
                                        
                                        # Show top strengths and weaknesses
                                        strengths = centroid_df.nlargest(5, 'Z_Score')[['Metric', 'Z_Score']]
                                        weaknesses = centroid_df.nsmallest(5, 'Z_Score')[['Metric', 'Z_Score']]
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Top Strengths (vs average):**")
                                            for _, row in strengths.iterrows():
                                                st.write(f"• {row['Metric'][:25]}: +{row['Z_Score']:.1f}σ")
                                        
                                        with col2:
                                            st.write("**Areas for Development:**")
                                            for _, row in weaknesses.iterrows():
                                                st.write(f"• {row['Metric'][:25]}: {row['Z_Score']:.1f}σ")
                else:
                    st.warning("Please select at least 3 metrics for PCA analysis.")
            
           

            else:
                st.info("👆 Select at least one metric category to perform enhanced PCA analysis.")
                st.markdown("""
                **Category Descriptions:**
                - **All Performance Metrics**: Complete analysis using all available metrics
                - **Force Production**: Peak forces, mean forces, relative force capabilities  
                - **Power Output**: Power generation and power-to-weight ratios
                - **Speed/Velocity**: Movement velocities and accelerations
                - **Jump Mechanics**: Jump height, flight time, stiffness, reactive strength
                - **Rate of Force Development**: Speed of force production at different time windows
                - **Impulse & Momentum**: Force-time characteristics and impulse ratios
                - **Timing & Phases**: Movement timing, phase durations, coordination
                - **Ratios & Performance Indices**: Calculated performance ratios and efficiency metrics
                """)


        with tab4:
            st.header("Performance Trends Over Time")

            # Get metrics first
            with st.spinner("Analyzing metrics..."):
                metrics_df, all_available_metrics, metric_categories = analyze_metrics(df_jump)

            st.subheader("Select Metrics to Track")
            
            # Use the metric categories from Tab 1
            st.markdown("**Choose categories and specific metrics for trend analysis:**")
            
            # Category selection - full width (no columns needed)
            trend_categories = st.multiselect(
                "Select Categories",
                options=list(metric_categories.keys()),
                default=list(metric_categories.keys())[:2] if len(metric_categories) > 0 else [],
                help="Choose which metric categories to include",
                key="trend_categories"
            )

            # Get metrics from selected categories
            metrics_from_categories = []
            if trend_categories:
                for category in trend_categories:
                    metrics_from_categories.extend(metric_categories[category])
                
                # Remove duplicates
                metrics_from_categories = list(dict.fromkeys(metrics_from_categories))

            # Now let user select specific metrics
            if metrics_from_categories:
                # Initialize session state if needed
                if 'trends_metrics' not in st.session_state:
                    st.session_state.trends_metrics = metrics_from_categories[:10]

                # Filter session state to only include metrics that exist in current options
                valid_defaults = [m for m in st.session_state.trends_metrics if m in metrics_from_categories]
                if not valid_defaults:
                    valid_defaults = metrics_from_categories[:10]

                metrics_for_trends = st.multiselect(
                    "Choose specific metrics to analyze",
                    options=metrics_from_categories,
                    default=valid_defaults[:10],
                    help="Select which metrics you want to see trends for (up to 20 recommended)",
                    format_func=lambda x: f"{x[:50]}..." if len(x) > 50 else x,
                    key="trend_metrics_selector"
                )
                                        
                # Show category breakdown
                with st.expander("View selected metrics by category"):
                    for cat in trend_categories:
                        cat_metrics = [m for m in metrics_for_trends if m in metric_categories[cat]]
                        if cat_metrics:
                            st.markdown(f"**{cat.title()}**: {len(cat_metrics)} metrics")
                            for m in cat_metrics[:5]:
                                st.write(f"  • {m}")
                            if len(cat_metrics) > 5:
                                st.write(f"  • ... and {len(cat_metrics) - 5} more")
                
                if len(metrics_for_trends) > 0:
                    with st.spinner("Analyzing trends..."):
                        fig_trends, improvement_df = analyze_trends(df_jump, selected_athletes, metrics_for_trends)
                    
                    st.plotly_chart(fig_trends, use_container_width=True)
                    
                    # Show improvement summary
                    st.subheader("📈 Improvement Summary")
                    
                    if not improvement_df.empty:
                        # Overall improvement by athlete
                        athlete_improvement = improvement_df.groupby('Athlete')['Improvement (%)'].mean().round(1).sort_values(ascending=False)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Top Improvers**")
                            top_improvers = athlete_improvement.head(5)
                            for athlete, improvement in top_improvers.items():
                                if improvement > 0:
                                    st.write(f"• {athlete}: +{improvement}%")
                        
                        with col2:
                            st.markdown("**Needs Attention**")
                            bottom_performers = athlete_improvement.tail(5)
                            for athlete, improvement in bottom_performers.items():
                                if improvement < 0:
                                    st.write(f"• {athlete}: {improvement}%")
                        
                        # Detailed improvement table with significance
                        st.subheader("📊 Detailed Improvement Analysis")

                        # Show significance legend
                        st.markdown("**Statistical Significance:** *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
                        st.markdown("**Effect Size (Cohen's d):** 0.2 = small, 0.5 = medium, 0.8 = large, 1.2+ = very large")

                        # Display table with significance and effect size
                        display_cols = ['Athlete', 'Metric', 'Improvement (%)', 'Significance', 'Data Points']
                        if 'Effect Size (d)' in improvement_df.columns:
                            display_cols.insert(4, 'Effect Size (d)')
                        st.dataframe(
                            improvement_df[display_cols].head(20),
                            use_container_width=True,
                            hide_index=True
                        )

                        improvement_pivot = improvement_df.pivot_table(
                            index='Athlete',
                            columns='Metric',
                            values='Improvement (%)',
                            aggfunc='mean'
                        ).round(1)
                        
                        # Create heatmap
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=improvement_pivot.values,
                            x=[col[:25] for col in improvement_pivot.columns],
                            y=improvement_pivot.index,
                            colorscale='RdYlGn',
                            colorbar=dict(title="Improvement %"),
                            text=improvement_pivot.values,
                            texttemplate="%{text}%",
                            textfont={"size": 10}
                        ))
                        
                        fig_heatmap.update_layout(
                            title="Improvement by Athlete and Metric (%)",
                            height=400 + len(selected_athletes) * 20
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.info("Not enough data points to calculate trends")
                else:
                    st.warning("⚠️ Please select at least one metric to analyze trends")
            else:
                st.info("👆 Please select at least one category to see available metrics")

        with tab5:
            st.header("👤 Athlete Profiles & Comparisons")

            # Get metrics first
            with st.spinner("Analyzing metrics..."):
                metrics_df, all_available_metrics, metric_categories = analyze_metrics(df_jump)

            st.markdown("View individual athlete profiles and compare athletes head-to-head")

            # Section 1: Individual Athlete Profiles
            st.subheader("🎯 Individual Athlete Profile")

            selected_athlete_profile = st.selectbox(
                "Select athlete to view profile",
                options=selected_athletes,
                key="profile_athlete"
            )

            if selected_athlete_profile:
                with st.spinner(f"Generating profile for {selected_athlete_profile}..."):
                    profile_df, strengths, weaknesses = create_athlete_profile_card(
                        df_jump, selected_athlete_profile, selected_athletes
                    )

                # Display profile summary
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Metrics", len(profile_df))

                with col2:
                    avg_percentile = profile_df['Percentile'].mean()
                    st.metric("Avg Percentile", f"{avg_percentile:.1f}%")

                with col3:
                    above_avg = len(profile_df[profile_df['Z-Score'] > 0])
                    st.metric("Above Average Metrics", f"{above_avg}/{len(profile_df)}")

                # Strengths and Weaknesses
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 💪 Top 5 Strengths")
                    for _, row in strengths.iterrows():
                        st.markdown(f"**{row['Metric'][:40]}**")
                        st.write(f"Value: {row['Value']:.2f} | Group Avg: {row['Group Avg']:.2f}")
                        st.write(f"Percentile: {row['Percentile']:.0f}% | Z-Score: +{row['Z-Score']:.2f}σ")
                        st.markdown("---")

                with col2:
                    st.markdown("### 📈 Top 5 Development Areas")
                    for _, row in weaknesses.iterrows():
                        st.markdown(f"**{row['Metric'][:40]}**")
                        st.write(f"Value: {row['Value']:.2f} | Group Avg: {row['Group Avg']:.2f}")
                        st.write(f"Percentile: {row['Percentile']:.0f}% | Z-Score: {row['Z-Score']:.2f}σ")
                        st.markdown("---")

                # Full profile table
                with st.expander("📊 View Full Profile"):
                    st.dataframe(
                        profile_df[['Metric', 'Value', 'Group Avg', 'Z-Score', 'Percentile']],
                        use_container_width=True,
                        hide_index=True
                    )

            # Section 2: Head-to-Head Comparison
            st.markdown("---")
            st.subheader("⚔️ Head-to-Head Athlete Comparison")

            col1, col2 = st.columns(2)

            with col1:
                athlete1 = st.selectbox(
                    "Select first athlete",
                    options=selected_athletes,
                    key="compare_athlete1"
                )

            with col2:
                athlete2_options = [a for a in selected_athletes if a != athlete1]
                athlete2 = st.selectbox(
                    "Select second athlete",
                    options=athlete2_options,
                    key="compare_athlete2"
                )

            if athlete1 and athlete2:
                # Category selection for comparison
                comparison_categories = st.multiselect(
                    "Select categories for comparison",
                    options=list(metric_categories.keys()),
                    default=list(metric_categories.keys())[:3] if len(metric_categories) >= 3 else list(metric_categories.keys()),
                    key="comparison_categories"
                )

                if comparison_categories:
                    # Get metrics from selected categories
                    comparison_metrics = []
                    for cat in comparison_categories:
                        comparison_metrics.extend(metric_categories[cat])

                    comparison_metrics = list(dict.fromkeys(comparison_metrics))[:15]  # Limit to 15 metrics

                    with st.spinner("Comparing athletes..."):
                        comparison_df, radar_fig = compare_athletes(df_jump, athlete1, athlete2, comparison_metrics)

                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)

                    if not comparison_df.empty:
                        # Summary stats
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            athlete1_wins = len(comparison_df[comparison_df['Winner'] == athlete1])
                            st.metric(f"{athlete1} Leads", athlete1_wins)

                        with col2:
                            athlete2_wins = len(comparison_df[comparison_df['Winner'] == athlete2])
                            st.metric(f"{athlete2} Leads", athlete2_wins)

                        with col3:
                            avg_diff = comparison_df['Difference (%)'].abs().mean()
                            st.metric("Avg Difference", f"{avg_diff:.1f}%")

                        # Detailed comparison table
                        st.subheader("📊 Detailed Comparison")
                        st.dataframe(
                            comparison_df[['Metric', athlete1, athlete2, 'Difference', 'Difference (%)', 'Winner']],
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.info("👆 Select at least one category for comparison")

        with tab6:
            st.header("Summary Report")
            
            # Generate summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Athletes Analyzed", len(selected_athletes))
                st.metric("Total Metrics Available", len(all_available_metrics))
            
            with col2:
                st.metric("Jump Type", selected_jump)
                st.metric("Data Points", len(df_jump[df_jump['ATHLETE_NAME'].isin(selected_athletes)]))
            
            with col3:
                st.metric("Time Period", f"{df_jump['YEAR'].min()}-{df_jump['YEAR'].max()}")
                st.metric("Unique Clusters", len(np.unique(clusters_overall)) if 'clusters_overall' in locals() else "N/A")
            
            # Export functionality
            st.markdown("---")
            st.subheader("📥 Export Results")

            if 'improvement_df' in locals() and not improvement_df.empty:
                csv = improvement_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis Report (CSV)",
                    data=csv,
                    file_name=f"force_plate_analysis_{selected_jump}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Complete the analysis to enable export")
    
    elif uploaded_file is None:
        # Welcome screen
        st.info("👈 Please upload a force plate CSV file to begin analysis")
        
        st.markdown("""
        ### How to use this application:

        1. **Upload your data**: Use the sidebar to upload your force plate CSV file
        2. **Select jump type**: Choose which type of jump to analyze
        3. **Choose athletes**: Select athletes for comparison (2-20 recommended)
        4. **Explore results**: Navigate through the six analysis tabs

        ### What this app does:

        #### 📊 Tab 1: Data Quality Dashboard
        - View data completeness heatmaps
        - Identify athletes with missing data
        - Get quality scores and recommendations

        #### 📈 Tab 2: Metrics Analysis
        - Search and filter through 100+ metrics
        - Quick filters for Power, Force, Speed, etc.
        - Categorized metric organization

        #### 🎯 Tab 3: PCA & Clustering
        - Overall PCA with ALL metrics
        - Category-specific PCA (e.g., only Power metrics)
        - Automatic cluster naming and insights
        - Export PCA results to CSV
        - 2D and 3D visualizations

        #### 📊 Tab 4: Performance Trends
        - Track metrics over time
        - Statistical significance testing (p-values)
        - Improvement heatmaps
        - Identify top improvers

        #### 👤 Tab 5: Athlete Profiles
        - Individual athlete strength/weakness analysis
        - Percentile rankings vs group
        - Head-to-head comparisons with radar charts
        - Win/loss tracking between athletes

        #### 📋 Tab 6: Summary Report
        - Export all analysis results
        - Overall statistics and insights
        
        ### Data Requirements:
        
        Your CSV can be in either format:
        
        **Format 1 (Wide):** Each row = one test, metrics as columns
        - Columns like: `Name`, `Jump Height [cm]`, `Peak Power [W]`, etc.
        
        **Format 2 (Long):** Each row = one measurement
        - Columns: `ATHLETE_NAME`, `TARGET_VARIABLE`, `RESULT`, `TEST_DATE`, etc.
        
        The app automatically detects and handles both formats!
        """)

if __name__ == "__main__":
    main()
