"""
Force Plate Analysis Streamlit Application
==========================================
A comprehensive, robust interface for analyzing force plate jump data
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import io

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
# HELPER FUNCTIONS
# ==========================================

@st.cache_data
def load_and_clean_data(uploaded_file):
    """Load and clean the force plate data"""
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        
        # Filter for Trial limb only (bilateral measurements)
        df_clean = df[df['LIMB'] == 'Trial'].copy()
        
        # Convert dates and add temporal columns
        df_clean['TEST_DATE'] = pd.to_datetime(df_clean['TEST_DATE'])
        df_clean['YEAR'] = df_clean['TEST_DATE'].dt.year
        df_clean['QUARTER'] = df_clean['TEST_DATE'].dt.quarter
        df_clean['MONTH'] = df_clean['TEST_DATE'].dt.month
        
        # Convert RESULT to numeric
        df_clean['RESULT'] = pd.to_numeric(df_clean['RESULT'], errors='coerce')
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_jump_types(df):
    """Extract unique jump types from the data"""
    if 'TEST_TYPE' in df.columns:
        return sorted(df['TEST_TYPE'].dropna().unique())
    elif 'JUMP_TYPE' in df.columns:
        return sorted(df['JUMP_TYPE'].dropna().unique())
    else:
        # Try to infer from TARGET_VARIABLE
        jump_indicators = ['CMJ', 'SJ', 'DJ', 'JUMP']
        jump_types = set()
        for var in df['TARGET_VARIABLE'].unique():
            for indicator in jump_indicators:
                if indicator in var:
                    jump_types.add(indicator)
        return sorted(list(jump_types)) if jump_types else ['All Jumps']

def filter_data_by_jump(df, jump_type):
    """Filter data for specific jump type"""
    if jump_type == 'All Jumps':
        return df
    
    # Try different column names
    if 'TEST_TYPE' in df.columns:
        return df[df['TEST_TYPE'] == jump_type]
    elif 'JUMP_TYPE' in df.columns:
        return df[df['JUMP_TYPE'] == jump_type]
    else:
        # Filter by TARGET_VARIABLE containing jump type
        return df[df['TARGET_VARIABLE'].str.contains(jump_type, na=False)]

def analyze_metrics(df, num_metrics=30):
    """Analyze and categorize available metrics"""
    all_metrics = df['TARGET_VARIABLE'].unique()
    
    # Analyze each metric's quality
    metric_analysis = []
    for metric in all_metrics:
        metric_data = df[df['TARGET_VARIABLE'] == metric]
        
        if len(metric_data) > 0:
            values = pd.to_numeric(metric_data['RESULT'], errors='coerce').dropna()
            
            if len(values) > 0:
                metric_analysis.append({
                    'metric': metric,
                    'completeness_pct': len(values) / len(metric_data) * 100,
                    'num_athletes': metric_data['ATHLETE_NAME'].nunique(),
                    'num_datapoints': len(values),
                    'mean': values.mean(),
                    'std': values.std(),
                    'cv': values.std() / values.mean() if values.mean() != 0 else 0
                })
    
    metrics_df = pd.DataFrame(metric_analysis).sort_values('completeness_pct', ascending=False)
    
    # Categorize metrics
    power_metrics = [m for m in all_metrics if 'POWER' in m.upper()]
    force_metrics = [m for m in all_metrics if 'FORCE' in m.upper() and 'RFD' not in m.upper()]
    velocity_metrics = [m for m in all_metrics if 'VELOCITY' in m.upper()]
    jump_metrics = [m for m in all_metrics if any(x in m.upper() for x in ['JUMP', 'HEIGHT', 'RSI'])]
    timing_metrics = [m for m in all_metrics if any(x in m.upper() for x in ['TIME', 'DURATION'])]
    rfd_metrics = [m for m in all_metrics if 'RFD' in m.upper()]
    
    # Select high-quality metrics
    high_quality = metrics_df[
        (metrics_df['completeness_pct'] >= 70) & 
        (metrics_df['num_athletes'] >= 3) &
        (metrics_df['cv'] >= 0.05) &
        (metrics_df['cv'] <= 2.0)
    ]
    
    # Prioritize key metrics
    priority_keywords = ['JUMP_HEIGHT', 'PEAK_TAKEOFF_FORCE', 'PEAK_POWER', 'RSI', 
                        'CONCENTRIC', 'ECCENTRIC', 'FLIGHT_TIME', 'CONTRACTION_TIME', 
                        'TAKEOFF_VELOCITY', 'LANDING_FORCE']
    
    selected_metrics = []
    for keyword in priority_keywords:
        matching = [m for m in high_quality['metric'] if keyword in m.upper()]
        selected_metrics.extend(matching[:2])
    
    # Remove duplicates
    selected_metrics = list(dict.fromkeys(selected_metrics))
    
    # Add more if needed
    if len(selected_metrics) < 10:
        additional = high_quality[~high_quality['metric'].isin(selected_metrics)]['metric'].head(10 - len(selected_metrics)).tolist()
        selected_metrics.extend(additional)
    
    return metrics_df, selected_metrics[:num_metrics], {
        'power': power_metrics,
        'force': force_metrics,
        'velocity': velocity_metrics,
        'jump': jump_metrics,
        'timing': timing_metrics,
        'rfd': rfd_metrics
    }

def perform_pca_analysis(df, selected_metrics, selected_athletes):
    """Perform PCA analysis on selected data"""
    # Filter data
    df_filtered = df[df['ATHLETE_NAME'].isin(selected_athletes)]
    
    # Create athlete × metric matrix
    athlete_metrics = df_filtered[df_filtered['TARGET_VARIABLE'].isin(selected_metrics)].pivot_table(
        index='ATHLETE_NAME',
        columns='TARGET_VARIABLE',
        values='RESULT',
        aggfunc='mean'
    )
    
    # Fill missing values
    athlete_metrics_filled = athlete_metrics.fillna(athlete_metrics.median())
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(athlete_metrics_filled)
    
    # PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    
    # Feature importance
    feature_importance = np.abs(pca.components_[:3]).mean(axis=0)
    
    return {
        'athlete_metrics': athlete_metrics_filled,
        'pca_data': pca_data,
        'pca_model': pca,
        'feature_importance': feature_importance,
        'feature_names': list(athlete_metrics_filled.columns)
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
    
    # 1. PCA Space - Make it bigger
    for i, athlete in enumerate(athlete_names):
        fig.add_trace(
            go.Scatter(
                x=[pca_data[i, 0]],
                y=[pca_data[i, 1]],
                mode='markers+text',
                name=athlete,
                text=[athlete[:15]],
                textposition="top center",
                marker=dict(color=colors[i % len(colors)], size=15),
                showlegend=True
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
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 3. K-means Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_data[:, :3])

    cluster_colors = px.colors.qualitative.Set1
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        cluster_athletes = np.array(athlete_names)[mask]
        fig.add_trace(
            go.Scatter(
                x=pca_data[mask, 0],
                y=pca_data[mask, 1],
                mode='markers+text',
                name=f'Cluster {cluster + 1}',
                text=[name[:12] for name in cluster_athletes],
                textposition="top center",
                marker=dict(color=cluster_colors[cluster % len(cluster_colors)], size=15),
                showlegend=True
            ),
            row=3, col=1
        )
    
    # 4. Explained Variance
    explained_var = pca_results['pca_model'].explained_variance_ratio_[:10]
    cumulative_var = np.cumsum(explained_var)
    
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(explained_var) + 1)),
            y=explained_var,
            marker_color='lightblue',
            name='Individual',
            showlegend=True
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
            showlegend=True
        ),
        row=4, col=1
    )
    
    # Update layout for better spacing
    fig.update_layout(
        height=2500,
        title_text="<b>Force Plate Analysis - PCA & Clustering Dashboard</b>",
        title_font_size=20,
        showlegend=True,
        font=dict(size=12)
    )
    
    # Update axes labels with better formatting
    fig.update_xaxes(title_text="<b>Principal Component 1</b>", row=1, col=1, title_font_size=14)
    fig.update_yaxes(title_text="<b>Principal Component 2</b>", row=1, col=1, title_font_size=14)
    
    fig.update_xaxes(title_text="<b>Importance Score</b>", row=2, col=1, title_font_size=14)
    fig.update_yaxes(title_text="<b>Features</b>", row=2, col=1, title_font_size=14)
    
    fig.update_xaxes(title_text="<b>Principal Component 1</b>", row=3, col=1, title_font_size=14)
    fig.update_yaxes(title_text="<b>Principal Component 2</b>", row=3, col=1, title_font_size=14)
    
    fig.update_xaxes(title_text="<b>Component Number</b>", row=4, col=1, title_font_size=14)
    fig.update_yaxes(title_text="<b>Variance Explained</b>", row=4, col=1, title_font_size=14)
    
    return fig, clusters

def get_metrics_by_pattern(patterns, available_metrics):
    """Get metrics that match any of the given patterns"""
    matching = []
    for metric in available_metrics:
        metric_upper = metric.upper()
        if any(pattern.upper() in metric_upper for pattern in patterns):
            matching.append(metric)
    return matching

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
        
        # Perform clustering
        actual_clusters = min(n_clusters, len(athlete_matrix))
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
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
            'scaler': scaler
        }, None
        
    except Exception as e:
        return None, f"Error in PCA analysis: {str(e)}"

def analyze_trends(df, selected_athletes, selected_metrics):
    """Analyze performance trends over time"""
    df_trends = df[
        (df['ATHLETE_NAME'].isin(selected_athletes)) & 
        (df['TARGET_VARIABLE'].isin(selected_metrics))
    ].copy()
    
    # Create proper date column for better plotting
    df_trends['DATE'] = pd.to_datetime(df_trends['TEST_DATE'])
    
    # Group by actual date (not period) for better plotting
    df_trends = df_trends.groupby(['ATHLETE_NAME', 'DATE', 'TARGET_VARIABLE'])['RESULT'].mean().reset_index()
    df_trends = df_trends.sort_values('DATE')
    
    # Calculate improvement rates
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
                
                improvement_data.append({
                    'Athlete': athlete,
                    'Metric': metric,
                    'Improvement (%)': improvement,
                    'First Value': first_val,
                    'Last Value': last_val,
                    'Data Points': len(athlete_metric)
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
                st.metric("Date Range", f"{df['TEST_DATE'].min().strftime('%Y-%m-%d')} to {df['TEST_DATE'].max().strftime('%Y-%m-%d')}")
                
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
                
                # Add metric count selector
                st.markdown("---")
                num_metrics_to_analyze = st.slider(
                    "Number of Metrics to Analyze",
                    min_value=10,
                    max_value=50,
                    value=30,
                    step=5,
                    help="Select how many metrics to include in analysis"
                )
                
                if len(selected_athletes) < 2:
                    st.warning("⚠️ Please select at least 2 athletes for analysis")
    
    # Main content area
    if uploaded_file is not None and df is not None and len(selected_athletes) >= 2:
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics Analysis", "🎯 PCA & Clustering", "📊 Performance Trends", "📋 Summary Report"])
        
        with tab1:
            st.header("Metrics Analysis")
            
            with st.spinner("Analyzing metrics..."):
                metrics_df, selected_metrics, metric_categories = analyze_metrics(df_jump, num_metrics_to_analyze)
            
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
            
            st.subheader("📝 Selected Metrics for Analysis")
            selected_display = pd.DataFrame({
                'Selected Metrics': selected_metrics,
                'Category': [
                    'Power' if 'POWER' in m.upper() else
                    'Force' if 'FORCE' in m.upper() else
                    'Velocity' if 'VELOCITY' in m.upper() else
                    'Jump' if any(x in m.upper() for x in ['JUMP', 'HEIGHT', 'RSI']) else
                    'Timing' if any(x in m.upper() for x in ['TIME', 'DURATION']) else
                    'RFD' if 'RFD' in m.upper() else
                    'Eccentric' if 'ECCENTRIC' in m.upper() else
                    'Concentric' if 'CONCENTRIC' in m.upper() else
                    'Other'
                    for m in selected_metrics
                ]
            })
            st.dataframe(selected_display, hide_index=True)
        
        with tab2:
            st.header("PCA Analysis & Clustering")
            
            if len(selected_metrics) > 0:
                # Add cluster selection control
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    num_clusters = st.slider(
                        "Number of Clusters",
                        min_value=2,
                        max_value=min(8, len(selected_athletes)),
                        value=3,
                        help="Choose how many groups to create"
                    )
                
                with st.spinner("Performing PCA analysis..."):
                    pca_results = perform_pca_analysis(df_jump, selected_metrics, selected_athletes)
                    fig_pca, clusters = create_visualizations(pca_results, num_clusters)
                
                st.plotly_chart(fig_pca, use_container_width=True)
                
                # Show cluster assignments
                st.subheader("🎯 Cluster Assignments")
                cluster_df = pd.DataFrame({
                    'Athlete': list(pca_results['athlete_metrics'].index),
                    'Cluster': [f"Group {c + 1}" for c in clusters]
                })
                
                # Create columns for cluster groups
                cols = st.columns(min(num_clusters, 3))
                for i in range(num_clusters):
                    with cols[i % 3]:
                        st.markdown(f"**Group {i + 1}**")
                        group_athletes = cluster_df[cluster_df['Cluster'] == f"Group {i + 1}"]['Athlete'].tolist()
                        for athlete in group_athletes:
                            st.write(f"• {athlete}")
            else:
                st.warning("⚠️ No metrics available for PCA analysis")

            # ==========================================
            # ENHANCED PCA ANALYSIS WITH METRIC CATEGORIES
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
            # Remove RFD metrics from force category
            force_metrics = [m for m in force_metrics if "RFD" not in m.upper() and "RPD" not in m.upper()]
            METRIC_CATEGORIES["Force Production"] = force_metrics
            
            # Power Output  
            power_patterns = ["POWER", "VELOCITY_AT_PEAK_POWER"]
            METRIC_CATEGORIES["Power Output"] = get_metrics_by_pattern(power_patterns, available_metrics_in_data)
            
            # Speed/Velocity
            velocity_patterns = ["VELOCITY", "ACCELERATION"]
            velocity_metrics = get_metrics_by_pattern(velocity_patterns, available_metrics_in_data)
            # Remove power-related velocity metrics to avoid duplication
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
            # Remove metrics already categorized elsewhere
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
            
            # Create dynamic columns based on available categories
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
                
                # Validate we have enough metrics
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

        with tab3:
            st.header("Performance Trends Over Time")
            
            if len(selected_metrics) > 0:
                # Add metric selector for trends
                st.subheader("Select Metrics to Track")
                
                # Organize metrics by category for easier selection
                metrics_by_category = {}
                for metric in selected_metrics:
                    if 'POWER' in metric.upper():
                        cat = 'Power'
                    elif 'FORCE' in metric.upper():
                        cat = 'Force'
                    elif 'VELOCITY' in metric.upper():
                        cat = 'Velocity'
                    elif any(x in metric.upper() for x in ['JUMP', 'HEIGHT', 'RSI']):
                        cat = 'Jump/Height'
                    elif any(x in metric.upper() for x in ['TIME', 'DURATION']):
                        cat = 'Timing'
                    elif 'RFD' in metric.upper():
                        cat = 'RFD'
                    elif 'ECCENTRIC' in metric.upper():
                        cat = 'Eccentric'
                    elif 'CONCENTRIC' in metric.upper():
                        cat = 'Concentric'
                    else:
                        cat = 'Other'
                    
                    if cat not in metrics_by_category:
                        metrics_by_category[cat] = []
                    metrics_by_category[cat].append(metric)
                
                # Create columns for metric selection
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Multi-select for metrics
                    metrics_for_trends = st.multiselect(
                        "Choose metrics to analyze trends",
                        options=selected_metrics,
                        default=selected_metrics[:10],
                        help="Select which metrics you want to see trends for (up to 10 recommended)",
                        format_func=lambda x: f"{x[:40]}..." if len(x) > 40 else x
                    )
                
                with col2:
                    # Quick select buttons
                    st.markdown("**Quick Select:**")
                    if st.button("Top 10"):
                        metrics_for_trends = selected_metrics[:10]
                    if st.button("Power Metrics"):
                        metrics_for_trends = [m for m in selected_metrics if 'POWER' in m.upper()]
                    if st.button("Jump Metrics"):
                        metrics_for_trends = [m for m in selected_metrics if any(x in m.upper() for x in ['JUMP', 'HEIGHT', 'RSI'])]
                    if st.button("Clear All"):
                        metrics_for_trends = []
                
                # Show category breakdown
                with st.expander("View metrics by category"):
                    for cat, metrics in metrics_by_category.items():
                        st.markdown(f"**{cat}:** {len(metrics)} metrics")
                        for m in metrics:
                            st.write(f"  • {m}")
                
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
                        
                        # Detailed improvement table
                        st.subheader("📊 Detailed Improvement Analysis")
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
                st.warning("⚠️ No metrics available for trend analysis")
        
        with tab4:
            st.header("Summary Report")
            
            # Generate summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Athletes Analyzed", len(selected_athletes))
                st.metric("Metrics Evaluated", len(selected_metrics))
            
            with col2:
                st.metric("Jump Type", selected_jump)
                st.metric("Data Points", len(df_jump[df_jump['ATHLETE_NAME'].isin(selected_athletes)]))
            
            with col3:
                st.metric("Time Period", f"{df_jump['YEAR'].min()}-{df_jump['YEAR'].max()}")
                st.metric("Unique Clusters", len(np.unique(clusters)) if 'clusters' in locals() else "N/A")
            
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
        3. **Choose athletes**: Select athletes for comparison (2-15 recommended)
        4. **Explore results**: Navigate through the tabs to see different analyses
        
        ### What this app does:
        
        - **Metrics Analysis**: Identifies and categorizes performance metrics
        - **PCA & Clustering**: Groups similar athletes based on performance patterns
        - **Enhanced PCA**: Category-based analysis for targeted insights
        - **Performance Trends**: Tracks improvements over time
        - **Summary Report**: Provides overview and export functionality
        
        ### Data Requirements:
        
        Your CSV should contain:
        - `ATHLETE_NAME`: Athlete identifiers
        - `TARGET_VARIABLE`: Metric names
        - `RESULT`: Measurement values
        - `TEST_DATE`: Date of testing
        - `LIMB`: Should include 'Trial' for bilateral measurements
        """)

if __name__ == "__main__":
    main()
