"""
Force Plate Analysis Streamlit Application
==========================================
A user-friendly interface for analyzing force plate jump data
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    """Create visualization dashboard"""

def create_visualizations(pca_results,num_clusters=3):
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
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)  # Use the parameter
    clusters = kmeans.fit_predict(pca_data[:, :3])

    cluster_colors = px.colors.qualitative.Set1  # Use full color palette
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
                marker=dict(color=cluster_colors[cluster % len(cluster_colors)], size=15),  # Handle more colors
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
        height=2500,  # Much taller for better visibility
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
    num_metrics = min(10, len(selected_metrics))  # Show up to 10 metrics
    num_rows = (num_metrics + 1) // 2  # Calculate rows needed (2 per row)
    
    fig_trends = make_subplots(
        rows=num_rows, 
        cols=2,
        subplot_titles=[f"{metric[:40]}..." if len(metric) > 40 else metric for metric in selected_metrics[:num_metrics]],
        vertical_spacing=0.08,  # Tighter vertical spacing since we have more rows
        horizontal_spacing=0.12  # Good horizontal spacing
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, metric in enumerate(selected_metrics[:num_metrics]):
        row = (i // 2) + 1  # 2 per row
        col = (i % 2) + 1   # 2 columns
        
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
                        legendgroup=athlete[:15],  # ADD THIS LINE - groups all traces by athlete
                        showlegend=(i == 0),  # Keep this as is
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x|%b %Y}<br>' +
                                    'Value: %{y:.2f}<br>' +
                                    '<extra></extra>'
                    ),
                    row=row, col=col
                )
    
    # Update all x-axes and y-axes
    for i in range(1, num_rows + 1):
        for j in range(1, 3):  # 2 columns
            fig_trends.update_xaxes(
                row=i, col=j,
                tickformat='%b %Y',  # Format as "Jan 2021"
                tickmode='auto',
                nticks=8,  # More ticks since charts are bigger
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
    
    # Calculate dynamic height based on number of rows
    chart_height = 400 * num_rows  # 400px per row
    
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
    st.title("🏃‍♂️ Force Plate Jump Analysis System")
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
                        value=30,  # Default to 30
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
                    # In tab3, update the default selection
                    metrics_for_trends = st.multiselect(
                        "Choose metrics to analyze trends",
                        options=selected_metrics,
                        default=selected_metrics[:10],  # Default to first 10 instead of 6
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
                        fig_trends, improvement_df = analyze_trends(df_jump, selected_athletes, metrics_for_trends)  # Use selected metrics
                    
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
            
            # # Export functionality
            # Export functionality (CSV version - no xlsxwriter needed)
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
            # st.markdown("---")
            # st.subheader("📥 Export Results")
            
            # # Prepare export data
            # export_data = {
            #     'Athlete Analysis': pd.DataFrame({
            #         'Athlete': selected_athletes,
            #         'Cluster': [f"Group {c + 1}" for c in clusters] if 'clusters' in locals() else ['N/A'] * len(selected_athletes)
            #     }),
            #     'Metrics Used': pd.DataFrame({'Metric': selected_metrics}),
            # }
            
            # if not improvement_df.empty:
            #     export_data['Improvement Analysis'] = improvement_df
            
            # # Create Excel file in memory
            # output = io.BytesIO()
            # with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            #     for sheet_name, df_export in export_data.items():
            #         df_export.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # output.seek(0)
            
            # st.download_button(
            #     label="📥 Download Analysis Report (Excel)",
            #     data=output,
            #     file_name=f"force_plate_analysis_{selected_jump}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
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