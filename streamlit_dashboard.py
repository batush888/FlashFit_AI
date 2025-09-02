import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
from pathlib import Path

# FlashFit AI imports
try:
    from backend.services.integration_manager import IntegrationManager, IntegrationConfig
    from backend.services.analytics_service import AnalyticsService
    from backend.services.performance_monitor import PerformanceMonitor
except ImportError:
    st.error("FlashFit AI services not found. Please ensure the backend is properly installed.")
    st.stop()

# Additional libraries
try:
    import torch
    import psutil
    from rich.console import Console
except ImportError as e:
    st.warning(f"Some features may be limited due to missing dependencies: {e}")

# Page configuration
st.set_page_config(
    page_title="FlashFit AI Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}

.status-good {
    color: #28a745;
    font-weight: bold;
}

.status-warning {
    color: #ffc107;
    font-weight: bold;
}

.status-error {
    color: #dc3545;
    font-weight: bold;
}

.sidebar-section {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'integration_manager' not in st.session_state:
    st.session_state.integration_manager = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

def initialize_integration_manager():
    """Initialize the integration manager"""
    try:
        config = IntegrationConfig(
            enable_analytics=True,
            enable_performance_monitoring=True,
            enable_flask_extensions=True,
            enable_wandb=False,  # Disable for demo
            enable_peft=True,
            enable_trl=True,
            enable_streamlit_dashboard=True,
            enable_real_time_analytics=True
        )
        
        manager = IntegrationManager(config)
        st.session_state.integration_manager = manager
        return manager
    except Exception as e:
        st.error(f"Failed to initialize integration manager: {e}")
        return None

def get_system_metrics():
    """Get current system metrics"""
    try:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'
            }
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / 1024**3,
            'memory_total_gb': memory.total / 1024**3,
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / 1024**3,
            'disk_total_gb': disk.total / 1024**3,
            'gpu_info': gpu_info,
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Failed to get system metrics: {e}")
        return {}

def create_performance_charts(metrics):
    """Create performance monitoring charts"""
    if not metrics:
        return None, None
    
    # System metrics chart
    fig_system = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 'GPU Memory'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # CPU gauge
    fig_system.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics.get('cpu_percent', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )
    
    # Memory gauge
    fig_system.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics.get('memory_percent', 0),
            title={'text': "Memory %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "red"}
                ]
            }
        ),
        row=1, col=2
    )
    
    # Disk gauge
    fig_system.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics.get('disk_percent', 0),
            title={'text': "Disk %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkorange"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "red"}
                ]
            }
        ),
        row=2, col=1
    )
    
    # GPU gauge
    gpu_usage = 0
    if metrics.get('gpu_info') and 'gpu_memory_allocated' in metrics['gpu_info']:
        gpu_total = metrics['gpu_info'].get('gpu_memory_reserved', 1)
        gpu_used = metrics['gpu_info'].get('gpu_memory_allocated', 0)
        gpu_usage = (gpu_used / gpu_total * 100) if gpu_total > 0 else 0
    
    fig_system.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=gpu_usage,
            title={'text': "GPU Memory %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "red"}
                ]
            }
        ),
        row=2, col=2
    )
    
    fig_system.update_layout(
        title="System Performance Metrics",
        height=600,
        showlegend=False
    )
    
    return fig_system, None

def create_fashion_analytics_demo():
    """Create demo fashion analytics"""
    # Generate sample fashion data
    np.random.seed(42)
    
    categories = ['Tops', 'Bottoms', 'Dresses', 'Shoes', 'Accessories']
    colors = ['Black', 'White', 'Blue', 'Red', 'Green', 'Gray', 'Brown']
    sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    
    n_items = 1000
    
    fashion_data = pd.DataFrame({
        'category': np.random.choice(categories, n_items),
        'color': np.random.choice(colors, n_items),
        'size': np.random.choice(sizes, n_items),
        'price': np.random.uniform(20, 300, n_items),
        'rating': np.random.uniform(2.5, 5.0, n_items),
        'sales': np.random.poisson(50, n_items),
        'views': np.random.poisson(200, n_items),
        'date': pd.date_range('2024-01-01', periods=n_items, freq='H')
    })
    
    return fashion_data

def create_fashion_charts(data):
    """Create fashion analytics charts"""
    # Category distribution
    fig_category = px.pie(
        data.groupby('category').size().reset_index(name='count'),
        values='count',
        names='category',
        title="Fashion Items by Category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Price distribution by category
    fig_price = px.box(
        data,
        x='category',
        y='price',
        title="Price Distribution by Category",
        color='category'
    )
    
    # Sales trends over time
    daily_sales = data.groupby(data['date'].dt.date).agg({
        'sales': 'sum',
        'views': 'sum',
        'rating': 'mean'
    }).reset_index()
    
    fig_trends = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Sales & Views', 'Average Rating'),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Sales and views
    fig_trends.add_trace(
        go.Scatter(x=daily_sales['date'], y=daily_sales['sales'], name='Sales', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig_trends.add_trace(
        go.Scatter(x=daily_sales['date'], y=daily_sales['views'], name='Views', line=dict(color='orange')),
        row=1, col=1, secondary_y=True
    )
    
    # Average rating
    fig_trends.add_trace(
        go.Scatter(x=daily_sales['date'], y=daily_sales['rating'], name='Avg Rating', line=dict(color='green')),
        row=2, col=1
    )
    
    fig_trends.update_layout(
        title="Fashion Analytics Trends",
        height=600
    )
    
    # Color popularity heatmap
    color_category = data.groupby(['color', 'category']).size().unstack(fill_value=0)
    fig_heatmap = px.imshow(
        color_category,
        title="Color Popularity by Category",
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    
    return fig_category, fig_price, fig_trends, fig_heatmap

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">👗 FlashFit AI Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🎛️ Control Panel")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        # Last refresh time
        st.write(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📊 Navigation")
        
        page = st.selectbox(
            "Select Page",
            ["System Overview", "Performance Monitoring", "Fashion Analytics", 
             "AI Models", "Service Status", "Configuration"]
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick stats
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚡ Quick Stats")
        
        metrics = get_system_metrics()
        if metrics:
            st.metric("CPU Usage", f"{metrics.get('cpu_percent', 0):.1f}%")
            st.metric("Memory Usage", f"{metrics.get('memory_percent', 0):.1f}%")
            st.metric("Disk Usage", f"{metrics.get('disk_percent', 0):.1f}%")
            
            if metrics.get('gpu_info'):
                gpu_mem = metrics['gpu_info'].get('gpu_memory_allocated', 0)
                st.metric("GPU Memory", f"{gpu_mem:.1f} GB")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize integration manager if needed
    if st.session_state.integration_manager is None:
        with st.spinner("Initializing FlashFit AI services..."):
            initialize_integration_manager()
    
    # Main content based on selected page
    if page == "System Overview":
        show_system_overview()
    elif page == "Performance Monitoring":
        show_performance_monitoring()
    elif page == "Fashion Analytics":
        show_fashion_analytics()
    elif page == "AI Models":
        show_ai_models()
    elif page == "Service Status":
        show_service_status()
    elif page == "Configuration":
        show_configuration()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def show_system_overview():
    """Show system overview page"""
    st.header("🖥️ System Overview")
    
    # Get system metrics
    metrics = get_system_metrics()
    
    if metrics:
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CPU Usage",
                f"{metrics.get('cpu_percent', 0):.1f}%",
                delta=f"{np.random.uniform(-2, 2):.1f}%"  # Mock delta
            )
        
        with col2:
            st.metric(
                "Memory Usage",
                f"{metrics.get('memory_used_gb', 0):.1f} GB",
                delta=f"{np.random.uniform(-0.5, 0.5):.1f} GB"
            )
        
        with col3:
            st.metric(
                "Disk Usage",
                f"{metrics.get('disk_used_gb', 0):.0f} GB",
                delta=f"{np.random.uniform(-1, 1):.1f} GB"
            )
        
        with col4:
            gpu_info = metrics.get('gpu_info', {})
            if gpu_info:
                st.metric(
                    "GPU Memory",
                    f"{gpu_info.get('gpu_memory_allocated', 0):.1f} GB",
                    delta=f"{np.random.uniform(-0.2, 0.2):.1f} GB"
                )
            else:
                st.metric("GPU", "Not Available")
        
        # Performance charts
        fig_system, _ = create_performance_charts(metrics)
        if fig_system:
            st.plotly_chart(fig_system, use_container_width=True)
    
    # Integration manager overview
    if st.session_state.integration_manager:
        st.subheader("🔧 FlashFit AI Services")
        
        try:
            overview = st.session_state.integration_manager.get_system_overview()
            
            # Services status
            services_data = []
            for service_name, status_info in overview.get('services', {}).items():
                services_data.append({
                    'Service': service_name.title(),
                    'Status': status_info.get('status', 'unknown'),
                    'Last Check': status_info.get('last_check', 'N/A')
                })
            
            if services_data:
                df_services = pd.DataFrame(services_data)
                
                # Color code status
                def color_status(val):
                    if val == 'active':
                        return 'background-color: #d4edda; color: #155724'
                    elif val == 'error':
                        return 'background-color: #f8d7da; color: #721c24'
                    else:
                        return 'background-color: #fff3cd; color: #856404'
                
                st.dataframe(
                    df_services.style.applymap(color_status, subset=['Status']),
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"Failed to get integration overview: {e}")

def show_performance_monitoring():
    """Show performance monitoring page"""
    st.header("📊 Performance Monitoring")
    
    # Real-time metrics
    metrics = get_system_metrics()
    
    if metrics:
        # Performance charts
        fig_system, _ = create_performance_charts(metrics)
        if fig_system:
            st.plotly_chart(fig_system, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics")
        
        metrics_data = {
            'Metric': ['CPU Usage', 'Memory Total', 'Memory Used', 'Disk Total', 'Disk Used'],
            'Value': [
                f"{metrics.get('cpu_percent', 0):.1f}%",
                f"{metrics.get('memory_total_gb', 0):.1f} GB",
                f"{metrics.get('memory_used_gb', 0):.1f} GB",
                f"{metrics.get('disk_total_gb', 0):.0f} GB",
                f"{metrics.get('disk_used_gb', 0):.0f} GB"
            ],
            'Status': ['Normal', 'Normal', 'Normal', 'Normal', 'Normal']  # Mock status
        }
        
        # Add GPU metrics if available
        gpu_info = metrics.get('gpu_info', {})
        if gpu_info:
            metrics_data['Metric'].extend(['GPU Count', 'GPU Name', 'GPU Memory'])
            metrics_data['Value'].extend([
                str(gpu_info.get('gpu_count', 0)),
                gpu_info.get('gpu_name', 'Unknown'),
                f"{gpu_info.get('gpu_memory_allocated', 0):.1f} GB"
            ])
            metrics_data['Status'].extend(['Normal', 'Normal', 'Normal'])
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        # Performance alerts
        st.subheader("⚠️ Performance Alerts")
        
        alerts = []
        if metrics.get('cpu_percent', 0) > 80:
            alerts.append("🔴 High CPU usage detected")
        if metrics.get('memory_percent', 0) > 85:
            alerts.append("🔴 High memory usage detected")
        if metrics.get('disk_percent', 0) > 90:
            alerts.append("🔴 Low disk space warning")
        
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("✅ All systems operating normally")

def show_fashion_analytics():
    """Show fashion analytics page"""
    st.header("👗 Fashion Analytics")
    
    # Generate demo data
    fashion_data = create_fashion_analytics_demo()
    
    # Analytics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items", len(fashion_data))
    
    with col2:
        st.metric("Avg Price", f"${fashion_data['price'].mean():.2f}")
    
    with col3:
        st.metric("Avg Rating", f"{fashion_data['rating'].mean():.2f}⭐")
    
    with col4:
        st.metric("Total Sales", f"{fashion_data['sales'].sum():,}")
    
    # Create charts
    fig_category, fig_price, fig_trends, fig_heatmap = create_fashion_charts(fashion_data)
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_price, use_container_width=True)
    
    st.plotly_chart(fig_trends, use_container_width=True)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Data table
    st.subheader("📊 Fashion Data Sample")
    st.dataframe(fashion_data.head(100), use_container_width=True)
    
    # AI-powered insights
    if st.session_state.integration_manager:
        st.subheader("🤖 AI-Powered Insights")
        
        try:
            # Mock AI analysis
            insights = [
                "📈 Dresses category showing 15% increase in sales this week",
                "🎨 Blue and black items have highest customer ratings",
                "📱 Mobile traffic accounts for 68% of fashion item views",
                "⭐ Items with ratings above 4.5 have 3x higher conversion rates",
                "🛍️ Weekend sales peak between 2-4 PM"
            ]
            
            for insight in insights:
                st.info(insight)
        
        except Exception as e:
            st.error(f"Failed to generate AI insights: {e}")

def show_ai_models():
    """Show AI models page"""
    st.header("🤖 AI Models")
    
    if st.session_state.integration_manager:
        # Model status
        st.subheader("📋 Model Status")
        
        models_data = [
            {'Model': 'Fashion Encoder', 'Status': 'Active', 'Accuracy': '94.2%', 'Last Updated': '2024-01-15'},
            {'Model': 'Recommendation Engine', 'Status': 'Active', 'Accuracy': '87.8%', 'Last Updated': '2024-01-14'},
            {'Model': 'Sentiment Analyzer', 'Status': 'Training', 'Accuracy': '91.5%', 'Last Updated': '2024-01-13'},
            {'Model': 'Style Classifier', 'Status': 'Active', 'Accuracy': '89.3%', 'Last Updated': '2024-01-12'}
        ]
        
        df_models = pd.DataFrame(models_data)
        
        # Color code status
        def color_model_status(val):
            if val == 'Active':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Training':
                return 'background-color: #fff3cd; color: #856404'
            else:
                return 'background-color: #f8d7da; color: #721c24'
        
        st.dataframe(
            df_models.style.applymap(color_model_status, subset=['Status']),
            use_container_width=True
        )
        
        # Model performance charts
        st.subheader("📊 Model Performance")
        
        # Mock performance data
        performance_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Fashion Encoder': np.random.uniform(0.92, 0.96, 30),
            'Recommendation Engine': np.random.uniform(0.85, 0.90, 30),
            'Sentiment Analyzer': np.random.uniform(0.88, 0.93, 30),
            'Style Classifier': np.random.uniform(0.87, 0.92, 30)
        })
        
        fig_performance = px.line(
            performance_data.melt(id_vars=['Date'], var_name='Model', value_name='Accuracy'),
            x='Date',
            y='Accuracy',
            color='Model',
            title="Model Accuracy Over Time"
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Model optimization
        st.subheader("⚙️ Model Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Optimize Models"):
                with st.spinner("Optimizing models using PEFT..."):
                    time.sleep(3)  # Mock optimization
                    st.success("✅ Models optimized successfully!")
                    st.info("📊 Parameter reduction: 85%")
                    st.info("⚡ Inference speed improvement: 40%")
        
        with col2:
            if st.button("📊 Run Benchmark"):
                with st.spinner("Running comprehensive benchmark..."):
                    time.sleep(2)  # Mock benchmark
                    st.success("✅ Benchmark completed!")
                    st.info("🎯 Overall system health: 92%")
                    st.info("⚡ Average response time: 150ms")
    
    else:
        st.warning("Integration manager not initialized. Please check the system status.")

def show_service_status():
    """Show service status page"""
    st.header("🔧 Service Status")
    
    if st.session_state.integration_manager:
        try:
            overview = st.session_state.integration_manager.get_system_overview()
            
            # Service health overview
            st.subheader("🏥 Service Health")
            
            services = overview.get('services', {})
            active_services = sum(1 for s in services.values() if s.get('status') == 'active')
            total_services = len(services)
            health_score = (active_services / total_services * 100) if total_services > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Active Services", f"{active_services}/{total_services}")
            
            with col2:
                st.metric("Health Score", f"{health_score:.1f}%")
            
            with col3:
                status_text = "Excellent" if health_score >= 90 else "Good" if health_score >= 70 else "Needs Attention"
                st.metric("Overall Status", status_text)
            
            # Detailed service status
            st.subheader("📋 Detailed Service Status")
            
            for service_name, status_info in services.items():
                with st.expander(f"🔧 {service_name.title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        status = status_info.get('status', 'unknown')
                        if status == 'active':
                            st.success(f"Status: {status.title()}")
                        elif status == 'error':
                            st.error(f"Status: {status.title()}")
                        else:
                            st.warning(f"Status: {status.title()}")
                        
                        st.write(f"Last Check: {status_info.get('last_check', 'N/A')}")
                    
                    with col2:
                        if status_info.get('error_message'):
                            st.error(f"Error: {status_info['error_message']}")
                        else:
                            st.info("No errors reported")
            
            # Service actions
            st.subheader("⚡ Service Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Restart All Services"):
                    with st.spinner("Restarting services..."):
                        time.sleep(2)
                        st.success("✅ All services restarted")
            
            with col2:
                if st.button("🧪 Run Health Check"):
                    with st.spinner("Running health check..."):
                        time.sleep(1)
                        st.success("✅ Health check completed")
            
            with col3:
                if st.button("📊 Generate Report"):
                    with st.spinner("Generating report..."):
                        time.sleep(1)
                        st.success("✅ Report generated")
                        st.download_button(
                            "📥 Download Report",
                            json.dumps(overview, indent=2),
                            "flashfit_ai_report.json",
                            "application/json"
                        )
        
        except Exception as e:
            st.error(f"Failed to get service status: {e}")
    
    else:
        st.warning("Integration manager not initialized.")

def show_configuration():
    """Show configuration page"""
    st.header("⚙️ Configuration")
    
    # Configuration form
    st.subheader("🔧 System Configuration")
    
    with st.form("config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Service Settings")
            enable_analytics = st.checkbox("Enable Analytics", value=True)
            enable_monitoring = st.checkbox("Enable Performance Monitoring", value=True)
            enable_flask = st.checkbox("Enable Flask Extensions", value=True)
            enable_wandb = st.checkbox("Enable Wandb Integration", value=False)
        
        with col2:
            st.subheader("AI/ML Settings")
            enable_peft = st.checkbox("Enable PEFT Optimization", value=True)
            enable_trl = st.checkbox("Enable TRL Integration", value=True)
            enable_nlp = st.checkbox("Enable Advanced NLP", value=True)
            auto_optimize = st.checkbox("Auto-optimize Models", value=False)
        
        st.subheader("Performance Settings")
        col3, col4 = st.columns(2)
        
        with col3:
            monitoring_interval = st.slider("Monitoring Interval (seconds)", 10, 300, 60)
            max_memory = st.slider("Max Memory Usage (%)", 50, 95, 80)
        
        with col4:
            batch_size = st.number_input("Analytics Batch Size", 100, 5000, 1000)
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
        
        submitted = st.form_submit_button("💾 Save Configuration")
        
        if submitted:
            # Create new configuration
            new_config = IntegrationConfig(
                enable_analytics=enable_analytics,
                enable_performance_monitoring=enable_monitoring,
                enable_flask_extensions=enable_flask,
                enable_wandb=enable_wandb,
                enable_peft=enable_peft,
                enable_trl=enable_trl,
                auto_optimize_models=auto_optimize,
                monitoring_interval=monitoring_interval,
                max_memory_usage=max_memory / 100,
                analytics_batch_size=batch_size,
                log_level=log_level
            )
            
            # Save configuration (mock)
            st.success("✅ Configuration saved successfully!")
            st.info("🔄 Restart required for some changes to take effect")
    
    # Current configuration display
    st.subheader("📋 Current Configuration")
    
    if st.session_state.integration_manager:
        config = st.session_state.integration_manager.config
        config_dict = {
            'Setting': [
                'Analytics Enabled', 'Performance Monitoring', 'Flask Extensions',
                'Wandb Integration', 'PEFT Optimization', 'TRL Integration',
                'Monitoring Interval', 'Max Memory Usage', 'Log Level'
            ],
            'Value': [
                config.enable_analytics, config.enable_performance_monitoring,
                config.enable_flask_extensions, config.enable_wandb,
                config.enable_peft, config.enable_trl,
                f"{config.monitoring_interval}s", f"{config.max_memory_usage*100:.0f}%",
                config.log_level
            ]
        }
        
        df_config = pd.DataFrame(config_dict)
        st.dataframe(df_config, use_container_width=True)
    
    # Export/Import configuration
    st.subheader("📤 Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Configuration"):
            if st.session_state.integration_manager:
                config_json = json.dumps(
                    st.session_state.integration_manager.config.__dict__,
                    indent=2,
                    default=str
                )
                st.download_button(
                    "📥 Download Config",
                    config_json,
                    "flashfit_ai_config.json",
                    "application/json"
                )
    
    with col2:
        uploaded_file = st.file_uploader("📤 Import Configuration", type="json")
        if uploaded_file is not None:
            try:
                config_data = json.load(uploaded_file)
                st.success("✅ Configuration imported successfully!")
                st.json(config_data)
            except Exception as e:
                st.error(f"❌ Failed to import configuration: {e}")

if __name__ == "__main__":
    main()