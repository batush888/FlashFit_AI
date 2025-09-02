import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import psutil
from rich.console import Console
import numpy as np
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="FlashFit AI Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize console for rich output
console = Console()

class FlashFitDashboard:
    def __init__(self):
        self.data_dir = Path("data")
        self.monitoring_dir = Path("data/monitoring")
        
    def load_data(self):
        """Load various data files for analysis"""
        data = {}
        
        # Load fashion items
        try:
            with open(self.data_dir / "fashion_items.json", "r") as f:
                data['fashion_items'] = json.load(f)
        except FileNotFoundError:
            data['fashion_items'] = []
            
        # Load feedback data
        try:
            with open(self.data_dir / "feedback.json", "r") as f:
                data['feedback'] = json.load(f)
        except FileNotFoundError:
            data['feedback'] = []
            
        # Load outfit history
        try:
            with open(self.data_dir / "outfit_history.json", "r") as f:
                data['outfit_history'] = json.load(f)
        except FileNotFoundError:
            data['outfit_history'] = []
            
        # Load outfit ratings
        try:
            with open(self.data_dir / "outfit_ratings.json", "r") as f:
                data['outfit_ratings'] = json.load(f)
        except FileNotFoundError:
            data['outfit_ratings'] = []
            
        return data
    
    def get_system_metrics(self):
        """Get current system performance metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters(),
            'process_count': len(psutil.pids())
        }
    
    def render_header(self):
        """Render dashboard header"""
        st.title("🎯 FlashFit AI Analytics Dashboard")
        st.markdown("""
        Welcome to the FlashFit AI monitoring and analytics dashboard. 
        This dashboard provides insights into AI model performance, user interactions, and system metrics.
        """)
        
    def render_system_metrics(self):
        """Render system performance metrics"""
        st.header("🖥️ System Performance")
        
        metrics = self.get_system_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="CPU Usage",
                value=f"{metrics['cpu_percent']:.1f}%",
                delta=None
            )
            
        with col2:
            st.metric(
                label="Memory Usage",
                value=f"{metrics['memory_percent']:.1f}%",
                delta=None
            )
            
        with col3:
            st.metric(
                label="Disk Usage",
                value=f"{metrics['disk_usage']:.1f}%",
                delta=None
            )
            
        with col4:
            st.metric(
                label="Active Processes",
                value=metrics['process_count'],
                delta=None
            )
    
    def render_ai_model_stats(self, data):
        """Render AI model performance statistics"""
        st.header("🤖 AI Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fashion Items Analysis")
            if data['fashion_items']:
                items_df = pd.DataFrame(data['fashion_items'])
                
                # Category distribution
                if 'category' in items_df.columns:
                    category_counts = items_df['category'].value_counts()
                    fig = px.pie(values=category_counts.values, names=category_counts.index,
                               title="Fashion Items by Category")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No category data available")
            else:
                st.info("No fashion items data available")
        
        with col2:
            st.subheader("User Feedback Analysis")
            if data['feedback']:
                feedback_df = pd.DataFrame(data['feedback'])
                
                # Rating distribution
                if 'rating' in feedback_df.columns:
                    rating_counts = feedback_df['rating'].value_counts().sort_index()
                    fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                               title="User Rating Distribution")
                    fig.update_xaxis(title="Rating")
                    fig.update_yaxis(title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No rating data available")
            else:
                st.info("No feedback data available")
    
    def render_recommendation_analytics(self, data):
        """Render recommendation system analytics"""
        st.header("📊 Recommendation Analytics")
        
        if data['outfit_history']:
            history_df = pd.DataFrame(data['outfit_history'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Outfit Generation Trends")
                if 'timestamp' in history_df.columns:
                    # Convert timestamp to datetime
                    history_df['date'] = pd.to_datetime(history_df['timestamp'], errors='coerce')
                    daily_counts = history_df.groupby(history_df['date'].dt.date).size()
                    
                    fig = px.line(x=daily_counts.index, y=daily_counts.values,
                                title="Daily Outfit Generations")
                    fig.update_xaxis(title="Date")
                    fig.update_yaxis(title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No timestamp data available")
            
            with col2:
                st.subheader("Popular Occasions")
                if 'occasion' in history_df.columns:
                    occasion_counts = history_df['occasion'].value_counts()
                    fig = px.bar(x=occasion_counts.values, y=occasion_counts.index,
                               orientation='h', title="Most Requested Occasions")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No occasion data available")
        else:
            st.info("No outfit history data available")
    
    def render_data_insights(self, data):
        """Render data insights and statistics"""
        st.header("📈 Data Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Fashion Items",
                value=len(data['fashion_items']),
                delta=None
            )
        
        with col2:
            st.metric(
                label="User Feedback Count",
                value=len(data['feedback']),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Outfit Generations",
                value=len(data['outfit_history']),
                delta=None
            )
        
        # Average rating if available
        if data['feedback']:
            feedback_df = pd.DataFrame(data['feedback'])
            if 'rating' in feedback_df.columns:
                avg_rating = feedback_df['rating'].mean()
                st.metric(
                    label="Average User Rating",
                    value=f"{avg_rating:.2f}/5.0",
                    delta=None
                )
    
    def render_model_health(self):
        """Render AI model health status"""
        st.header("🏥 Model Health Status")
        
        # Check if model files exist
        model_files = {
            "CLIP Fashion Index": self.data_dir / "clip_fashion.index",
            "BLIP Fashion Index": self.data_dir / "blip_fashion.index",
            "Fashion Items DB": self.data_dir / "fashion_items.json",
            "Fusion Model": self.data_dir / "simple_fusion_model.pkl"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Files Status")
            for model_name, file_path in model_files.items():
                if file_path.exists():
                    st.success(f"✅ {model_name}: Available")
                    # Show file size
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    st.caption(f"Size: {size_mb:.2f} MB")
                else:
                    st.error(f"❌ {model_name}: Missing")
        
        with col2:
            st.subheader("Data Quality Metrics")
            data = self.load_data()
            
            # Check data completeness
            if data['fashion_items']:
                items_df = pd.DataFrame(data['fashion_items'])
                completeness = {}
                for col in items_df.columns:
                    non_null_pct = (items_df[col].notna().sum() / len(items_df)) * 100
                    completeness[col] = non_null_pct
                
                completeness_df = pd.DataFrame(list(completeness.items()), 
                                             columns=['Field', 'Completeness %'])
                st.dataframe(completeness_df, use_container_width=True)
            else:
                st.info("No fashion items data to analyze")
    
    def run(self):
        """Main dashboard runner"""
        self.render_header()
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a section",
            ["Overview", "System Metrics", "AI Models", "Recommendations", "Model Health"]
        )
        
        # Load data once
        data = self.load_data()
        
        if page == "Overview":
            self.render_data_insights(data)
            self.render_system_metrics()
        elif page == "System Metrics":
            self.render_system_metrics()
        elif page == "AI Models":
            self.render_ai_model_stats(data)
        elif page == "Recommendations":
            self.render_recommendation_analytics(data)
        elif page == "Model Health":
            self.render_model_health()
        
        # Auto-refresh option
        if st.sidebar.checkbox("Auto-refresh (30s)"):
            st.rerun()

if __name__ == "__main__":
    dashboard = FlashFitDashboard()
    dashboard.run()