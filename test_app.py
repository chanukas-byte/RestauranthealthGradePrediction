import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.title("🧪 Analytics Test Page")

# Load data
@st.cache_data
def load_test_data():
    try:
        df = pd.read_csv('data/cleaned_restaurant_dataset.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_test_data()

if df is not None:
    st.success(f"✅ Data loaded successfully: {len(df):,} records")
    
    # Test 1: Simple grade distribution
    st.subheader("Test 1: Grade Distribution")
    grade_counts = df['Grade'].value_counts().sort_index()
    st.write("Grade counts:", grade_counts.to_dict())
    
    # Create simple bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(grade_counts.index),
        y=list(grade_counts.values),
        text=list(grade_counts.values),
        textposition='outside'
    ))
    fig.update_layout(title="Grade Distribution Test", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Test 2: Simple metrics
    st.subheader("Test 2: Basic Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        grade_a_pct = (df['Grade'] == 'A').mean() * 100
        st.metric("Grade A %", f"{grade_a_pct:.1f}%")
    with col3:
        if 'Critical Flag' in df.columns:
            critical_pct = (df['Critical Flag'] == 'Critical').mean() * 100
            st.metric("Critical %", f"{critical_pct:.1f}%")
    
    # Test 3: Sample data view
    st.subheader("Test 3: Sample Data")
    st.dataframe(df.head(10))
    
else:
    st.error("❌ Data loading failed")
