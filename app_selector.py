import streamlit as st

# Page configuration
st.set_page_config(
    page_title="🍽️ NYC Restaurant Health Apps - Version Selector",
    page_icon="🍽️",
    layout="wide"
)

st.markdown("""
<style>
    .app-selector {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .version-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .version-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .status-running { color: #28a745; font-weight: bold; }
    .status-available { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-selector">
    <h1>🍽️ NYC Restaurant Health Prediction Apps</h1>
    <p>Choose which version you'd like to run</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## 🚀 Available App Versions")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="version-card">
        <h3>📱 Original App</h3>
        <p><strong>File:</strong> streamlit_app.py</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Single-user interface</li>
            <li>Grade prediction</li>
            <li>Analytics dashboard</li>
            <li>Fully debugged & optimized</li>
        </ul>
        <p><strong>Status:</strong> <span class="status-available">Ready to Run</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch Original App", key="original", use_container_width=True):
        st.code("streamlit run streamlit_app.py --server.port 8501", language="bash")
        st.success("Copy and run the command above in your terminal!")

with col2:
    st.markdown("""
    <div class="version-card">
        <h3>👥 Previous Stakeholder App</h3>
        <p><strong>File:</strong> stakeholder_app.py</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Multi-stakeholder interface</li>
            <li>Role-based dashboards</li>
            <li>Restaurant owner tools</li>
            <li>Your previous work preserved</li>
        </ul>
        <p><strong>Status:</strong> <span class="status-running">Currently Running on Port 8504</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🌐 Open Previous Stakeholder App", key="previous", use_container_width=True):
        st.markdown("**App is running at:** http://localhost:8504")
        st.balloons()

with col3:
    st.markdown("""
    <div class="version-card">
        <h3>⭐ Enhanced Multi-Stakeholder App</h3>
        <p><strong>File:</strong> multi_stakeholder_app.py</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Advanced stakeholder interface</li>
            <li>Health inspector dashboard</li>
            <li>Public search interface</li>
            <li>Enhanced visualizations</li>
        </ul>
        <p><strong>Status:</strong> <span class="status-running">Currently Running on Port 8503</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✨ Open Enhanced App", key="enhanced", use_container_width=True):
        st.markdown("**App is running at:** http://localhost:8503")
        st.balloons()

st.markdown("---")

st.markdown("## 🔧 Quick Commands")

st.markdown("### Terminal Commands to Launch Each Version:")

st.code("""
# Original App (Port 8501)
streamlit run streamlit_app.py --server.port 8501

# Previous Stakeholder App (Port 8504) - Currently Running
streamlit run stakeholder_app.py --server.port 8504

# Enhanced Multi-Stakeholder App (Port 8503) - Currently Running  
streamlit run multi_stakeholder_app.py --server.port 8503

# This Version Selector (Port 8505)
streamlit run app_selector.py --server.port 8505
""", language="bash")

st.markdown("### 📊 Current Status:")

status_data = {
    "App Version": ["Original App", "Previous Stakeholder", "Enhanced Multi-Stakeholder", "Version Selector"],
    "File": ["streamlit_app.py", "stakeholder_app.py", "multi_stakeholder_app.py", "app_selector.py"],
    "Port": ["8501", "8504", "8503", "8505"],
    "Status": ["Available", "🟢 Running", "🟢 Running", "🟢 Current"]
}

import pandas as pd
st.dataframe(pd.DataFrame(status_data), use_container_width=True)

st.markdown("---")

st.markdown("## 📝 Version Comparison")

comparison_data = {
    "Feature": [
        "Grade Prediction",
        "Analytics Dashboard", 
        "Multi-Stakeholder Support",
        "Restaurant Owner Interface",
        "Health Inspector Tools",
        "Public Search",
        "Advanced Visualizations",
        "Error Handling",
        "Performance Optimization",
        "Modern UI Design"
    ],
    "Original App": ["✅", "✅", "❌", "✅", "❌", "❌", "✅", "✅", "✅", "✅"],
    "Previous Stakeholder": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅"],
    "Enhanced Multi-Stakeholder": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅"]
}

st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

st.markdown("---")
st.markdown("### 💡 **Recommendation:** All versions are preserved and working. Choose based on your needs!")
