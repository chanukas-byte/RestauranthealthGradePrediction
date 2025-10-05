import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🍽️ NYC Restaurant Health Grade Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling to match your vision
st.markdown("""
<style>
    /* Global Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Role Selection Cards */
    .role-selector {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .role-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        flex: 1;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .role-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    .role-card.active {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #e9ecff 100%);
    }
    
    .role-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .role-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .role-description {
        color: #6c757d;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    /* Portal Headers */
    .portal-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .portal-header h2 {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .portal-header p {
        opacity: 0.9;
        margin: 0;
    }
    
    /* Form Styling */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stSlider > div > div > div {
        background-color: #667eea;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Results Card */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    .grade-display {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .grade-letter {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .grade-a { color: #28a745; }
    .grade-b { color: #ffc107; }
    .grade-c { color: #fd7e14; }
    .grade-z { color: #dc3545; }
    .grade-n { color: #6c757d; }
    .grade-p { color: #17a2b8; }
    
    .grade-meaning {
        font-size: 1.2rem;
        color: #495057;
        font-weight: 500;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Metric Cards */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Alert Boxes */
    .alert-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Load model and data functions
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_path = Path(__file__).parent / "models" / "restaurant_balanced_model.joblib"
        if model_path.exists():
            return joblib.load(model_path)
        else:
            # Alternative paths
            alternative_paths = [
                "models/restaurant_balanced_model.joblib",
                "restaurant_balanced_model.joblib"
            ]
            for alt_path in alternative_paths:
                try:
                    return joblib.load(alt_path)
                except:
                    continue
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_dataset():
    """Load the restaurant dataset"""
    try:
        data_path = Path(__file__).parent / "data" / "cleaned_restaurant_dataset.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            if 'Inspection Date' in df.columns:
                df['Inspection Date'] = pd.to_datetime(df['Inspection Date'], errors='coerce')
            return df
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def predict_grade(model_objects, input_data):
    """Make grade prediction"""
    try:
        if model_objects is None:
            return None, "Model not available"
        
        model = model_objects['model']
        scaler = model_objects['scaler']
        label_encoders = model_objects['label_encoders']
        features = model_objects['features']
        
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        categorical_features = ['Inspection Type', 'Critical Flag', 'Violation Code']
        for feature in categorical_features:
            if feature in label_encoders:
                try:
                    value_to_encode = str(input_data[feature])
                    input_df[feature] = label_encoders[feature].transform([value_to_encode])
                except ValueError:
                    input_df[feature] = [0]
        
        # Scale features
        X = input_df[features].copy()
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Get grade name
        grade_encoder = label_encoders['GRADE']
        predicted_grade = grade_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        grade_probs = {}
        for i, grade in enumerate(grade_encoder.classes_):
            grade_probs[grade] = probabilities[i]
        
        return predicted_grade, grade_probs
        
    except Exception as e:
        return None, f"Error making prediction: {str(e)}"

def render_main_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ NYC Restaurant Health Grade Predictor</h1>
        <p>Advanced ML-powered health grade prediction system</p>
    </div>
    """, unsafe_allow_html=True)

def render_role_selection():
    """Render modern role selection interface"""
    st.markdown("### Select Your Role")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👤 Customer", key="customer_btn", use_container_width=True):
            st.session_state.user_role = "customer"
            st.rerun()
    
    with col2:
        if st.button("🏪 Restaurant Owner", key="owner_btn", use_container_width=True):
            st.session_state.user_role = "restaurant_owner"
            st.rerun()
    
    with col3:
        if st.button("🏛️ Health Authority", key="authority_btn", use_container_width=True):
            st.session_state.user_role = "health_authority"
            st.rerun()

def render_customer_portal():
    """Customer portal for checking restaurant grades"""
    st.markdown("""
    <div class="portal-header">
        <h2>👤 Customer Portal</h2>
        <p>Check the predicted health grade for a restaurant inspection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_objects = load_model()
    
    if model_objects is None:
        st.error("⚠️ Prediction model not available")
        return
    
    # Form inputs
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Inspection Details")
        
        inspection_type = st.selectbox(
            "Inspection Type",
            [
                'Administrative Miscellaneous / Initial Inspection',
                'Cycle Inspection / Initial Inspection',
                'Cycle Inspection / Re-inspection',
                'Pre-permit (Operational) / Initial Inspection',
                'Pre-permit (Operational) / Re-inspection'
            ]
        )
        
        critical_flag = st.selectbox(
            "Critical Flag",
            ['Critical', 'Not Critical', 'Not Applicable']
        )
        
        violation_code = st.selectbox(
            "Violation Code",
            ['02A', '02B', '02G', '04A', '04L', '04M', '06D', '08A', '09B', '10F', '10B', 'Null']
        )
    
    with col2:
        st.markdown("#### Assessment Details")
        
        score = st.slider("Score", min_value=0, max_value=100, value=10)
        
        inspection_date = st.date_input(
            "Inspection Date",
            value=datetime.date.today()
        )
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔮 Predict Grade", type="primary", use_container_width=True):
        # Calculate date features
        inspection_year = inspection_date.year
        inspection_month = inspection_date.month
        inspection_day_of_week = inspection_date.weekday()
        
        input_data = {
            'Inspection Type': inspection_type,
            'Critical Flag': critical_flag,
            'Violation Code': violation_code,
            'Inspection Score': score,
            'inspection_year': inspection_year,
            'inspection_month': inspection_month,
            'inspection_day_of_week': inspection_day_of_week
        }
        
        predicted_grade, grade_probs = predict_grade(model_objects, input_data)
        
        if predicted_grade:
            # Display result
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="grade-display">
                    <div class="grade-letter grade-{predicted_grade.lower()}">{predicted_grade}</div>
                    <div class="grade-meaning">Predicted Grade</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if isinstance(grade_probs, dict):
                    grades = list(grade_probs.keys())
                    probs = [grade_probs[grade] * 100 for grade in grades]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=grades, 
                            y=probs,
                            marker_color='#667eea',
                            text=[f'{p:.1f}%' for p in probs],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Grade Probability Distribution",
                        xaxis_title="Grade",
                        yaxis_title="Probability (%)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def render_restaurant_owner_portal():
    """Restaurant owner portal"""
    st.markdown("""
    <div class="portal-header">
        <h2>🏪 Restaurant Owner Portal</h2>
        <p>As a restaurant owner, enter your inspection details to predict the grade and see improvement guidance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_objects = load_model()
    
    if model_objects is None:
        st.error("⚠️ Prediction model not available")
        return
    
    # Restaurant details
    st.markdown("#### Restaurant Information")
    restaurant_name = st.text_input("Restaurant Name", placeholder="e.g., Pizza Place in Brooklyn")
    
    # Form inputs in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Inspection Details")
        
        inspection_type = st.selectbox(
            "Inspection Type",
            [
                'Administrative Miscellaneous / Initial Inspection',
                'Cycle Inspection / Initial Inspection',
                'Cycle Inspection / Re-inspection',
                'Pre-permit (Operational) / Initial Inspection',
                'Pre-permit (Operational) / Re-inspection'
            ]
        )
        
        critical_flag = st.selectbox(
            "Critical Flag",
            ['Critical', 'Not Critical', 'Not Applicable']
        )
        
        violation_code = st.selectbox(
            "Violation Code",
            ['02A', '02B', '02G', '04A', '04L', '04M', '06D', '08A', '09B', '10F', '10B', 'Null']
        )
    
    with col2:
        st.markdown("#### Self-Assessment")
        
        score = st.slider("Self-audited Score", min_value=0, max_value=100, value=18)
        
        inspection_date = st.date_input(
            "Scheduled Inspection Date",
            value=datetime.date.today()
        )
    
    # Prediction
    st.markdown("---")
    
    if st.button("🎯 Predict as Owner", type="primary", use_container_width=True):
        # Calculate date features
        inspection_year = inspection_date.year
        inspection_month = inspection_date.month
        inspection_day_of_week = inspection_date.weekday()
        
        input_data = {
            'Inspection Type': inspection_type,
            'Critical Flag': critical_flag,
            'Violation Code': violation_code,
            'Inspection Score': score,
            'inspection_year': inspection_year,
            'inspection_month': inspection_month,
            'inspection_day_of_week': inspection_day_of_week
        }
        
        predicted_grade, grade_probs = predict_grade(model_objects, input_data)
        
        if predicted_grade:
            # Results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="grade-display">
                    <div class="grade-letter grade-{predicted_grade.lower()}">{predicted_grade}</div>
                    <div class="grade-meaning">Predicted Grade</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Grade interpretation
                grade_meanings = {
                    'A': ('Excellent (0-13 points)', 'alert-success'),
                    'B': ('Good (14-27 points)', 'alert-warning'),
                    'C': ('Fair (28+ points)', 'alert-warning'),
                    'N': ('Not Yet Graded', 'alert-warning'),
                    'Z': ('Closed for Violations', 'alert-danger'),
                    'P': ('Permit Pending', 'alert-warning')
                }
                
                if predicted_grade in grade_meanings:
                    meaning, alert_class = grade_meanings[predicted_grade]
                    st.markdown(f"""
                    <div class="{alert_class}">
                        <strong>Grade Meaning:</strong> {meaning}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if isinstance(grade_probs, dict):
                    grades = list(grade_probs.keys())
                    probs = [grade_probs[grade] * 100 for grade in grades]
                    
                    colors = ['#28a745' if g == predicted_grade else '#6c757d' for g in grades]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=grades, 
                            y=probs,
                            marker_color=colors,
                            text=[f'{p:.1f}%' for p in probs],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Grade Probability Distribution",
                        xaxis_title="Grade",
                        yaxis_title="Probability (%)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Improvement recommendations
            st.markdown("### 💡 Improvement Recommendations")
            
            recommendations = []
            
            if score > 27:
                recommendations.append("🚨 **Critical**: Score too high for good grade. Focus on major violations first")
            elif score > 13:
                recommendations.append("🎯 **Improve Score**: Target score ≤13 points for Grade A")
            
            if critical_flag == 'Critical':
                recommendations.append("⚠️ **Address Critical Violations**: These directly impact public health")
            
            if violation_code in ['02B', '04M']:
                recommendations.append("🌡️ **Critical Food Safety**: Address temperature control and pest issues immediately")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

def render_health_authority_portal():
    """Health authority portal"""
    st.markdown("""
    <div class="portal-header">
        <h2>🏛️ Health Authority Dashboard</h2>
        <p>Enter inspection details to simulate an inspection prediction and help prioritize efforts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model_objects = load_model()
    df = load_dataset()
    
    if model_objects is None:
        st.error("⚠️ Prediction model not available")
        return
    
    # Location and restaurant details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Location Details")
        
        location = st.selectbox(
            "Location / Borough",
            ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        )
        
        cuisine_type = st.selectbox(
            "Cuisine Type",
            ['American', 'Chinese', 'Italian', 'Mexican', 'Pizza', 'Japanese', 'Indian', 'Other']
        )
    
    with col2:
        st.markdown("#### Inspection Assessment")
        
        inspection_type = st.selectbox(
            "Inspection Type",
            [
                'Administrative Miscellaneous / Initial Inspection',
                'Cycle Inspection / Initial Inspection',
                'Cycle Inspection / Re-inspection',
                'Pre-permit (Operational) / Initial Inspection'
            ]
        )
        
        critical_flag = st.selectbox(
            "Critical Flag",
            ['Critical', 'Not Critical', 'Not Applicable']
        )
    
    violation_code = st.selectbox(
        "Violation Code",
        ['02A', '02B', '02G', '04A', '04L', '04M', '06D', '08A', '09B', '10F', '10B', 'Null']
    )
    
    score = st.slider("Inspection Score", min_value=0, max_value=100, value=32)
    
    # Analytics section
    if df is not None:
        st.markdown("### 📊 System Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_restaurants = df['Restaurant ID'].nunique()
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{total_restaurants:,}</div>
                <div class="metric-label">Total Restaurants</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_inspections = len(df)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{total_inspections:,}</div>
                <div class="metric-label">Total Inspections</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            critical_violations = len(df[df['Critical Flag'] == 'Critical'])
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{critical_violations:,}</div>
                <div class="metric-label">Critical Violations</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            grade_a_percent = len(df[df['Grade'] == 'A']) / len(df) * 100
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{grade_a_percent:.1f}%</div>
                <div class="metric-label">Grade A Rate</div>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Initialize session state
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    # Render main header
    render_main_header()
    
    # Back button for role selection
    if st.session_state.user_role is not None:
        if st.button("← Back to Role Selection", key="back_btn"):
            st.session_state.user_role = None
            st.rerun()
    
    # Route to appropriate portal
    if st.session_state.user_role == "customer":
        render_customer_portal()
    elif st.session_state.user_role == "restaurant_owner":
        render_restaurant_owner_portal()
    elif st.session_state.user_role == "health_authority":
        render_health_authority_portal()
    else:
        render_role_selection()

if __name__ == "__main__":
    main()
