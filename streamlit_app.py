import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🍽️ NYC Restaurant Health Grade Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        # Try to load the model from the correct path
        model_path = Path(__file__).parent / "models" / "restaurant_balanced_model.joblib"
        
        if model_path.exists():
            model_objects = joblib.load(model_path)
            return model_objects
        else:
            # Alternative paths for cloud deployment
            alternative_paths = [
                "models/restaurant_balanced_model.joblib",
                "restaurant_balanced_model.joblib",
                Path(__file__).parent / "restaurant_balanced_model.joblib"
            ]
            
            for alt_path in alternative_paths:
                try:
                    model_objects = joblib.load(alt_path)
                    return model_objects
                except:
                    continue
            
            st.error("⚠️ Model file not found")
            st.info("Creating a demo model for preview purposes...")
            return create_demo_model()
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Creating a demo model for preview purposes...")
        return create_demo_model()

def create_demo_model():
    """Create a demo model structure for cloud deployment when actual model isn't available"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    # Create dummy encoders and model
    demo_model = {
        'model': RandomForestClassifier(n_estimators=100, random_state=42),
        'model_type': 'demo',
        'scaler': StandardScaler(),
        'label_encoders': {
            'Inspection Type': LabelEncoder(),
            'Critical Flag': LabelEncoder(),
            'Violation Code': LabelEncoder()
        },
        'features': ['Inspection Type', 'Critical Flag', 'Violation Code', 'Inspection Score', 
                    'inspection_year', 'inspection_month', 'inspection_day_of_week'],
        'grade_encoder': LabelEncoder(),
        'grade_classes': np.array(['A', 'B', 'C', 'N', 'Z', 'P']),
        'class_weights': {0: 1.0, 1: 2.0, 2: 3.0, 3: 2.5, 4: 2.5, 5: 2.5}
    }
    
    # Fit dummy data with actual values from dataset
    demo_model['grade_encoder'].fit(['A', 'B', 'C', 'N', 'Z', 'P'])
    demo_model['label_encoders']['Inspection Type'].fit([
        'Cycle Inspection / Initial Inspection',
        'Cycle Inspection / Re-inspection', 
        'Pre-permit (Operational) / Initial Inspection',
        'Pre-permit (Operational) / Re-inspection',
        'Cycle Inspection / Reopening Inspection'
    ])
    demo_model['label_encoders']['Critical Flag'].fit(['Critical', 'Not Critical', 'Not Applicable'])
    demo_model['label_encoders']['Violation Code'].fit([
        'Null', '10F', '08A', '06D', '04L', '10B', '06C', '02G', '02B', '04N', 
        '04A', '04H', '08C', '06A', '09C', '04M', '06E', '10H', '06F'
    ])
    
    return demo_model

@st.cache_data
def load_dataset():
    """Load the actual dataset"""
    try:
        # Try to load the actual dataset
        data_path = Path(__file__).parent / "data" / "cleaned_restaurant_dataset.csv"
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            return df
        else:
            # Alternative paths for cloud deployment
            alternative_paths = [
                "data/cleaned_restaurant_dataset.csv",
                "cleaned_restaurant_dataset.csv",
                "sample_dataset.csv",
                Path(__file__).parent / "sample_dataset.csv"
            ]
            
            for alt_path in alternative_paths:
                try:
                    df = pd.read_csv(alt_path)
                    return df
                except:
                    continue
            
            st.warning("⚠️ Dataset not found. Using sample data for demo.")
            return generate_sample_data()
            
    except Exception as e:
        st.warning(f"Error loading dataset: {str(e)}. Using sample data.")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data that matches your actual dataset structure"""
    np.random.seed(42)
    
    # Sample data that matches your actual dataset statistics
    restaurants = [
        "PEKING GARDEN", "BAO BY KAYA", "WOODROW DINER", "JANE FAST FOOD", 
        "GREENHOUSE CAFE", "AGEHA JAPANESE FUSION", "THE STRAND BISTRO", "BABY BO'S CANTINA",
        "LAO JIE HOTPOT", "LOVELL'S GUIDING LIGHT", "YANKEES CLUBHOUSE KITCHEN", "THALASSA"
    ]
    
    locations = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    location_weights = [0.37, 0.27, 0.24, 0.09, 0.03]  # Based on actual distribution
    
    cuisine_types = [
        "American", "Chinese", "Coffee/Tea", "Pizza", "Italian", "Mexican",
        "Latin American", "Bakery Products/Desserts", "Caribbean", "Japanese", 
        "Chicken", "Spanish", "Donuts", "Sandwiches", "Hamburgers"
    ]
    
    inspection_types = [
        "Cycle Inspection / Initial Inspection",
        "Cycle Inspection / Re-inspection", 
        "Pre-permit (Operational) / Initial Inspection",
        "Pre-permit (Operational) / Re-inspection",
        "Cycle Inspection / Reopening Inspection"
    ]
    inspection_weights = [0.44, 0.36, 0.10, 0.07, 0.03]  # Based on actual distribution
    
    critical_flags = ["Critical", "Not Critical", "Not Applicable"]
    critical_weights = [0.52, 0.47, 0.01]  # Based on actual distribution
    
    violation_codes = [
        "Null", "10F", "08A", "06D", "04L", "10B", "06C", "02G", "02B", "04N", 
        "04A", "04H", "08C", "06A", "09C", "04M", "06E", "10H", "06F"
    ]
    
    # Grades with realistic distribution (A is most common)
    grades = ["A", "B", "C", "N", "Z", "P"]
    grade_weights = [0.66, 0.10, 0.05, 0.07, 0.05, 0.07]  # Based on actual distribution
    
    n_samples = 1000
    
    sample_data = []
    for i in range(n_samples):
        # Select grade first (influences score)
        grade = np.random.choice(grades, p=grade_weights)
        
        # Generate realistic inspection scores based on grade
        if grade == "A":
            score = max(0, np.random.gamma(2, 2))  # Lower scores for A (0-13 typical)
        elif grade == "B":
            score = np.random.gamma(3, 4) + 14  # Mid scores for B (14-27 typical)
        elif grade == "C":
            score = np.random.gamma(2, 8) + 28  # Higher scores for C (28+ typical)
        else:  # N, Z, P grades can have various scores
            score = np.random.gamma(2, 6)
        
        score = max(0, min(150, score))  # Clamp between 0-150
        
        # Select critical flag based on score
        if score > 25:
            critical_flag = np.random.choice(critical_flags, p=[0.7, 0.25, 0.05])
        else:
            critical_flag = np.random.choice(critical_flags, p=[0.3, 0.65, 0.05])
        
        sample_data.append({
            'Restaurant ID': 40000000 + i,
            'Restaurant Name': np.random.choice(restaurants),
            'Location': np.random.choice(locations, p=location_weights),
            'Cuisine Type': np.random.choice(cuisine_types),
            'Inspection Date': f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
            'Inspection Type': np.random.choice(inspection_types, p=inspection_weights),
            'Violation Code': np.random.choice(violation_codes),
            'Critical Flag': critical_flag,
            'Inspection Score': score,
            'Grade': grade
        })
    
    return pd.DataFrame(sample_data)

# Helper functions
def get_grade_emoji(grade):
    emojis = {'A': '🏆', 'B': '🥈', 'C': '🥉', 'N': '❓', 'Z': '📊', 'P': '📝'}
    return emojis.get(grade, '❓')

def get_grade_description(grade):
    descriptions = {
        'A': 'Excellent - Restaurant meets all health requirements with minimal violations.',
        'B': 'Good - Restaurant has some violations that need to be addressed promptly.',
        'C': 'Fair - Restaurant has significant violations requiring immediate attention.',
        'N': 'Not Yet Graded - The restaurant has not received an official grade yet.',
        'Z': 'Grade Pending - The restaurant is awaiting grade assignment from recent inspection.',
        'P': 'Grade Pending - The restaurant has a pending inspection grade issuance.'
    }
    return descriptions.get(grade, 'Unknown grade')

def get_color_for_grade(grade):
    colors = {'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545', 'N': '#6c757d', 'Z': '#17a2b8', 'P': '#6610f2'}
    return colors.get(grade, '#6c757d')

def get_score_range_for_grade(grade):
    ranges = {
        'A': '0-13 points',
        'B': '14-27 points', 
        'C': '28+ points',
        'N': 'Not applicable',
        'Z': 'Pending assessment',
        'P': 'Pending assessment'
    }
    return ranges.get(grade, 'Unknown range')
@st.cache_data
def load_sample_data():
    """Load sample data for the application"""
    data_file = Path("data/cleaned_restaurant_dataset.csv")
    
    if not data_file.exists():
        # Create minimal sample data if file doesn't exist
        sample_data = {
            'Restaurant Name': ['Sample Restaurant A', 'Sample Restaurant B', 'Sample Restaurant C'] * 100,
            'Cuisine Type': ['American', 'Chinese', 'Italian'] * 100,
            'Inspection Type': ['Cycle Inspection / Initial Inspection'] * 300,
            'Critical Flag': ['Critical', 'Not Critical', 'Critical'] * 100,
            'Violation Code': ['02D', '04L', '06C'] * 100,
            'Inspection Score': [13, 25, 35] * 100,
            'Grade': ['A', 'B', 'C'] * 100
        }
        return pd.DataFrame(sample_data)
    
    try:
        return pd.read_csv(data_file)
    except:
        return pd.DataFrame({'Grade': ['A', 'B', 'C'], 'Inspection Score': [10, 20, 30]})

# Simple prediction function
def predict_grade_with_model(model_objects, inspection_type, critical_flag, violation_code, 
                           inspection_score, inspection_date):
    """Make prediction using the trained model"""
    
    if model_objects is None or model_objects.get('model_type') == 'demo':
        # Fallback to simple rule-based prediction
        return predict_grade_simple(inspection_score, critical_flag, violation_code)
    
    try:
        # Extract model components
        model = model_objects['model']
        scaler = model_objects['scaler']
        label_encoders = model_objects['label_encoders']
        grade_encoder = model_objects['grade_encoder']
        features = model_objects['features']
        
        # Parse inspection date
        if isinstance(inspection_date, str):
            inspection_date = pd.to_datetime(inspection_date)
        
        # Create feature vector
        feature_vector = {}
        
        # Handle categorical features
        try:
            feature_vector['Inspection Type'] = label_encoders['Inspection Type'].transform([inspection_type])[0]
        except:
            feature_vector['Inspection Type'] = 0  # Default value
            
        try:
            feature_vector['Critical Flag'] = label_encoders['Critical Flag'].transform([critical_flag])[0]
        except:
            feature_vector['Critical Flag'] = 0  # Default value
            
        try:
            feature_vector['Violation Code'] = label_encoders['Violation Code'].transform([violation_code])[0]
        except:
            feature_vector['Violation Code'] = 0  # Default value
        
        # Handle numerical features
        feature_vector['Inspection Score'] = float(inspection_score)
        feature_vector['inspection_year'] = inspection_date.year
        feature_vector['inspection_month'] = inspection_date.month
        feature_vector['inspection_day_of_week'] = inspection_date.weekday()
        
        # Create feature array in correct order
        X = np.array([feature_vector[feature] for feature in features]).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Convert prediction back to grade
        predicted_grade = grade_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary and ensure all values are positive
        grade_probabilities = {}
        for i, grade in enumerate(grade_encoder.classes_):
            grade_probabilities[grade] = max(0.0, float(probabilities[i]))
        
        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(grade_probabilities.values())
        if total_prob > 0:
            grade_probabilities = {k: v/total_prob for k, v in grade_probabilities.items()}
        else:
            # Fallback if all probabilities are zero
            grade_probabilities = {predicted_grade: 1.0}
            for grade in grade_encoder.classes_:
                if grade != predicted_grade:
                    grade_probabilities[grade] = 0.0
        
        return predicted_grade, grade_probabilities
        
    except Exception as e:
        st.warning(f"Model prediction failed: {str(e)}. Using fallback prediction.")
        return predict_grade_simple(inspection_score, critical_flag, violation_code)

def predict_grade_simple(score, critical_flag, violation_code):
    """Enhanced rule-based prediction that matches actual dataset patterns"""
    
    # Initialize base probabilities for all possible grades
    base_prob = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'N': 0.0, 'Z': 0.0, 'P': 0.0}
    
    # Primary scoring logic (based on actual NYC grading system)
    if score <= 13:
        base_grade = 'A'
        base_prob = {'A': 0.85, 'B': 0.10, 'C': 0.03, 'N': 0.01, 'Z': 0.005, 'P': 0.005}
    elif score <= 27:
        base_grade = 'B' 
        base_prob = {'A': 0.15, 'B': 0.70, 'C': 0.12, 'N': 0.02, 'Z': 0.005, 'P': 0.005}
    else:
        base_grade = 'C'
        base_prob = {'A': 0.05, 'B': 0.15, 'C': 0.75, 'N': 0.03, 'Z': 0.01, 'P': 0.01}
    
    # Adjust based on critical violations
    if critical_flag == 'Critical':
        # Critical violations increase chance of lower grades
        if base_grade == 'A':
            base_grade = 'B'
            base_prob = {'A': 0.40, 'B': 0.50, 'C': 0.08, 'N': 0.015, 'Z': 0.003, 'P': 0.002}
        elif base_grade == 'B':
            base_grade = 'C'
            base_prob = {'A': 0.05, 'B': 0.35, 'C': 0.55, 'N': 0.03, 'Z': 0.01, 'P': 0.01}
        # C grade gets worse with critical violations
        elif base_grade == 'C':
            base_prob = {'A': 0.02, 'B': 0.08, 'C': 0.80, 'N': 0.05, 'Z': 0.03, 'P': 0.02}
    
    # Specific violation code adjustments
    serious_violations = ['04M', '04L', '04H', '08A']  # Roaches, mice, adulterated food, vermin
    if violation_code in serious_violations:
        # These violations often lead to pending grades or closures
        if base_grade in ['B', 'C']:
            base_prob['Z'] += 0.10  # Increase pending grade probability
            base_prob['C'] += 0.05  # Increase C grade probability
            # Normalize
            total = sum(base_prob.values())
            base_prob = {k: v/total for k, v in base_prob.items()}
    
    # Handle no violations case
    if violation_code == 'Null' and score == 0:
        base_grade = 'A'
        base_prob = {'A': 0.95, 'B': 0.03, 'C': 0.01, 'N': 0.005, 'Z': 0.003, 'P': 0.002}
    
    # Ensure all probabilities are positive and sum to 1
    base_prob = {k: max(0.0, v) for k, v in base_prob.items()}
    total = sum(base_prob.values())
    if total > 0:
        base_prob = {k: v/total for k, v in base_prob.items()}
    else:
        # Fallback if something went wrong
        base_prob = {base_grade: 1.0}
        for grade in ['A', 'B', 'C', 'N', 'Z', 'P']:
            if grade not in base_prob:
                base_prob[grade] = 0.0
    
    return base_grade, base_prob

def main():
    # App header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1E3A8A; font-size: 3rem; margin-bottom: 0.5rem;'>
            🍽️ NYC Restaurant Health Grade Predictor
        </h1>
        <p style='color: #6B7280; font-size: 1.2rem; margin-bottom: 2rem;'>
            AI-powered restaurant health grade prediction system
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model_objects = load_model()
    df = load_dataset()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 About This App")
        st.info("""
        This system predicts NYC restaurant health grades using machine learning patterns from real inspection data.
        
        **Features:**
        - Real-time grade prediction
        - Interactive analytics
        - Performance metrics
        - Educational insights
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        
        if model_objects and model_objects.get('model_type') != 'demo':
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Type", model_objects['model_type'].title())
            with col2:
                st.metric("Features", len(model_objects['features']))
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", "Demo Mode")
            with col2:
                st.metric("Type", "Rule-based")
        
        if len(df) > 0:
            st.markdown("---")
            st.markdown("### 📈 Dataset Stats")
            st.metric("Total Records", f"{len(df):,}")
            
            grade_counts = df['Grade'].value_counts()
            fig_pie = px.pie(
                values=grade_counts.values,
                names=grade_counts.index,
                title="Grade Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict Grade", "📊 Analytics", "ℹ️ Information"])
    
    with tab1:
        st.markdown("### 🔮 Restaurant Grade Prediction")
        st.markdown("Enter inspection details to get an AI-powered grade prediction.")
        
        # Prediction form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏢 Restaurant Details")
            
            restaurant_name = st.text_input("Restaurant Name", "PEKING GARDEN")
            
            # Use actual cuisine types from dataset (top 15 most common)
            cuisine_options = [
                'American', 'Chinese', 'Coffee/Tea', 'Pizza', 'Italian', 'Mexican',
                'Latin American', 'Bakery Products/Desserts', 'Caribbean', 'Japanese', 
                'Chicken', 'Spanish', 'Donuts', 'Sandwiches', 'Hamburgers', 'Thai',
                'Indian', 'Greek', 'French', 'Seafood', 'Asian/Asian Fusion', 'Other'
            ]
            cuisine = st.selectbox("Cuisine Type", cuisine_options)
            
            # Use actual inspection types from dataset (top 5 most common)
            inspection_type_options = [
                'Cycle Inspection / Initial Inspection',
                'Cycle Inspection / Re-inspection', 
                'Pre-permit (Operational) / Initial Inspection',
                'Pre-permit (Operational) / Re-inspection',
                'Cycle Inspection / Reopening Inspection'
            ]
            inspection_type = st.selectbox("Inspection Type", inspection_type_options)
            
            # Borough selection
            location_options = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
            location = st.selectbox("Borough", location_options)
        
        with col2:
            st.markdown("#### 📊 Inspection Results")
            
            score = st.slider(
                "Inspection Score",
                min_value=0,
                max_value=150,
                value=15,
                help="Lower scores indicate better compliance (0 = perfect, 150+ = severe violations)"
            )
            
            critical_flag = st.selectbox(
                "Critical Violations",
                ['Not Critical', 'Critical', 'Not Applicable'],
                help="Whether violations pose immediate health risks"
            )
            
            # Use actual violation codes from dataset (top 15 most common)
            violation_code_options = [
                'Null',  # For no violations
                '10F', '08A', '06D', '04L', '10B', '06C', '02G', '02B', '04N', 
                '04A', '04H', '08C', '06A', '09C', '04M', '06E', '10H', '06F', 'Other'
            ]
            
            # Create user-friendly descriptions
            violation_descriptions = {
                'Null': 'No violations recorded',
                '10F': '10F - Wiping cloths not stored clean and dry',
                '08A': '08A - Facility not vermin proof', 
                '06D': '06D - Food contact surface not properly washed',
                '04L': '04L - Evidence of mice or live mice in establishment',
                '10B': '10B - Plumbing not properly installed or maintained',
                '06C': '06C - Food not protected from contamination',
                '02G': '02G - Cold food item held above 41° F',
                '02B': '02B - Hot food item not held at or above 140° F',
                '04N': '04N - Filth flies or food/refuse/sewage associated flies',
                '04A': '04A - Food Protection Certificate not held by supervisor',
                '04H': '04H - Raw, cooked or prepared food is adulterated',
                '08C': '08C - Pesticide use not in accordance with label',
                '06A': '06A - Personal cleanliness inadequate',
                '09C': '09C - Food contact surface not properly maintained',
                '04M': '04M - Live roaches present in facility',
                '06E': '06E - Sanitized equipment or utensil improperly used',
                '10H': '10H - Proper sanitization not provided for utensil ware',
                '06F': '06F - Wiping cloths soiled or not stored in sanitizing solution',
                'Other': 'Other violation type'
            }
            
            violation_code = st.selectbox(
                "Primary Violation Type", 
                violation_code_options,
                format_func=lambda x: violation_descriptions.get(x, x)
            )
            
            inspection_date = st.date_input("Inspection Date", datetime.date.today())
        
        # Prediction button
        if st.button("🚀 Predict Health Grade", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing inspection data..."):
                
                # Get prediction using the trained model
                predicted_grade, probabilities = predict_grade_with_model(
                    model_objects, inspection_type, critical_flag, violation_code, 
                    score, inspection_date
                )
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                # Main result
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    # Grade circle
                    colors = {'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545', 'N': '#6c757d', 'Z': '#17a2b8', 'P': '#6610f2'}
                    emojis = {'A': '🏆', 'B': '🥈', 'C': '🥉', 'N': '❓', 'Z': '📊', 'P': '📝'}
                    
                    grade_color = colors.get(predicted_grade, '#6c757d')
                    emoji = emojis.get(predicted_grade, '❓')
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem;'>
                        <div style='background: {grade_color}; width: 120px; height: 120px; border-radius: 50%; 
                                   display: flex; flex-direction: column; justify-content: center; align-items: center; 
                                   margin: 0 auto; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                            <div style='font-size: 48px; font-weight: 800; color: white;'>{predicted_grade}</div>
                            <div style='font-size: 24px;'>{emoji}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"### Predicted Grade: **{predicted_grade}**")
                    
                    descriptions = {
                        'A': '🌟 Excellent! Restaurant meets all health requirements.',
                        'B': '⚠️ Good with minor issues. Some improvements needed.',
                        'C': '🚨 Poor conditions. Significant improvements required.',
                        'N': '❓ Not yet graded - awaiting full assessment.',
                        'Z': '📊 Grade pending - inspection results being processed.',
                        'P': '📝 Grade pending - awaiting final determination.'
                    }
                    
                    st.markdown(descriptions.get(predicted_grade, 'Unknown grade'))
                    
                    # Calculate confidence based on probabilities
                    if isinstance(probabilities, dict):
                        confidence = max(probabilities.values()) * 100
                    else:
                        confidence = max(probabilities) * 100
                        
                    st.markdown(f"**Confidence:** {confidence:.1f}%")
                    st.progress(confidence/100)
                    
                    score_ranges = {
                        'A': '0-13 points', 'B': '14-27 points', 'C': '28+ points',
                        'N': 'Not applicable', 'Z': 'Pending assessment', 'P': 'Pending assessment'
                    }
                    st.markdown(f"**Expected Score Range:** {score_ranges.get(predicted_grade, 'Unknown')}")
                
                # Probability chart
                st.markdown("### 📊 Grade Probabilities")
                
                # Ensure probabilities are properly formatted and positive
                if isinstance(probabilities, dict):
                    # Filter out any negative or zero probabilities for display
                    filtered_probs = {k: max(0, v) for k, v in probabilities.items() if v > 0.001}
                    
                    if filtered_probs:
                        # Normalize probabilities to ensure they sum to 1
                        total_prob = sum(filtered_probs.values())
                        if total_prob > 0:
                            filtered_probs = {k: v/total_prob for k, v in filtered_probs.items()}
                        
                        prob_df = pd.DataFrame(list(filtered_probs.items()), columns=['Grade', 'Probability'])
                        prob_df['Probability'] = prob_df['Probability'] * 100
                        prob_df = prob_df.sort_values('Probability', ascending=False)
                    else:
                        # Fallback if no valid probabilities
                        prob_df = pd.DataFrame({
                            'Grade': [predicted_grade],
                            'Probability': [100.0]
                        })
                else:
                    # Handle list/array format
                    grade_names = ['A', 'B', 'C']
                    if len(probabilities) >= 3:
                        probs = [max(0, p) for p in probabilities[:3]]  # Ensure positive
                        total = sum(probs)
                        if total > 0:
                            probs = [p/total for p in probs]  # Normalize
                        prob_df = pd.DataFrame({
                            'Grade': grade_names,
                            'Probability': [p * 100 for p in probs]
                        })
                        prob_df = prob_df[prob_df['Probability'] > 0.1]  # Filter very small probabilities
                    else:
                        prob_df = pd.DataFrame({
                            'Grade': [predicted_grade],
                            'Probability': [100.0]
                        })
                
                # Create the chart
                fig_bar = px.bar(
                    prob_df,
                    x='Grade',
                    y='Probability',
                    color='Grade',
                    color_discrete_map={'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545', 
                                      'N': '#6c757d', 'Z': '#17a2b8', 'P': '#6610f2'},
                    title="Probability Distribution (%)"
                )
                
                # Ensure Y-axis starts at 0 and shows percentages
                fig_bar.update_layout(
                    showlegend=False, 
                    height=400,
                    yaxis=dict(
                        title="Probability (%)",
                        range=[0, max(prob_df['Probability'].max() * 1.1, 10)]  # Ensure minimum range
                    ),
                    xaxis=dict(title="Grade")
                )
                
                # Add value labels on bars
                fig_bar.update_traces(
                    texttemplate='%{y:.1f}%',
                    textposition='outside'
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                if predicted_grade == 'A':
                    st.success("🎉 **Excellent Performance!** Maintain high standards and continue best practices.")
                elif predicted_grade == 'B':
                    st.warning("⚠️ **Good, but improvable.** Address violations promptly and enhance procedures.")
                elif predicted_grade == 'C':
                    st.error("🚨 **Immediate action required!** Address all violations and implement corrective measures.")
                else:
                    st.info("ℹ️ **Grade pending.** Follow up on inspection status and address any identified issues.")
    
    with tab2:
        st.markdown("## 📊 Analytics Dashboard")
        
        if len(df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Grade distribution
                grade_counts = df['Grade'].value_counts()
                
                fig_grades = px.bar(
                    x=grade_counts.index,
                    y=grade_counts.values,
                    color=grade_counts.index,
                    color_discrete_map={'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545', 
                                      'N': '#6c757d', 'Z': '#17a2b8', 'P': '#6610f2'},
                    title="Grade Distribution in Dataset"
                )
                fig_grades.update_layout(showlegend=False)
                st.plotly_chart(fig_grades, use_container_width=True)
            
            with col2:
                # Score distribution
                if 'Inspection Score' in df.columns:
                    fig_scores = px.histogram(
                        df,
                        x='Inspection Score',
                        color='Grade',
                        title="Score Distribution by Grade",
                        nbins=20
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)
                else:
                    st.info("Score distribution not available")
            
            # Additional analytics
            st.markdown("### 📈 Additional Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Cuisine Type' in df.columns:
                    cuisine_grades = df.groupby('Cuisine Type')['Grade'].value_counts().unstack(fill_value=0)
                    fig_cuisine = px.bar(
                        cuisine_grades.head(10),
                        title="Top 10 Cuisine Types by Grade Distribution",
                        color_discrete_map={'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545'}
                    )
                    st.plotly_chart(fig_cuisine, use_container_width=True)
            
            with col2:
                if 'Critical Flag' in df.columns:
                    critical_dist = df['Critical Flag'].value_counts()
                    fig_critical = px.pie(
                        values=critical_dist.values,
                        names=critical_dist.index,
                        title="Critical vs Non-Critical Violations"
                    )
                    st.plotly_chart(fig_critical, use_container_width=True)
                    
        else:
            st.info("📊 Analytics will be available when dataset is loaded.")
    
    with tab3:
        st.markdown("## ℹ️ Understanding Restaurant Grades")
        
        # Grade explanations
        col1, col2, col3 = st.columns(3)
        
        grades_info = [
            ('A', '🏆', '#28a745', '0-13 points', 'Excellent compliance'),
            ('B', '🥈', '#ffc107', '14-27 points', 'Good with minor issues'),
            ('C', '🥉', '#dc3545', '28+ points', 'Needs improvement')
        ]
        
        for i, (grade, emoji, color, range_text, desc) in enumerate(grades_info):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div style='background: {color}; color: white; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                    <div style='font-size: 3rem;'>{emoji}</div>
                    <h3 style='color: white; margin: 0.5rem 0;'>Grade {grade}</h3>
                    <p style='color: white; margin: 0;'><strong>{range_text}</strong></p>
                    <p style='color: white; margin: 0.5rem 0 0 0;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### 📊 How Scoring Works")
        st.info("""
        **NYC Restaurant Inspection Scoring:**
        - Each violation receives points based on severity
        - Critical violations (food safety risks) receive more points
        - Lower total scores = better grades
        - Inspectors evaluate food temperature, cleanliness, pest control, and staff hygiene
        """)
        
        st.markdown("### 🔬 About This Model")
        
        if model_objects and model_objects.get('model_type') != 'demo':
            st.markdown(f"""
            This prediction system uses a **{model_objects['model_type'].title()} Classifier** trained on real NYC restaurant inspection data:
            
            - **Training Data**: {len(df):,} restaurant inspection records
            - **Features Used**: {len(model_objects['features'])} key inspection metrics
            - **Model Type**: {model_objects['model_type'].title()} with class balancing
            - **Prediction Accuracy**: Based on historical inspection patterns
            
            **Key Features:**
            - Restaurant inspection scores and violation types
            - Critical vs non-critical violation classifications
            - Temporal patterns (year, month, day of week)
            - Inspection types and procedures
            """)
        else:
            st.markdown("""
            This demonstration system shows machine learning concepts using:
            - **Rule-based Logic** for basic predictions
            - **Feature Analysis** of inspection components
            - **Grade Distribution** patterns from real data
            - **Interactive Visualizations** for data exploration
            """)
        
        st.warning("""
        **Important Note**: This is a demonstration system for educational purposes. 
        Actual restaurant grades should only be determined by official NYC Department of Health inspections.
        """)

if __name__ == "__main__":
    main()
