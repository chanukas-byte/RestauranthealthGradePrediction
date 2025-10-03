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
        'grade_classes': np.array(['A', 'B', 'C']),
        'class_weights': {0: 1.0, 1: 2.0, 2: 3.0}
    }
    
    # Fit dummy data
    demo_model['grade_encoder'].fit(['A', 'B', 'C'])
    demo_model['label_encoders']['Inspection Type'].fit(['Cycle Inspection / Re-inspection', 'Pre-permit (Operational) / Re-inspection'])
    demo_model['label_encoders']['Critical Flag'].fit(['Critical', 'Not Critical', 'Not Applicable'])
    demo_model['label_encoders']['Violation Code'].fit(['02B', '02D', '04A', '06D', '08A', '09B', '10F'])
    
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
    """Generate sample data that matches your dataset structure"""
    np.random.seed(42)
    
    # Sample data that matches your actual dataset columns
    restaurants = [
        "PEKING GARDEN", "BAO BY KAYA", "WOODROW DINER", "JANE FAST FOOD", 
        "GREENHOUSE CAFE", "GOLDEN DRAGON", "PIZZA PALACE", "SUBWAY",
        "MCDONALD'S", "BURGER KING", "TACO BELL", "KFC", "WENDY'S"
    ]
    
    locations = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    cuisine_types = ["Chinese", "American", "Asian/Asian Fusion", "Coffee/Tea", "Pizza", "Fast Food"]
    inspection_types = ["Cycle Inspection / Re-inspection", "Pre-permit (Operational) / Re-inspection"]
    critical_flags = ["Critical", "Not Critical", "Not Applicable"]
    violation_codes = ["02B", "02D", "04A", "06D", "08A", "09B", "10F", "Null"]
    grades = ["A", "B", "C"]
    
    n_samples = 1000
    
    sample_data = []
    for i in range(n_samples):
        # Generate realistic inspection scores based on grade
        grade = np.random.choice(grades, p=[0.7, 0.2, 0.1])  # More A grades
        if grade == "A":
            score = np.random.gamma(2, 3)  # Lower scores for A
        elif grade == "B":
            score = np.random.gamma(3, 5) + 14  # Mid scores for B
        else:
            score = np.random.gamma(2, 8) + 28  # Higher scores for C
        
        score = max(0, min(100, score))  # Clamp between 0-100
        
        sample_data.append({
            'Restaurant ID': 40000000 + i,
            'Restaurant Name': np.random.choice(restaurants),
            'Location': np.random.choice(locations),
            'Cuisine Type': np.random.choice(cuisine_types),
            'Inspection Date': f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
            'Inspection Type': np.random.choice(inspection_types),
            'Violation Code': np.random.choice(violation_codes),
            'Critical Flag': np.random.choice(critical_flags),
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
        
        # Create probability dictionary
        grade_probabilities = {}
        for i, grade in enumerate(grade_encoder.classes_):
            grade_probabilities[grade] = probabilities[i]
        
        return predicted_grade, grade_probabilities
        
    except Exception as e:
        st.warning(f"Model prediction failed: {str(e)}. Using fallback prediction.")
        return predict_grade_simple(inspection_score, critical_flag, violation_code)

def predict_grade_simple(score, critical_flag, violation_type):
    """Simple rule-based prediction for demo purposes"""
    
    # Basic prediction logic
    if score <= 13:
        base_grade = 'A'
        base_prob = {'A': 0.8, 'B': 0.15, 'C': 0.05}
    elif score <= 27:
        base_grade = 'B' 
        base_prob = {'A': 0.2, 'B': 0.7, 'C': 0.1}
    else:
        base_grade = 'C'
        base_prob = {'A': 0.1, 'B': 0.2, 'C': 0.7}
    
    # Adjust based on critical flag
    if critical_flag == 'Critical':
        if base_grade == 'A':
            base_grade = 'B'
            base_prob = {'A': 0.3, 'B': 0.6, 'C': 0.1}
        elif base_grade == 'B':
            base_grade = 'C'
            base_prob = {'A': 0.1, 'B': 0.3, 'C': 0.6}
    
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
            
            restaurant_name = st.text_input("Restaurant Name", "Sample Restaurant")
            
            # Get cuisine options from dataset or use defaults
            if 'Cuisine Type' in df.columns:
                cuisine_options = ['Other'] + sorted(df['Cuisine Type'].dropna().unique().tolist()[:20])
            else:
                cuisine_options = ['American', 'Chinese', 'Italian', 'Mexican', 'Pizza', 'Japanese', 'Thai', 'Other']
            cuisine = st.selectbox("Cuisine Type", cuisine_options)
            
            # Get inspection type options from dataset or use defaults
            if 'Inspection Type' in df.columns:
                inspection_type_options = sorted(df['Inspection Type'].dropna().unique().tolist())
            else:
                inspection_type_options = ['Cycle Inspection / Re-inspection', 'Pre-permit (Operational) / Re-inspection']
            
            inspection_type = st.selectbox("Inspection Type", inspection_type_options)
        
        with col2:
            st.markdown("#### 📊 Inspection Results")
            
            score = st.slider(
                "Inspection Score",
                min_value=0,
                max_value=100,
                value=15,
                help="Lower scores indicate better compliance"
            )
            
            critical_flag = st.selectbox(
                "Critical Violations",
                ['Not Critical', 'Critical', 'Not Applicable'],
                help="Whether violations pose immediate health risks"
            )
            
            # Get violation codes from dataset or use defaults
            if 'Violation Code' in df.columns:
                violation_codes = ['None'] + sorted([str(x) for x in df['Violation Code'].dropna().unique().tolist()[:20]])
            else:
                violation_codes = ['02B', '02D', '04A', '04L', '06C', '06D', '08A', '09B', '10F', 'Other']
            
            violation_code = st.selectbox("Primary Violation Code", violation_codes)
            
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
                
                if isinstance(probabilities, dict):
                    prob_df = pd.DataFrame(list(probabilities.items()), columns=['Grade', 'Probability'])
                    prob_df['Probability'] = prob_df['Probability'] * 100
                else:
                    prob_df = pd.DataFrame({
                        'Grade': ['A', 'B', 'C'],
                        'Probability': [p * 100 for p in probabilities]
                    })
                
                fig_bar = px.bar(
                    prob_df,
                    x='Grade',
                    y='Probability',
                    color='Grade',
                    color_discrete_map={'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545', 
                                      'N': '#6c757d', 'Z': '#17a2b8', 'P': '#6610f2'},
                    title="Probability Distribution"
                )
                
                fig_bar.update_layout(showlegend=False, height=400)
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
