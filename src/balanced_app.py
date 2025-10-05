import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Restaurant Health Grade Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), '../assets/styles.css')
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../models/restaurant_balanced_model.joblib')
        if not os.path.exists(model_path):
            st.error("Model file not found. Please train the model first.")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Try to load model
model_data = load_model()

# Helper functions
def get_grade_emoji(grade):
    emojis = {'A': '🏆', 'B': '🥈', 'C': '🥉', 'N': '❓', 'Z': '📊', 'P': '📝'}
    return emojis.get(grade, '❓')

def get_grade_description(grade):
    descriptions = {
        'A': 'Excellent - Minimal violations.',
        'B': 'Good - Some violations to address.',
        'C': 'Fair - Significant violations.',
        'N': 'Not Yet Graded.',
        'Z': 'Grade Pending.',
        'P': 'Grade Pending.'
    }
    return descriptions.get(grade, 'Unknown grade')

def get_color_for_grade(grade):
    colors = {'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545', 'N': '#6c757d', 'Z': '#17a2b8', 'P': '#6610f2'}
    return colors.get(grade, '#6c757d')

# Main app content
if model_data is None:
    st.error("⚠️ Model not loaded! Please train the model first.")
    st.info("Run the following command to train the model:")
    st.code("python health-prediction-app/src/train_model.py")
    st.stop()

# Sidebar
st.sidebar.header("About this App")
st.sidebar.info("""
This app predicts NYC restaurant health grades using ML.

**Grades:**
- A: Excellent
- B: Good
- C: Fair
- N/Z/P: Pending/Not Graded
""")

st.sidebar.header("Model Info")
st.sidebar.success(f"Model: {model_data['model_type'].upper()}")

# Feature importance
st.sidebar.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': model_data['features'],
    'Importance': model_data['model'].feature_importances_
}).sort_values(by='Importance', ascending=False)
fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
plt.tight_layout()
st.sidebar.pyplot(fig)

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4080/4080037.png", width=100)
with col2:
    st.title("Restaurant Health Grade Predictor")
    st.subheader("Predict NYC restaurant inspection grades with ML")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "Grades Info", "Model Info"])

with tab1:
    st.write("### Enter Inspection Details")
    col1, col2 = st.columns(2)
    with col1:
        inspection_types = list(model_data['label_encoders']['Inspection Type'].classes_)
        inspection_type = st.selectbox("Inspection Type", inspection_types)
        critical_flags = list(model_data['label_encoders']['Critical Flag'].classes_)
        critical_flag = st.selectbox("Critical Flag", critical_flags)
        violation_codes = list(model_data['label_encoders']['Violation Code'].classes_)
        common_codes = violation_codes[:10] if len(violation_codes) > 10 else violation_codes
        common_codes.append("Other (Enter below)")
        selected_code = st.selectbox("Violation Code", common_codes)
        if selected_code == "Other (Enter below)":
            violation_code = st.text_input("Enter Violation Code")
        else:
            violation_code = selected_code
    with col2:
        score = st.slider("Inspection Score (Higher is worse)", 0, 100, 13)
        inspection_date = st.date_input("Inspection Date", datetime.date.today())
        inspection_year = inspection_date.year
        inspection_month = inspection_date.month
        inspection_day_of_week = inspection_date.weekday()
    if st.button("Predict Health Grade", type="primary"):
        with st.spinner("Analyzing inspection data..."):
            try:
                it_enc = model_data['label_encoders']['Inspection Type'].transform([inspection_type])[0]
                cf_enc = model_data['label_encoders']['Critical Flag'].transform([critical_flag])[0]
                try:
                    vc_enc = model_data['label_encoders']['Violation Code'].transform([violation_code])[0]
                except:
                    vc_enc = 0
                    st.warning(f"Violation code '{violation_code}' not found. Using default value.")
                X = np.array([[it_enc, cf_enc, vc_enc, score, inspection_year, inspection_month, inspection_day_of_week]])
                X_scaled = model_data['scaler'].transform(X)
                prediction = model_data['model'].predict(X_scaled)
                predicted_grade = model_data['grade_encoder'].inverse_transform(prediction)[0]
                probabilities = model_data['model'].predict_proba(X_scaled)[0]
                grade_probs = {model_data['grade_encoder'].inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}
                st.divider()
                st.subheader("Prediction Result")
                col1, col2 = st.columns([1, 3])
                with col1:
                    grade_color = get_color_for_grade(predicted_grade)
                    emoji = get_grade_emoji(predicted_grade)
                    st.markdown(f"""
                        <div class='grade-container' style='background-color: {grade_color}'>
                            <div class='grade'>{predicted_grade}</div>
                            <div class='emoji'>{emoji}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"### Predicted Grade: {predicted_grade}")
                    st.write(get_grade_description(predicted_grade))
                    confidence = grade_probs[predicted_grade] * 100
                    st.write(f"Confidence: {confidence:.1f}%")
                    st.progress(confidence/100)
                st.subheader("Grade Probability Distribution")
                prob_df = pd.DataFrame({'Grade': list(grade_probs.keys()), 'Probability': list(grade_probs.values())}).sort_values('Probability', ascending=False)
                fig, ax = plt.subplots(figsize=(8, 3))
                bars = sns.barplot(x='Probability', y='Grade', data=prob_df, ax=ax, palette='viridis')
                for i, v in enumerate(prob_df['Probability']):
                    ax.text(v + 0.01, i, f"{v*100:.1f}%", va='center')
                ax.set_xlim(0, 1.1)
                ax.set_xlabel('Probability')
                ax.set_ylabel('Grade')
                ax.set_title('Probability Distribution of Predicted Grades')
                st.pyplot(fig)
                st.subheader("Recommendations")
                if predicted_grade == 'A':
                    st.success("✅ Excellent! Maintain your high standards.")
                elif predicted_grade == 'B':
                    st.warning("⚠️ Good, but improve on flagged issues.")
                elif predicted_grade == 'C':
                    st.error("🚨 Immediate action required! Address all violations.")
                else:
                    st.info("ℹ️ Grade is pending or not yet determined.")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

with tab2:
    st.header("Understanding Restaurant Health Grades")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### Grade A 🏆\n**Score Range:** 0-13\nExcellent compliance.
        """)
    with col2:
        st.markdown("""
        ### Grade B 🥈\n**Score Range:** 14-27\nGood, some violations.
        """)
    with col3:
        st.markdown("""
        ### Grade C 🥉\n**Score Range:** 28+\nSignificant violations.
        """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Grade N ❓\nNot Yet Graded.")
    with col2:
        st.markdown("### Grade Z 📊\nGrade Pending.")
    with col3:
        st.markdown("### Grade P 📝\nGrade Pending.")
    st.info("""
    **How Scores Work:** Points are assigned for violations. Higher score = worse grade.
    """)

with tab3:
    st.header("About the Model")
    st.write("""
    This app uses a Random Forest classifier trained on NYC inspection data.\n\nFeatures:\n- Inspection type\n- Critical flag\n- Violation code\n- Score\n- Inspection date (year, month, day of week)\n\nModel is balanced for all grade categories.
    """)
    try:
        cm_img = Image.open(os.path.join(os.path.dirname(__file__), '../confusion_matrix.png'))
        st.image(cm_img, caption="Model Confusion Matrix", use_column_width=True)
    except:
        st.write("Confusion matrix not available.")
    try:
        dist_img = Image.open(os.path.join(os.path.dirname(__file__), '../grade_distribution.png'))
        st.image(dist_img, caption="Grade Distribution in Training Data", use_column_width=True)
    except:
        st.write("Grade distribution not available.")