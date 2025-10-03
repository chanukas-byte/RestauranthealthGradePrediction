import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_class_distribution(df):
    """Analyze and visualize class distribution"""
    print("Class Distribution Analysis:")
    print("=" * 50)
    
    grade_counts = df['Grade'].value_counts()
    grade_percentages = df['Grade'].value_counts(normalize=True) * 100
    
    # Create a DataFrame for better display
    distribution_df = pd.DataFrame({
        'Count': grade_counts,
        'Percentage': grade_percentages
    })
    
    print(distribution_df)
    print("\n")
    
    # Visualize the distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Grade', order=grade_counts.index)
    plt.title('Distribution of Restaurant Grades', fontsize=16)
    plt.xlabel('Grade', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels on bars
    for i, count in enumerate(grade_counts):
        plt.text(i, count + 10, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('grade_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return grade_counts

def train_and_save_model():
    print("Starting balanced model training with class imbalance analysis...")
    
    # Get the correct data path relative to this script
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '../data/cleaned_restaurant_dataset.csv')
    
    # Check if the data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please make sure the data file is in the correct location.")
        return False
    
    # Load the data
    try:
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False
    
    # Create a copy of the data for preprocessing
    df = data.copy()
      # Handle missing values in the GRADE column
    df['Grade'] = df['Grade'].fillna('Not Graded')
    
    # Filter out 'Not Graded' grades
    df = df[df['Grade'].isin(['A', 'B', 'C', 'N', 'Z', 'P'])]
    
    # Analyze class distribution
    grade_counts = analyze_class_distribution(df)
    
    # Check if we have sufficient samples for each class
    min_samples_threshold = 50  # Minimum samples per class
    rare_classes = grade_counts[grade_counts < min_samples_threshold].index.tolist()
    
    if rare_classes:
        print(f"Warning: The following classes have fewer than {min_samples_threshold} samples: {rare_classes}")
        print("Consider collecting more data or using techniques for handling rare classes.")
      # Clean up the columns we'll use for prediction
    df['Inspection Type'] = df['Inspection Type'].fillna('Unknown')
    df['Critical Flag'] = df['Critical Flag'].fillna('Not Applicable')
    df['Violation Code'] = df['Violation Code'].fillna('No Violation')
    df['Inspection Score'] = df['Inspection Score'].fillna(df['Inspection Score'].median())
    
    # Extract useful features from dates
    df['Inspection Date'] = pd.to_datetime(df['Inspection Date'], errors='coerce')
    df['inspection_year'] = df['Inspection Date'].dt.year
    df['inspection_month'] = df['Inspection Date'].dt.month
    df['inspection_day_of_week'] = df['Inspection Date'].dt.dayofweek
    
    # Drop rows with missing target
    df = df.dropna(subset=['Grade'])
    
    # Select the features for our model
    features = ['Inspection Type', 'Critical Flag', 'Violation Code', 'Inspection Score', 
                'inspection_year', 'inspection_month', 'inspection_day_of_week']
    
    # Create a new dataframe with only the features and target
    model_df = df[features + ['Grade']].copy()
    
    # Handle categorical variables with label encoding
    label_encoders = {}
    categorical_features = ['Inspection Type', 'Critical Flag', 'Violation Code']
    
    for feature in categorical_features:
        le = LabelEncoder()
        model_df[feature] = le.fit_transform(model_df[feature].astype(str))
        label_encoders[feature] = le
    
    # Encode the target variable (GRADE)
    grade_encoder = LabelEncoder()
    model_df['GRADE_encoded'] = grade_encoder.fit_transform(model_df['Grade'])
    label_encoders['GRADE'] = grade_encoder
    
    # Store the mapping between encoded values and actual grades
    grade_mapping = dict(zip(grade_encoder.classes_, grade_encoder.transform(grade_encoder.classes_)))
    print("Grade mapping:", grade_mapping)
      # Split the data into features and target
    X = model_df.drop(['Grade', 'GRADE_encoded'], axis=1)
    y = model_df['GRADE_encoded']
    
    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Create a dictionary of class weights
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print("\nClass Weights:")
    for grade_encoded, weight in class_weight_dict.items():
        grade_name = grade_encoder.inverse_transform([grade_encoded])[0]
        print(f"{grade_name}: {weight:.4f}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models with different approaches
    models = {}
    
    # Model 1: Standard Random Forest (no class weighting)
    print("\nTraining Model 1: Standard Random Forest")
    rf_standard = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10
    )
    rf_standard.fit(X_train_scaled, y_train)
    models['standard'] = rf_standard
    
    # Model 2: Random Forest with class weights
    print("Training Model 2: Random Forest with Class Weights")
    rf_weighted = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        class_weight=class_weight_dict
    )
    rf_weighted.fit(X_train_scaled, y_train)
    models['weighted'] = rf_weighted
    
    # Model 3: Random Forest with balanced subsample
    print("Training Model 3: Random Forest with Balanced Subsample")
    rf_balanced = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        class_weight='balanced_subsample'
    )
    rf_balanced.fit(X_train_scaled, y_train)
    models['balanced_subsample'] = rf_balanced
    
    # Evaluate all models
    print("\nModel Evaluation:")
    print("=" * 50)
    
    best_model = None
    best_f1_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\n{name.upper()} MODEL:")
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, 
            target_names=grade_encoder.classes_,
            output_dict=True
        )
        
        # Calculate macro F1-score (average across all classes)
        macro_f1 = report['macro avg']['f1-score']
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        
        # Print detailed classification report
        print(classification_report(
            y_test, y_pred, 
            target_names=grade_encoder.classes_
        ))
        
        # Track the best model based on macro F1-score
        if macro_f1 > best_f1_score:
            best_f1_score = macro_f1
            best_model = model
            best_model_name = name
    
    print(f"\nBEST MODEL: {best_model_name.upper()} with Macro F1-Score: {best_f1_score:.4f}")
    
    # Create confusion matrix for the best model
    y_pred_best = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=grade_encoder.classes_,
        yticklabels=grade_encoder.classes_
    )
    plt.title(f'Confusion Matrix - {best_model_name.upper()} Model', fontsize=16)
    plt.xlabel('Predicted Grade', fontsize=12)
    plt.ylabel('True Grade', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the best model and preprocessing objects
    model_objects = {
        'model': best_model,
        'model_type': best_model_name,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'features': features,
        'grade_encoder': grade_encoder,
        'grade_classes': grade_encoder.classes_,
        'class_weights': class_weight_dict
    }
      # Save to a file using joblib
    try:
        model_path = os.path.join(current_dir, '../models/restaurant_balanced_model.joblib')
        joblib.dump(model_objects, model_path)
        print(f"\nBest model ({best_model_name}) and preprocessing objects saved to '{model_path}'")
        
        # Verify the file was created
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"Model file created successfully. Size: {file_size} bytes")
            return True
        else:
            print("Error: Model file was not created!")
            return False
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

if __name__ == "__main__":
    success = train_and_save_model()
    if success:
        print("\nBalanced model training completed successfully!")
        print("You can now run the Streamlit app with: streamlit run balanced_app.py")
    else:
        print("\nModel training failed. Please check the error messages above.")