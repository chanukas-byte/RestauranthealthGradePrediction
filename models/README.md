# Model Files

This directory contains the trained machine learning models for the restaurant health grade prediction system.

## How to Generate the Model

The trained model file `restaurant_balanced_model.joblib` is not included in the repository due to its large size (10MB+). To generate the model:

1. **Run the training script:**
   ```bash
   python src/train_model.py
   ```

2. **The script will:**
   - Load and preprocess the restaurant inspection data
   - Train a Random Forest classifier with balanced class weights
   - Save the trained model and preprocessing objects to `models/restaurant_balanced_model.joblib`

## Model Details

- **Type**: Random Forest Classifier
- **Performance**: 95.8% accuracy, 84.8% F1-score
- **Training Data**: 67,000+ NYC restaurant inspection records
- **Features**: 7 engineered features including temporal data
- **File Size**: ~10MB

## Model Components

The saved model file contains:
- Trained Random Forest model
- Label encoders for categorical features
- Standard scaler for numerical features
- Grade encoder for target labels
- Feature names and preprocessing metadata

## Note

If you encounter any issues with model generation, ensure you have:
- Python 3.8+ installed
- All required dependencies from `requirements.txt`
- The dataset file `data/cleaned_restaurant_dataset.csv`
