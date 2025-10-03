# 🍽️ NYC Restaurant Health Grade Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An intelligent machine learning system that predicts NYC restaurant health inspection grades using advanced AI algorithms. This project provides a modern web interface for real-time grade predictions and comprehensive analytics.

## 🌟 Features

- **🤖 AI-Powered Predictions**: Advanced Random Forest classifier with 95.8% accuracy
- **🎯 Real-time Analysis**: Instant grade predictions based on inspection parameters
- **📊 Interactive Dashboard**: Comprehensive analytics and visualizations
- **🔍 Feature Importance**: Detailed analysis of factors affecting health grades
- **📱 Modern UI**: Clean, responsive web interface built with Streamlit
- **📈 Performance Metrics**: Model evaluation with detailed statistics
- **☁️ Cloud Deployment**: Optimized for Streamlit Cloud with fallback mechanisms

## 🚀 Quick Start

### 🌐 **Live Demo on Streamlit Cloud**
**🔗 [View Live Application](https://share.streamlit.io/)** *(Available after deployment)*

The cloud-deployed version automatically handles:
- Model loading with fallback to demo mode
- Sample data generation if dataset is unavailable  
- Responsive design for all screen sizes
- Real-time predictions with confidence scores

### 💻 **Local Installation**

#### Prerequisites

- Python 3.8 or higher
- pip package manager

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chanukas-byte/RestauranthealthGradePrediction.git
   cd RestauranthealthGradePrediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional - for full features)
   ```bash
   python src/train_model.py
   ```

4. **Run the application**
   ```bash
   # For cloud-optimized version (recommended)
   streamlit run streamlit_app.py
   
   # Or run the full-featured local version
   streamlit run src/modern_app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## ☁️ Cloud Deployment

### 🌐 **Streamlit Cloud Deployment**

The application is optimized for easy deployment on Streamlit Cloud:

1. **Fork this repository** on GitHub
2. **Visit [share.streamlit.io](https://share.streamlit.io/)**
3. **Connect your GitHub account**
4. **Select your forked repository**
5. **Set main file path**: `streamlit_app.py`
6. **Deploy!** 🚀

The `streamlit_app.py` file is specifically designed for cloud deployment with:
- **Smart Model Loading**: Automatically loads the trained model or falls back to demo mode
- **Dataset Fallbacks**: Multiple fallback options including sample data generation
- **Optimized Dependencies**: Cloud-friendly requirements and configurations
- **Responsive Design**: Works perfectly on all screen sizes
- **Error Handling**: Graceful degradation when resources are unavailable

### 📁 **Cloud-Ready Files**

The following files are specifically created for cloud deployment:

- **`streamlit_app.py`** - Main application optimized for cloud
- **`requirements.txt`** - Minimal dependencies for cloud deployment
- **`packages.txt`** - System packages for Streamlit Cloud
- **`.streamlit/config.toml`** - Streamlit configuration
- **`sample_dataset.csv`** - Lightweight dataset for demo purposes

### 🔧 **Configuration Options**

For advanced deployment, you can customize:

```toml
# .streamlit/config.toml
[server]
headless = true
enableCORS = false
port = $PORT

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 📊 Model Performance

- **Accuracy**: 95.8%
- **F1-Score**: 84.8%
- **Training Data**: 67,000+ real NYC restaurant inspection records
- **Model Type**: Balanced Random Forest Classifier

## 🏗️ Project Structure

```
health-prediction-app/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── assets/                  # Static assets
│   ├── styles.css          # Custom CSS styles
│   └── modern_styles.css   # Modern UI styles
├── data/                   # Dataset files
│   └── cleaned_restaurant_dataset.csv
├── models/                 # Trained models
│   └── restaurant_balanced_model.joblib
├── src/                    # Source code
│   ├── train_model.py     # Model training script
│   ├── modern_app.py      # Main Streamlit application
│   ├── balanced_app.py    # Alternative app version
│   └── ultra_modern_app.py # Ultra-modern UI version
└── webapp/                 # Additional web components
```

## 🎯 How It Works

1. **Data Input**: Enter restaurant inspection details including:
   - Inspection type and date
   - Critical violations flag
   - Violation codes
   - Inspection score

2. **AI Analysis**: The machine learning model analyzes patterns from:
   - 67,000+ historical inspection records
   - 7 key features including temporal patterns
   - Weighted class balancing for accurate predictions

3. **Grade Prediction**: Get instant predictions with:
   - Confidence scores
   - Probability distributions
   - Actionable recommendations

## 📈 Grade System

| Grade | Score Range | Description |
|-------|-------------|-------------|
| **A** | 0-13 points | Excellent compliance with health codes |
| **B** | 14-27 points | Good compliance with minor violations |
| **C** | 28+ points | Significant violations requiring attention |

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **ML Framework**: Scikit-learn
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Model Persistence**: Joblib

## 📊 Key Features

### Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Features**: 7 engineered features including temporal data
- **Preprocessing**: Label encoding, standard scaling
- **Validation**: Cross-validation with balanced sampling

### Web Application
- **Modern UI**: Clean, professional interface
- **Real-time Predictions**: Instant grade predictions
- **Interactive Charts**: Dynamic visualizations
- **Responsive Design**: Works on desktop and mobile
- **Performance Metrics**: Live model statistics

## 🔧 Usage Examples

### Basic Prediction
```python
# Example inspection data
inspection_data = {
    'inspection_type': 'Cycle Inspection / Initial Inspection',
    'critical_flag': 'Critical',
    'violation_code': '02D',
    'score': 15,
    'inspection_year': 2024,
    'inspection_month': 10,
    'inspection_day_of_week': 1
}

# Get prediction
predicted_grade = model.predict(processed_data)
confidence = model.predict_proba(processed_data)
```

### Running Different App Versions
```bash
# Modern interface (recommended)
streamlit run src/modern_app.py

# Basic interface
streamlit run src/balanced_app.py

# Ultra-modern interface
streamlit run src/ultra_modern_app.py
```

## 📝 Model Training

The model is trained on real NYC Department of Health inspection data with the following process:

1. **Data Cleaning**: Remove invalid records and handle missing values
2. **Feature Engineering**: Extract temporal features from inspection dates
3. **Encoding**: Label encode categorical variables
4. **Scaling**: Standardize numerical features
5. **Training**: Random Forest with balanced class weights
6. **Validation**: Cross-validation and performance evaluation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NYC Department of Health for providing the inspection data
- Streamlit team for the amazing web framework
- Scikit-learn developers for the machine learning tools

## 📞 Contact

**Project Link**: [https://github.com/chanukas-byte/RestauranthealthGradePrediction](https://github.com/chanukas-byte/RestauranthealthGradePrediction)

---

⭐ **Star this repository if you found it helpful!** ⭐