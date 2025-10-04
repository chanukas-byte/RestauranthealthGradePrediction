#!/bin/bash

# 🚀 Streamlit Cloud Deployment Script
# This script prepares and pushes your app for Streamlit Cloud deployment

echo "🚀 Preparing Restaurant Health Prediction App for Streamlit Cloud Deployment"
echo "============================================================================"

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found. Please run this script from the project root."
    exit 1
fi

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Error: This is not a git repository. Please initialize git first."
    exit 1
fi

echo "✅ Project structure validated"

# Check for required files
echo "📋 Checking deployment files..."

required_files=(
    "streamlit_app.py"
    "requirements.txt" 
    ".streamlit/config.toml"
    "data/cleaned_restaurant_dataset.csv"
    "models/restaurant_balanced_model.joblib"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file found"
    else
        echo "❌ $file missing"
        missing_files=true
    fi
done

if [ "$missing_files" = true ]; then
    echo "❌ Missing required files. Please ensure all files are present."
    exit 1
fi

echo "✅ All required files present"

# Check git status
echo "📊 Checking git status..."
if git diff --quiet; then
    echo "✅ Working directory clean"
else
    echo "📝 Changes detected. Adding files to git..."
    git add .
    git status --short
    
    echo "💬 Enter commit message (or press Enter for default):"
    read -r commit_message
    
    if [ -z "$commit_message" ]; then
        commit_message="🚀 Prepare for Streamlit Cloud deployment"
    fi
    
    git commit -m "$commit_message"
    echo "✅ Changes committed"
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to GitHub"
else
    echo "❌ Failed to push to GitHub. Please check your git configuration."
    exit 1
fi

echo ""
echo "🎉 Deployment Preparation Complete!"
echo "======================================="
echo ""
echo "📋 Next Steps:"
echo "1. Go to https://share.streamlit.io"
echo "2. Sign in with your GitHub account"
echo "3. Click 'New app' or 'Deploy an app'"
echo "4. Fill in the deployment form:"
echo "   Repository: chanukas-byte/RestauranthealthGradePrediction"
echo "   Branch: main"
echo "   Main file path: streamlit_app.py"
echo "   App URL: restaurant-health-predictor (optional)"
echo "5. Click 'Deploy!'"
echo ""
echo "🌐 Your app will be available at:"
echo "   https://[your-app-name].streamlit.app"
echo ""
echo "📚 For troubleshooting, see:"
echo "   ./STREAMLIT_CLOUD_DEPLOYMENT.md"
echo ""
echo "🎯 App Features Ready for Production:"
echo "   ✅ Multi-stakeholder interface (Health Authority, Restaurant Owner, Customer)"
echo "   ✅ Real-time AI predictions with 95.8% accuracy"
echo "   ✅ Interactive analytics dashboard"
echo "   ✅ Modern glassmorphism UI with bold black text"
echo "   ✅ Responsive design for all devices"
echo "   ✅ Optimized performance with caching"
echo ""
echo "🚀 Happy deploying!"
