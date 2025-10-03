#!/usr/bin/env python3
"""
Setup script for Streamlit Cloud deployment
Downloads the NYC restaurant dataset if not present
"""

import os
import pandas as pd
import urllib.request
import zipfile
from pathlib import Path

def download_and_setup_data():
    """Download and prepare the dataset for the application"""
    
    data_dir = Path("data")
    data_file = data_dir / "cleaned_restaurant_dataset.csv"
    
    # Check if data already exists
    if data_file.exists():
        print("✅ Dataset already exists!")
        return True
    
    print("📥 Dataset not found. Setting up sample data...")
    
    try:
        # For Streamlit Cloud, we'll create a sample dataset
        # Since the full 28MB dataset is too large for GitHub
        sample_data = {
            'Restaurant ID': [41647571, 50128737, 40510804] * 1000,
            'Restaurant Name': ['Sample Restaurant A', 'Sample Restaurant B', 'Sample Restaurant C'] * 1000,
            'Location': ['Manhattan', 'Brooklyn', 'Queens'] * 1000,
            'Cuisine Type': ['American', 'Chinese', 'Italian'] * 1000,
            'Inspection Date': ['2023-01-01', '2023-02-01', '2023-03-01'] * 1000,
            'Inspection Type': ['Cycle Inspection / Initial Inspection'] * 3000,
            'Violation Code': ['02D', '04L', '06C'] * 1000,
            'Violation Description': ['Hot food item not held at or above 140º F.'] * 3000,
            'Critical Flag': ['Critical', 'Not Critical', 'Critical'] * 1000,
            'Inspection Score': [13, 25, 35] * 1000,
            'Record Date': ['2023-12-01'] * 3000,
            'Grade': ['A', 'B', 'C'] * 1000
        }
        
        # Create sample DataFrame
        df = pd.DataFrame(sample_data)
        
        # Save to CSV
        data_dir.mkdir(exist_ok=True)
        df.to_csv(data_file, index=False)
        
        print(f"✅ Sample dataset created with {len(df)} records")
        print(f"📁 Saved to: {data_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up data: {e}")
        return False

if __name__ == "__main__":
    download_and_setup_data()
