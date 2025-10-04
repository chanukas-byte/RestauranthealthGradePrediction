# Dataset Directory

This directory contains the dataset files for the restaurant health grade prediction system.

## Required Dataset

The application requires the NYC restaurant inspection dataset file:
**`cleaned_restaurant_dataset.csv`** (28MB)

## How to Obtain the Dataset

### Option 1: Use NYC Open Data (Recommended)

1. Visit the [NYC Open Data - Restaurant Inspection Results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j)
2. Download the CSV file
3. Rename it to `cleaned_restaurant_dataset.csv`
4. Place it in this `data/` directory

### Option 2: Data Processing Required

If you download the raw dataset, you may need to clean it:

1. **Download raw data** from NYC Open Data
2. **Clean the data** by running:

   ```python
   # Example data cleaning steps
   import pandas as pd

   # Load raw data
   df = pd.read_csv('raw_dataset.csv')

   # Basic cleaning
   df = df.dropna(subset=['Grade'])
   df = df[df['Grade'].isin(['A', 'B', 'C'])]

   # Save cleaned data
   df.to_csv('data/cleaned_restaurant_dataset.csv', index=False)
   ```

## Dataset Information

**Expected columns:**

- `Restaurant ID` - Unique identifier
- `Restaurant Name` - Name of the establishment
- `Borough` - NYC borough location
- `Cuisine Type` - Type of cuisine served
- `Inspection Date` - Date of inspection
- `Inspection Type` - Type of inspection conducted
- `Violation Code` - Specific violation code
- `Violation Description` - Description of violation
- `Critical Flag` - Whether violation is critical
- `Inspection Score` - Total violation points
- `Grade` - Final health grade (A, B, C)

**Dataset Size:** ~67,000 records after cleaning

## Note

The dataset file is excluded from the repository due to GitHub's 25MB file size limit. Make sure to download and place the dataset file before running the application or training the model.
