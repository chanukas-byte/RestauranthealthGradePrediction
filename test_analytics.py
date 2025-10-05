#!/usr/bin/env python3
"""
Test script to verify analytics functionality
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def test_data_loading():
    """Test if data loads correctly"""
    print("🔍 Testing data loading...")
    
    try:
        # Try to load the actual dataset
        data_path = Path("data/cleaned_restaurant_dataset.csv")
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            print(f"✅ Dataset loaded successfully: {len(df):,} records")
            print(f"📊 Columns: {list(df.columns)}")
            print(f"🏆 Grade distribution: {df['Grade'].value_counts().to_dict()}")
            return df
        else:
            print("❌ Dataset file not found")
            return None
            
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def test_chart_creation(df):
    """Test if charts can be created"""
    print("\n📊 Testing chart creation...")
    
    if df is None:
        print("❌ No data available for chart testing")
        return False
    
    try:
        # Test Grade Distribution Chart
        print("🧪 Testing grade distribution chart...")
        grade_counts = df['Grade'].value_counts().sort_index()
        
        fig_grades = go.Figure()
        colors = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c', 
                 'N': '#95a5a6', 'Z': '#3498db', 'P': '#9b59b6'}
        
        for grade in grade_counts.index:
            count = grade_counts[grade]
            percentage = (count / len(df)) * 100
            
            fig_grades.add_trace(go.Bar(
                x=[grade],
                y=[count],
                name=f'Grade {grade}',
                text=f'{count:,}<br>({percentage:.1f}%)',
                textposition='outside',
                marker_color=colors.get(grade, '#95a5a6')
            ))
        
        print("✅ Grade distribution chart created successfully")
        
        # Test Score Distribution Chart
        print("🧪 Testing score distribution chart...")
        if 'Inspection Score' in df.columns:
            df_scores = df.copy()
            df_scores['Inspection Score'] = pd.to_numeric(df_scores['Inspection Score'], errors='coerce')
            df_scores = df_scores.dropna(subset=['Inspection Score'])
            
            if len(df_scores) > 0:
                fig_scores = go.Figure()
                grades_order = ['A', 'B', 'C', 'N', 'Z', 'P']
                
                for grade in grades_order:
                    if grade in df_scores['Grade'].values:
                        grade_scores = df_scores[df_scores['Grade'] == grade]['Inspection Score']
                        
                        fig_scores.add_trace(go.Box(
                            y=grade_scores,
                            name=f'Grade {grade}',
                            marker_color=colors.get(grade, '#95a5a6')
                        ))
                
                print("✅ Score distribution chart created successfully")
            else:
                print("⚠️ No valid inspection scores found")
        
        # Test Cuisine Analysis
        print("🧪 Testing cuisine analysis...")
        if 'Cuisine Type' in df.columns:
            top_cuisines = df['Cuisine Type'].value_counts().head(15).index
            cuisine_subset = df[df['Cuisine Type'].isin(top_cuisines)]
            
            cuisine_performance = cuisine_subset.groupby('Cuisine Type').apply(
                lambda x: (x['Grade'] == 'A').mean() * 100
            ).sort_values(ascending=False)
            
            print(f"✅ Cuisine analysis completed. Top performer: {cuisine_performance.index[0]} ({cuisine_performance.iloc[0]:.1f}% Grade A)")
        
        # Test Critical Flag Analysis
        print("🧪 Testing critical flag analysis...")
        if 'Critical Flag' in df.columns:
            critical_dist = df['Critical Flag'].value_counts()
            print(f"✅ Critical flag analysis completed: {critical_dist.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating charts: {str(e)}")
        return False

def test_borough_analysis(df):
    """Test borough analysis"""
    print("\n🏙️ Testing borough analysis...")
    
    if df is None:
        print("❌ No data available for borough testing")
        return False
    
    try:
        if 'Location' in df.columns:
            borough_performance = df.groupby('Location').agg({
                'Grade': lambda x: (x == 'A').mean() * 100,
                'Restaurant ID': 'count'
            }).round(2)
            borough_performance.columns = ['Grade A Rate (%)', 'Total Restaurants']
            borough_performance = borough_performance.sort_values('Grade A Rate (%)', ascending=False)
            
            print("✅ Borough analysis completed:")
            print(borough_performance)
            return True
        else:
            print("⚠️ Location column not found")
            return False
            
    except Exception as e:
        print(f"❌ Error in borough analysis: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Analytics Dashboard Tests\n")
    
    # Test data loading
    df = test_data_loading()
    
    # Test chart creation
    charts_work = test_chart_creation(df)
    
    # Test borough analysis
    borough_work = test_borough_analysis(df)
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"Data Loading: {'✅ PASS' if df is not None else '❌ FAIL'}")
    print(f"Chart Creation: {'✅ PASS' if charts_work else '❌ FAIL'}")
    print(f"Borough Analysis: {'✅ PASS' if borough_work else '❌ FAIL'}")
    
    if df is not None and charts_work and borough_work:
        print("\n🎉 All tests PASSED! Analytics dashboard should work perfectly.")
    else:
        print("\n⚠️ Some tests FAILED. Analytics dashboard may have issues.")

if __name__ == "__main__":
    main()
