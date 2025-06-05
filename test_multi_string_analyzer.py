#!/usr/bin/env python3
"""
Multi-String Data Generator and Test System
Generates realistic 32-string I-V data for testing the multi-string analyzer
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

def generate_multi_string_iv_data(num_strings=32, num_points=100, 
                                 base_voc=40.0, base_isc=8.5,
                                 variation_factor=0.1, output_file=None):
    """Generate realistic 32-string I-V curve data"""
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"multi_string_iv_data_{timestamp}.csv"
    
    print(f"🔧 Generating {num_strings}-string I-V data with {num_points} points per string...")
    
    # Create voltage column names
    voltage_cols = [f"Vstr{i}(V)" for i in range(1, num_strings + 1)]
    current_cols = [f"Istr{i}(A)" for i in range(1, num_strings + 1)]
    
    # Initialize data dictionary
    data = {}
    
    # Generate I-V data for each string
    for i in range(num_strings):
        string_num = i + 1
        
        # Add realistic variation between strings
        string_variation = 1 + (np.random.random() - 0.5) * variation_factor
        
        # String-specific parameters
        voc = base_voc * string_variation
        isc = base_isc * string_variation
        
        # Create voltage points (from 0 to Voc)
        voltage_points = np.linspace(0, voc, num_points)
        
        # Generate realistic I-V curve using single-diode model approximation
        # I = Isc * (1 - exp((V - Voc)/Vt))
        vt = 1.5  # Thermal voltage approximation
        current_points = isc * (1 - np.exp((voltage_points - voc) / vt))
        
        # Add realistic noise
        voltage_noise = np.random.normal(0, 0.1, num_points)
        current_noise = np.random.normal(0, 0.05, num_points)
        
        voltage_points += voltage_noise
        current_points += current_noise
        
        # Ensure physical constraints
        voltage_points = np.clip(voltage_points, 0, voc + 2)
        current_points = np.clip(current_points, 0, isc + 1)
        
        # Store in data dictionary
        data[voltage_cols[i]] = voltage_points
        data[current_cols[i]] = current_points
        
        if string_num % 8 == 0:
            print(f"✅ Generated string {string_num}: Voc={voc:.1f}V, Isc={isc:.1f}A")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"📁 Multi-string data saved: {output_file}")
    print(f"📊 Data shape: {df.shape}")
    print(f"🔌 Columns: {len(voltage_cols)} voltage + {len(current_cols)} current")
    
    return output_file, df

def test_multi_string_analyzer():
    """Test the multi-string analyzer with generated data"""
    
    print("🚀 Testing Multi-String I-V Analyzer")
    print("=" * 50)
    
    # Import the analyzer
    from multi_string_iv_analyzer import MultiStringIVAnalyzer
    
    # Generate test data
    print("📊 Step 1: Generating Test Data")
    data_file, df = generate_multi_string_iv_data(
        num_strings=32,
        num_points=150,
        variation_factor=0.15  # 15% variation between strings
    )
    
    # Initialize analyzer
    print("\n🔌 Step 2: Initializing Multi-String Analyzer")
    analyzer = MultiStringIVAnalyzer(max_strings=32)
    
    # Run comprehensive analysis
    print("\n🔍 Step 3: Running Comprehensive Analysis")
    try:
        result = analyzer.run_comprehensive_analysis(data_file)
        
        print(f"\n✅ Analysis Results:")
        print(f"📊 Active Strings: {result.active_strings}/{result.total_strings}")
        print(f"⚡ Total System Power: {result.system_performance['total_power_w']:.0f} W")
        print(f"📈 Average Efficiency: {result.system_performance['average_efficiency_pct']:.1f}%")
        print(f"⚖️ Power Imbalance: {result.imbalance_analysis['imbalance_severity']}")
        print(f"🎯 Recommendations: {len(result.optimization_recommendations)}")
        
        # Show top performing strings
        print(f"\n🏆 Top 5 Performing Strings:")
        for i, (string_id, power) in enumerate(result.performance_ranking[:5], 1):
            print(f"  {i}. String {string_id}: {power:.0f} W")
        
        # Show recommendations
        print(f"\n💡 Key Recommendations:")
        for i, rec in enumerate(result.optimization_recommendations[:3], 1):
            print(f"  {i}. {rec}")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the test
    test_result = test_multi_string_analyzer()
    
    if test_result:
        print("\n🎉 Multi-String Analysis Test Completed Successfully!")
    else:
        print("\n❌ Multi-String Analysis Test Failed!")
