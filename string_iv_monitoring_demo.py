#!/usr/bin/env python3
"""
String I-V Curve Monitoring Demonstration
Shows how to use the enhanced string I-V analyzer for ongoing monitoring
and per-string performance tracking
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import the enhanced string I-V analyzer
from enhanced_string_iv_analyzer import EnhancedStringIVAnalyzer

def demonstrate_string_iv_monitoring():
    """
    Demonstrate comprehensive string I-V curve monitoring capabilities
    """
    print("🔍 String I-V Curve Monitoring Demonstration")
    print("=" * 60)
    
    # Define data file path
    data_file = Path("inverter/INVERTER_01_2025-04-04_2025-04-05.csv")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📁 Analyzing data from: {data_file}")
    
    # Initialize the analyzer with the data file
    analyzer = EnhancedStringIVAnalyzer(str(data_file))
    
    # Load and analyze the data
    try:
        # Run complete analysis (this loads data internally)
        print("\n🔄 Running comprehensive string I-V analysis...")
        results = analyzer.run_complete_string_analysis()
        
        if not results:
            print("❌ Analysis failed")
            return
        
        print(f"✅ Analysis completed with {results['summary']['total_data_points']} data points")
        
        # Display key findings
        print("\n📈 KEY FINDINGS:")
        print("-" * 40)
        
        # String 1 Analysis
        string1_analysis = results['string1_analysis']
        print(f"🔌 String 1:")
        print(f"   • Average Power: {string1_analysis['performance']['average_power']:.1f} W")
        print(f"   • MPP Voltage: {string1_analysis['iv_curve']['mpp']['voltage']:.1f} V")
        print(f"   • MPP Current: {string1_analysis['iv_curve']['mpp']['current']:.3f} A")
        print(f"   • MPP Power: {string1_analysis['iv_curve']['mpp']['power']:.1f} W")
        print(f"   • Efficiency: {string1_analysis['efficiency']['average_efficiency']:.1f}%")
        print(f"   • Fill Factor: {string1_analysis['iv_curve']['characteristics']['fill_factor']:.3f}")
        
        # String 2 Analysis
        string2_analysis = results['string2_analysis']
        print(f"\n🔌 String 2:")
        print(f"   • Average Power: {string2_analysis['performance']['average_power']:.1f} W")
        print(f"   • MPP Voltage: {string2_analysis['iv_curve']['mpp']['voltage']:.1f} V")
        print(f"   • MPP Current: {string2_analysis['iv_curve']['mpp']['current']:.3f} A")
        print(f"   • MPP Power: {string2_analysis['iv_curve']['mpp']['power']:.1f} W")
        print(f"   • Efficiency: {string2_analysis['efficiency']['average_efficiency']:.1f}%")
        print(f"   • Fill Factor: {string2_analysis['iv_curve']['characteristics']['fill_factor']:.3f}")
        
        # Summary Analysis
        summary = results['summary']
        print(f"\n⚖️ Comparative Analysis:")
        print(f"   • Power Imbalance: {summary['power_imbalance_percent']:.1f}%")
        eff_diff = abs(string1_analysis['efficiency']['average_efficiency'] - 
                      string2_analysis['efficiency']['average_efficiency'])
        print(f"   • Efficiency Difference: {eff_diff:.1f}%")
        
        higher_power_string = "String 1" if string1_analysis['performance']['average_power'] > string2_analysis['performance']['average_power'] else "String 2"
        print(f"   • Higher Performing String: {higher_power_string}")
        
        # Generated files
        files = results['files_generated']
        dashboard_file = files['dashboard']
        report_file = files['report']
        
        print(f"\n🎨 Dashboard saved: {dashboard_file}")
        print(f"📝 Report saved: {report_file}")
        
        # Display monitoring recommendations
        print("\n💡 MONITORING RECOMMENDATIONS:")
        print("-" * 40)
        
        # Check for issues and provide recommendations
        issues = []
        
        # Power imbalance check
        power_imbalance = abs(summary['power_imbalance_percent'])
        if power_imbalance > 5:
            issues.append(f"High power imbalance ({power_imbalance:.1f}%) - Check for shading or module mismatch")
        
        # Efficiency check
        s1_eff = string1_analysis['efficiency']['average_efficiency']
        s2_eff = string2_analysis['efficiency']['average_efficiency']
        if s1_eff < 70 or s2_eff < 70:
            issues.append("Low efficiency detected - Consider cleaning and maintenance")
        
        # Fill factor check
        s1_ff = string1_analysis['iv_curve']['characteristics']['fill_factor']
        s2_ff = string2_analysis['iv_curve']['characteristics']['fill_factor']
        if s1_ff < 0.7 or s2_ff < 0.7:
            issues.append("Low fill factor - Check for series resistance issues")
        
        if issues:
            for i, issue in enumerate(issues, 1):
                print(f"{i}. {issue}")
        else:
            print("✅ No major issues detected - System operating normally")
        
        print(f"\n🎯 NEXT STEPS:")
        print("-" * 40)
        print("1. Review the detailed dashboard for I-V curve analysis")
        print("2. Check the comprehensive report for performance metrics")
        print("3. Set up regular monitoring using this analysis system")
        print("4. Compare results with historical data for trend analysis")
        print("5. Implement preventive maintenance based on recommendations")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return False
    
    return True

def demonstrate_batch_monitoring():
    """
    Demonstrate batch monitoring capabilities for multiple files
    """
    print("\n🔄 Batch Monitoring Demonstration")
    print("=" * 40)
    
    # Look for multiple data files
    data_dir = Path("inverter")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        print(f"📁 Found {len(csv_files)} data files in {data_dir}")
        
        for file_path in csv_files[:3]:  # Process first 3 files as demo
            print(f"\n📊 Processing: {file_path.name}")
            
            try:
                # Quick analysis for each file
                analyzer = EnhancedStringIVAnalyzer(str(file_path))
                results = analyzer.run_complete_string_analysis()
                
                if results:
                    s1_power = results['string1_analysis']['performance']['average_power']
                    s2_power = results['string2_analysis']['performance']['average_power']
                    imbalance = results['summary']['power_imbalance_percent']
                    
                    print(f"   String 1: {s1_power:.1f}W, String 2: {s2_power:.1f}W")
                    print(f"   Power Imbalance: {imbalance:.1f}%")
                    
                    if abs(imbalance) > 5:
                        print("   ⚠️  High imbalance detected!")
                else:
                    print("   ❌ Analysis failed")
            
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
    
    else:
        print(f"📁 Data directory not found: {data_dir}")

if __name__ == "__main__":
    print("🚀 Enhanced String I-V Curve Analysis System")
    print("=" * 60)
    print("This demonstration shows comprehensive per-string I-V analysis")
    print("capabilities including detailed performance metrics, curve")
    print("characteristics, and comparative analysis between strings.")
    print()
    
    # Run main demonstration
    success = demonstrate_string_iv_monitoring()
    
    if success:
        # Run batch monitoring demo
        demonstrate_batch_monitoring()
        
        print("\n🎉 Demonstration Complete!")
        print("=" * 60)
        print("The enhanced string I-V analyzer provides:")
        print("• Individual I-V curve analysis per string")
        print("• Comprehensive performance metrics")
        print("• MPP tracking and efficiency analysis")
        print("• Comparative analysis between strings")
        print("• Quality assessment and recommendations")
        print("• Visual dashboards and detailed reports")
        print()
        print("Use this system for ongoing monitoring and optimization")
        print("of your dual-string MPPT configuration!")
    else:
        print("\n❌ Demonstration failed. Check data files and try again.")
