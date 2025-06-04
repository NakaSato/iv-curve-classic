#!/usr/bin/env python3
"""
I-V Curve Analysis Summary Generator

This script generates a comprehensive summary report comparing all analyzed inverters,
including both real and synthetic data for demonstration purposes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

# Set styling
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def create_summary_report():
    """Create a comprehensive analysis summary comparing all inverters."""
    
    # Sample data from the recent analysis run
    inverter_data = {
        'INVERTER_01 (Real)': {
            'VOC': 750.50, 'ISC': 20.20, 'P_MAX': 101854.90, 'V_MP': 675.45, 
            'I_MP': 19.19, 'Fill_Factor': 0.850, 'RS': 0.50, 'RSH': 5458.00,
            'Health': 'Good', 'Severity': 'Low', 'Type': 'Real Data'
        },
        'INVERTER_02 (High Perf)': {
            'VOC': 788.02, 'ISC': 20.60, 'P_MAX': 110003.29, 'V_MP': 709.22,
            'I_MP': 19.33, 'Fill_Factor': 0.845, 'RS': 0.40, 'RSH': 6549.60,
            'Health': 'Good', 'Severity': 'Low', 'Type': 'Synthetic'
        },
        'INVERTER_03 (Normal)': {
            'VOC': 750.50, 'ISC': 20.20, 'P_MAX': 101854.90, 'V_MP': 652.93,
            'I_MP': 18.58, 'Fill_Factor': 0.800, 'RS': 0.50, 'RSH': 5458.00,
            'Health': 'Good', 'Severity': 'Low', 'Type': 'Synthetic'
        },
        'INVERTER_04 (Degraded)': {
            'VOC': 712.98, 'ISC': 19.59, 'P_MAX': 93706.51, 'V_MP': 589.27,
            'I_MP': 17.49, 'Fill_Factor': 0.738, 'RS': 0.65, 'RSH': 4366.40,
            'Health': 'Fair', 'Severity': 'Medium', 'Type': 'Synthetic'
        }
    }
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(inverter_data).T
    
    # Create comprehensive comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('I-V Curve Analysis Summary - Multi-Inverter Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Power Output Comparison
    ax1 = axes[0, 0]
    power_values = [data['P_MAX']/1000 for data in inverter_data.values()]  # Convert to kW
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars1 = ax1.bar(range(len(inverter_data)), power_values, color=colors, alpha=0.7)
    ax1.set_xlabel('Inverter')
    ax1.set_ylabel('Maximum Power (kW)')
    ax1.set_title('Power Output Comparison')
    ax1.set_xticks(range(len(inverter_data)))
    ax1.set_xticklabels([name.split('_')[1].split(' ')[0] for name in inverter_data.keys()], rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, power_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}kW', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Fill Factor Comparison
    ax2 = axes[0, 1]
    ff_values = [data['Fill_Factor'] for data in inverter_data.values()]
    bars2 = ax2.bar(range(len(inverter_data)), ff_values, color=colors, alpha=0.7)
    ax2.set_xlabel('Inverter')
    ax2.set_ylabel('Fill Factor')
    ax2.set_title('Fill Factor Comparison')
    ax2.set_xticks(range(len(inverter_data)))
    ax2.set_xticklabels([name.split('_')[1].split(' ')[0] for name in inverter_data.keys()], rotation=45)
    ax2.set_ylim([0.7, 0.9])
    
    # Add value labels
    for bar, value in zip(bars2, ff_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Voltage-Current Operating Points
    ax3 = axes[0, 2]
    for i, (name, data) in enumerate(inverter_data.items()):
        ax3.scatter(data['V_MP'], data['I_MP'], s=150, color=colors[i], 
                   label=name.split('_')[1].split(' ')[0], alpha=0.7)
    ax3.set_xlabel('Maximum Power Voltage (V)')
    ax3.set_ylabel('Maximum Power Current (A)')
    ax3.set_title('Operating Point Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Series Resistance Comparison
    ax4 = axes[1, 0]
    rs_values = [data['RS'] for data in inverter_data.values()]
    bars4 = ax4.bar(range(len(inverter_data)), rs_values, color=colors, alpha=0.7)
    ax4.set_xlabel('Inverter')
    ax4.set_ylabel('Series Resistance (Ω)')
    ax4.set_title('Series Resistance Comparison')
    ax4.set_xticks(range(len(inverter_data)))
    ax4.set_xticklabels([name.split('_')[1].split(' ')[0] for name in inverter_data.keys()], rotation=45)
    
    # Add value labels
    for bar, value in zip(bars4, rs_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}Ω', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Health Status Distribution
    ax5 = axes[1, 1]
    health_counts = {}
    for data in inverter_data.values():
        health = data['Health']
        health_counts[health] = health_counts.get(health, 0) + 1
    
    wedges, texts, autotexts = ax5.pie(health_counts.values(), labels=health_counts.keys(), 
                                      autopct='%1.0f%%', startangle=90, colors=['#2ca02c', '#ff7f0e'])
    ax5.set_title('Health Status Distribution')
    
    # Plot 6: Performance Summary Table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table data
    table_data = []
    for name, data in inverter_data.items():
        inv_name = name.split('_')[1].split(' ')[0]
        table_data.append([
            inv_name,
            f"{data['P_MAX']/1000:.1f}kW",
            f"{data['Fill_Factor']:.3f}",
            data['Health'],
            data['Type']
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Inverter', 'Power', 'Fill Factor', 'Health', 'Data Type'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f5f5f5' if i % 2 == 0 else 'white')
    
    ax6.set_title('Performance Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed summary report
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE I-V CURVE ANALYSIS SUMMARY")
    print("="*80)
    print(f"📊 Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 Inverters Analyzed: {len(inverter_data)}")
    print(f"🔍 Real Data Sources: {sum(1 for d in inverter_data.values() if d['Type'] == 'Real Data')}")
    print(f"🧪 Synthetic Scenarios: {sum(1 for d in inverter_data.values() if d['Type'] == 'Synthetic')}")
    
    print("\n📊 PERFORMANCE RANKING (by Power Output):")
    sorted_inverters = sorted(inverter_data.items(), key=lambda x: x[1]['P_MAX'], reverse=True)
    for i, (name, data) in enumerate(sorted_inverters, 1):
        inv_name = name.split('_')[1].split(' ')[0]
        print(f"  {i}. {inv_name:<12} | {data['P_MAX']/1000:>6.1f}kW | FF: {data['Fill_Factor']:>5.3f} | {data['Health']}")
    
    print("\n🏥 HEALTH STATUS SUMMARY:")
    for health, count in health_counts.items():
        percentage = (count / len(inverter_data)) * 100
        print(f"  {health:<10} | {count} inverters ({percentage:.0f}%)")
    
    print("\n⚠️  MAINTENANCE RECOMMENDATIONS:")
    for name, data in inverter_data.items():
        inv_name = name.split('_')[1].split(' ')[0]
        if data['Health'] == 'Fair':
            print(f"  🔧 {inv_name}: Schedule maintenance within 3 months (Fill Factor: {data['Fill_Factor']:.3f})")
        elif data['Fill_Factor'] < 0.80:
            print(f"  🔍 {inv_name}: Monitor closely (Fill Factor: {data['Fill_Factor']:.3f})")
    
    print("\n✅ ANALYSIS COMPLETE")
    print("📊 Comprehensive summary plot saved as: comprehensive_analysis_summary.png")
    print("="*80)

if __name__ == "__main__":
    create_summary_report()
