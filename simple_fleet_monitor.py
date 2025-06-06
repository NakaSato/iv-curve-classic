#!/usr/bin/env python3
"""
Simple Fleet Monitoring System
==============================
A lightweight monitoring system for PV inverter fleet analysis.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

def simple_fleet_monitor():
    """Simple fleet monitoring function."""
    print("🔄 Simple Fleet Monitoring System")
    print("=" * 40)
    
    # Load fleet data from cleaned files
    csv_files = [f for f in os.listdir('.') if f.startswith('cleaned_INVERTER_') and f.endswith('.csv')]
    csv_files.sort()
    
    print(f'📁 Found {len(csv_files)} cleaned data files')
    
    fleet_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_systems': 0,
        'fleet_metrics': {},
        'system_analyses': {},
        'alerts': [],
        'recommendations': []
    }
    
    # Process each file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            inverter_id = csv_file.split('_')[1]
            
            # Basic analysis
            analysis = {
                'data_points': len(df),
                'efficiency': df['Efficiency'].mean() if 'Efficiency' in df.columns else 0,
                'health_score': 100.0,
                'status': 'normal',
                'max_temp': 0,
                'avg_power': 0
            }
            
            # Temperature analysis
            temp_cols = [col for col in df.columns if 'Temp' in col or 'temp' in col]
            if temp_cols:
                analysis['max_temp'] = df[temp_cols[0]].max()
                if analysis['max_temp'] > 85:
                    analysis['status'] = 'warning'
                    analysis['health_score'] -= 15
                    fleet_summary['alerts'].append({
                        'system': inverter_id,
                        'type': 'high_temperature',
                        'value': analysis['max_temp']
                    })
            
            # Power analysis
            if 'DC_Power' in df.columns:
                analysis['avg_power'] = df['DC_Power'].mean()
            
            # Efficiency check
            if 'Efficiency' in df.columns:
                if df['Efficiency'].mean() < 90:
                    analysis['status'] = 'warning'
                    analysis['health_score'] -= 10
                    fleet_summary['alerts'].append({
                        'system': inverter_id,
                        'type': 'efficiency_low',
                        'value': df['Efficiency'].mean()
                    })
                
                if df['Efficiency'].mean() < 80:
                    analysis['status'] = 'alert'
                    analysis['health_score'] -= 20
            
            fleet_summary['system_analyses'][inverter_id] = analysis
            fleet_summary['total_systems'] += 1
            
            status_icon = "✅" if analysis['status'] == 'normal' else "⚠️" if analysis['status'] == 'warning' else "🚨"
            print(f'  {status_icon} {inverter_id}: {len(df)} points, Eff: {analysis["efficiency"]:.1f}%, Health: {analysis["health_score"]:.1f}%')
            
        except Exception as e:
            print(f'  ❌ Error processing {csv_file}: {str(e)}')
    
    # Calculate fleet metrics
    if fleet_summary['system_analyses']:
        efficiencies = [s['efficiency'] for s in fleet_summary['system_analyses'].values()]
        health_scores = [s['health_score'] for s in fleet_summary['system_analyses'].values()]
        powers = [s['avg_power'] for s in fleet_summary['system_analyses'].values()]
        temps = [s['max_temp'] for s in fleet_summary['system_analyses'].values()]
        
        fleet_summary['fleet_metrics'] = {
            'average_efficiency': float(np.mean(efficiencies)),
            'average_health': float(np.mean(health_scores)),
            'total_power': float(np.sum(powers)),
            'max_temperature': float(np.max(temps)) if temps else 0,
            'systems_normal': sum(1 for s in fleet_summary['system_analyses'].values() if s['status'] == 'normal'),
            'systems_warning': sum(1 for s in fleet_summary['system_analyses'].values() if s['status'] == 'warning'),
            'systems_alert': sum(1 for s in fleet_summary['system_analyses'].values() if s['status'] == 'alert'),
            'total_alerts': len(fleet_summary['alerts'])
        }
    
    # Generate recommendations
    metrics = fleet_summary['fleet_metrics']
    if metrics['average_efficiency'] < 90:
        fleet_summary['recommendations'].append('Fleet efficiency below 90% - schedule maintenance checks')
    if metrics['total_alerts'] > 3:
        fleet_summary['recommendations'].append('Multiple systems showing issues - prioritize inspection')
    if metrics['max_temperature'] > 85:
        fleet_summary['recommendations'].append('High temperatures detected - check cooling systems')
    if metrics['systems_alert'] > 0:
        fleet_summary['recommendations'].append('Critical alerts present - immediate attention required')
    
    # Save report
    report_file = f"simple_fleet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(fleet_summary, f, indent=2, default=str)
    
    print(f'\n📊 Fleet Summary:')
    print(f'  🏭 Total Systems: {fleet_summary["total_systems"]}')
    print(f'  ⚡ Average Efficiency: {metrics["average_efficiency"]:.1f}%')
    print(f'  💚 Average Health: {metrics["average_health"]:.1f}%')
    print(f'  🔋 Total Power: {metrics["total_power"]:.1f} W')
    print(f'  🌡️  Max Temperature: {metrics["max_temperature"]:.1f}°C')
    print(f'  ✅ Systems Normal: {metrics["systems_normal"]}')
    print(f'  ⚠️  Systems Warning: {metrics["systems_warning"]}')
    print(f'  🚨 Systems Alert: {metrics["systems_alert"]}')
    print(f'  📢 Total Alerts: {metrics["total_alerts"]}')
    
    if fleet_summary['alerts']:
        print(f'\n⚠️  Top Alerts:')
        for i, alert in enumerate(fleet_summary['alerts'][:5], 1):
            print(f'    {i}. {alert["system"]}: {alert["type"]} = {alert["value"]:.1f}')
    
    if fleet_summary['recommendations']:
        print(f'\n💡 Recommendations:')
        for i, rec in enumerate(fleet_summary['recommendations'], 1):
            print(f'    {i}. {rec}')
    
    print(f'\n📋 Report saved: {report_file}')
    
    return fleet_summary

if __name__ == "__main__":
    simple_fleet_monitor()
