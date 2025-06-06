#!/usr/bin/env python3
"""
Enhanced Real Inverter Fleet Monitoring System
===============================================
Advanced monitoring system using actual inverter log data with 258 parameters.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

def enhanced_fleet_monitor():
    """Enhanced fleet monitoring with real inverter parameters."""
    print("🔄 Enhanced Real Inverter Fleet Monitoring System")
    print("=" * 55)
    
    # Load fleet data from cleaned files
    csv_files = [f for f in os.listdir('.') if f.startswith('cleaned_INVERTER_') and f.endswith('.csv')]
    csv_files.sort()
    
    print(f'📁 Found {len(csv_files)} cleaned inverter data files')
    
    fleet_summary = {
        'timestamp': datetime.now().isoformat(),
        'analysis_metadata': {
            'data_source': 'real_inverter_logs',
            'parameters_analyzed': 258,
            'string_configuration': '32-string',
            'analysis_type': 'production_fleet_monitoring'
        },
        'total_systems': 0,
        'fleet_metrics': {},
        'system_analyses': {},
        'alerts': [],
        'recommendations': [],
        'string_analysis': {},
        'performance_rankings': {}
    }
    
    all_systems_data = []
    
    # Process each inverter file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            inverter_id = csv_file.split('_')[1]
            
            print(f'\n🔍 Analyzing {inverter_id} ({len(df)} data points)...')
            
            # Comprehensive analysis
            analysis = {
                'inverter_id': inverter_id,
                'data_points': len(df),
                'date_range': {
                    'start': df['Time'].min() if 'Time' in df.columns else 'Unknown',
                    'end': df['Time'].max() if 'Time' in df.columns else 'Unknown'
                },
                'efficiency': 0,
                'health_score': 100.0,
                'status': 'normal',
                'performance_metrics': {},
                'string_analysis': {},
                'alerts': []
            }
            
            # Conversion Efficiency Analysis
            if 'Conversion_Efficiency(%)' in df.columns:
                eff_data = df['Conversion_Efficiency(%)'].dropna()
                if len(eff_data) > 0:
                    analysis['efficiency'] = float(eff_data.mean())
                    analysis['performance_metrics']['efficiency'] = {
                        'mean': float(eff_data.mean()),
                        'median': float(eff_data.median()),
                        'std': float(eff_data.std()),
                        'min': float(eff_data.min()),
                        'max': float(eff_data.max())
                    }
                    
                    # Efficiency alerts
                    if eff_data.mean() < 85:
                        analysis['status'] = 'alert'
                        analysis['health_score'] -= 25
                        analysis['alerts'].append({
                            'type': 'critical_efficiency',
                            'value': float(eff_data.mean()),
                            'threshold': 85,
                            'severity': 'critical'
                        })
                    elif eff_data.mean() < 90:
                        analysis['status'] = 'warning'
                        analysis['health_score'] -= 15
                        analysis['alerts'].append({
                            'type': 'low_efficiency',
                            'value': float(eff_data.mean()),
                            'threshold': 90,
                            'severity': 'medium'
                        })
            
            # Total Power Analysis
            if 'Total_String_Power(W)' in df.columns:
                power_data = df['Total_String_Power(W)'].dropna()
                if len(power_data) > 0:
                    analysis['performance_metrics']['total_power'] = {
                        'mean': float(power_data.mean()),
                        'peak': float(power_data.max()),
                        'min': float(power_data.min())
                    }
            
            # DC Power from individual PV strings (Ppv1-Ppv16)
            pv_power_cols = [f'Ppv{i}(W)' for i in range(1, 17)]
            pv_powers = []
            for col in pv_power_cols:
                if col in df.columns:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        pv_powers.append(float(col_data.mean()))
            
            if pv_powers:
                analysis['performance_metrics']['pv_string_power'] = {
                    'total_mean': float(np.sum(pv_powers)),
                    'average_per_string': float(np.mean(pv_powers)),
                    'string_count_active': len(pv_powers),
                    'imbalance_cv': float(np.std(pv_powers) / np.mean(pv_powers) * 100) if np.mean(pv_powers) > 0 else 0
                }
                
                # String imbalance alert
                cv = analysis['performance_metrics']['pv_string_power']['imbalance_cv']
                if cv > 20:
                    analysis['status'] = 'warning' if analysis['status'] == 'normal' else analysis['status']
                    analysis['health_score'] -= 10
                    analysis['alerts'].append({
                        'type': 'string_imbalance',
                        'value': float(cv),
                        'threshold': 20,
                        'severity': 'medium'
                    })
            
            # Temperature Analysis
            temp_cols = ['INVTemp(℃)', 'AMTemp1(℃)', 'BTTemp(℃)', 'OUTTemp(℃)', 'AMTemp2(℃)']
            temp_analysis = {}
            max_temp = 0
            
            for temp_col in temp_cols:
                if temp_col in df.columns:
                    temp_data = df[temp_col].dropna()
                    if len(temp_data) > 0:
                        temp_mean = float(temp_data.mean())
                        temp_max = float(temp_data.max())
                        temp_analysis[temp_col] = {
                            'mean': temp_mean,
                            'max': temp_max,
                            'min': float(temp_data.min())
                        }
                        max_temp = max(max_temp, temp_max)
                        
                        # Temperature alerts
                        if temp_max > 85:
                            analysis['status'] = 'alert' if temp_max > 90 else ('warning' if analysis['status'] == 'normal' else analysis['status'])
                            analysis['health_score'] -= 20 if temp_max > 90 else 10
                            analysis['alerts'].append({
                                'type': 'high_temperature',
                                'location': temp_col,
                                'value': temp_max,
                                'threshold': 85,
                                'severity': 'critical' if temp_max > 90 else 'medium'
                            })
            
            analysis['performance_metrics']['temperature'] = temp_analysis
            analysis['performance_metrics']['max_temperature'] = max_temp
            
            # AC Power Analysis
            if 'Pac(W)' in df.columns:
                ac_power_data = df['Pac(W)'].dropna()
                if len(ac_power_data) > 0:
                    analysis['performance_metrics']['ac_power'] = {
                        'mean': float(ac_power_data.mean()),
                        'peak': float(ac_power_data.max()),
                        'min': float(ac_power_data.min())
                    }
            
            # Frequency Analysis
            if 'Fac(Hz)' in df.columns:
                freq_data = df['Fac(Hz)'].dropna()
                if len(freq_data) > 0:
                    freq_mean = float(freq_data.mean())
                    analysis['performance_metrics']['frequency'] = {
                        'mean': freq_mean,
                        'std': float(freq_data.std()),
                        'min': float(freq_data.min()),
                        'max': float(freq_data.max())
                    }
                    
                    # Frequency alerts
                    if abs(freq_mean - 50.0) > 1.0:  # Assuming 50Hz grid
                        analysis['status'] = 'warning' if analysis['status'] == 'normal' else analysis['status']
                        analysis['health_score'] -= 5
                        analysis['alerts'].append({
                            'type': 'frequency_deviation',
                            'value': freq_mean,
                            'nominal': 50.0,
                            'deviation': abs(freq_mean - 50.0),
                            'severity': 'low'
                        })
            
            # String Power Analysis (32 strings)
            string_powers = []
            string_analysis = {}
            for i in range(1, 33):
                col = f'Pstr{i}(W)'
                if col in df.columns:
                    str_data = df[col].dropna()
                    if len(str_data) > 0 and str_data.mean() > 10:  # Active strings only
                        power_mean = float(str_data.mean())
                        string_powers.append(power_mean)
                        string_analysis[f'String_{i}'] = {
                            'mean_power': power_mean,
                            'peak_power': float(str_data.max()),
                            'active': True
                        }
                    else:
                        string_analysis[f'String_{i}'] = {
                            'mean_power': 0,
                            'peak_power': 0,
                            'active': False
                        }
            
            if string_powers:
                analysis['string_analysis'] = {
                    'active_strings': len(string_powers),
                    'total_strings': 32,
                    'total_power': float(np.sum(string_powers)),
                    'average_power': float(np.mean(string_powers)),
                    'power_std': float(np.std(string_powers)),
                    'imbalance_percentage': float(np.std(string_powers) / np.mean(string_powers) * 100) if np.mean(string_powers) > 0 else 0,
                    'string_details': string_analysis
                }
            
            # Energy Production Analysis
            if 'EacToday(kWh)' in df.columns:
                energy_data = df['EacToday(kWh)'].dropna()
                if len(energy_data) > 0:
                    analysis['performance_metrics']['daily_energy'] = {
                        'latest': float(energy_data.iloc[-1]) if len(energy_data) > 0 else 0,
                        'max': float(energy_data.max()),
                        'mean': float(energy_data.mean())
                    }
            
            # Fault Analysis
            fault_indicators = ['FaultCode', 'WarnCode', 'PidFaultCode']
            fault_summary = {}
            for fault_col in fault_indicators:
                if fault_col in df.columns:
                    fault_data = df[fault_col].dropna()
                    fault_count = len(fault_data[fault_data != 0]) if len(fault_data) > 0 else 0
                    fault_summary[fault_col] = fault_count
                    
                    if fault_count > 0:
                        analysis['status'] = 'alert'
                        analysis['health_score'] -= 15
                        analysis['alerts'].append({
                            'type': 'fault_detected',
                            'fault_type': fault_col,
                            'count': fault_count,
                            'severity': 'high'
                        })
            
            analysis['performance_metrics']['faults'] = fault_summary
            
            # Store analysis
            fleet_summary['system_analyses'][inverter_id] = analysis
            fleet_summary['total_systems'] += 1
            all_systems_data.append(analysis)
            
            # Add system alerts to fleet alerts
            for alert in analysis['alerts']:
                alert['inverter_id'] = inverter_id
                fleet_summary['alerts'].append(alert)
            
            # Print system summary
            status_icon = "✅" if analysis['status'] == 'normal' else "⚠️" if analysis['status'] == 'warning' else "🚨"
            active_strings = analysis['string_analysis']['active_strings'] if 'string_analysis' in analysis else 0
            print(f'  {status_icon} Efficiency: {analysis["efficiency"]:.1f}%, Health: {analysis["health_score"]:.1f}%, Active Strings: {active_strings}, Alerts: {len(analysis["alerts"])}')
            
        except Exception as e:
            print(f'  ❌ Error processing {csv_file}: {str(e)}')
            continue
    
    # Calculate fleet-wide metrics
    if all_systems_data:
        efficiencies = [s['efficiency'] for s in all_systems_data if s['efficiency'] > 0]
        health_scores = [s['health_score'] for s in all_systems_data]
        total_powers = [s['performance_metrics'].get('total_power', {}).get('mean', 0) for s in all_systems_data]
        max_temps = [s['performance_metrics'].get('max_temperature', 0) for s in all_systems_data]
        active_strings = [s['string_analysis'].get('active_strings', 0) for s in all_systems_data if 'string_analysis' in s]
        
        fleet_summary['fleet_metrics'] = {
            'average_efficiency': float(np.mean(efficiencies)) if efficiencies else 0,
            'efficiency_std': float(np.std(efficiencies)) if efficiencies else 0,
            'average_health': float(np.mean(health_scores)),
            'health_std': float(np.std(health_scores)),
            'total_fleet_power': float(np.sum(total_powers)),
            'average_system_power': float(np.mean(total_powers)),
            'max_fleet_temperature': float(np.max(max_temps)) if max_temps else 0,
            'average_temperature': float(np.mean(max_temps)) if max_temps else 0,
            'total_active_strings': int(np.sum(active_strings)),
            'average_strings_per_system': float(np.mean(active_strings)) if active_strings else 0,
            'systems_normal': sum(1 for s in all_systems_data if s['status'] == 'normal'),
            'systems_warning': sum(1 for s in all_systems_data if s['status'] == 'warning'),
            'systems_alert': sum(1 for s in all_systems_data if s['status'] == 'alert'),
            'total_alerts': len(fleet_summary['alerts']),
            'efficiency_range': {
                'min': float(np.min(efficiencies)) if efficiencies else 0,
                'max': float(np.max(efficiencies)) if efficiencies else 0
            }
        }
        
        # Performance rankings
        sorted_systems = sorted(all_systems_data, key=lambda x: x['efficiency'], reverse=True)
        fleet_summary['performance_rankings'] = {
            'top_performers': [
                {
                    'inverter_id': s['inverter_id'],
                    'efficiency': s['efficiency'],
                    'health_score': s['health_score']
                } for s in sorted_systems[:5]
            ],
            'needs_attention': [
                {
                    'inverter_id': s['inverter_id'],
                    'efficiency': s['efficiency'],
                    'health_score': s['health_score'],
                    'alert_count': len(s['alerts'])
                } for s in sorted_systems if s['health_score'] < 85
            ]
        }
    
    # Generate intelligent recommendations
    metrics = fleet_summary['fleet_metrics']
    recommendations = []
    
    if metrics['average_efficiency'] < 90:
        recommendations.append('🔧 Fleet efficiency below 90% - Schedule comprehensive maintenance')
    if metrics['total_alerts'] > 5:
        recommendations.append('⚠️ Multiple alerts detected - Prioritize system inspections')
    if metrics['max_fleet_temperature'] > 85:
        recommendations.append('🌡️ High temperatures detected - Check cooling and ventilation systems')
    if metrics['systems_alert'] > 0:
        recommendations.append('🚨 Critical systems detected - Immediate intervention required')
    if len(fleet_summary['performance_rankings']['needs_attention']) > 3:
        recommendations.append('📊 Multiple underperforming systems - Consider fleet-wide optimization')
    
    # String-specific recommendations
    low_string_systems = [s for s in all_systems_data if 'string_analysis' in s and s['string_analysis'].get('active_strings', 0) < 20]
    if low_string_systems:
        recommendations.append(f'🔌 {len(low_string_systems)} systems have low active string counts - Check string connections')
    
    fleet_summary['recommendations'] = recommendations
    
    # Save comprehensive report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"enhanced_fleet_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(fleet_summary, f, indent=2, default=str)
    
    # Display comprehensive summary
    print(f'\n📊 Enhanced Fleet Analysis Summary')
    print('=' * 50)
    print(f'🏭 Total Systems Analyzed: {fleet_summary["total_systems"]}')
    print(f'⚡ Average Fleet Efficiency: {metrics["average_efficiency"]:.2f}% ± {metrics["efficiency_std"]:.2f}%')
    print(f'💚 Average Health Score: {metrics["average_health"]:.1f}%')
    print(f'🔋 Total Fleet Power: {metrics["total_fleet_power"]:.1f} W')
    print(f'🔌 Total Active Strings: {metrics["total_active_strings"]} / {fleet_summary["total_systems"] * 32}')
    print(f'🌡️ Max Fleet Temperature: {metrics["max_fleet_temperature"]:.1f}°C')
    print(f'✅ Systems Normal: {metrics["systems_normal"]}')
    print(f'⚠️ Systems Warning: {metrics["systems_warning"]}')
    print(f'🚨 Systems Alert: {metrics["systems_alert"]}')
    print(f'📢 Total Alerts: {metrics["total_alerts"]}')
    
    if fleet_summary['performance_rankings']['top_performers']:
        print(f'\n🏆 Top 3 Performers:')
        for i, perf in enumerate(fleet_summary['performance_rankings']['top_performers'][:3], 1):
            print(f'   {i}. {perf["inverter_id"]}: {perf["efficiency"]:.1f}% efficiency')
    
    if fleet_summary['alerts']:
        print(f'\n⚠️ Critical Alerts (Top 5):')
        sorted_alerts = sorted(fleet_summary['alerts'], 
                             key=lambda x: {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}.get(x['severity'], 0), 
                             reverse=True)
        for i, alert in enumerate(sorted_alerts[:5], 1):
            print(f'   {i}. [{alert["severity"].upper()}] {alert["inverter_id"]}: {alert["type"]} = {alert.get("value", "N/A")}')
    
    if recommendations:
        print(f'\n💡 Fleet Optimization Recommendations:')
        for i, rec in enumerate(recommendations, 1):
            print(f'   {i}. {rec}')
    
    print(f'\n📋 Comprehensive Report: {report_file}')
    print(f'🔍 Analysis includes {fleet_summary["analysis_metadata"]["parameters_analyzed"]} parameters per system')
    
    return fleet_summary

if __name__ == "__main__":
    enhanced_fleet_monitor()
