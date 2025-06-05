#!/usr/bin/env python3
"""
Real-Time Dual-String MPPT Monitoring System

This module provides continuous monitoring capabilities for dual-string MPPT configurations:
- Real-time performance tracking
- Automated alert generation
- Historical trend analysis
- Performance benchmarking
- Predictive maintenance scheduling

Author: Advanced PV Analysis System
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dual_string_mppt_analysis import DualStringMPPTAnalyzer

warnings.filterwarnings('ignore')

class StringMonitoringSystem:
    """Advanced monitoring system for dual-string MPPT configurations."""
    
    def __init__(self, config_file: str = "string_monitoring_config.json"):
        """Initialize the monitoring system."""
        self.config_file = Path(config_file)
        self.config = self._load_or_create_config()
        self.monitoring_data = []
        self.alert_history = []
        
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load or create monitoring configuration."""
        default_config = {
            "monitoring_intervals": {
                "real_time": 300,  # 5 minutes
                "daily_summary": 86400,  # 24 hours
                "weekly_report": 604800  # 7 days
            },
            "alert_thresholds": {
                "power_imbalance_critical": 15.0,  # %
                "power_imbalance_warning": 8.0,   # %
                "voltage_mismatch_critical": 10.0,  # V
                "voltage_mismatch_warning": 5.0,   # V
                "current_mismatch_critical": 3.0,  # A
                "current_mismatch_warning": 1.5,  # A
                "efficiency_drop_critical": 85.0,  # %
                "efficiency_drop_warning": 90.0   # %
            },
            "data_retention": {
                "raw_data_days": 30,
                "summary_data_days": 365,
                "alert_history_days": 90
            },
            "notification_settings": {
                "email_alerts": False,
                "sms_alerts": False,
                "dashboard_alerts": True,
                "log_file_path": "string_monitoring.log"
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"⚠️ Error loading config, using defaults: {e}")
                return default_config
        else:
            # Create default config file
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"✅ Created default monitoring configuration: {self.config_file}")
            return default_config
    
    def analyze_csv_file(self, csv_file_path: str) -> Dict[str, Any]:
        """Analyze a CSV file and return results with alerts."""
        analyzer = DualStringMPPTAnalyzer(csv_file_path)
        results = analyzer.run_complete_analysis()
        
        # Generate alerts based on analysis
        alerts = self._generate_alerts(results)
        
        # Store monitoring data
        monitoring_entry = {
            'timestamp': datetime.now(),
            'file_path': csv_file_path,
            'results': results,
            'alerts': alerts
        }
        
        self.monitoring_data.append(monitoring_entry)
        self.alert_history.extend(alerts)
        
        # Clean old data according to retention policy
        self._cleanup_old_data()
        
        return {
            'analysis_results': results,
            'alerts': alerts,
            'monitoring_status': self._get_system_status()
        }
    
    def _generate_alerts(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on analysis results and configured thresholds."""
        alerts = []
        timestamp = datetime.now()
        
        if not analysis_results:
            return alerts
        
        # Performance-based alerts
        if 'performance' in analysis_results:
            perf = analysis_results['performance']
            power_diff = perf['comparison']['power_diff_percent']
            voltage_diff = perf['comparison']['voltage_diff_avg']
            current_diff = perf['comparison']['current_diff_avg']
            
            # Power imbalance alerts
            if power_diff >= self.config['alert_thresholds']['power_imbalance_critical']:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'CRITICAL',
                    'category': 'power_imbalance',
                    'message': f"Critical power imbalance detected: {power_diff:.1f}% difference between strings",
                    'value': power_diff,
                    'threshold': self.config['alert_thresholds']['power_imbalance_critical'],
                    'action_required': True
                })
            elif power_diff >= self.config['alert_thresholds']['power_imbalance_warning']:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'WARNING',
                    'category': 'power_imbalance',
                    'message': f"Power imbalance warning: {power_diff:.1f}% difference between strings",
                    'value': power_diff,
                    'threshold': self.config['alert_thresholds']['power_imbalance_warning'],
                    'action_required': False
                })
            
            # Voltage mismatch alerts
            if voltage_diff >= self.config['alert_thresholds']['voltage_mismatch_critical']:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'CRITICAL',
                    'category': 'voltage_mismatch',
                    'message': f"Critical voltage mismatch: {voltage_diff:.1f}V difference between strings",
                    'value': voltage_diff,
                    'threshold': self.config['alert_thresholds']['voltage_mismatch_critical'],
                    'action_required': True
                })
            elif voltage_diff >= self.config['alert_thresholds']['voltage_mismatch_warning']:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'WARNING',
                    'category': 'voltage_mismatch',
                    'message': f"Voltage mismatch warning: {voltage_diff:.1f}V difference between strings",
                    'value': voltage_diff,
                    'threshold': self.config['alert_thresholds']['voltage_mismatch_warning'],
                    'action_required': False
                })
        
        # MPPT efficiency alerts
        if 'mppt_efficiency' in analysis_results:
            mppt = analysis_results['mppt_efficiency']
            avg_efficiency = mppt['average_efficiency']
            
            if avg_efficiency <= self.config['alert_thresholds']['efficiency_drop_critical']:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'CRITICAL',
                    'category': 'efficiency_drop',
                    'message': f"Critical efficiency drop: {avg_efficiency:.1f}% MPPT efficiency",
                    'value': avg_efficiency,
                    'threshold': self.config['alert_thresholds']['efficiency_drop_critical'],
                    'action_required': True
                })
            elif avg_efficiency <= self.config['alert_thresholds']['efficiency_drop_warning']:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'WARNING',
                    'category': 'efficiency_drop',
                    'message': f"Efficiency warning: {avg_efficiency:.1f}% MPPT efficiency",
                    'value': avg_efficiency,
                    'threshold': self.config['alert_thresholds']['efficiency_drop_warning'],
                    'action_required': False
                })
        
        # Issues-based alerts
        if 'issues' in analysis_results:
            issues = analysis_results['issues']
            
            if len(issues['critical_events']) > 0:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'CRITICAL',
                    'category': 'critical_events',
                    'message': f"Critical events detected: {len(issues['critical_events'])} simultaneous faults",
                    'value': len(issues['critical_events']),
                    'threshold': 0,
                    'action_required': True
                })
            
            if len(issues['performance_degradation']) > 10:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'WARNING',
                    'category': 'performance_degradation',
                    'message': f"Multiple performance degradation events: {len(issues['performance_degradation'])} detected",
                    'value': len(issues['performance_degradation']),
                    'threshold': 10,
                    'action_required': False
                })
        
        return alerts
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data according to retention policy."""
        now = datetime.now()
        retention_days = self.config['data_retention']['raw_data_days']
        cutoff_date = now - timedelta(days=retention_days)
        
        # Filter out old monitoring data
        self.monitoring_data = [
            entry for entry in self.monitoring_data 
            if entry['timestamp'] > cutoff_date
        ]
        
        # Filter out old alerts
        alert_retention_days = self.config['data_retention']['alert_history_days']
        alert_cutoff_date = now - timedelta(days=alert_retention_days)
        
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert['timestamp'] > alert_cutoff_date
        ]
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system monitoring status."""
        if not self.monitoring_data:
            return {
                'status': 'NO_DATA',
                'last_analysis': None,
                'active_alerts': 0,
                'system_health': 'UNKNOWN'
            }
        
        latest_entry = self.monitoring_data[-1]
        
        # Count active alerts (last 24 hours)
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=24)
        active_alerts = len([
            alert for alert in self.alert_history 
            if alert['timestamp'] > recent_cutoff and alert['type'] == 'CRITICAL'
        ])
        
        # Determine system health
        if active_alerts > 0:
            health = 'CRITICAL'
        elif len([a for a in self.alert_history if a['timestamp'] > recent_cutoff]) > 0:
            health = 'WARNING'
        else:
            health = 'HEALTHY'
        
        return {
            'status': 'ACTIVE',
            'last_analysis': latest_entry['timestamp'],
            'active_alerts': active_alerts,
            'system_health': health,
            'data_points_stored': len(self.monitoring_data),
            'total_alerts': len(self.alert_history)
        }
    
    def generate_monitoring_dashboard(self) -> None:
        """Generate a comprehensive monitoring dashboard."""
        if not self.monitoring_data:
            print("❌ No monitoring data available")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.patch.set_facecolor('#0a0a0a')
        
        # Title
        fig.suptitle('🔍 Dual-String MPPT Monitoring Dashboard', 
                    fontsize=20, fontweight='bold', color='#ffffff', y=0.96)
        
        # 1. System Health Status
        ax1 = axes[0, 0]
        self._plot_system_health(ax1)
        
        # 2. Alert Timeline
        ax2 = axes[0, 1]
        self._plot_alert_timeline(ax2)
        
        # 3. Performance Trends
        ax3 = axes[1, 0]
        self._plot_performance_trends(ax3)
        
        # 4. Efficiency Trends
        ax4 = axes[1, 1]
        self._plot_efficiency_trends(ax4)
        
        # 5. Alert Distribution
        ax5 = axes[2, 0]
        self._plot_alert_distribution(ax5)
        
        # 6. Maintenance Scheduler
        ax6 = axes[2, 1]
        self._plot_maintenance_schedule(ax6)
        
        plt.tight_layout()
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"string_monitoring_dashboard_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='#0a0a0a', edgecolor='none')
        plt.show()
        
        print(f"✅ Monitoring dashboard saved as: {filename}")
    
    def _plot_system_health(self, ax):
        """Plot system health status."""
        status = self._get_system_status()
        health_colors = {
            'HEALTHY': '#2ed573',
            'WARNING': '#ff8c42',
            'CRITICAL': '#ff4757',
            'UNKNOWN': '#747d8c'
        }
        
        health = status['system_health']
        color = health_colors.get(health, '#747d8c')
        
        ax.pie([1], colors=[color], startangle=90)
        ax.text(0, 0, f'{health}\n{status["active_alerts"]} Active Alerts', 
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.set_title('🏥 System Health Status', fontsize=14, fontweight='bold', color='white', pad=20)
    
    def _plot_alert_timeline(self, ax):
        """Plot alert timeline for the last 7 days."""
        if not self.alert_history:
            ax.text(0.5, 0.5, 'No alerts in timeline', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='white')
            ax.set_title('📅 Alert Timeline (7 Days)', fontsize=14, fontweight='bold', color='white', pad=20)
            return
        
        # Filter last 7 days
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > week_ago]
        
        if not recent_alerts:
            ax.text(0.5, 0.5, 'No alerts in last 7 days', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='white')
            ax.set_title('📅 Alert Timeline (7 Days)', fontsize=14, fontweight='bold', color='white', pad=20)
            return
        
        # Group alerts by day
        alert_counts = {}
        for alert in recent_alerts:
            day = alert['timestamp'].date()
            if day not in alert_counts:
                alert_counts[day] = {'CRITICAL': 0, 'WARNING': 0}
            alert_counts[day][alert['type']] += 1
        
        days = sorted(alert_counts.keys())
        critical_counts = [alert_counts[day]['CRITICAL'] for day in days]
        warning_counts = [alert_counts[day]['WARNING'] for day in days]
        
        ax.bar(days, critical_counts, color='#ff4757', label='Critical', alpha=0.8)
        ax.bar(days, warning_counts, bottom=critical_counts, color='#ff8c42', label='Warning', alpha=0.8)
        
        ax.set_title('📅 Alert Timeline (7 Days)', fontsize=14, fontweight='bold', color='white', pad=20)
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Alert Count', fontsize=12, color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_trends(self, ax):
        """Plot performance trends over time."""
        if len(self.monitoring_data) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for trends', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='white')
            ax.set_title('📈 Performance Trends', fontsize=14, fontweight='bold', color='white', pad=20)
            return
        
        timestamps = [entry['timestamp'] for entry in self.monitoring_data]
        power_diffs = []
        
        for entry in self.monitoring_data:
            if 'performance' in entry['results']:
                power_diffs.append(entry['results']['performance']['comparison']['power_diff_percent'])
            else:
                power_diffs.append(0)
        
        ax.plot(timestamps, power_diffs, color='#ff6b6b', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=self.config['alert_thresholds']['power_imbalance_warning'], 
                  color='#ff8c42', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.axhline(y=self.config['alert_thresholds']['power_imbalance_critical'], 
                  color='#ff4757', linestyle='--', alpha=0.7, label='Critical Threshold')
        
        ax.set_title('📈 Power Imbalance Trends', fontsize=14, fontweight='bold', color='white', pad=20)
        ax.set_xlabel('Time', fontsize=12, color='white')
        ax.set_ylabel('Power Difference (%)', fontsize=12, color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_trends(self, ax):
        """Plot MPPT efficiency trends."""
        if len(self.monitoring_data) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for trends', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='white')
            ax.set_title('⚙️ MPPT Efficiency Trends', fontsize=14, fontweight='bold', color='white', pad=20)
            return
        
        timestamps = [entry['timestamp'] for entry in self.monitoring_data]
        efficiencies = []
        
        for entry in self.monitoring_data:
            if 'mppt_efficiency' in entry['results']:
                efficiencies.append(entry['results']['mppt_efficiency']['average_efficiency'])
            else:
                efficiencies.append(100)
        
        ax.plot(timestamps, efficiencies, color='#4fd1c7', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=self.config['alert_thresholds']['efficiency_drop_warning'], 
                  color='#ff8c42', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.axhline(y=self.config['alert_thresholds']['efficiency_drop_critical'], 
                  color='#ff4757', linestyle='--', alpha=0.7, label='Critical Threshold')
        
        ax.set_title('⚙️ MPPT Efficiency Trends', fontsize=14, fontweight='bold', color='white', pad=20)
        ax.set_xlabel('Time', fontsize=12, color='white')
        ax.set_ylabel('Efficiency (%)', fontsize=12, color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_alert_distribution(self, ax):
        """Plot alert distribution by category."""
        if not self.alert_history:
            ax.text(0.5, 0.5, 'No alerts to display', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='white')
            ax.set_title('🚨 Alert Distribution', fontsize=14, fontweight='bold', color='white', pad=20)
            return
        
        # Count alerts by category
        categories = {}
        for alert in self.alert_history:
            cat = alert['category']
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        if categories:
            labels = list(categories.keys())
            sizes = list(categories.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('🚨 Alert Distribution', fontsize=14, fontweight='bold', color='white', pad=20)
    
    def _plot_maintenance_schedule(self, ax):
        """Plot maintenance schedule recommendations."""
        # Calculate maintenance priority based on alert frequency
        maintenance_items = [
            'String Inspection',
            'Connection Check',
            'Module Cleaning',
            'Thermal Imaging',
            'Performance Testing'
        ]
        
        # Simple scoring based on recent alerts
        now = datetime.now()
        recent_cutoff = now - timedelta(days=30)
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > recent_cutoff]
        
        # Calculate priority scores (0-10 scale)
        priority_scores = []
        for item in maintenance_items:
            # Base score
            score = 3
            
            # Increase based on alert types
            for alert in recent_alerts:
                if alert['category'] in ['power_imbalance', 'voltage_mismatch']:
                    score += 1
                elif alert['category'] in ['critical_events', 'efficiency_drop']:
                    score += 2
            
            priority_scores.append(min(score, 10))  # Cap at 10
        
        colors = ['#ff4757' if score >= 8 else '#ff8c42' if score >= 6 else '#2ed573' 
                 for score in priority_scores]
        
        bars = ax.barh(maintenance_items, priority_scores, color=colors, alpha=0.8)
        
        ax.set_title('🔧 Maintenance Priority', fontsize=14, fontweight='bold', color='white', pad=20)
        ax.set_xlabel('Priority Score (0-10)', fontsize=12, color='white')
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add score labels
        for bar, score in zip(bars, priority_scores):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{score}', va='center', fontsize=10, color='white', fontweight='bold')
    
    def generate_monitoring_report(self) -> str:
        """Generate a comprehensive monitoring report."""
        report = []
        report.append("=" * 80)
        report.append("🔍 DUAL-STRING MPPT MONITORING SYSTEM REPORT")
        report.append("=" * 80)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        status = self._get_system_status()
        report.append(f"🏥 System Status: {status['status']}")
        report.append(f"💚 System Health: {status['system_health']}")
        report.append(f"🚨 Active Critical Alerts: {status['active_alerts']}")
        report.append(f"📊 Monitoring Sessions: {status['data_points_stored']}")
        report.append(f"📈 Total Alerts: {status['total_alerts']}")
        report.append("")
        
        # Recent alerts summary
        if self.alert_history:
            now = datetime.now()
            recent_cutoff = now - timedelta(hours=24)
            recent_alerts = [a for a in self.alert_history if a['timestamp'] > recent_cutoff]
            
            report.append("🚨 RECENT ALERTS (24 Hours)")
            report.append("-" * 40)
            
            if recent_alerts:
                for alert in recent_alerts[-10:]:  # Last 10 alerts
                    report.append(
                        f"{alert['timestamp'].strftime('%H:%M:%S')} | "
                        f"{alert['type']} | {alert['message']}"
                    )
            else:
                report.append("✅ No alerts in the last 24 hours")
            report.append("")
        
        # Performance summary
        if self.monitoring_data:
            latest = self.monitoring_data[-1]
            if 'performance' in latest['results']:
                perf = latest['results']['performance']
                report.append("📈 LATEST PERFORMANCE METRICS")
                report.append("-" * 40)
                report.append(f"String 1 Power: {perf['string1']['avg_power']:.1f} W")
                report.append(f"String 2 Power: {perf['string2']['avg_power']:.1f} W")
                report.append(f"Power Imbalance: {perf['comparison']['power_diff_percent']:.1f}%")
                
                if 'mppt_efficiency' in latest['results']:
                    mppt = latest['results']['mppt_efficiency']
                    report.append(f"MPPT Efficiency: {mppt['average_efficiency']:.1f}%")
                report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)

def main():
    """Main function for monitoring system demonstration."""
    # Initialize monitoring system
    monitor = StringMonitoringSystem()
    
    # Analyze the sample data file
    csv_file_path = "/Users/chanthawat/Development/py-dev/iv-curve-classic/inverter/INVERTER_01_2025-04-04_2025-04-05.csv"
    
    print("🚀 Starting String Monitoring System...")
    
    # Run analysis and monitoring
    results = monitor.analyze_csv_file(csv_file_path)
    
    # Generate monitoring dashboard
    print("📊 Generating monitoring dashboard...")
    monitor.generate_monitoring_dashboard()
    
    # Generate monitoring report
    print("📝 Generating monitoring report...")
    report = monitor.generate_monitoring_report()
    
    # Save monitoring report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"string_monitoring_report_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"✅ Monitoring report saved as: {report_filename}")
    print("\n" + report)
    
    return results

if __name__ == "__main__":
    main()
