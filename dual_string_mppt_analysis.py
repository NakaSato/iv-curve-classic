#!/usr/bin/env python3
"""
Comprehensive Dual-String MPPT Configuration Analysis System

This script provides advanced analysis for 2-string 1 MPPT configuration, including:
- String mismatch detection and quantification
- Individual string performance evaluation
- MPPT efficiency analysis for dual-string configuration
- Imbalance impact assessment
- Performance optimization recommendations
- Interactive visualizations with modern UI

Author: Advanced PV Analysis System
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Enhanced plotting configuration
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor': '#1a1a1a',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.color': '#333333',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.4,
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,
    'text.color': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#ffffff',
    'ytick.color': '#ffffff',
    'legend.edgecolor': '#333333',
    'legend.facecolor': '#1a1a1a'
})

class DualStringMPPTAnalyzer:
    """Advanced analyzer for dual-string MPPT configuration performance."""
    
    def __init__(self, csv_file_path: str):
        """Initialize the analyzer with CSV data file."""
        self.csv_file_path = Path(csv_file_path)
        self.data = None
        self.analysis_results = {}
        
        # Analysis parameters
        self.voltage_mismatch_threshold = 5.0  # Volts
        self.current_mismatch_threshold = 2.0  # Amps
        self.power_loss_threshold = 5.0  # Percentage
        
        # Color schemes for modern visualization
        self.colors = {
            'string1': '#00ff9d',  # Neon green
            'string2': '#ff6b6b',  # Coral red
            'combined': '#ffd93d', # Golden yellow
            'warning': '#ff8c42',  # Orange
            'critical': '#ff4757', # Red
            'excellent': '#2ed573', # Green
            'background': '#1a1a1a',
            'surface': '#2d3748',
            'accent': '#4fd1c7'
        }
        
    def load_and_process_data(self) -> pd.DataFrame:
        """Load and process the CSV data for dual-string analysis."""
        try:
            # Read CSV with proper handling of the header structure
            df = pd.read_csv(self.csv_file_path, skiprows=4)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert time column to datetime
            df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')
            
            # Extract dual-string data
            string_columns = ['Vstr1(V)', 'Vstr2(V)', 'Istr1(A)', 'Istr2(A)']
            
            # Ensure we have the required columns
            missing_cols = [col for col in string_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                return None
            
            # Convert string data to numeric, handling any non-numeric values
            for col in string_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate additional parameters
            df['P_str1(W)'] = df['Vstr1(V)'] * df['Istr1(A)']
            df['P_str2(W)'] = df['Vstr2(V)'] * df['Istr2(A)']
            df['P_total(W)'] = df['P_str1(W)'] + df['P_str2(W)']
            
            # Calculate mismatch parameters
            df['V_mismatch(V)'] = abs(df['Vstr1(V)'] - df['Vstr2(V)'])
            df['I_mismatch(A)'] = abs(df['Istr1(A)'] - df['Istr2(A)'])
            df['P_mismatch(%)'] = abs(df['P_str1(W)'] - df['P_str2(W)']) / (df['P_total(W)'] / 2) * 100
            
            # Filter out invalid/zero data points
            df = df[(df['Vstr1(V)'] > 0) & (df['Vstr2(V)'] > 0) & 
                   (df['Istr1(A)'] >= 0) & (df['Istr2(A)'] >= 0)]
            
            self.data = df
            print(f"✅ Loaded {len(df)} valid data points for dual-string analysis")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def analyze_string_performance(self) -> Dict[str, Any]:
        """Comprehensive analysis of individual string performance."""
        if self.data is None:
            return {}
        
        analysis = {
            'string1': {},
            'string2': {},
            'comparison': {},
            'statistics': {}
        }
        
        # String 1 Analysis
        str1_data = self.data[self.data['P_str1(W)'] > 0]
        analysis['string1'] = {
            'avg_voltage': str1_data['Vstr1(V)'].mean(),
            'avg_current': str1_data['Istr1(A)'].mean(),
            'avg_power': str1_data['P_str1(W)'].mean(),
            'max_power': str1_data['P_str1(W)'].max(),
            'min_power': str1_data['P_str1(W)'].min(),
            'std_voltage': str1_data['Vstr1(V)'].std(),
            'std_current': str1_data['Istr1(A)'].std(),
            'std_power': str1_data['P_str1(W)'].std(),
            'data_points': len(str1_data)
        }
        
        # String 2 Analysis
        str2_data = self.data[self.data['P_str2(W)'] > 0]
        analysis['string2'] = {
            'avg_voltage': str2_data['Vstr2(V)'].mean(),
            'avg_current': str2_data['Istr2(A)'].mean(),
            'avg_power': str2_data['P_str2(W)'].mean(),
            'max_power': str2_data['P_str2(W)'].max(),
            'min_power': str2_data['P_str2(W)'].min(),
            'std_voltage': str2_data['Vstr2(V)'].std(),
            'std_current': str2_data['Istr2(A)'].std(),
            'std_power': str2_data['P_str2(W)'].std(),
            'data_points': len(str2_data)
        }
        
        # Comparison and Mismatch Analysis
        analysis['comparison'] = {
            'voltage_diff_avg': abs(analysis['string1']['avg_voltage'] - analysis['string2']['avg_voltage']),
            'current_diff_avg': abs(analysis['string1']['avg_current'] - analysis['string2']['avg_current']),
            'power_diff_avg': abs(analysis['string1']['avg_power'] - analysis['string2']['avg_power']),
            'power_diff_percent': abs(analysis['string1']['avg_power'] - analysis['string2']['avg_power']) / 
                                 ((analysis['string1']['avg_power'] + analysis['string2']['avg_power']) / 2) * 100
        }
        
        # Overall Statistics
        analysis['statistics'] = {
            'total_avg_power': analysis['string1']['avg_power'] + analysis['string2']['avg_power'],
            'mismatch_frequency': len(self.data[self.data['V_mismatch(V)'] > self.voltage_mismatch_threshold]) / len(self.data) * 100,
            'severe_mismatch_frequency': len(self.data[self.data['P_mismatch(%)'] > self.power_loss_threshold]) / len(self.data) * 100
        }
        
        self.analysis_results['performance'] = analysis
        return analysis
    
    def detect_string_issues(self) -> Dict[str, Any]:
        """Advanced string issue detection and classification."""
        if self.data is None:
            return {}
        
        issues = {
            'voltage_mismatch': [],
            'current_mismatch': [],
            'power_imbalance': [],
            'performance_degradation': [],
            'critical_events': []
        }
        
        # Voltage mismatch detection
        voltage_issues = self.data[self.data['V_mismatch(V)'] > self.voltage_mismatch_threshold]
        for _, row in voltage_issues.iterrows():
            issues['voltage_mismatch'].append({
                'timestamp': row['Time'],
                'v_str1': row['Vstr1(V)'],
                'v_str2': row['Vstr2(V)'],
                'mismatch': row['V_mismatch(V)'],
                'severity': 'High' if row['V_mismatch(V)'] > self.voltage_mismatch_threshold * 2 else 'Moderate'
            })
        
        # Current mismatch detection
        current_issues = self.data[self.data['I_mismatch(A)'] > self.current_mismatch_threshold]
        for _, row in current_issues.iterrows():
            issues['current_mismatch'].append({
                'timestamp': row['Time'],
                'i_str1': row['Istr1(A)'],
                'i_str2': row['Istr2(A)'],
                'mismatch': row['I_mismatch(A)'],
                'severity': 'High' if row['I_mismatch(A)'] > self.current_mismatch_threshold * 2 else 'Moderate'
            })
        
        # Power imbalance detection
        power_issues = self.data[self.data['P_mismatch(%)'] > self.power_loss_threshold]
        for _, row in power_issues.iterrows():
            issues['power_imbalance'].append({
                'timestamp': row['Time'],
                'p_str1': row['P_str1(W)'],
                'p_str2': row['P_str2(W)'],
                'imbalance_percent': row['P_mismatch(%)'],
                'severity': 'Critical' if row['P_mismatch(%)'] > self.power_loss_threshold * 2 else 'Moderate'
            })
        
        # Performance degradation analysis
        if len(self.data) > 100:  # Need sufficient data points
            # Rolling average to detect trends
            window = min(50, len(self.data) // 4)
            self.data['P_total_rolling'] = self.data['P_total(W)'].rolling(window=window).mean()
            
            # Detect significant power drops
            power_drop_threshold = 0.15  # 15% drop
            for i in range(window, len(self.data)):
                current_avg = self.data['P_total_rolling'].iloc[i]
                previous_avg = self.data['P_total_rolling'].iloc[i-window//2]
                
                if previous_avg > 0 and (previous_avg - current_avg) / previous_avg > power_drop_threshold:
                    issues['performance_degradation'].append({
                        'timestamp': self.data['Time'].iloc[i],
                        'power_drop_percent': (previous_avg - current_avg) / previous_avg * 100,
                        'current_power': current_avg,
                        'previous_power': previous_avg
                    })
        
        # Critical events (combination of multiple issues)
        critical_data = self.data[
            (self.data['V_mismatch(V)'] > self.voltage_mismatch_threshold) &
            (self.data['I_mismatch(A)'] > self.current_mismatch_threshold) &
            (self.data['P_mismatch(%)'] > self.power_loss_threshold)
        ]
        
        for _, row in critical_data.iterrows():
            issues['critical_events'].append({
                'timestamp': row['Time'],
                'v_mismatch': row['V_mismatch(V)'],
                'i_mismatch': row['I_mismatch(A)'],
                'p_mismatch': row['P_mismatch(%)'],
                'total_power': row['P_total(W)']
            })
        
        self.analysis_results['issues'] = issues
        return issues
    
    def calculate_mppt_efficiency(self) -> Dict[str, Any]:
        """Calculate MPPT efficiency for dual-string configuration."""
        if self.data is None:
            return {}
        
        # Theoretical maximum power calculation
        # Assuming optimal MPPT would track each string independently
        theoretical_power = []
        actual_power = []
        
        for _, row in self.data.iterrows():
            # Theoretical: each string at its optimal power point
            str1_optimal = row['Vstr1(V)'] * row['Istr1(A)']
            str2_optimal = row['Vstr2(V)'] * row['Istr2(A)']
            
            # In dual-string single MPPT, the MPPT finds a compromise
            # This may not be optimal for both strings simultaneously
            actual_total = row['P_total(W)']
            theoretical_total = str1_optimal + str2_optimal
            
            theoretical_power.append(theoretical_total)
            actual_power.append(actual_total)
        
        theoretical_power = np.array(theoretical_power)
        actual_power = np.array(actual_power)
        
        # Calculate efficiency metrics
        valid_indices = (theoretical_power > 0) & (actual_power > 0)
        efficiency = actual_power[valid_indices] / theoretical_power[valid_indices] * 100
        
        mppt_analysis = {
            'average_efficiency': efficiency.mean(),
            'min_efficiency': efficiency.min(),
            'max_efficiency': efficiency.max(),
            'std_efficiency': efficiency.std(),
            'efficiency_data': efficiency,
            'power_loss_due_to_mismatch': (theoretical_power.mean() - actual_power.mean()),
            'relative_power_loss_percent': (theoretical_power.mean() - actual_power.mean()) / theoretical_power.mean() * 100
        }
        
        self.analysis_results['mppt_efficiency'] = mppt_analysis
        return mppt_analysis
    
    def generate_performance_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        if not self.analysis_results:
            return ["❌ No analysis results available. Please run analysis first."]
        
        # Check performance analysis
        if 'performance' in self.analysis_results:
            perf = self.analysis_results['performance']
            power_diff = perf['comparison']['power_diff_percent']
            
            if power_diff > 10:
                recommendations.append(
                    f"🔧 **Critical Power Imbalance**: {power_diff:.1f}% difference between strings. "
                    "Consider checking for soiling, shading, or module degradation."
                )
            elif power_diff > 5:
                recommendations.append(
                    f"⚠️ **Moderate Power Imbalance**: {power_diff:.1f}% difference between strings. "
                    "Monitor for potential issues."
                )
        
        # Check MPPT efficiency
        if 'mppt_efficiency' in self.analysis_results:
            mppt = self.analysis_results['mppt_efficiency']
            avg_eff = mppt['average_efficiency']
            power_loss = mppt['relative_power_loss_percent']
            
            if avg_eff < 90:
                recommendations.append(
                    f"🔧 **Low MPPT Efficiency**: {avg_eff:.1f}% average efficiency. "
                    f"Power loss due to string mismatch: {power_loss:.1f}%. "
                    "Consider installing individual string optimizers."
                )
            elif power_loss > 5:
                recommendations.append(
                    f"💡 **Optimization Opportunity**: {power_loss:.1f}% power loss from string mismatch. "
                    "String-level optimization could improve performance."
                )
        
        # Check for issues
        if 'issues' in self.analysis_results:
            issues = self.analysis_results['issues']
            
            if len(issues['critical_events']) > 0:
                recommendations.append(
                    f"🚨 **Critical Events Detected**: {len(issues['critical_events'])} instances of "
                    "simultaneous voltage, current, and power mismatches. Immediate inspection required."
                )
            
            if len(issues['voltage_mismatch']) > len(self.data) * 0.1:
                recommendations.append(
                    "⚡ **Frequent Voltage Mismatches**: Check for module degradation, "
                    "bypass diode failures, or connection issues."
                )
            
            if len(issues['performance_degradation']) > 0:
                recommendations.append(
                    f"📉 **Performance Degradation Detected**: {len(issues['performance_degradation'])} "
                    "instances of significant power drops. Schedule maintenance inspection."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ **System Operating Well**: No major issues detected in dual-string configuration.")
            
        recommendations.extend([
            "📊 **Monitoring**: Continue regular monitoring of string-level performance",
            "🔍 **Maintenance**: Schedule periodic inspection of connections and modules",
            "📈 **Optimization**: Consider string-level power optimizers for maximum efficiency"
        ])
        
        return recommendations
    
    def create_comprehensive_dashboard(self) -> None:
        """Create a comprehensive analysis dashboard with modern styling."""
        if self.data is None:
            print("❌ No data available for visualization")
            return
        
        # Create the main figure with subplots
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid layout
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('🔍 Dual-String MPPT Comprehensive Analysis Dashboard', 
                    fontsize=24, fontweight='bold', color='#ffffff', y=0.96)
        
        # 1. String Power Comparison (Time Series)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_string_power_comparison(ax1)
        
        # 2. Voltage vs Current Scatter
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_voltage_current_scatter(ax2)
        
        # 3. Mismatch Analysis
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_mismatch_analysis(ax3)
        
        # 4. Power Distribution
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_power_distribution(ax4)
        
        # 5. MPPT Efficiency Analysis
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_mppt_efficiency(ax5)
        
        # 6. Performance Metrics
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_performance_metrics(ax6)
        
        # 7. Issue Detection Timeline
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_issue_timeline(ax7)
        
        # Save the dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dual_string_mppt_comprehensive_dashboard_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='#0a0a0a', edgecolor='none')
        plt.show()
        
        print(f"✅ Dashboard saved as: {filename}")
    
    def _plot_string_power_comparison(self, ax):
        """Plot string power comparison over time."""
        time_data = self.data['Time']
        
        ax.plot(time_data, self.data['P_str1(W)'], 
               color=self.colors['string1'], linewidth=2, label='String 1 Power', alpha=0.8)
        ax.plot(time_data, self.data['P_str2(W)'], 
               color=self.colors['string2'], linewidth=2, label='String 2 Power', alpha=0.8)
        ax.plot(time_data, self.data['P_total(W)'], 
               color=self.colors['combined'], linewidth=3, label='Total Power', alpha=0.9)
        
        ax.set_title('⚡ String Power Performance Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Power (W)', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add performance statistics
        avg_str1 = self.data['P_str1(W)'].mean()
        avg_str2 = self.data['P_str2(W)'].mean()
        ax.text(0.02, 0.95, f'Avg String 1: {avg_str1:.0f}W\nAvg String 2: {avg_str2:.0f}W', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))
    
    def _plot_voltage_current_scatter(self, ax):
        """Plot voltage vs current scatter for both strings."""
        ax.scatter(self.data['Vstr1(V)'], self.data['Istr1(A)'], 
                  c=self.colors['string1'], alpha=0.6, s=30, label='String 1', edgecolors='white', linewidths=0.5)
        ax.scatter(self.data['Vstr2(V)'], self.data['Istr2(A)'], 
                  c=self.colors['string2'], alpha=0.6, s=30, label='String 2', edgecolors='white', linewidths=0.5)
        
        ax.set_title('🔌 Voltage vs Current Characteristics', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Voltage (V)', fontsize=12)
        ax.set_ylabel('Current (A)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_mismatch_analysis(self, ax):
        """Plot mismatch analysis over time."""
        time_data = self.data['Time']
        
        ax.plot(time_data, self.data['V_mismatch(V)'], 
               color=self.colors['warning'], linewidth=2, label='Voltage Mismatch (V)')
        ax.plot(time_data, self.data['I_mismatch(A)'], 
               color=self.colors['critical'], linewidth=2, label='Current Mismatch (A)')
        
        # Add threshold lines
        ax.axhline(y=self.voltage_mismatch_threshold, color=self.colors['warning'], 
                  linestyle='--', alpha=0.7, label=f'V Threshold ({self.voltage_mismatch_threshold}V)')
        ax.axhline(y=self.current_mismatch_threshold, color=self.colors['critical'], 
                  linestyle='--', alpha=0.7, label=f'I Threshold ({self.current_mismatch_threshold}A)')
        
        ax.set_title('⚠️ String Mismatch Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Mismatch Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_power_distribution(self, ax):
        """Plot power distribution histogram."""
        ax.hist(self.data['P_str1(W)'], bins=30, alpha=0.7, color=self.colors['string1'], 
               label='String 1', edgecolor='white', linewidth=0.5)
        ax.hist(self.data['P_str2(W)'], bins=30, alpha=0.7, color=self.colors['string2'], 
               label='String 2', edgecolor='white', linewidth=0.5)
        
        ax.set_title('📊 Power Distribution Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Power (W)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_mppt_efficiency(self, ax):
        """Plot MPPT efficiency analysis."""
        if 'mppt_efficiency' in self.analysis_results:
            efficiency_data = self.analysis_results['mppt_efficiency']['efficiency_data']
            
            ax.hist(efficiency_data, bins=30, alpha=0.8, color=self.colors['accent'], 
                   edgecolor='white', linewidth=0.5)
            
            avg_eff = efficiency_data.mean()
            ax.axvline(x=avg_eff, color=self.colors['warning'], linewidth=3, 
                      label=f'Average: {avg_eff:.1f}%')
            
            ax.set_title('⚙️ MPPT Efficiency Distribution', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Efficiency (%)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_performance_metrics(self, ax):
        """Plot key performance metrics as a bar chart."""
        if 'performance' in self.analysis_results:
            perf = self.analysis_results['performance']
            
            metrics = ['Avg Power\nString 1', 'Avg Power\nString 2', 'Power Diff\n(%)', 'Mismatch\nFreq (%)']
            values = [
                perf['string1']['avg_power'],
                perf['string2']['avg_power'],
                perf['comparison']['power_diff_percent'],
                perf['statistics']['mismatch_frequency']
            ]
            
            colors = [self.colors['string1'], self.colors['string2'], 
                     self.colors['warning'], self.colors['critical']]
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_title('📈 Key Performance Metrics', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_issue_timeline(self, ax):
        """Plot issue detection timeline."""
        if 'issues' in self.analysis_results:
            issues = self.analysis_results['issues']
            
            # Create timeline markers for different types of issues
            y_positions = {'voltage_mismatch': 1, 'current_mismatch': 2, 'power_imbalance': 3, 'critical_events': 4}
            colors_map = {'voltage_mismatch': self.colors['warning'], 'current_mismatch': self.colors['critical'],
                         'power_imbalance': '#ff6b6b', 'critical_events': '#ff4757'}
            
            for issue_type, issue_list in issues.items():
                if issue_type in y_positions and issue_list:
                    timestamps = [item['timestamp'] for item in issue_list]
                    y_vals = [y_positions[issue_type]] * len(timestamps)
                    ax.scatter(timestamps, y_vals, c=colors_map[issue_type], 
                             s=60, alpha=0.8, label=issue_type.replace('_', ' ').title())
            
            ax.set_title('🚨 Issue Detection Timeline', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Issue Type', fontsize=12)
            ax.set_yticks(list(y_positions.values()))
            ax.set_yticklabels([name.replace('_', ' ').title() for name in y_positions.keys()])
            if any(issues.values()):
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
    
    def generate_analysis_report(self) -> str:
        """Generate a comprehensive analysis report."""
        if not self.analysis_results:
            return "❌ No analysis results available. Please run analysis first."
        
        report = []
        report.append("=" * 80)
        report.append("🔍 DUAL-STRING MPPT CONFIGURATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📁 Data File: {self.csv_file_path.name}")
        report.append(f"📊 Data Points: {len(self.data)} valid measurements")
        report.append("")
        
        # Performance Summary
        if 'performance' in self.analysis_results:
            perf = self.analysis_results['performance']
            report.append("📈 PERFORMANCE SUMMARY")
            report.append("-" * 40)
            report.append(f"String 1 Average Power: {perf['string1']['avg_power']:.1f} W")
            report.append(f"String 2 Average Power: {perf['string2']['avg_power']:.1f} W")
            report.append(f"Total Average Power: {perf['statistics']['total_avg_power']:.1f} W")
            report.append(f"Power Difference: {perf['comparison']['power_diff_percent']:.1f}%")
            report.append(f"Voltage Difference: {perf['comparison']['voltage_diff_avg']:.2f} V")
            report.append(f"Current Difference: {perf['comparison']['current_diff_avg']:.2f} A")
            report.append("")
        
        # MPPT Efficiency
        if 'mppt_efficiency' in self.analysis_results:
            mppt = self.analysis_results['mppt_efficiency']
            report.append("⚙️ MPPT EFFICIENCY ANALYSIS")
            report.append("-" * 40)
            report.append(f"Average MPPT Efficiency: {mppt['average_efficiency']:.1f}%")
            report.append(f"Efficiency Range: {mppt['min_efficiency']:.1f}% - {mppt['max_efficiency']:.1f}%")
            report.append(f"Power Loss from Mismatch: {mppt['relative_power_loss_percent']:.1f}%")
            report.append("")
        
        # Issues Summary
        if 'issues' in self.analysis_results:
            issues = self.analysis_results['issues']
            report.append("🚨 ISSUES DETECTION SUMMARY")
            report.append("-" * 40)
            report.append(f"Voltage Mismatch Events: {len(issues['voltage_mismatch'])}")
            report.append(f"Current Mismatch Events: {len(issues['current_mismatch'])}")
            report.append(f"Power Imbalance Events: {len(issues['power_imbalance'])}")
            report.append(f"Critical Events: {len(issues['critical_events'])}")
            report.append(f"Performance Degradation Events: {len(issues['performance_degradation'])}")
            report.append("")
        
        # Recommendations
        recommendations = self.generate_performance_recommendations()
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete dual-string MPPT analysis pipeline."""
        print("🚀 Starting Dual-String MPPT Analysis...")
        
        # Load and process data
        if self.load_and_process_data() is None:
            return {}
        
        # Run all analysis components
        print("📊 Analyzing string performance...")
        self.analyze_string_performance()
        
        print("🔍 Detecting string issues...")
        self.detect_string_issues()
        
        print("⚙️ Calculating MPPT efficiency...")
        self.calculate_mppt_efficiency()
        
        # Generate visualizations
        print("📈 Creating comprehensive dashboard...")
        self.create_comprehensive_dashboard()
        
        # Generate report
        print("📝 Generating analysis report...")
        report = self.generate_analysis_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"dual_string_mppt_analysis_report_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"✅ Analysis complete! Report saved as: {report_filename}")
        print("\n" + "="*60)
        print(report)
        
        return self.analysis_results

def main():
    """Main function to run dual-string MPPT analysis."""
    # Path to the inverter data file
    csv_file_path = "/Users/chanthawat/Development/py-dev/iv-curve-classic/inverter/INVERTER_01_2025-04-04_2025-04-05.csv"
    
    # Create analyzer instance
    analyzer = DualStringMPPTAnalyzer(csv_file_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main()