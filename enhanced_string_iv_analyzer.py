#!/usr/bin/env python3
"""
Enhanced Dual-String I-V Curve Analysis System
Provides comprehensive I-V curve analysis per string with detailed reporting

This system extends the dual-string MPPT analysis to include:
- Individual I-V curve generation per string
- Per-string performance metrics and characteristics
- String-specific power, voltage, current analysis
- Individual string efficiency calculations
- Comparative I-V curve analysis between strings
- Detailed per-string reporting with recommendations
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
import json

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Enhanced plotting configuration for I-V curves
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

class EnhancedStringIVAnalyzer:
    """Enhanced analyzer for individual string I-V curve analysis and reporting."""
    
    def __init__(self, csv_file_path: str):
        """Initialize the enhanced string I-V analyzer."""
        self.csv_file_path = Path(csv_file_path)
        self.data = None
        self.string1_data = {}
        self.string2_data = {}
        self.analysis_results = {}
        
        # I-V curve analysis parameters
        self.iv_analysis_params = {
            'voltage_bins': 50,  # Number of voltage bins for I-V curve
            'power_threshold': 0.1,  # Minimum power threshold (fraction of max)
            'efficiency_threshold': 0.85,  # Minimum efficiency threshold
            'curve_smoothing': 3  # Smoothing factor for I-V curves
        }
        
        # Color schemes for string visualization
        self.colors = {
            'string1': '#00ff9d',  # Neon green
            'string2': '#ff6b6b',  # Coral red
            'string1_secondary': '#4ecdc4',  # Teal
            'string2_secondary': '#ff8a80',  # Light coral
            'combined': '#ffd93d',  # Golden yellow
            'mpp': '#ffffff',  # White for MPP
            'warning': '#ff8c42',  # Orange
            'critical': '#ff4757',  # Red
            'excellent': '#2ed573',  # Green
            'background': '#1a1a1a',
            'surface': '#2d3748',
            'accent': '#4fd1c7'
        }
        
    def load_and_process_data(self) -> pd.DataFrame:
        """Load and process CSV data for dual-string I-V analysis."""
        try:
            # Read CSV with proper handling
            df = pd.read_csv(self.csv_file_path, skiprows=4)
            df.columns = df.columns.str.strip()
            
            # Convert time to datetime
            df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')
            
            # Extract string data
            string_columns = ['Vstr1(V)', 'Vstr2(V)', 'Istr1(A)', 'Istr2(A)']
            
            # Verify columns exist
            missing_columns = [col for col in string_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing dual-string columns: {missing_columns}")
            
            # Clean and process data
            for col in string_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate power for each string
            df['Pstr1(W)'] = df['Vstr1(V)'] * df['Istr1(A)']
            df['Pstr2(W)'] = df['Vstr2(V)'] * df['Istr2(A)']
            
            # Filter valid data points
            valid_mask = (
                (df['Vstr1(V)'] > 0) & (df['Istr1(A)'] >= 0) &
                (df['Vstr2(V)'] > 0) & (df['Istr2(A)'] >= 0) &
                df[string_columns].notna().all(axis=1)
            )
            
            self.data = df[valid_mask].copy()
            print(f"✅ Loaded {len(self.data)} valid data points for dual-string I-V analysis")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def analyze_string_iv_characteristics(self, string_num: int) -> Dict[str, Any]:
        """Analyze I-V characteristics for a specific string."""
        if self.data is None:
            self.load_and_process_data()
        
        # Get string data
        v_col = f'Vstr{string_num}(V)'
        i_col = f'Istr{string_num}(A)'
        p_col = f'Pstr{string_num}(W)'
        
        voltage = self.data[v_col].values
        current = self.data[i_col].values
        power = self.data[p_col].values
        
        # I-V curve analysis
        iv_analysis = self._analyze_iv_curve(voltage, current, power, string_num)
        
        # Performance metrics
        performance_metrics = self._calculate_string_performance_metrics(
            voltage, current, power, string_num
        )
        
        # Operating point analysis
        operating_points = self._analyze_operating_points(voltage, current, power)
        
        # Efficiency analysis
        efficiency_analysis = self._analyze_string_efficiency(voltage, current, power)
        
        # Combine all analyses
        string_analysis = {
            'string_number': string_num,
            'iv_curve': iv_analysis,
            'performance': performance_metrics,
            'operating_points': operating_points,
            'efficiency': efficiency_analysis,
            'data_points': len(voltage),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return string_analysis
    
    def _analyze_iv_curve(self, voltage: np.ndarray, current: np.ndarray, 
                         power: np.ndarray, string_num: int) -> Dict[str, Any]:
        """Analyze I-V curve characteristics for a string."""
        
        # Find maximum power point (MPP)
        mpp_idx = np.argmax(power)
        mpp_voltage = voltage[mpp_idx]
        mpp_current = current[mpp_idx]
        mpp_power = power[mpp_idx]
        
        # Open circuit voltage (Voc) - extrapolate to I=0
        voc_indices = current < 0.1  # Near zero current
        if np.any(voc_indices):
            voc = np.max(voltage[voc_indices])
        else:
            voc = np.max(voltage)
        
        # Short circuit current (Isc) - extrapolate to V=0
        isc_indices = voltage < 50  # Near zero voltage
        if np.any(isc_indices):
            isc = np.max(current[isc_indices])
        else:
            isc = np.max(current)
        
        # Fill factor calculation
        fill_factor = mpp_power / (voc * isc) if (voc * isc) > 0 else 0
        
        # Create binned I-V curve for characteristic analysis
        voltage_bins = np.linspace(np.min(voltage), np.max(voltage), 
                                 self.iv_analysis_params['voltage_bins'])
        binned_iv = self._create_binned_iv_curve(voltage, current, voltage_bins)
        
        # Curve shape analysis
        curve_shape = self._analyze_curve_shape(voltage, current, power)
        
        # Series resistance estimation
        series_resistance = self._estimate_series_resistance(voltage, current, mpp_voltage, mpp_current)
        
        # Shunt resistance estimation
        shunt_resistance = self._estimate_shunt_resistance(voltage, current)
        
        return {
            'mpp': {
                'voltage': mpp_voltage,
                'current': mpp_current,
                'power': mpp_power
            },
            'characteristics': {
                'voc': voc,
                'isc': isc,
                'fill_factor': fill_factor,
                'series_resistance': series_resistance,
                'shunt_resistance': shunt_resistance
            },
            'binned_curve': binned_iv,
            'curve_shape': curve_shape,
            'quality_metrics': self._calculate_iv_quality_metrics(voltage, current, power)
        }
    
    def _create_binned_iv_curve(self, voltage: np.ndarray, current: np.ndarray, 
                              voltage_bins: np.ndarray) -> Dict[str, np.ndarray]:
        """Create binned I-V curve for characteristic analysis."""
        binned_current = []
        binned_voltage = []
        
        for i in range(len(voltage_bins) - 1):
            v_min, v_max = voltage_bins[i], voltage_bins[i + 1]
            mask = (voltage >= v_min) & (voltage < v_max)
            
            if np.any(mask):
                avg_voltage = np.mean(voltage[mask])
                avg_current = np.mean(current[mask])
                binned_voltage.append(avg_voltage)
                binned_current.append(avg_current)
        
        return {
            'voltage': np.array(binned_voltage),
            'current': np.array(binned_current),
            'power': np.array(binned_voltage) * np.array(binned_current)
        }
    
    def _analyze_curve_shape(self, voltage: np.ndarray, current: np.ndarray, 
                           power: np.ndarray) -> Dict[str, Any]:
        """Analyze I-V curve shape characteristics."""
        
        # Calculate derivatives
        dI_dV = np.gradient(current, voltage)
        dP_dV = np.gradient(power, voltage)
        
        # Find knee point (maximum dP/dV)
        knee_idx = np.argmax(np.abs(dP_dV))
        knee_voltage = voltage[knee_idx]
        knee_current = current[knee_idx]
        
        # Curve linearity in different regions
        low_v_mask = voltage < np.percentile(voltage, 25)
        high_v_mask = voltage > np.percentile(voltage, 75)
        
        low_v_linearity = self._calculate_linearity(voltage[low_v_mask], current[low_v_mask])
        high_v_linearity = self._calculate_linearity(voltage[high_v_mask], current[high_v_mask])
        
        return {
            'knee_point': {'voltage': knee_voltage, 'current': knee_current},
            'max_derivative': np.max(np.abs(dI_dV)),
            'linearity': {
                'low_voltage_region': low_v_linearity,
                'high_voltage_region': high_v_linearity
            },
            'curve_smoothness': np.std(dI_dV)
        }
    
    def _calculate_linearity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate linearity coefficient (R²) for a data segment."""
        if len(x) < 3:
            return 0.0
        
        try:
            correlation_matrix = np.corrcoef(x, y)
            return correlation_matrix[0, 1] ** 2
        except:
            return 0.0
    
    def _estimate_series_resistance(self, voltage: np.ndarray, current: np.ndarray,
                                  mpp_voltage: float, mpp_current: float) -> float:
        """Estimate series resistance from I-V curve."""
        
        # Use high current region for series resistance estimation
        high_current_mask = current > 0.8 * np.max(current)
        if np.sum(high_current_mask) < 3:
            return 0.0
        
        v_hc = voltage[high_current_mask]
        i_hc = current[high_current_mask]
        
        # Linear fit in high current region: V = V0 - Rs * I
        try:
            coeffs = np.polyfit(i_hc, v_hc, 1)
            series_resistance = -coeffs[0]  # Negative slope
            return max(0, series_resistance)  # Ensure positive
        except:
            return 0.0
    
    def _estimate_shunt_resistance(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Estimate shunt resistance from I-V curve."""
        
        # Use low voltage region for shunt resistance estimation
        low_voltage_mask = voltage < 0.2 * np.max(voltage)
        if np.sum(low_voltage_mask) < 3:
            return float('inf')
        
        v_lv = voltage[low_voltage_mask]
        i_lv = current[low_voltage_mask]
        
        # Linear fit in low voltage region: I = V/Rsh + I0
        try:
            if len(v_lv) > 1:
                coeffs = np.polyfit(v_lv, i_lv, 1)
                shunt_conductance = coeffs[0]
                shunt_resistance = 1 / shunt_conductance if shunt_conductance > 0 else float('inf')
                return shunt_resistance
        except:
            pass
        
        return float('inf')
    
    def _calculate_string_performance_metrics(self, voltage: np.ndarray, current: np.ndarray,
                                            power: np.ndarray, string_num: int) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for a string."""
        
        return {
            'average_power': np.mean(power),
            'max_power': np.max(power),
            'min_power': np.min(power),
            'power_std': np.std(power),
            'average_voltage': np.mean(voltage),
            'voltage_range': {'min': np.min(voltage), 'max': np.max(voltage)},
            'average_current': np.mean(current),
            'current_range': {'min': np.min(current), 'max': np.max(current)},
            'power_factor': np.mean(power) / np.max(power) if np.max(power) > 0 else 0,
            'stability_index': 1 - (np.std(power) / np.mean(power)) if np.mean(power) > 0 else 0,
            'performance_ratio': np.mean(power) / np.max(power) if np.max(power) > 0 else 0
        }
    
    def _analyze_operating_points(self, voltage: np.ndarray, current: np.ndarray,
                                power: np.ndarray) -> Dict[str, Any]:
        """Analyze operating point distribution and characteristics."""
        
        # Operating point clusters
        power_percentiles = np.percentile(power, [25, 50, 75, 90, 95])
        
        operating_regions = {
            'low_power': np.sum(power < power_percentiles[0]),
            'medium_power': np.sum((power >= power_percentiles[0]) & (power < power_percentiles[1])),
            'high_power': np.sum((power >= power_percentiles[1]) & (power < power_percentiles[2])),
            'peak_power': np.sum(power >= power_percentiles[2])
        }
        
        # Time spent at different power levels
        total_points = len(power)
        operating_time_distribution = {
            region: count / total_points * 100 
            for region, count in operating_regions.items()
        }
        
        # MPP tracking efficiency
        max_power = np.max(power)
        mpp_efficiency_threshold = 0.95 * max_power
        mpp_tracking_time = np.sum(power >= mpp_efficiency_threshold) / total_points * 100
        
        return {
            'operating_regions': operating_regions,
            'time_distribution': operating_time_distribution,
            'mpp_tracking_efficiency': mpp_tracking_time,
            'power_percentiles': power_percentiles.tolist()
        }
    
    def _analyze_string_efficiency(self, voltage: np.ndarray, current: np.ndarray,
                                 power: np.ndarray) -> Dict[str, Any]:
        """Analyze string efficiency characteristics."""
        
        # Theoretical maximum power (assuming ideal conditions)
        theoretical_max = np.max(power) * 1.1  # 10% margin for ideal conditions
        
        # Efficiency metrics
        instantaneous_efficiency = power / theoretical_max if theoretical_max > 0 else np.zeros_like(power)
        average_efficiency = np.mean(instantaneous_efficiency) * 100
        
        # Efficiency distribution
        efficiency_percentiles = np.percentile(instantaneous_efficiency, [10, 25, 50, 75, 90])
        
        # Low efficiency periods
        low_efficiency_threshold = 0.7
        low_efficiency_periods = np.sum(instantaneous_efficiency < low_efficiency_threshold)
        
        return {
            'average_efficiency': average_efficiency,
            'efficiency_range': {
                'min': np.min(instantaneous_efficiency) * 100,
                'max': np.max(instantaneous_efficiency) * 100
            },
            'efficiency_percentiles': (efficiency_percentiles * 100).tolist(),
            'low_efficiency_periods': low_efficiency_periods,
            'efficiency_stability': 1 - np.std(instantaneous_efficiency)
        }
    
    def _calculate_iv_quality_metrics(self, voltage: np.ndarray, current: np.ndarray,
                                    power: np.ndarray) -> Dict[str, Any]:
        """Calculate I-V curve quality metrics."""
        
        # Data quality indicators
        voltage_noise = np.std(np.diff(voltage))
        current_noise = np.std(np.diff(current))
        power_noise = np.std(np.diff(power))
        
        # Curve completeness
        voltage_span = (np.max(voltage) - np.min(voltage)) / np.max(voltage)
        current_span = (np.max(current) - np.min(current)) / np.max(current)
        
        # Data density
        unique_voltage_points = len(np.unique(np.round(voltage, 1)))
        unique_current_points = len(np.unique(np.round(current, 2)))
        
        return {
            'noise_levels': {
                'voltage': voltage_noise,
                'current': current_noise,
                'power': power_noise
            },
            'curve_completeness': {
                'voltage_span': voltage_span,
                'current_span': current_span
            },
            'data_density': {
                'voltage_points': unique_voltage_points,
                'current_points': unique_current_points
            },
            'overall_quality_score': self._calculate_quality_score(
                voltage_noise, current_noise, voltage_span, current_span
            )
        }
    
    def _calculate_quality_score(self, v_noise: float, i_noise: float,
                               v_span: float, i_span: float) -> float:
        """Calculate overall I-V curve quality score (0-100)."""
        
        # Normalize noise (lower is better)
        noise_score = max(0, 100 - (v_noise + i_noise) * 1000)
        
        # Normalize span (higher is better)
        span_score = min(100, (v_span + i_span) * 50)
        
        # Combined score
        quality_score = (noise_score * 0.6 + span_score * 0.4)
        return max(0, min(100, quality_score))
    
    def create_string_iv_dashboard(self) -> None:
        """Create comprehensive I-V curve dashboard for both strings."""
        
        print("📊 Creating comprehensive string I-V curve dashboard...")
        
        # Analyze both strings
        string1_analysis = self.analyze_string_iv_characteristics(1)
        string2_analysis = self.analyze_string_iv_characteristics(2)
        
        # Create dashboard
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Dual-String I-V Curve Comprehensive Analysis Dashboard', 
                    fontsize=24, fontweight='bold', color='white', y=0.98)
        
        # 1. I-V Curves Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_iv_curves_comparison(ax1, string1_analysis, string2_analysis)
        
        # 2. Power Curves Comparison
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_power_curves_comparison(ax2, string1_analysis, string2_analysis)
        
        # 3. String 1 Detailed Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_string_detailed_analysis(ax3, string1_analysis, 1)
        
        # 4. String 2 Detailed Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_string_detailed_analysis(ax4, string2_analysis, 2)
        
        # 5. Performance Comparison
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_performance_comparison(ax5, string1_analysis, string2_analysis)
        
        # 6. Operating Points Analysis
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_operating_points_analysis(ax6, string1_analysis, string2_analysis)
        
        # 7. Efficiency Analysis
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_efficiency_analysis(ax7, string1_analysis, string2_analysis)
        
        # 8. I-V Characteristics Summary
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_iv_characteristics_summary(ax8, string1_analysis, string2_analysis)
        
        # 9. Quality Metrics
        ax9 = fig.add_subplot(gs[3, :])
        self._plot_quality_metrics(ax9, string1_analysis, string2_analysis)
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_filename = f'string_iv_analysis_dashboard_{timestamp}.png'
        plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight', 
                   facecolor='#0a0a0a', edgecolor='none')
        plt.close()
        
        print(f"✅ String I-V analysis dashboard saved as: {dashboard_filename}")
        
        # Store analysis results
        self.string1_data = string1_analysis
        self.string2_data = string2_analysis
        
        return dashboard_filename
    
    def _plot_iv_curves_comparison(self, ax, string1_analysis, string2_analysis):
        """Plot I-V curves comparison between strings."""
        
        # Get I-V data
        if self.data is not None:
            v1, i1 = self.data['Vstr1(V)'], self.data['Istr1(A)']
            v2, i2 = self.data['Vstr2(V)'], self.data['Istr2(A)']
            
            # Plot scatter points
            ax.scatter(v1, i1, c=self.colors['string1'], alpha=0.6, s=2, label='String 1')
            ax.scatter(v2, i2, c=self.colors['string2'], alpha=0.6, s=2, label='String 2')
            
            # Plot binned curves
            if 'binned_curve' in string1_analysis['iv_curve']:
                binned1 = string1_analysis['iv_curve']['binned_curve']
                ax.plot(binned1['voltage'], binned1['current'], 
                       color=self.colors['string1'], linewidth=3, alpha=0.8)
            
            if 'binned_curve' in string2_analysis['iv_curve']:
                binned2 = string2_analysis['iv_curve']['binned_curve']
                ax.plot(binned2['voltage'], binned2['current'], 
                       color=self.colors['string2'], linewidth=3, alpha=0.8)
            
            # Mark MPP points
            mpp1 = string1_analysis['iv_curve']['mpp']
            mpp2 = string2_analysis['iv_curve']['mpp']
            
            ax.scatter(mpp1['voltage'], mpp1['current'], 
                      c=self.colors['mpp'], s=100, marker='*', 
                      edgecolor=self.colors['string1'], linewidth=2, label='String 1 MPP')
            ax.scatter(mpp2['voltage'], mpp2['current'], 
                      c=self.colors['mpp'], s=100, marker='*', 
                      edgecolor=self.colors['string2'], linewidth=2, label='String 2 MPP')
        
        ax.set_xlabel('Voltage (V)', fontweight='bold')
        ax.set_ylabel('Current (A)', fontweight='bold')
        ax.set_title('I-V Curves Comparison', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_power_curves_comparison(self, ax, string1_analysis, string2_analysis):
        """Plot power curves comparison."""
        
        if self.data is not None:
            v1, p1 = self.data['Vstr1(V)'], self.data['Pstr1(W)']
            v2, p2 = self.data['Vstr2(V)'], self.data['Pstr2(W)']
            
            ax.scatter(v1, p1, c=self.colors['string1'], alpha=0.6, s=2, label='String 1')
            ax.scatter(v2, p2, c=self.colors['string2'], alpha=0.6, s=2, label='String 2')
            
            # Mark MPP
            mpp1 = string1_analysis['iv_curve']['mpp']
            mpp2 = string2_analysis['iv_curve']['mpp']
            
            ax.scatter(mpp1['voltage'], mpp1['power'], 
                      c=self.colors['mpp'], s=100, marker='*', 
                      edgecolor=self.colors['string1'], linewidth=2)
            ax.scatter(mpp2['voltage'], mpp2['power'], 
                      c=self.colors['mpp'], s=100, marker='*', 
                      edgecolor=self.colors['string2'], linewidth=2)
        
        ax.set_xlabel('Voltage (V)', fontweight='bold')
        ax.set_ylabel('Power (W)', fontweight='bold')
        ax.set_title('P-V Curves Comparison', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_string_detailed_analysis(self, ax, string_analysis, string_num):
        """Plot detailed analysis for a specific string."""
        
        characteristics = string_analysis['iv_curve']['characteristics']
        performance = string_analysis['performance']
        
        # Create text summary
        text_content = f"STRING {string_num} ANALYSIS\n"
        text_content += f"{'=' * 20}\n"
        text_content += f"Voc: {characteristics['voc']:.1f} V\n"
        text_content += f"Isc: {characteristics['isc']:.2f} A\n"
        text_content += f"Fill Factor: {characteristics['fill_factor']:.3f}\n"
        text_content += f"Avg Power: {performance['average_power']:.1f} W\n"
        text_content += f"Max Power: {performance['max_power']:.1f} W\n"
        text_content += f"Stability: {performance['stability_index']:.3f}\n"
        text_content += f"Rs: {characteristics['series_resistance']:.3f} Ω\n"
        
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['surface'], alpha=0.8),
               color='white', fontfamily='monospace')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'String {string_num} Characteristics', fontweight='bold', fontsize=14)
    
    def _plot_performance_comparison(self, ax, string1_analysis, string2_analysis):
        """Plot performance comparison between strings."""
        
        metrics = ['average_power', 'max_power', 'stability_index', 'performance_ratio']
        string1_values = [string1_analysis['performance'][m] for m in metrics]
        string2_values = [string2_analysis['performance'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, string1_values, width, label='String 1', 
               color=self.colors['string1'], alpha=0.8)
        ax.bar(x + width/2, string2_values, width, label='String 2', 
               color=self.colors['string2'], alpha=0.8)
        
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_operating_points_analysis(self, ax, string1_analysis, string2_analysis):
        """Plot operating points analysis."""
        
        # Get operating time distribution
        op1 = string1_analysis['operating_points']['time_distribution']
        op2 = string2_analysis['operating_points']['time_distribution']
        
        categories = list(op1.keys())
        string1_values = list(op1.values())
        string2_values = list(op2.values())
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, string1_values, width, label='String 1', 
               color=self.colors['string1'], alpha=0.8)
        ax.bar(x + width/2, string2_values, width, label='String 2', 
               color=self.colors['string2'], alpha=0.8)
        
        ax.set_xlabel('Operating Regions')
        ax.set_ylabel('Time Distribution (%)')
        ax.set_title('Operating Points Distribution', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_analysis(self, ax, string1_analysis, string2_analysis):
        """Plot efficiency analysis."""
        
        eff1 = string1_analysis['efficiency']
        eff2 = string2_analysis['efficiency']
        
        # Efficiency comparison
        metrics = ['average_efficiency', 'efficiency_stability']
        string1_values = [eff1['average_efficiency'], eff1['efficiency_stability'] * 100]
        string2_values = [eff2['average_efficiency'], eff2['efficiency_stability'] * 100]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, string1_values, width, label='String 1', 
               color=self.colors['string1'], alpha=0.8)
        ax.bar(x + width/2, string2_values, width, label='String 2', 
               color=self.colors['string2'], alpha=0.8)
        
        ax.set_xlabel('Efficiency Metrics')
        ax.set_ylabel('Values (%)')
        ax.set_title('Efficiency Analysis', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['Average Efficiency', 'Efficiency Stability'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_iv_characteristics_summary(self, ax, string1_analysis, string2_analysis):
        """Plot I-V characteristics summary."""
        
        char1 = string1_analysis['iv_curve']['characteristics']
        char2 = string2_analysis['iv_curve']['characteristics']
        
        characteristics = ['voc', 'isc', 'fill_factor']
        string1_values = [char1[c] for c in characteristics]
        string2_values = [char2[c] for c in characteristics]
        
        # Normalize values for comparison
        max_values = [max(v1, v2) for v1, v2 in zip(string1_values, string2_values)]
        norm1 = [v/mv if mv > 0 else 0 for v, mv in zip(string1_values, max_values)]
        norm2 = [v/mv if mv > 0 else 0 for v, mv in zip(string2_values, max_values)]
        
        x = np.arange(len(characteristics))
        width = 0.35
        
        ax.bar(x - width/2, norm1, width, label='String 1 (Normalized)', 
               color=self.colors['string1'], alpha=0.8)
        ax.bar(x + width/2, norm2, width, label='String 2 (Normalized)', 
               color=self.colors['string2'], alpha=0.8)
        
        ax.set_xlabel('I-V Characteristics')
        ax.set_ylabel('Normalized Values')
        ax.set_title('I-V Characteristics Summary', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['Voc', 'Isc', 'Fill Factor'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_quality_metrics(self, ax, string1_analysis, string2_analysis):
        """Plot quality metrics comparison."""
        
        quality1 = string1_analysis['iv_curve']['quality_metrics']['overall_quality_score']
        quality2 = string2_analysis['iv_curve']['quality_metrics']['overall_quality_score']
        
        # Create quality comparison
        strings = ['String 1', 'String 2']
        quality_scores = [quality1, quality2]
        colors = [self.colors['string1'], self.colors['string2']]
        
        bars = ax.bar(strings, quality_scores, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Quality Score (%)')
        ax.set_title('I-V Curve Quality Assessment', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add quality threshold line
        ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Good Quality Threshold')
        ax.legend()
    
    def generate_per_string_report(self) -> str:
        """Generate comprehensive per-string analysis report."""
        
        if not hasattr(self, 'string1_data') or not hasattr(self, 'string2_data'):
            # Run analysis if not already done
            self.create_string_iv_dashboard()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_lines = []
        
        # Report header
        report_lines.extend([
            "=" * 80,
            "🔍 COMPREHENSIVE DUAL-STRING I-V CURVE ANALYSIS REPORT",
            "=" * 80,
            f"📅 Generated: {timestamp}",
            f"📁 Data File: {self.csv_file_path.name}",
            f"📊 Data Points: {len(self.data)} valid measurements",
            ""
        ])
        
        # String 1 Analysis
        report_lines.extend(self._generate_string_section_report(self.string1_data, 1))
        
        # String 2 Analysis
        report_lines.extend(self._generate_string_section_report(self.string2_data, 2))
        
        # Comparative Analysis
        report_lines.extend(self._generate_comparative_analysis())
        
        # Recommendations
        report_lines.extend(self._generate_string_recommendations())
        
        report_lines.append("=" * 80)
        
        # Save report
        report_content = "\n".join(report_lines)
        timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f'string_iv_analysis_report_{timestamp_file}.txt'
        
        with open(report_filename, 'w') as f:
            f.write(report_content)
        
        print(f"✅ Per-string analysis report saved as: {report_filename}")
        return report_filename
    
    def _generate_string_section_report(self, string_data: Dict, string_num: int) -> List[str]:
        """Generate report section for a specific string."""
        
        lines = []
        lines.extend([
            f"📈 STRING {string_num} COMPREHENSIVE ANALYSIS",
            "-" * 40
        ])
        
        # I-V Curve Characteristics
        iv_char = string_data['iv_curve']['characteristics']
        mpp = string_data['iv_curve']['mpp']
        
        lines.extend([
            "🔌 I-V Curve Characteristics:",
            f"   • Open Circuit Voltage (Voc): {iv_char['voc']:.2f} V",
            f"   • Short Circuit Current (Isc): {iv_char['isc']:.3f} A",
            f"   • Fill Factor: {iv_char['fill_factor']:.4f}",
            f"   • Series Resistance: {iv_char['series_resistance']:.4f} Ω",
            f"   • Shunt Resistance: {iv_char['shunt_resistance']:.2f} Ω" if iv_char['shunt_resistance'] != float('inf') else "   • Shunt Resistance: ∞ Ω",
            ""
        ])
        
        # Maximum Power Point
        lines.extend([
            "⚡ Maximum Power Point (MPP):",
            f"   • MPP Voltage: {mpp['voltage']:.2f} V",
            f"   • MPP Current: {mpp['current']:.3f} A",
            f"   • MPP Power: {mpp['power']:.2f} W",
            ""
        ])
        
        # Performance Metrics
        perf = string_data['performance']
        lines.extend([
            "📊 Performance Metrics:",
            f"   • Average Power: {perf['average_power']:.2f} W",
            f"   • Maximum Power: {perf['max_power']:.2f} W",
            f"   • Power Standard Deviation: {perf['power_std']:.2f} W",
            f"   • Average Voltage: {perf['average_voltage']:.2f} V",
            f"   • Voltage Range: {perf['voltage_range']['min']:.1f} - {perf['voltage_range']['max']:.1f} V",
            f"   • Average Current: {perf['average_current']:.3f} A",
            f"   • Current Range: {perf['current_range']['min']:.3f} - {perf['current_range']['max']:.3f} A",
            f"   • Stability Index: {perf['stability_index']:.4f}",
            f"   • Performance Ratio: {perf['performance_ratio']:.4f}",
            ""
        ])
        
        # Operating Points Analysis
        op_points = string_data['operating_points']
        lines.extend([
            "🎯 Operating Points Analysis:",
            f"   • Low Power Operation: {op_points['time_distribution']['low_power']:.1f}%",
            f"   • Medium Power Operation: {op_points['time_distribution']['medium_power']:.1f}%",
            f"   • High Power Operation: {op_points['time_distribution']['high_power']:.1f}%",
            f"   • Peak Power Operation: {op_points['time_distribution']['peak_power']:.1f}%",
            f"   • MPP Tracking Efficiency: {op_points['mpp_tracking_efficiency']:.1f}%",
            ""
        ])
        
        # Efficiency Analysis
        eff = string_data['efficiency']
        lines.extend([
            "⚙️ Efficiency Analysis:",
            f"   • Average Efficiency: {eff['average_efficiency']:.2f}%",
            f"   • Efficiency Range: {eff['efficiency_range']['min']:.2f}% - {eff['efficiency_range']['max']:.2f}%",
            f"   • Efficiency Stability: {eff['efficiency_stability']:.4f}",
            f"   • Low Efficiency Periods: {eff['low_efficiency_periods']} occurrences",
            ""
        ])
        
        # Quality Assessment
        quality = string_data['iv_curve']['quality_metrics']
        lines.extend([
            "🔍 I-V Curve Quality Assessment:",
            f"   • Overall Quality Score: {quality['overall_quality_score']:.1f}/100",
            f"   • Voltage Noise Level: {quality['noise_levels']['voltage']:.4f}",
            f"   • Current Noise Level: {quality['noise_levels']['current']:.4f}",
            f"   • Curve Completeness (V): {quality['curve_completeness']['voltage_span']:.3f}",
            f"   • Curve Completeness (I): {quality['curve_completeness']['current_span']:.3f}",
            "",
            ""
        ])
        
        return lines
    
    def _generate_comparative_analysis(self) -> List[str]:
        """Generate comparative analysis between strings."""
        
        lines = []
        lines.extend([
            "⚖️ COMPARATIVE STRING ANALYSIS",
            "-" * 40
        ])
        
        # Power comparison
        p1_avg = self.string1_data['performance']['average_power']
        p2_avg = self.string2_data['performance']['average_power']
        power_diff = abs(p1_avg - p2_avg)
        power_diff_percent = (power_diff / max(p1_avg, p2_avg)) * 100
        
        lines.extend([
            "🔋 Power Analysis:",
            f"   • String 1 Average Power: {p1_avg:.2f} W",
            f"   • String 2 Average Power: {p2_avg:.2f} W",
            f"   • Power Difference: {power_diff:.2f} W ({power_diff_percent:.1f}%)",
            f"   • Higher Performing String: {'String 1' if p1_avg > p2_avg else 'String 2'}",
            ""
        ])
        
        # I-V characteristics comparison
        char1 = self.string1_data['iv_curve']['characteristics']
        char2 = self.string2_data['iv_curve']['characteristics']
        
        lines.extend([
            "🔌 I-V Characteristics Comparison:",
            f"   • Voc Difference: {abs(char1['voc'] - char2['voc']):.2f} V",
            f"   • Isc Difference: {abs(char1['isc'] - char2['isc']):.3f} A",
            f"   • Fill Factor Difference: {abs(char1['fill_factor'] - char2['fill_factor']):.4f}",
            ""
        ])
        
        # Efficiency comparison
        eff1 = self.string1_data['efficiency']['average_efficiency']
        eff2 = self.string2_data['efficiency']['average_efficiency']
        
        lines.extend([
            "⚙️ Efficiency Comparison:",
            f"   • String 1 Efficiency: {eff1:.2f}%",
            f"   • String 2 Efficiency: {eff2:.2f}%",
            f"   • Efficiency Difference: {abs(eff1 - eff2):.2f}%",
            ""
        ])
        
        # Quality comparison
        quality1 = self.string1_data['iv_curve']['quality_metrics']['overall_quality_score']
        quality2 = self.string2_data['iv_curve']['quality_metrics']['overall_quality_score']
        
        lines.extend([
            "🔍 Quality Comparison:",
            f"   • String 1 Quality Score: {quality1:.1f}/100",
            f"   • String 2 Quality Score: {quality2:.1f}/100",
            f"   • Quality Difference: {abs(quality1 - quality2):.1f} points",
            "",
            ""
        ])
        
        return lines
    
    def _generate_string_recommendations(self) -> List[str]:
        """Generate recommendations based on string analysis."""
        
        lines = []
        lines.extend([
            "💡 RECOMMENDATIONS",
            "-" * 40
        ])
        
        recommendations = []
        
        # Power imbalance check
        p1_avg = self.string1_data['performance']['average_power']
        p2_avg = self.string2_data['performance']['average_power']
        power_diff_percent = (abs(p1_avg - p2_avg) / max(p1_avg, p2_avg)) * 100
        
        if power_diff_percent > 10:
            recommendations.append("🔧 **Critical Power Imbalance**: Significant power difference between strings detected. Inspect for soiling, shading, or module degradation.")
        elif power_diff_percent > 5:
            recommendations.append("⚠️ **Power Imbalance Warning**: Moderate power difference detected. Schedule inspection.")
        
        # Fill factor check
        ff1 = self.string1_data['iv_curve']['characteristics']['fill_factor']
        ff2 = self.string2_data['iv_curve']['characteristics']['fill_factor']
        
        if ff1 < 0.7 or ff2 < 0.7:
            recommendations.append("📉 **Low Fill Factor**: One or both strings show low fill factor. Check for series resistance issues or cell degradation.")
        
        # Quality assessment
        quality1 = self.string1_data['iv_curve']['quality_metrics']['overall_quality_score']
        quality2 = self.string2_data['iv_curve']['quality_metrics']['overall_quality_score']
        
        if quality1 < 70 or quality2 < 70:
            recommendations.append("🔍 **Data Quality Issues**: I-V curve quality concerns detected. Verify monitoring equipment and connections.")
        
        # Efficiency assessment
        eff1 = self.string1_data['efficiency']['average_efficiency']
        eff2 = self.string2_data['efficiency']['average_efficiency']
        
        if eff1 < 85 or eff2 < 85:
            recommendations.append("⚡ **Low Efficiency Alert**: String efficiency below optimal levels. Consider cleaning and maintenance.")
        
        # Series resistance check
        rs1 = self.string1_data['iv_curve']['characteristics']['series_resistance']
        rs2 = self.string2_data['iv_curve']['characteristics']['series_resistance']
        
        if rs1 > 0.5 or rs2 > 0.5:
            recommendations.append("🔧 **High Series Resistance**: Elevated series resistance detected. Inspect connections and wiring.")
        
        # MPP tracking
        mpp_eff1 = self.string1_data['operating_points']['mpp_tracking_efficiency']
        mpp_eff2 = self.string2_data['operating_points']['mpp_tracking_efficiency']
        
        if mpp_eff1 < 85 or mpp_eff2 < 85:
            recommendations.append("🎯 **MPP Tracking Issues**: Poor maximum power point tracking detected. Review MPPT controller settings.")
        
        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "✅ **System Performance**: Both strings operating within normal parameters.",
                "📊 **Continued Monitoring**: Maintain regular I-V curve analysis for trend detection.",
                "🛠️ **Preventive Maintenance**: Schedule routine cleaning and inspection."
            ])
        
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        lines.append("")
        
        return lines
    
    def run_complete_string_analysis(self) -> Dict[str, Any]:
        """Run complete string I-V analysis and generate all outputs."""
        
        print("🚀 Starting Comprehensive String I-V Analysis...")
        
        # Load data
        self.load_and_process_data()
        
        # Create dashboard
        dashboard_file = self.create_string_iv_dashboard()
        
        # Generate report
        report_file = self.generate_per_string_report()
        
        # Prepare summary results
        results = {
            'string1_analysis': self.string1_data,
            'string2_analysis': self.string2_data,
            'files_generated': {
                'dashboard': dashboard_file,
                'report': report_file
            },
            'summary': {
                'total_data_points': len(self.data),
                'string1_avg_power': self.string1_data['performance']['average_power'],
                'string2_avg_power': self.string2_data['performance']['average_power'],
                'power_imbalance_percent': abs(
                    self.string1_data['performance']['average_power'] - 
                    self.string2_data['performance']['average_power']
                ) / max(
                    self.string1_data['performance']['average_power'],
                    self.string2_data['performance']['average_power']
                ) * 100,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        print("✅ Comprehensive String I-V Analysis Complete!")
        print(f"📊 Dashboard: {dashboard_file}")
        print(f"📝 Report: {report_file}")
        
        return results

def main():
    """Main execution function for string I-V analysis."""
    
    print("🔍 Enhanced Dual-String I-V Curve Analysis System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = EnhancedStringIVAnalyzer('./inverter/INVERTER_01_2025-04-04_2025-04-05.csv')
    
    # Run complete analysis
    results = analyzer.run_complete_string_analysis()
    
    # Display summary
    summary = results['summary']
    print(f"\n📈 ANALYSIS SUMMARY")
    print("-" * 30)
    print(f"Data Points Analyzed: {summary['total_data_points']}")
    print(f"String 1 Average Power: {summary['string1_avg_power']:.1f} W")
    print(f"String 2 Average Power: {summary['string2_avg_power']:.1f} W")
    print(f"Power Imbalance: {summary['power_imbalance_percent']:.1f}%")
    
    return results

if __name__ == "__main__":
    main()
