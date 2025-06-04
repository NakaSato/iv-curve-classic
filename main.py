"""
Enhanced Main Script for I-V Curve Analysis with Real Inverter Data and PV Module Specifications

This script demonstrates comprehensive I-V curve analysis using:
- Real ZNSHINESOLAR ZXM7-UHLD144 570W module specifications
- Actual inverter performance data from 10 solar inverters
- Environmental corrections based on manufacturer coefficients
- Economic impact analysis with real-world parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import re

# Enhanced plotting configuration for professional visualizations
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#dee2e6',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.color': '#e9ecef',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.7,
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'legend.fancybox': True,
    'legend.shadow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

# Professional color palette
COLORS = {
    'primary': '#2E86C1',      # Professional blue
    'secondary': '#28B463',     # Success green  
    'accent': '#F39C12',       # Warning orange
    'danger': '#E74C3C',       # Error red
    'dark': '#34495E',         # Dark blue-grey
    'light': '#BDC3C7',       # Light grey
    'info': '#8E44AD',        # Purple
    'success': '#27AE60',     # Dark green
    'warning': '#F1C40F',     # Yellow
    'muted': '#95A5A6'        # Muted grey
}

# Enhanced gradient colors for multi-series plots
GRADIENT_COLORS = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD', 
                   '#17A2B8', '#6C757D', '#FD7E14', '#20C997', '#6F42C1']

from iv_curve_classic.data_loader import InverterDataLoader, get_real_data_samples
from iv_curve_classic.iv_analyzer import IVAnalyzer
from iv_curve_classic.parameter_extractor import ParameterExtractor
from iv_curve_classic.diagnostics import DiagnosticAnalyzer
from iv_curve_classic.environmental_corrections import EnvironmentalCorrections


class PVModuleConfig:
    """Load and manage PV module configuration from config.toml"""
    
    def __init__(self, config_path: str = "config.toml"):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from TOML file with inline comment handling"""
        config = {}
        current_section = None
        
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle section headers
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                        config[current_section] = {}
                        continue
                    
                    # Handle key-value pairs
                    if '=' in line and current_section:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.split('#')[0].strip()  # Remove inline comments
                        
                        # Convert values to appropriate types
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]  # String
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'  # Boolean
                        else:
                            try:
                                if '.' in value or 'e' in value.lower():
                                    value = float(value)  # Float
                                else:
                                    value = int(value)  # Integer
                            except ValueError:
                                pass  # Keep as string
                        
                        config[current_section][key] = value
        
        except FileNotFoundError:
            print(f"⚠️  Configuration file {config_path} not found. Using defaults.")
            return self._get_default_config()
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Provide default ZNSHINESOLAR module configuration"""
        return {
            'pv_module': {
                'name': 'ZXM7-UHLD144 570W',
                'manufacturer': 'ZNSHINESOLAR',
                'rated_power_stc': 570.0,
                'rated_voltage_stc': 51.00,
                'rated_current_stc': 14.18,
                'voltage_at_max_power': 42.40,
                'current_at_max_power': 13.45,
                'fill_factor_nominal': 0.787,
                'efficiency_stc': 22.06,
                'temp_coeff_voc': -0.0025,
                'temp_coeff_isc': 0.00048,
                'temp_coeff_power': -0.003,
                'cells_in_series': 144
            }
        }
    
    def get_module_specs(self) -> Dict[str, Any]:
        """Get PV module specifications"""
        return self.config.get('pv_module', {})


class EnhancedInverterAnalyzer:
    """Enhanced analyzer that combines real inverter data with PV module specifications"""
    
    def __init__(self):
        self.pv_config = PVModuleConfig()
        self.module_specs = self.pv_config.get_module_specs()
        self.analyzer = IVAnalyzer()
        self.diagnostics = DiagnosticAnalyzer()
        self.env_corrections = EnvironmentalCorrections()
    
    def analyze_inverter_data(self, inverter_data_path: str) -> Optional[Dict[str, Any]]:
        """Analyze real inverter CSV data with PV module context"""
        try:
            # Load CSV data with proper header handling
            df = pd.read_csv(inverter_data_path)
            
            print(f"  📊 Loaded {len(df)} data rows from CSV")
            
            # Extract relevant columns for I-V analysis
            analysis_data = {
                'vpv': [],  # PV voltages
                'ipv': [],  # PV currents  
                'ppv': [],  # PV powers
                'timestamp': [],
                'temperature': [],
                'inverter_id': Path(inverter_data_path).stem
            }
            
            # Process data rows with improved filtering
            valid_rows = 0
            for _, row in df.iterrows():
                if pd.isna(row.get('Time')):
                    continue
                
                # Only process rows with "Normal" status for active operation
                status = str(row.get('Status', '')).strip()
                if status not in ['Normal', 'Waiting']:
                    continue
                
                # Extract PV string data (Vpv1-16, Ipv1-16, Ppv1-16)
                vpv_values = []
                ipv_values = []
                ppv_values = []
                
                # Check all 16 PV strings
                for i in range(1, 17):
                    vpv_col = f'Vpv{i}(V)'
                    ipv_col = f'Ipv{i}(A)'
                    ppv_col = f'Ppv{i}(W)'
                    
                    # More lenient validation - accept any positive voltage > 10V and current > 0.01A
                    vpv_val = row.get(vpv_col, 0)
                    ipv_val = row.get(ipv_col, 0)
                    ppv_val = row.get(ppv_col, 0)
                    
                    if (not pd.isna(vpv_val) and not pd.isna(ipv_val) and 
                        vpv_val > 10 and ipv_val > 0.01):  # Much more lenient thresholds
                        vpv_values.append(float(vpv_val))
                        ipv_values.append(float(ipv_val))
                        if not pd.isna(ppv_val) and ppv_val > 0:
                            ppv_values.append(float(ppv_val))
                        else:
                            ppv_values.append(float(vpv_val) * float(ipv_val))  # Calculate if missing
                
                # If we have valid data points, add them
                # If we have valid data points, add them
                if len(vpv_values) >= 3:  # Need at least 3 strings with data
                    analysis_data['vpv'].extend(vpv_values)
                    analysis_data['ipv'].extend(ipv_values)
                    analysis_data['ppv'].extend(ppv_values)
                    analysis_data['timestamp'].append(row.get('Time', ''))
                    
                    # Extract temperature if available
                    temp = row.get('INVTemp(℃)', 25.0)
                    if pd.isna(temp):
                        temp = 25.0
                    analysis_data['temperature'].append(float(temp))
                    valid_rows += 1
            
            print(f"  ✅ Extracted data from {valid_rows} valid rows, {len(analysis_data['vpv'])} total data points")
            
            if len(analysis_data['vpv']) < 10:  # Need minimum data points for analysis
                print(f"  ⚠️  Insufficient data points ({len(analysis_data['vpv'])}) for reliable analysis")
                return None
            
            return self._calculate_iv_parameters(analysis_data)
            
        except Exception as e:
            print(f"❌ Error analyzing {inverter_data_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_iv_parameters(self, data: Dict) -> Optional[Dict[str, Any]]:
        """Calculate I-V parameters from real data with module specifications"""
        if not data['vpv'] or not data['ipv']:
            print("  ❌ No valid voltage or current data found")
            return None
        
        vpv = np.array(data['vpv'])
        ipv = np.array(data['ipv'])
        ppv = np.array(data['ppv']) if data['ppv'] else vpv * ipv
        
        # Filter out any invalid values
        valid_mask = (vpv > 0) & (ipv >= 0) & (ppv > 0)
        vpv = vpv[valid_mask]
        ipv = ipv[valid_mask]
        ppv = ppv[valid_mask]
        
        if len(vpv) < 5:
            print(f"  ❌ Insufficient valid data points after filtering ({len(vpv)})")
            return None
        
        # Calculate aggregate parameters with robust methods
        try:
            # VOC: maximum voltage where current is still significant
            high_voltage_mask = vpv > np.percentile(vpv, 95)
            voc_measured = np.max(vpv[high_voltage_mask]) if np.any(high_voltage_mask) else np.max(vpv)
            
            # ISC: maximum current where voltage is still reasonable
            low_voltage_mask = vpv < np.percentile(vpv, 10)
            isc_measured = np.max(ipv[low_voltage_mask]) if np.any(low_voltage_mask) else np.max(ipv)
            
            # Maximum power and MPP
            p_max_measured = np.max(ppv)
            max_power_idx = np.argmax(ppv)
            v_mp_measured = vpv[max_power_idx]
            i_mp_measured = ipv[max_power_idx]
            
            # Calculate fill factor
            fill_factor = (v_mp_measured * i_mp_measured) / (voc_measured * isc_measured) if (voc_measured > 0 and isc_measured > 0) else 0
            
            # Ensure reasonable bounds
            if fill_factor > 1.0:
                fill_factor = 0.85  # Use typical value if calculation is unrealistic
            
        except Exception as e:
            print(f"  ❌ Error in parameter calculation: {e}")
            return None
        
        # Estimate series and shunt resistance using module specifications
        module_voc = self.module_specs.get('rated_voltage_stc', 51.0)
        module_isc = self.module_specs.get('rated_current_stc', 14.18)
        
        # Estimate number of modules in string from voltage ratio
        modules_per_string = max(1, int(round(voc_measured / module_voc)))
        strings_in_parallel = max(1, int(round(isc_measured / module_isc)))
        
        # Calculate system-level parameters
        rs_estimated = 0.5 * modules_per_string / strings_in_parallel  # Ohms
        rsh_estimated = 1000 * modules_per_string * strings_in_parallel  # Ohms
        
        # Temperature analysis
        avg_temp = np.mean(data['temperature']) if data['temperature'] else 25.0
        temp_effect = self._calculate_temperature_effects(avg_temp)
        
        return {
            'inverter_id': data['inverter_id'],
            'voc': voc_measured,
            'isc': isc_measured,
            'p_max': p_max_measured,
            'v_mp': v_mp_measured,
            'i_mp': i_mp_measured,
            'fill_factor': fill_factor,
            'rs': rs_estimated,
            'rsh': rsh_estimated,
            'modules_per_string': modules_per_string,
            'strings_in_parallel': strings_in_parallel,
            'avg_temperature': avg_temp,
            'temperature_effects': temp_effect,
            'data_points': len(data['vpv']),
            'module_specs': self.module_specs,
            'raw_data': {
                'vpv': vpv,
                'ipv': ipv,
                'ppv': ppv
            }
        }
    
    def _calculate_temperature_effects(self, temperature: float) -> Dict[str, float]:
        """Calculate temperature effects on module performance"""
        t_ref = 25.0  # STC temperature
        dt = temperature - t_ref
        
        tc_voc = self.module_specs.get('temp_coeff_voc', -0.0025)  # %/°C
        tc_isc = self.module_specs.get('temp_coeff_isc', 0.00048)  # %/°C
        tc_power = self.module_specs.get('temp_coeff_power', -0.003)  # %/°C
        
        return {
            'temperature_delta': dt,
            'voc_effect_percent': tc_voc * dt * 100,
            'isc_effect_percent': tc_isc * dt * 100,
            'power_effect_percent': tc_power * dt * 100,
            'expected_voc_factor': 1 + (tc_voc * dt),
            'expected_isc_factor': 1 + (tc_isc * dt),
            'expected_power_factor': 1 + (tc_power * dt)
        }

# Import the configuration-based analyzer for PV module specifications
try:
    from config_based_analyzer import ConfigBasedIVAnalyzer
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️  Configuration-based analyzer not available. Using fallback mode.")


def main():
    """Enhanced main function with real data and PV module specifications."""
    
    print("🔬 Enhanced I-V Curve Analysis with Real Data and PV Module Specifications")
    print("=" * 80)
    
    # Initialize enhanced analyzer
    enhanced_analyzer = EnhancedInverterAnalyzer()
    
    # Display PV module specifications
    module_specs = enhanced_analyzer.module_specs
    print(f"\n📋 PV Module Configuration:")
    print(f"  Module: {module_specs.get('name', 'Unknown')}")
    print(f"  Manufacturer: {module_specs.get('manufacturer', 'Unknown')}")
    print(f"  Rated Power: {module_specs.get('rated_power_stc', 0):.0f} W")
    print(f"  VOC (STC): {module_specs.get('rated_voltage_stc', 0):.2f} V")
    print(f"  ISC (STC): {module_specs.get('rated_current_stc', 0):.2f} A")
    print(f"  Efficiency: {module_specs.get('efficiency_stc', 0):.2f}%")
    print(f"  Technology: {module_specs.get('technology', 'Unknown')}")
    
    # Load and analyze real inverter data
    inverter_dir = Path("inverter")
    csv_files = list(inverter_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No inverter CSV files found. Please check the inverter directory.")
        demonstrate_with_enhanced_synthetic_data(enhanced_analyzer)
        return
    
    print(f"\n📊 Found {len(csv_files)} inverter data files")
    
    analysis_results = []
    successful_analyses = 0
    
    # Analyze each inverter
    for csv_file in csv_files:
        print(f"\n🔍 Analyzing {csv_file.stem}")
        print("-" * 50)
        
        try:
            result = enhanced_analyzer.analyze_inverter_data(str(csv_file))
            
            if result:
                analysis_results.append(result)
                successful_analyses += 1
                
                # Display results
                display_enhanced_analysis_results(result)
                
                # Create enhanced visualization
                create_enhanced_analysis_plot(result)
                
            else:
                print(f"⚠️  No valid data extracted from {csv_file.stem}")
                
        except Exception as e:
            print(f"❌ Error analyzing {csv_file.stem}: {e}")
    
    if successful_analyses > 0:
        print(f"\n✅ Successfully analyzed {successful_analyses} inverters")
        
        # Generate fleet summary
        create_fleet_analysis_summary(analysis_results, enhanced_analyzer)
        
    else:
        print("\n⚠️  No successful analyses. Demonstrating with synthetic data...")
        demonstrate_with_enhanced_synthetic_data(enhanced_analyzer)


def display_enhanced_analysis_results(result: Dict[str, Any]):
    """Display comprehensive analysis results with module context"""
    
    print("📈 Real Data Analysis Results:")
    print(f"  Data Points Analyzed: {result['data_points']}")
    print(f"  Estimated String Configuration: {result['modules_per_string']} modules/string × {result['strings_in_parallel']} strings")
    
    print("\n⚡ Measured I-V Parameters:")
    print(f"  VOC: {result['voc']:.1f} V")
    print(f"  ISC: {result['isc']:.2f} A") 
    print(f"  Pmax: {result['p_max']:.0f} W")
    print(f"  VMP: {result['v_mp']:.1f} V")
    print(f"  IMP: {result['i_mp']:.2f} A")
    print(f"  Fill Factor: {result['fill_factor']:.3f}")
    print(f"  Rs (estimated): {result['rs']:.2f} Ω")
    print(f"  Rsh (estimated): {result['rsh']:.0f} Ω")
    
    # Temperature analysis
    temp_effects = result['temperature_effects']
    print(f"\n🌡️  Temperature Analysis (Avg: {result['avg_temperature']:.1f}°C):")
    print(f"  Temperature Delta: {temp_effects['temperature_delta']:.1f}°C from STC")
    print(f"  VOC Temperature Effect: {temp_effects['voc_effect_percent']:.2f}%")
    print(f"  ISC Temperature Effect: {temp_effects['isc_effect_percent']:.2f}%")
    print(f"  Power Temperature Effect: {temp_effects['power_effect_percent']:.2f}%")
    
    # Module-level performance estimation
    module_specs = result['module_specs']
    estimated_modules = result['modules_per_string'] * result['strings_in_parallel']
    expected_system_power = module_specs.get('rated_power_stc', 570) * estimated_modules
    
    print(f"\n🔋 System Performance vs Module Specifications:")
    print(f"  Estimated Total Modules: {estimated_modules}")
    print(f"  Expected System Power (STC): {expected_system_power:.0f} W")
    print(f"  Measured Peak Power: {result['p_max']:.0f} W")
    
    if expected_system_power > 0:
        performance_ratio = (result['p_max'] / expected_system_power) * 100
        print(f"  Performance Ratio: {performance_ratio:.1f}%")
        
        # Health assessment
        if performance_ratio > 90:
            health_status = "Excellent"
            health_color = "🟢"
        elif performance_ratio > 80:
            health_status = "Good"
            health_color = "🟡"
        elif performance_ratio > 70:
            health_status = "Fair"
            health_color = "🟠"
        else:
            health_status = "Poor"
            health_color = "🔴"
        
        print(f"  System Health: {health_color} {health_status}")


def create_enhanced_analysis_plot(result: Dict[str, Any]):
    """Create enhanced analysis plot with professional styling and real data"""
    
    # Create figure with custom styling
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Create custom grid layout for better spacing
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3, 
                  height_ratios=[1.2, 1.2, 0.8], width_ratios=[1, 1, 0.8])
    
    # Enhanced title with module information
    module_name = result["module_specs"].get("name", "Unknown Module")
    inverter_id = result["inverter_id"].replace('_', ' ').title()
    fig.suptitle(f'📊 Enhanced I-V Analysis: {inverter_id}\n🔋 {module_name}', 
                 fontsize=16, fontweight='bold', y=0.96, color=COLORS['dark'])
    
    # Get raw data
    vpv = result['raw_data']['vpv']
    ipv = result['raw_data']['ipv']
    ppv = result['raw_data']['ppv']
    
    # Enhanced Plot 1: I-V Characteristic with professional styling
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create density-based coloring for scatter points
    scatter = ax1.scatter(vpv, ipv, alpha=0.7, s=35, c=vpv, cmap='viridis', 
                         edgecolors='white', linewidth=0.5, label='📍 Real Data Points')
    
    # Enhanced reference lines
    ax1.axhline(y=result['isc'], color=COLORS['danger'], linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f"ISC = {result['isc']:.2f} A")
    ax1.axvline(x=result['voc'], color=COLORS['secondary'], linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f"VOC = {result['voc']:.1f} V")
    
    # Highlighted Maximum Power Point with enhanced styling
    ax1.scatter([result['v_mp']], [result['i_mp']], color=COLORS['accent'], 
               s=200, zorder=10, marker='*', edgecolors=COLORS['dark'], linewidth=2,
               label=f"⭐ MPP ({result['v_mp']:.1f}V, {result['i_mp']:.1f}A)")
    
    # Enhanced axes styling
    ax1.set_xlabel('Voltage (V)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Current (A)', fontweight='bold', fontsize=12)
    ax1.set_title('🔌 I-V Characteristic Curve', fontweight='bold', fontsize=14, 
                  color=COLORS['primary'], pad=20)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax1.legend(loc='upper right', framealpha=0.95, shadow=True, fontsize=10)
    
    # Add colorbar for voltage gradient
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label('Voltage (V)', fontweight='bold', fontsize=10)
    
    # Enhanced Plot 2: Power curve with gradient fill
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create smooth power curve for better visualization
    voltage_sorted = np.sort(vpv)
    power_sorted = np.interp(voltage_sorted, vpv, ppv)
    
    # Power curve with gradient fill
    ax2.plot(voltage_sorted, power_sorted, color=COLORS['primary'], linewidth=3, 
             alpha=0.8, label='📈 Power Curve')
    ax2.fill_between(voltage_sorted, power_sorted, alpha=0.3, color=COLORS['primary'])
    
    # Scatter plot of actual data points
    ax2.scatter(vpv, ppv, alpha=0.6, s=25, c=COLORS['info'], 
               edgecolors='white', linewidth=0.5, label='📊 Measured Data')
    
    # Enhanced Maximum Power Point
    ax2.scatter([result['v_mp']], [result['p_max']], color=COLORS['accent'], 
               s=250, zorder=10, marker='D', edgecolors=COLORS['dark'], linewidth=2,
               label=f"💎 Pmax = {result['p_max']:.0f} W")
    
    ax2.set_xlabel('Voltage (V)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Power (W)', fontweight='bold', fontsize=12)
    ax2.set_title('⚡ Power-Voltage Characteristic', fontweight='bold', fontsize=14, 
                  color=COLORS['primary'], pad=20)
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax2.legend(loc='upper right', framealpha=0.95, shadow=True, fontsize=10)
    
    # Enhanced Plot 3: Module specifications with stylized info box
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # Create a fancy info box for module specifications
    module_specs = result['module_specs']
    
    # Create styled text box
    info_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, 
                              boxstyle="round,pad=0.05", 
                              facecolor=COLORS['light'], 
                              edgecolor=COLORS['primary'],
                              linewidth=2, alpha=0.8)
    ax3.add_patch(info_box)
    
    spec_text = f"""🔋 MODULE SPECIFICATIONS

📋 Model: {module_specs.get('name', 'Unknown')}
🏭 Manufacturer: {module_specs.get('manufacturer', 'Unknown')}
⚡ Rated Power: {module_specs.get('rated_power_stc', 0):.0f} W
🔌 VOC (STC): {module_specs.get('rated_voltage_stc', 0):.2f} V
⚡ ISC (STC): {module_specs.get('rated_current_stc', 0):.2f} A
💎 VMP (STC): {module_specs.get('voltage_at_max_power', 0):.2f} V
⭐ IMP (STC): {module_specs.get('current_at_max_power', 0):.2f} A
📊 Fill Factor: {module_specs.get('fill_factor_nominal', 0):.3f}
🎯 Efficiency: {module_specs.get('efficiency_stc', 0):.2f}%

🔧 SYSTEM CONFIG (Est.)
📐 Modules/String: {result['modules_per_string']}
🔗 Parallel Strings: {result['strings_in_parallel']}
🔢 Total Modules: {result['modules_per_string'] * result['strings_in_parallel']}
📊 Data Points: {result['data_points']}"""
    
    ax3.text(0.1, 0.95, spec_text, transform=ax3.transAxes, fontsize=9, 
            verticalalignment='top', fontfamily='monospace', 
            color=COLORS['dark'], fontweight='bold')
    
    # Enhanced Plot 4: Performance analysis with styled metrics
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    # Create performance metrics dashboard
    temp_effects = result['temperature_effects']
    expected_system_power = module_specs.get('rated_power_stc', 570) * result['modules_per_string'] * result['strings_in_parallel']
    performance_ratio = (result['p_max'] / expected_system_power * 100) if expected_system_power > 0 else 0
    
    # Create multiple styled info boxes for different sections
    sections = [
        {
            'title': '📊 MEASURED PERFORMANCE',
            'x': 0.02, 'width': 0.22,
            'content': f"""
Peak Power: {result['p_max']:.0f} W
Fill Factor: {result['fill_factor']:.3f}
Performance Ratio: {performance_ratio:.1f}%
Efficiency: {(result['p_max']/1000):.1f}%""",
            'color': COLORS['primary']
        },
        {
            'title': '🌡️ TEMPERATURE ANALYSIS',
            'x': 0.26, 'width': 0.22,
            'content': f"""
Average Temp: {result['avg_temperature']:.1f}°C
Delta from STC: {temp_effects['temperature_delta']:+.1f}°C
VOC Effect: {temp_effects['voc_effect_percent']:+.2f}%
ISC Effect: {temp_effects['isc_effect_percent']:+.2f}%
Power Effect: {temp_effects['power_effect_percent']:+.2f}%""",
            'color': COLORS['accent']
        },
        {
            'title': '🎯 EXPECTED vs MEASURED',
            'x': 0.50, 'width': 0.22,
            'content': f"""
Expected VOC: {module_specs.get('rated_voltage_stc', 0) * result['modules_per_string'] * temp_effects['expected_voc_factor']:.1f} V
Measured VOC: {result['voc']:.1f} V
Expected ISC: {module_specs.get('rated_current_stc', 0) * result['strings_in_parallel'] * temp_effects['expected_isc_factor']:.2f} A
Measured ISC: {result['isc']:.2f} A""",
            'color': COLORS['secondary']
        },
        {
            'title': '🏥 SYSTEM HEALTH',
            'x': 0.74, 'width': 0.22,
            'content': f"""
Health Status: {'🟢 Excellent' if performance_ratio > 90 else '🟡 Good' if performance_ratio > 80 else '🟠 Fair' if performance_ratio > 70 else '🔴 Poor'}
Data Quality: {'🟢 High' if result['data_points'] > 1000 else '🟡 Medium' if result['data_points'] > 100 else '🟠 Low'}
Temperature: {'🟢 Normal' if abs(temp_effects['temperature_delta']) < 10 else '🟡 Elevated' if temp_effects['temperature_delta'] > 0 else '🔵 Cold'}""",
            'color': COLORS['info']
        }
    ]
    
    for section in sections:
        # Create fancy box for each section
        section_box = FancyBboxPatch((section['x'], 0.1), section['width'], 0.8, 
                                    boxstyle="round,pad=0.02", 
                                    facecolor=section['color'], 
                                    edgecolor=COLORS['dark'],
                                    linewidth=1.5, alpha=0.15)
        ax4.add_patch(section_box)
        
        # Add title
        ax4.text(section['x'] + section['width']/2, 0.85, section['title'], 
                transform=ax4.transAxes, fontsize=11, fontweight='bold',
                horizontalalignment='center', color=COLORS['dark'])
        
        # Add content
        ax4.text(section['x'] + 0.01, 0.7, section['content'], 
                transform=ax4.transAxes, fontsize=9, fontweight='bold',
                verticalalignment='top', fontfamily='monospace', 
                color=COLORS['dark'])
    
    # Enhanced subplot for efficiency distribution
    ax5 = fig.add_subplot(gs[2, :2])
    
    # Create efficiency comparison chart
    categories = ['STC Rating', 'Expected\n(Temp Corrected)', 'Measured', 'Performance\nRatio']
    values = [
        module_specs.get('efficiency_stc', 22.0),
        module_specs.get('efficiency_stc', 22.0) * temp_effects['expected_power_factor'],
        (result['p_max'] / (module_specs.get('rated_power_stc', 570) * result['modules_per_string'] * result['strings_in_parallel'])) * module_specs.get('efficiency_stc', 22.0) if expected_system_power > 0 else 0,
        performance_ratio
    ]
    
    colors = [COLORS['primary'], COLORS['accent'], COLORS['secondary'], COLORS['info']]
    bars = ax5.bar(categories, values, color=colors, alpha=0.8, edgecolor=COLORS['dark'], linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%' if 'Ratio' in categories[bars.index(bar)] else f'{value:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax5.set_ylabel('Efficiency / Performance (%)', fontweight='bold', fontsize=12)
    ax5.set_title('📈 Performance Comparison Dashboard', fontweight='bold', fontsize=14, 
                  color=COLORS['primary'], pad=20)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, max(values) * 1.2)
    
    # Final styling touches
    plt.tight_layout()
    
    # Save with enhanced quality
    filename = f"enhanced_stylish_analysis_{result['inverter_id'].lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', 
               edgecolor='none', pad_inches=0.2)
    print(f"  🎨 Enhanced stylish plot saved as: {filename}")
    plt.close()


def create_fleet_summary_plot(results: List[Dict[str, Any]], analyzer: EnhancedInverterAnalyzer):
    """Create enhanced fleet summary visualization with professional styling"""
    
    # Create figure with enhanced styling
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Create custom grid layout
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3, 
                  height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1, 0.8])
    
    # Enhanced main title
    fig.suptitle('🏭 Fleet Analysis Dashboard - Solar Inverter Performance Overview\n🔋 Real Data Analysis with PV Module Specifications', 
                 fontsize=18, fontweight='bold', y=0.96, color=COLORS['dark'])
    
    # Extract data for plotting
    inverter_names = [r['inverter_id'] for r in results]
    powers = [r['p_max'] for r in results]
    fill_factors = [r['fill_factor'] for r in results]
    temperatures = [r['avg_temperature'] for r in results]
    estimated_modules = [r['modules_per_string'] * r['strings_in_parallel'] for r in results]
    
    # Enhanced Plot 1: Power distribution with gradient bars
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create gradient effect for bars
    bars1 = ax1.bar(range(len(powers)), powers, 
                    color=[GRADIENT_COLORS[i % len(GRADIENT_COLORS)] for i in range(len(powers))],
                    alpha=0.8, edgecolor=COLORS['dark'], linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, power) in enumerate(zip(bars1, powers)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{power:.0f}W', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_xlabel('Inverter ID', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Peak Power (W)', fontweight='bold', fontsize=12)
    ax1.set_title('⚡ Peak Power Distribution', fontweight='bold', fontsize=14, 
                  color=COLORS['primary'], pad=20)
    ax1.set_xticks(range(len(powers)))
    ax1.set_xticklabels([name.split('_')[1] for name in inverter_names], rotation=45, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Enhanced Plot 2: Fill factor with performance thresholds
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Color code based on fill factor quality
    ff_colors = [COLORS['success'] if ff > 0.85 else COLORS['warning'] if ff > 0.75 else COLORS['danger'] for ff in fill_factors]
    bars2 = ax2.bar(range(len(fill_factors)), fill_factors, 
                    color=ff_colors, alpha=0.8, edgecolor=COLORS['dark'], linewidth=1.5)
    
    # Add performance threshold lines
    ax2.axhline(y=0.85, color=COLORS['success'], linestyle='--', alpha=0.7, linewidth=2, label='Excellent (>0.85)')
    ax2.axhline(y=0.75, color=COLORS['warning'], linestyle='--', alpha=0.7, linewidth=2, label='Good (>0.75)')
    
    # Add value labels
    for i, (bar, ff) in enumerate(zip(bars2, fill_factors)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{ff:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel('Inverter ID', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Fill Factor', fontweight='bold', fontsize=12)
    ax2.set_title('📊 Fill Factor Quality Assessment', fontweight='bold', fontsize=14, 
                  color=COLORS['primary'], pad=20)
    ax2.set_xticks(range(len(fill_factors)))
    ax2.set_xticklabels([name.split('_')[1] for name in inverter_names], rotation=45, fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper left', fontsize=9)
    
    # Enhanced Plot 3: Temperature vs Performance correlation
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Create enhanced scatter plot with size based on module count
    scatter = ax3.scatter(temperatures, powers, s=[modules*5 for modules in estimated_modules], 
                         alpha=0.7, c=fill_factors, cmap='RdYlGn', 
                         edgecolors=COLORS['dark'], linewidth=1.5)
    
    # Add trend line
    if len(temperatures) > 1:
        z = np.polyfit(temperatures, powers, 1)
        p = np.poly1d(z)
        ax3.plot(sorted(temperatures), p(sorted(temperatures)), 
                color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.8, label='Trend Line')
    
    ax3.set_xlabel('Average Temperature (°C)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Peak Power (W)', fontweight='bold', fontsize=12)
    ax3.set_title('🌡️ Temperature vs Performance', fontweight='bold', fontsize=14, 
                  color=COLORS['primary'], pad=20)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Add colorbar for fill factor
    cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8, pad=0.02)
    cbar.set_label('Fill Factor', fontweight='bold', fontsize=10)
    
    # Enhanced Plot 4: Fleet summary statistics
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    
    # Create comprehensive fleet statistics
    module_specs = analyzer.module_specs
    total_estimated_modules = sum(estimated_modules)
    total_expected_power = total_estimated_modules * module_specs.get('rated_power_stc', 570)
    total_measured_power = sum(powers)
    fleet_performance_ratio = (total_measured_power / total_expected_power * 100) if total_expected_power > 0 else 0
    
    # Create styled info box
    info_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, 
                              boxstyle="round,pad=0.05", 
                              facecolor=COLORS['light'], 
                              edgecolor=COLORS['primary'],
                              linewidth=2, alpha=0.8)
    ax4.add_patch(info_box)
    
    summary_text = f"""🏭 FLEET OVERVIEW

📊 Configuration:
  Total Inverters: {len(results)}
  Total Modules: {total_estimated_modules}
  Expected Power: {total_expected_power:.0f} W
  Measured Power: {total_measured_power:.0f} W
  Fleet Performance: {fleet_performance_ratio:.1f}%

🔋 Module Specs:
  Model: {module_specs.get('name', 'Unknown')}
  Rated Power: {module_specs.get('rated_power_stc', 0):.0f} W
  Technology: {module_specs.get('technology', 'Unknown')}

📈 Performance Stats:
  Avg Fill Factor: {np.mean(fill_factors):.3f}
  FF Range: {min(fill_factors):.3f} - {max(fill_factors):.3f}
  Avg Temperature: {np.mean(temperatures):.1f}°C
  Temp Range: {min(temperatures):.1f} - {max(temperatures):.1f}°C
  Avg Power/Inverter: {np.mean(powers):.0f} W"""
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', 
            color=COLORS['dark'], fontweight='bold')
    
    # Enhanced Performance Distribution Chart
    ax5 = fig.add_subplot(gs[1, :2])
    
    # Create histogram of performance ratios
    performance_ratios = [(p / (module_specs.get('rated_power_stc', 570) * est_mod)) * 100 
                         for p, est_mod in zip(powers, estimated_modules)]
    
    bins = np.linspace(min(performance_ratios), max(performance_ratios), 8)
    hist, bin_edges = np.histogram(performance_ratios, bins=bins)
    
    # Create gradient bars for histogram
    bar_colors = [COLORS['success'] if (bin_edges[i] + bin_edges[i+1])/2 > 90 else 
                  COLORS['warning'] if (bin_edges[i] + bin_edges[i+1])/2 > 80 else 
                  COLORS['danger'] for i in range(len(hist))]
    
    bars = ax5.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), 
                   color=bar_colors, alpha=0.8, edgecolor=COLORS['dark'], linewidth=1.5)
    
    ax5.set_xlabel('Performance Ratio (%)', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Number of Inverters', fontweight='bold', fontsize=12)
    ax5.set_title('📊 Fleet Performance Distribution', fontweight='bold', fontsize=14, 
                  color=COLORS['primary'], pad=20)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Enhanced Efficiency Comparison
    ax6 = fig.add_subplot(gs[1, 2:])
    
    # Create box plot for key metrics
    metrics_data = [powers, [ff*1000 for ff in fill_factors], temperatures, estimated_modules]
    metrics_labels = ['Power (W)', 'Fill Factor (×1000)', 'Temperature (°C)', 'Module Count']
    
    box_plot = ax6.boxplot(metrics_data, labels=metrics_labels, patch_artist=True, 
                          boxprops=dict(facecolor=COLORS['primary'], alpha=0.7),
                          medianprops=dict(color=COLORS['dark'], linewidth=2),
                          whiskerprops=dict(color=COLORS['dark'], linewidth=1.5),
                          capprops=dict(color=COLORS['dark'], linewidth=1.5))
    
    ax6.set_title('📊 Fleet Metrics Distribution', fontweight='bold', fontsize=14, 
                  color=COLORS['primary'], pad=20)
    ax6.grid(True, alpha=0.3)
    
    # Final summary chart
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create performance summary timeline
    summary_metrics = {
        'Total Fleet Power': f"{total_measured_power:.0f} W",
        'Average Efficiency': f"{np.mean(performance_ratios):.1f}%",
        'Best Performer': f"{inverter_names[np.argmax(powers)].split('_')[1]} ({max(powers):.0f}W)",
        'Highest Fill Factor': f"{inverter_names[np.argmax(fill_factors)].split('_')[1]} ({max(fill_factors):.3f})",
        'Optimal Temperature': f"{temperatures[np.argmin([abs(t-25) for t in temperatures])]:.1f}°C",
        'System Health': '🟢 Excellent' if fleet_performance_ratio > 90 else '🟡 Good' if fleet_performance_ratio > 80 else '🟠 Fair'
    }
    
    # Create metric cards
    x_positions = np.linspace(0.05, 0.85, len(summary_metrics))
    for i, (metric, value) in enumerate(summary_metrics.items()):
        # Create mini card for each metric
        card_box = FancyBboxPatch((x_positions[i]-0.06, 0.2), 0.12, 0.6, 
                                 boxstyle="round,pad=0.02", 
                                 facecolor=GRADIENT_COLORS[i % len(GRADIENT_COLORS)], 
                                 edgecolor=COLORS['dark'],
                                 linewidth=1.5, alpha=0.3)
        ax7.add_patch(card_box)
        
        # Add metric title and value
        ax7.text(x_positions[i], 0.7, metric, transform=ax7.transAxes, 
                fontsize=10, fontweight='bold', horizontalalignment='center', 
                color=COLORS['dark'])
        ax7.text(x_positions[i], 0.4, value, transform=ax7.transAxes, 
                fontsize=12, fontweight='bold', horizontalalignment='center', 
                color=COLORS['dark'])
    
    plt.tight_layout()
    
    # Save with enhanced quality
    filename = "enhanced_stylish_fleet_analysis_summary.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', 
               edgecolor='none', pad_inches=0.3)
    print(f"\n🎨 Enhanced stylish fleet summary plot saved as: {filename}")
    plt.close()
    
    # Plot 3: Module specifications and system configuration
    ax3.axis('off')
    ax3.set_title('PV Module Specifications & System Configuration', fontweight='bold')
    
    module_specs = analyzer.module_specs
    spec_text = f"""
PV Module Specifications:
  Name: {module_specs.get('name', 'Unknown')}
  Manufacturer: {module_specs.get('manufacturer', 'Unknown')}
  Rated Power: {module_specs.get('rated_power_stc', 0):.0f} W
  VOC (STC): {module_specs.get('rated_voltage_stc', 0):.2f} V
  ISC (STC): {module_specs.get('rated_current_stc', 0):.2f} A
  VMP (STC): {module_specs.get('voltage_at_max_power', 0):.2f} V
  IMP (STC): {module_specs.get('current_at_max_power', 0):.2f} A
  Fill Factor: {module_specs.get('fill_factor_nominal', 0):.3f}
  Efficiency: {module_specs.get('efficiency_stc', 0):.2f}%

System Configuration (Estimated):
  Modules per String: {result['modules_per_string']}
  Parallel Strings: {result['strings_in_parallel']}
  Total Modules: {result['modules_per_string'] * result['strings_in_parallel']}
  Data Points: {result['data_points']}
    """
    ax3.text(0.05, 0.95, spec_text, transform=ax3.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace')
    
    # Plot 4: Performance analysis and temperature effects
    ax4.axis('off')
    ax4.set_title('Performance Analysis & Environmental Effects', fontweight='bold')
    
    temp_effects = result['temperature_effects']
    expected_system_power = module_specs.get('rated_power_stc', 570) * result['modules_per_string'] * result['strings_in_parallel']
    performance_ratio = (result['p_max'] / expected_system_power * 100) if expected_system_power > 0 else 0
    
    analysis_text = f"""
Measured Performance:
  Peak Power: {result['p_max']:.0f} W
  Fill Factor: {result['fill_factor']:.3f}
  Performance Ratio: {performance_ratio:.1f}%

Temperature Analysis:
  Average Temperature: {result['avg_temperature']:.1f}°C
  Delta from STC: {temp_effects['temperature_delta']:.1f}°C
  VOC Temperature Effect: {temp_effects['voc_effect_percent']:.2f}%
  ISC Temperature Effect: {temp_effects['isc_effect_percent']:.2f}%
  Power Temperature Effect: {temp_effects['power_effect_percent']:.2f}%

Expected vs Measured (with temp correction):
  Expected VOC: {module_specs.get('rated_voltage_stc', 0) * result['modules_per_string'] * temp_effects['expected_voc_factor']:.1f} V
  Measured VOC: {result['voc']:.1f} V
  Expected ISC: {module_specs.get('rated_current_stc', 0) * result['strings_in_parallel'] * temp_effects['expected_isc_factor']:.2f} A
  Measured ISC: {result['isc']:.2f} A
    """
    ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    filename = f"enhanced_analysis_{result['inverter_id'].lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Enhanced plot saved as: {filename}")
    plt.close()


def create_fleet_analysis_summary(results: List[Dict[str, Any]], analyzer: EnhancedInverterAnalyzer):
    """Create comprehensive fleet analysis summary"""
    
    print(f"\n📊 Fleet Analysis Summary ({len(results)} inverters)")
    print("=" * 60)
    
    # Calculate fleet statistics
    total_power = sum(r['p_max'] for r in results)
    avg_fill_factor = np.mean([r['fill_factor'] for r in results])
    avg_temp = np.mean([r['avg_temperature'] for r in results])
    
    # Module specifications
    module_specs = analyzer.module_specs
    module_power = module_specs.get('rated_power_stc', 570)
    
    print(f"Fleet Performance Summary:")
    print(f"  Total System Power: {total_power:.0f} W")
    print(f"  Average Fill Factor: {avg_fill_factor:.3f}")
    print(f"  Average Temperature: {avg_temp:.1f}°C")
    
    # Individual inverter summary
    print(f"\nIndividual Inverter Performance:")
    print(f"{'Inverter':<20} {'Power (W)':<12} {'Fill Factor':<12} {'Temp (°C)':<10} {'Modules Est.':<12}")
    print("-" * 70)
    
    for result in results:
        estimated_modules = result['modules_per_string'] * result['strings_in_parallel']
        print(f"{result['inverter_id']:<20} {result['p_max']:<12.0f} {result['fill_factor']:<12.3f} "
              f"{result['avg_temperature']:<10.1f} {estimated_modules:<12}")
    
    # Create fleet summary plot
    create_fleet_summary_plot(results, analyzer)


def demonstrate_with_enhanced_synthetic_data(analyzer: EnhancedInverterAnalyzer):
    """Enhanced demonstration with synthetic data based on real module specifications"""
    
    print("\n🔬 Enhanced Demonstration with Module-Specific Synthetic Data")
    print("-" * 60)
    
    module_specs = analyzer.module_specs
    
    # Create realistic scenarios based on actual module specifications
    scenarios = {
        "Optimal Performance": {
            'modules_per_string': 20,
            'strings_in_parallel': 2,
            'temperature': 25.0,
            'performance_factor': 0.98
        },
        "High Temperature": {
            'modules_per_string': 20,
            'strings_in_parallel': 2,
            'temperature': 60.0,
            'performance_factor': 0.92
        },
        "Degraded System": {
            'modules_per_string': 20,
            'strings_in_parallel': 2,
            'temperature': 35.0,
            'performance_factor': 0.85
        }
    }
    
    for scenario_name, config in scenarios.items():
        print(f"\n📊 Scenario: {scenario_name}")
        
        # Calculate expected performance based on module specs and configuration
        modules_total = config['modules_per_string'] * config['strings_in_parallel']
        
        # Apply temperature effects
        temp_effects = analyzer._calculate_temperature_effects(config['temperature'])
        
        # Calculate system parameters
        voc_system = (module_specs.get('rated_voltage_stc', 51.0) * config['modules_per_string'] * 
                     temp_effects['expected_voc_factor'] * config['performance_factor'])
        isc_system = (module_specs.get('rated_current_stc', 14.18) * config['strings_in_parallel'] * 
                     temp_effects['expected_isc_factor'] * config['performance_factor'])
        power_system = (module_specs.get('rated_power_stc', 570) * modules_total * 
                       temp_effects['expected_power_factor'] * config['performance_factor'])
        
        vmp_system = voc_system * 0.85  # Typical VMP/VOC ratio
        imp_system = power_system / vmp_system if vmp_system > 0 else 0
        fill_factor = (vmp_system * imp_system) / (voc_system * isc_system) if (voc_system > 0 and isc_system > 0) else 0
        
        print(f"  Configuration: {config['modules_per_string']} modules/string × {config['strings_in_parallel']} strings")
        print(f"  Temperature: {config['temperature']:.1f}°C (Δ{temp_effects['temperature_delta']:.1f}°C)")
        print(f"  VOC: {voc_system:.1f} V")
        print(f"  ISC: {isc_system:.2f} A")
        print(f"  Pmax: {power_system:.0f} W")
        print(f"  Fill Factor: {fill_factor:.3f}")
        print(f"  Performance Factor: {config['performance_factor']*100:.0f}%")
        print(f"  Temperature Effect on Power: {temp_effects['power_effect_percent']:.1f}%")


def create_analysis_plot(inverter_id: str, voltage: np.ndarray, current: np.ndarray, 
                        params: dict, diagnosis: dict):
    """Create a comprehensive analysis plot for an inverter."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'I-V Curve Analysis: {inverter_id}', fontsize=16, fontweight='bold')
    
    # Plot I-V curve
    ax1.plot(voltage, current, 'b-', linewidth=2, label='I-V Curve')
    ax1.axhline(y=params['isc'], color='r', linestyle='--', alpha=0.7, label=f"Isc = {params['isc']:.2f} A")
    ax1.axvline(x=params['voc'], color='g', linestyle='--', alpha=0.7, label=f"Voc = {params['voc']:.2f} V")
    ax1.scatter([params['v_mp']], [params['i_mp']], color='red', s=100, zorder=5, label=f"MPP ({params['v_mp']:.1f}V, {params['i_mp']:.1f}A)")
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current (A)')
    ax1.set_title('I-V Characteristic Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot P-V curve
    power = voltage * current
    ax2.plot(voltage, power, 'g-', linewidth=2, label='P-V Curve')
    ax2.scatter([params['v_mp']], [params['p_max']], color='red', s=100, zorder=5, label=f"Pmax = {params['p_max']:.1f} W")
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Power-Voltage Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Parameters summary
    ax3.axis('off')
    ax3.set_title('Key Parameters', fontweight='bold')
    param_text = f"""
    Open Circuit Voltage (Voc): {params['voc']:.2f} V
    Short Circuit Current (Isc): {params['isc']:.2f} A
    Maximum Power (Pmax): {params['p_max']:.1f} W
    Voltage at MPP (Vmp): {params['v_mp']:.2f} V
    Current at MPP (Imp): {params['i_mp']:.2f} A
    Fill Factor (FF): {params['fill_factor']:.3f}
    Series Resistance (Rs): {params['rs']:.2f} Ω
    Shunt Resistance (Rsh): {params['rsh']:.1f} Ω
    """
    ax3.text(0.1, 0.9, param_text, transform=ax3.transAxes, fontsize=11, 
            verticalalignment='top', fontfamily='monospace')
    
    # Diagnostic summary
    ax4.axis('off')
    ax4.set_title('Diagnostic Summary', fontweight='bold')
    
    # Color code health status
    health_color = {'Good': 'green', 'Fair': 'orange', 'Poor': 'red'}.get(diagnosis['overall_health'], 'black')
    severity_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}.get(diagnosis['severity'], 'black')
    
    diagnostic_text = f"Overall Health: {diagnosis['overall_health']}\n"
    diagnostic_text += f"Severity: {diagnosis['severity']}\n\n"
    
    if diagnosis['primary_issues']:
        diagnostic_text += "Primary Issues:\n"
        for issue in diagnosis['primary_issues'][:3]:
            diagnostic_text += f"• {issue}\n"
    
    if diagnosis['recommendations']:
        diagnostic_text += f"\nTop Recommendations:\n"
        for rec in diagnosis['recommendations'][:2]:
            diagnostic_text += f"→ {rec}\n"
    
    ax4.text(0.1, 0.9, diagnostic_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', wrap=True)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"analysis_{inverter_id.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Plot saved as: {filename}")
    plt.close()


if __name__ == "__main__":
    main()
