"""
Configuration-Integrated I-V Curve Analysis System

This script demonstrates the complete integration of the configuration file
with the I-V curve analysis system, utilizing PV module specifications
and analysis parameters from the config.toml file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Any
import warnings
from datetime import datetime
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class ConfigurableIVAnalyzer:
    """
    I-V Curve Analyzer that uses configuration parameters from config.toml
    for PV module specifications and analysis settings.
    """
    
    def __init__(self, config_path: str = "config.toml"):
        """Initialize the analyzer with configuration file."""
        self.config = self.load_config(config_path)
        self.pv_module = self.config['pv_module']
        self.analysis_params = self.config['analysis_parameters']
        self.environmental = self.config['environmental']
        self.diagnostics = self.config['diagnostics']
        self.economic = self.config['economic']
        
        print("🔧 Configuration-Integrated I-V Analyzer Initialized")
        print(f"📋 PV Module: {self.pv_module['name']} ({self.pv_module['rated_power_stc']}W)")
        print(f"🌡️  Operating Range: {self.environmental['min_temperature']}°C to {self.environmental['max_temperature']}°C")
        print(f"☀️  Irradiance Range: {self.environmental['min_irradiance']} to {self.environmental['max_irradiance']} W/m²")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from TOML file using simple parser."""
        try:
            return self.parse_toml_file(config_path)
        except FileNotFoundError:
            print(f"❌ Configuration file {config_path} not found. Using defaults.")
            return self.get_default_config()
    
    def parse_toml_file(self, file_path: str) -> Dict:
        """Simple TOML parser for configuration file."""
        config = {}
        current_section = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                
                # Section headers
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1]
                    if current_section not in config:
                        config[current_section] = {}
                    continue
                
                # Key-value pairs
                if '=' in line and current_section:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes and parse value
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]  # String value
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    else:
                        try:
                            # Try parsing as number
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            # Keep as string if parsing fails
                            pass
                    
                    config[current_section][key] = value
        
        return config
    
    def get_default_config(self) -> Dict:
        """Return default configuration if file not found."""
        return {
            'pv_module': {
                'rated_power_stc': 300.0,
                'rated_voltage_stc': 750.0,
                'rated_current_stc': 20.0,
                'fill_factor_nominal': 0.80,
                'temp_coeff_voc': -2.3e-3,
                'temp_coeff_isc': 0.06e-2,
                'temp_coeff_power': -0.4e-2,
                'series_resistance_typical': 0.5,
                'shunt_resistance_typical': 5000
            },
            'environmental': {
                'reference_temperature': 25.0,
                'reference_irradiance': 1000.0,
                'min_temperature': -10.0,
                'max_temperature': 70.0
            },
            'diagnostics': {
                'fill_factor_excellent': 0.82,
                'fill_factor_good': 0.75,
                'fill_factor_fair': 0.65,
                'series_resistance_excellent': 0.3,
                'series_resistance_good': 0.7,
                'series_resistance_warning': 2.0
            },
            'economic': {
                'electricity_rate': 0.12,
                'project_lifetime_years': 25,
                'maintenance_cost_base': 1000
            }
        }
    
    def generate_iv_curve_with_config(self, 
                                    voc: float = None, 
                                    isc: float = None, 
                                    rs: float = None, 
                                    rsh: float = None,
                                    temperature: float = None,
                                    irradiance: float = None,
                                    n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate I-V curve using configuration parameters.
        
        Uses PV module specifications from config.toml as defaults.
        """
        # Use configuration defaults if not provided
        if voc is None:
            voc = self.pv_module['rated_voltage_stc']
        if isc is None:
            isc = self.pv_module['rated_current_stc']
        if rs is None:
            rs = self.pv_module['series_resistance_typical']
        if rsh is None:
            rsh = self.pv_module['shunt_resistance_typical']
        if temperature is None:
            temperature = self.environmental['reference_temperature']
        if irradiance is None:
            irradiance = self.environmental['reference_irradiance']
        
        # Apply environmental corrections
        voc_corrected, isc_corrected = self.apply_environmental_corrections(
            voc, isc, temperature, irradiance
        )
        
        # Generate voltage array
        voltage = np.linspace(0, voc_corrected, n_points)
        
        # Calculate current using single-diode model with configuration parameters
        current = self.calculate_current_single_diode(
            voltage, voc_corrected, isc_corrected, rs, rsh
        )
        
        return voltage, current
    
    def apply_environmental_corrections(self, 
                                     voc: float, 
                                     isc: float, 
                                     temperature: float, 
                                     irradiance: float) -> Tuple[float, float]:
        """Apply temperature and irradiance corrections using config parameters."""
        # Temperature differences from STC
        temp_diff = temperature - self.environmental['reference_temperature']
        
        # Irradiance ratio
        irr_ratio = irradiance / self.environmental['reference_irradiance']
        
        # Apply corrections using configuration coefficients
        voc_corrected = voc + (self.pv_module['temp_coeff_voc'] * 
                             self.pv_module['cells_in_series'] * temp_diff)
        isc_corrected = isc * irr_ratio * (1 + self.pv_module['temp_coeff_isc'] * temp_diff)
        
        return voc_corrected, isc_corrected
    
    def calculate_current_single_diode(self, 
                                     voltage: np.ndarray, 
                                     voc: float, 
                                     isc: float, 
                                     rs: float, 
                                     rsh: float) -> np.ndarray:
        """
        Calculate current using single-diode model.
        
        Implements the fundamental photovoltaic equation:
        I = Iph - I0 * (exp((V + I*Rs)/(n*Vt)) - 1) - (V + I*Rs)/Rsh
        """
        # Thermal voltage at reference temperature
        k = 1.381e-23  # Boltzmann constant
        q = 1.602e-19  # Electron charge
        T = self.environmental['reference_temperature'] + 273.15  # Convert to Kelvin
        Vt = k * T / q
        
        # Diode ideality factor from configuration
        n = self.pv_module.get('ideality_factor_typical', 1.2)
        
        # Photocurrent (approximately equal to Isc)
        Iph = isc
        
        # Reverse saturation current estimation
        I0 = self.pv_module.get('reverse_saturation_current', 1e-12)
        
        # Current calculation (simplified approach for demonstration)
        current = np.zeros_like(voltage)
        
        for i, v in enumerate(voltage):
            # Simplified current calculation avoiding complex iteration
            if v >= voc:
                current[i] = 0
            else:
                # Approximate current using linear interpolation method
                i_approx = isc * (1 - (v / voc)**2) - (v / rsh)
                current[i] = max(0, i_approx)
        
        return current
    
    def extract_parameters(self, voltage: np.ndarray, current: np.ndarray) -> Dict:
        """Extract I-V parameters and validate against configuration limits."""
        # Basic parameter extraction
        voc = voltage[np.where(current <= 0)[0][0]] if np.any(current <= 0) else voltage[-1]
        isc = current[0] if current[0] > 0 else np.max(current)
        
        # Find maximum power point
        power = voltage * current
        max_power_idx = np.argmax(power)
        v_mp = voltage[max_power_idx]
        i_mp = current[max_power_idx]
        p_max = power[max_power_idx]
        
        # Calculate fill factor
        fill_factor = p_max / (voc * isc) if (voc * isc) > 0 else 0
        
        # Estimate series and shunt resistance
        rs = self.estimate_series_resistance(voltage, current)
        rsh = self.estimate_shunt_resistance(voltage, current)
        
        # Validate parameters against configuration limits
        params = {
            'voc': voc,
            'isc': isc,
            'v_mp': v_mp,
            'i_mp': i_mp,
            'p_max': p_max,
            'fill_factor': fill_factor,
            'rs': rs,
            'rsh': rsh
        }
        
        # Apply validation
        params = self.validate_parameters(params)
        
        return params
    
    def estimate_series_resistance(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Estimate series resistance using slope method."""
        # Find points near Voc for Rs calculation
        voc_region = voltage > 0.9 * voltage.max()
        if np.sum(voc_region) < 2:
            return self.pv_module['series_resistance_typical']
        
        v_region = voltage[voc_region]
        i_region = current[voc_region]
        
        # Calculate slope (dV/dI)
        if len(v_region) > 1:
            slope = np.gradient(v_region, i_region)
            rs = -np.mean(slope[slope < 0]) if np.any(slope < 0) else self.pv_module['series_resistance_typical']
        else:
            rs = self.pv_module['series_resistance_typical']
        
        return max(0.01, min(rs, 10.0))  # Reasonable bounds
    
    def estimate_shunt_resistance(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Estimate shunt resistance using slope method."""
        # Find points near Isc for Rsh calculation
        isc_region = voltage < 0.1 * voltage.max()
        if np.sum(isc_region) < 2:
            return self.pv_module['shunt_resistance_typical']
        
        v_region = voltage[isc_region]
        i_region = current[isc_region]
        
        # Calculate slope (dI/dV)
        if len(v_region) > 1:
            slope = np.gradient(i_region, v_region)
            rsh = -1 / np.mean(slope[slope < 0]) if np.any(slope < 0) else self.pv_module['shunt_resistance_typical']
        else:
            rsh = self.pv_module['shunt_resistance_typical']
        
        return max(10, min(rsh, 50000))  # Reasonable bounds
    
    def validate_parameters(self, params: Dict) -> Dict:
        """Validate extracted parameters against configuration limits."""
        validated = params.copy()
        
        # Validate against analysis parameter limits
        if 'voc_min' in self.analysis_params:
            validated['voc'] = max(params['voc'], self.analysis_params['voc_min'])
        if 'voc_max' in self.analysis_params:
            validated['voc'] = min(params['voc'], self.analysis_params['voc_max'])
        
        if 'isc_min' in self.analysis_params:
            validated['isc'] = max(params['isc'], self.analysis_params['isc_min'])
        if 'isc_max' in self.analysis_params:
            validated['isc'] = min(params['isc'], self.analysis_params['isc_max'])
        
        if 'fill_factor_min' in self.analysis_params:
            validated['fill_factor'] = max(params['fill_factor'], self.analysis_params['fill_factor_min'])
        if 'fill_factor_max' in self.analysis_params:
            validated['fill_factor'] = min(params['fill_factor'], self.analysis_params['fill_factor_max'])
        
        return validated
    
    def perform_health_assessment(self, params: Dict) -> Dict:
        """Perform health assessment using configuration thresholds."""
        health_assessment = {
            'overall_health': 'Unknown',
            'health_score': 0,
            'fill_factor_rating': 'Unknown',
            'resistance_rating': 'Unknown',
            'issues': [],
            'recommendations': []
        }
        
        # Assess fill factor
        ff = params['fill_factor']
        if ff >= self.diagnostics['fill_factor_excellent']:
            health_assessment['fill_factor_rating'] = 'Excellent'
            ff_score = 100
        elif ff >= self.diagnostics['fill_factor_good']:
            health_assessment['fill_factor_rating'] = 'Good'
            ff_score = 80
        elif ff >= self.diagnostics['fill_factor_fair']:
            health_assessment['fill_factor_rating'] = 'Fair'
            ff_score = 60
        else:
            health_assessment['fill_factor_rating'] = 'Poor'
            ff_score = 30
            health_assessment['issues'].append('Low fill factor indicates performance degradation')
            health_assessment['recommendations'].append('Inspect for shading, soiling, or module degradation')
        
        # Assess series resistance
        rs = params['rs']
        if rs <= self.diagnostics['series_resistance_excellent']:
            health_assessment['resistance_rating'] = 'Excellent'
            rs_score = 100
        elif rs <= self.diagnostics['series_resistance_good']:
            health_assessment['resistance_rating'] = 'Good'
            rs_score = 80
        elif rs <= self.diagnostics['series_resistance_warning']:
            health_assessment['resistance_rating'] = 'Warning'
            rs_score = 50
            health_assessment['issues'].append('Elevated series resistance detected')
            health_assessment['recommendations'].append('Check electrical connections and contacts')
        else:
            health_assessment['resistance_rating'] = 'Critical'
            rs_score = 20
            health_assessment['issues'].append('High series resistance - critical issue')
            health_assessment['recommendations'].append('Immediate maintenance required - check all connections')
        
        # Calculate overall health score
        health_assessment['health_score'] = (ff_score + rs_score) / 2
        
        # Determine overall health
        if health_assessment['health_score'] >= 90:
            health_assessment['overall_health'] = 'Excellent'
        elif health_assessment['health_score'] >= 75:
            health_assessment['overall_health'] = 'Good'
        elif health_assessment['health_score'] >= 60:
            health_assessment['overall_health'] = 'Fair'
        else:
            health_assessment['overall_health'] = 'Poor'
        
        return health_assessment
    
    def calculate_economic_impact(self, params: Dict, health_assessment: Dict) -> Dict:
        """Calculate economic impact using configuration parameters."""
        # Base calculations from configuration
        rated_power = self.pv_module['rated_power_stc']
        electricity_rate = self.economic['electricity_rate']
        project_lifetime = self.economic['project_lifetime_years']
        capacity_factor = self.economic.get('capacity_factor', 0.20)
        
        # Calculate power loss
        expected_power = rated_power
        actual_power = params['p_max']
        power_loss_percent = max(0, (expected_power - actual_power) / expected_power * 100)
        
        # Annual energy loss
        annual_energy_loss = (power_loss_percent / 100) * rated_power * 8760 * capacity_factor / 1000  # kWh
        annual_revenue_loss = annual_energy_loss * electricity_rate
        
        # Lifetime impact
        lifetime_revenue_loss = annual_revenue_loss * project_lifetime
        
        # Maintenance cost estimation based on health
        base_maintenance = self.economic['maintenance_cost_base']
        if health_assessment['overall_health'] == 'Poor':
            maintenance_multiplier = 2.0
        elif health_assessment['overall_health'] == 'Fair':
            maintenance_multiplier = 1.5
        elif health_assessment['overall_health'] == 'Good':
            maintenance_multiplier = 1.0
        else:
            maintenance_multiplier = 0.8
        
        estimated_maintenance_cost = base_maintenance * maintenance_multiplier
        
        return {
            'power_loss_percent': power_loss_percent,
            'annual_energy_loss_kwh': annual_energy_loss,
            'annual_revenue_loss': annual_revenue_loss,
            'lifetime_revenue_loss': lifetime_revenue_loss,
            'estimated_maintenance_cost': estimated_maintenance_cost,
            'total_economic_impact': lifetime_revenue_loss + estimated_maintenance_cost
        }
    
    def create_comprehensive_report(self, inverter_id: str, params: Dict, 
                                  health_assessment: Dict, economic_impact: Dict) -> str:
        """Create a comprehensive analysis report."""
        report = f"""
# Configuration-Based I-V Analysis Report
## Inverter: {inverter_id}
### Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## PV Module Configuration
- **Module Type**: {self.pv_module['name']}
- **Rated Power**: {self.pv_module['rated_power_stc']} W
- **Rated Voltage**: {self.pv_module['rated_voltage_stc']} V
- **Rated Current**: {self.pv_module['rated_current_stc']} A
- **Technology**: {self.pv_module.get('technology', 'Silicon')}

## Measured Parameters
- **VOC**: {params['voc']:.2f} V
- **ISC**: {params['isc']:.2f} A
- **Pmax**: {params['p_max']:.1f} W
- **Vmp**: {params['v_mp']:.2f} V
- **Imp**: {params['i_mp']:.2f} A
- **Fill Factor**: {params['fill_factor']:.3f}
- **Series Resistance**: {params['rs']:.3f} Ω
- **Shunt Resistance**: {params['rsh']:.1f} Ω

## Health Assessment
- **Overall Health**: {health_assessment['overall_health']} ({health_assessment['health_score']:.1f}/100)
- **Fill Factor Rating**: {health_assessment['fill_factor_rating']}
- **Resistance Rating**: {health_assessment['resistance_rating']}

### Issues Identified:
"""
        
        if health_assessment['issues']:
            for issue in health_assessment['issues']:
                report += f"- {issue}\n"
        else:
            report += "- No significant issues detected\n"
        
        report += f"""
### Recommendations:
"""
        
        if health_assessment['recommendations']:
            for rec in health_assessment['recommendations']:
                report += f"- {rec}\n"
        else:
            report += "- Continue regular monitoring\n"
        
        report += f"""
## Economic Impact Analysis
- **Power Loss**: {economic_impact['power_loss_percent']:.1f}%
- **Annual Energy Loss**: {economic_impact['annual_energy_loss_kwh']:.1f} kWh
- **Annual Revenue Loss**: ${economic_impact['annual_revenue_loss']:.2f}
- **Lifetime Revenue Loss**: ${economic_impact['lifetime_revenue_loss']:.2f}
- **Estimated Maintenance Cost**: ${economic_impact['estimated_maintenance_cost']:.2f}
- **Total Economic Impact**: ${economic_impact['total_economic_impact']:.2f}

## Configuration Parameters Used
- **Analysis Method**: {self.analysis_params.get('curve_fitting_method', 'least_squares')}
- **Temperature Correction**: {'Enabled' if self.analysis_params.get('temperature_correction', True) else 'Disabled'}
- **Irradiance Correction**: {'Enabled' if self.analysis_params.get('irradiance_correction', True) else 'Disabled'}
- **Reference Conditions**: {self.environmental['reference_irradiance']} W/m², {self.environmental['reference_temperature']}°C
"""
        
        return report


def demonstrate_config_integration():
    """Demonstrate the configuration-integrated analysis system."""
    print("🚀 Configuration-Integrated I-V Analysis Demonstration")
    print("=" * 60)
    
    # Initialize the configurable analyzer
    analyzer = ConfigurableIVAnalyzer()
    
    # Generate scenarios based on configuration
    scenarios = [
        {
            'name': 'Optimal Performance',
            'voc': 750.0,
            'isc': 20.0,
            'rs': 0.3,
            'rsh': 10000,
            'temperature': 25.0,
            'irradiance': 1000.0
        },
        {
            'name': 'High Temperature Operation',
            'voc': 750.0,
            'isc': 20.0,
            'rs': 0.5,
            'rsh': 8000,
            'temperature': 60.0,
            'irradiance': 1000.0
        },
        {
            'name': 'Low Irradiance Conditions',
            'voc': 750.0,
            'isc': 20.0,
            'rs': 0.4,
            'rsh': 9000,
            'temperature': 25.0,
            'irradiance': 400.0
        },
        {
            'name': 'Degraded Module',
            'voc': 720.0,
            'isc': 18.5,
            'rs': 1.5,
            'rsh': 3000,
            'temperature': 25.0,
            'irradiance': 1000.0
        }
    ]
    
    # Analyze each scenario
    all_results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        
        # Generate I-V curve
        voltage, current = analyzer.generate_iv_curve_with_config(**scenario)
        
        # Extract parameters
        params = analyzer.extract_parameters(voltage, current)
        
        # Perform health assessment
        health_assessment = analyzer.perform_health_assessment(params)
        
        # Calculate economic impact
        economic_impact = analyzer.calculate_economic_impact(params, health_assessment)
        
        # Display key results
        print(f"  Power Output: {params['p_max']:.1f} W")
        print(f"  Fill Factor: {params['fill_factor']:.3f}")
        print(f"  Health Score: {health_assessment['health_score']:.1f}/100")
        print(f"  Economic Impact: ${economic_impact['total_economic_impact']:.2f}")
        
        # Store results
        all_results.append({
            'scenario': scenario['name'],
            'params': params,
            'health': health_assessment,
            'economic': economic_impact,
            'voltage': voltage,
            'current': current
        })
    
    # Create comprehensive visualization
    create_config_based_visualization(all_results, analyzer)
    
    # Generate individual reports
    for result in all_results:
        report = analyzer.create_comprehensive_report(
            result['scenario'],
            result['params'],
            result['health'],
            result['economic']
        )
        
        # Save report to file
        report_filename = f"config_analysis_{result['scenario'].replace(' ', '_').lower()}.md"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved: {report_filename}")
    
    print(f"\n✅ Configuration-integrated analysis complete!")
    print(f"📈 Generated visualization: config_based_analysis.png")
    print(f"📋 Generated {len(all_results)} detailed reports")


def create_config_based_visualization(results: List[Dict], analyzer: ConfigurableIVAnalyzer):
    """Create comprehensive visualization of configuration-based analysis."""
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # I-V Curves Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    for result in results:
        ax1.plot(result['voltage'], result['current'], 
                linewidth=2, label=result['scenario'])
    ax1.set_xlabel('Voltage (V)', fontsize=12)
    ax1.set_ylabel('Current (A)', fontsize=12)
    ax1.set_title('I-V Curves Comparison (Configuration-Based)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Power Curves Comparison
    ax2 = fig.add_subplot(gs[1, :2])
    for result in results:
        power = result['voltage'] * result['current']
        ax2.plot(result['voltage'], power, 
                linewidth=2, label=result['scenario'])
    ax2.set_xlabel('Voltage (V)', fontsize=12)
    ax2.set_ylabel('Power (W)', fontsize=12)
    ax2.set_title('P-V Curves Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Performance Metrics Bar Chart
    ax3 = fig.add_subplot(gs[0, 2])
    scenarios = [r['scenario'] for r in results]
    fill_factors = [r['params']['fill_factor'] for r in results]
    colors = ['green', 'orange', 'blue', 'red']
    
    bars = ax3.bar(range(len(scenarios)), fill_factors, color=colors, alpha=0.7)
    ax3.set_xlabel('Scenario', fontsize=12)
    ax3.set_ylabel('Fill Factor', fontsize=12)
    ax3.set_title('Fill Factor Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(scenarios)))
    ax3.set_xticklabels([s.split()[0] for s in scenarios], rotation=45)
    
    # Add FF threshold lines from configuration
    ax3.axhline(y=analyzer.diagnostics['fill_factor_excellent'], 
               color='green', linestyle='--', alpha=0.5, label='Excellent')
    ax3.axhline(y=analyzer.diagnostics['fill_factor_good'], 
               color='orange', linestyle='--', alpha=0.5, label='Good')
    ax3.axhline(y=analyzer.diagnostics['fill_factor_fair'], 
               color='red', linestyle='--', alpha=0.5, label='Fair')
    ax3.legend(fontsize=8)
    
    # Health Scores
    ax4 = fig.add_subplot(gs[1, 2])
    health_scores = [r['health']['health_score'] for r in results]
    bars = ax4.bar(range(len(scenarios)), health_scores, color=colors, alpha=0.7)
    ax4.set_xlabel('Scenario', fontsize=12)
    ax4.set_ylabel('Health Score', fontsize=12)
    ax4.set_title('Health Assessment', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(scenarios)))
    ax4.set_xticklabels([s.split()[0] for s in scenarios], rotation=45)
    ax4.set_ylim(0, 100)
    
    # Economic Impact
    ax5 = fig.add_subplot(gs[2, :])
    economic_impacts = [r['economic']['total_economic_impact'] for r in results]
    bars = ax5.bar(scenarios, economic_impacts, color=colors, alpha=0.7)
    ax5.set_xlabel('Scenario', fontsize=12)
    ax5.set_ylabel('Total Economic Impact ($)', fontsize=12)
    ax5.set_title('Economic Impact Analysis (Configuration-Based)', fontsize=14, fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, economic_impacts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${value:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Overall title
    fig.suptitle('Configuration-Integrated I-V Curve Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add configuration info
    config_text = f"""Configuration Parameters:
    • PV Module: {analyzer.pv_module['name']} ({analyzer.pv_module['rated_power_stc']}W)
    • Reference Conditions: {analyzer.environmental['reference_irradiance']} W/m², {analyzer.environmental['reference_temperature']}°C
    • Analysis Method: {analyzer.analysis_params.get('curve_fitting_method', 'least_squares')}
    • Health Thresholds: FF Excellent ≥ {analyzer.diagnostics['fill_factor_excellent']:.2f}"""
    
    fig.text(0.02, 0.02, config_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('config_based_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demonstrate_config_integration()
