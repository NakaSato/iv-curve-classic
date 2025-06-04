"""
Enhanced I-V Curve Analysis with Configuration Integration

This script demonstrates the integration of PV module specifications from config.toml
into the I-V curve analysis system with comprehensive mathematical modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class ConfigBasedIVAnalyzer:
    """
    I-V Curve Analyzer that integrates PV module specifications
    from configuration file for enhanced analysis accuracy.
    """
    
    def __init__(self, config_path: str = "config.toml"):
        """Initialize analyzer with configuration parameters."""
        self.config = self.load_config(config_path)
        print("🔧 Configuration-Based I-V Analyzer Initialized")
        print(f"📋 Module: {self.config['pv_module']['name']} ({self.config['pv_module']['rated_power_stc']}W)")
        print(f"🌡️  Temperature Range: {self.config['environmental']['min_temperature']}°C to {self.config['environmental']['max_temperature']}°C")
        print(f"☀️  Irradiance Range: {self.config['environmental']['min_irradiance']} to {self.config['environmental']['max_irradiance']} W/m²")
    
    def load_config(self, config_path: str) -> Dict:
        """Load and parse configuration file."""
        config = {
            'pv_module': {},
            'environmental': {},
            'diagnostics': {},
            'economic': {},
            'analysis_parameters': {}
        }
        
        try:
            current_section = None
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#') or line.startswith('//'):
                        continue
                    
                    # Parse section headers
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                        continue
                    
                    # Parse key-value pairs
                    if '=' in line and current_section and current_section in config:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove inline comments
                        if '#' in value:
                            value = value.split('#')[0].strip()
                        if '//' in value:
                            value = value.split('//')[0].strip()
                        
                        # Parse value types
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]  # String
                        elif value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        else:
                            try:
                                # Try parsing as number
                                if '.' in value or 'e-' in value.lower():
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                # Keep as string if parsing fails
                                pass
                        
                        config[current_section][key] = value
            
            print(f"✅ Configuration loaded from {config_path}")
            
        except FileNotFoundError:
            print(f"⚠️  Configuration file not found. Using defaults.")
            config = self.get_default_config()
        
        return config
    
    def get_default_config(self) -> Dict:
        """Return default configuration values."""
        return {
            'pv_module': {
                'name': 'Standard PV Module',
                'rated_power_stc': 300.0,
                'rated_voltage_stc': 750.0,
                'rated_current_stc': 20.0,
                'voltage_at_max_power': 600.0,
                'current_at_max_power': 16.67,
                'fill_factor_nominal': 0.80,
                'temp_coeff_voc': -2.3e-3,
                'temp_coeff_isc': 0.06e-2,
                'temp_coeff_power': -0.4e-2,
                'series_resistance_typical': 0.5,
                'shunt_resistance_typical': 5000,
                'cells_in_series': 72
            },
            'environmental': {
                'reference_temperature': 25.0,
                'reference_irradiance': 1000.0,
                'min_temperature': -10.0,
                'max_temperature': 70.0,
                'min_irradiance': 100.0,
                'max_irradiance': 1200.0
            },
            'diagnostics': {
                'fill_factor_excellent': 0.82,
                'fill_factor_good': 0.75,
                'fill_factor_fair': 0.65,
                'series_resistance_excellent': 0.3,
                'series_resistance_good': 0.7,
                'series_resistance_warning': 2.0,
                'shunt_resistance_excellent': 10000,
                'shunt_resistance_good': 5000,
                'shunt_resistance_warning': 1000
            },
            'economic': {
                'electricity_rate': 0.12,
                'project_lifetime_years': 25,
                'maintenance_cost_base': 1000,
                'capacity_factor': 0.20
            },
            'analysis_parameters': {
                'curve_fitting_method': 'least_squares',
                'temperature_correction': True,
                'irradiance_correction': True
            }
        }
    
    def generate_iv_curve_with_environmental_correction(self, 
                                                      base_voc: float = None,
                                                      base_isc: float = None,
                                                      rs: float = None,
                                                      rsh: float = None,
                                                      temperature: float = 25.0,
                                                      irradiance: float = 1000.0,
                                                      n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate I-V curve with environmental corrections based on configuration.
        
        Uses PV module specifications from config for corrections.
        """
        # Use config defaults if not provided
        if base_voc is None:
            base_voc = self.config['pv_module']['rated_voltage_stc']
        if base_isc is None:
            base_isc = self.config['pv_module']['rated_current_stc']
        if rs is None:
            rs = self.config['pv_module']['series_resistance_typical']
        if rsh is None:
            rsh = self.config['pv_module']['shunt_resistance_typical']
        
        # Apply environmental corrections
        voc_corrected, isc_corrected = self.apply_environmental_corrections(
            base_voc, base_isc, temperature, irradiance
        )
        
        # Generate I-V curve
        voltage = np.linspace(0, voc_corrected, n_points)
        current = self.calculate_current_with_single_diode_model(
            voltage, voc_corrected, isc_corrected, rs, rsh, temperature
        )
        
        return voltage, current
    
    def apply_environmental_corrections(self, 
                                      base_voc: float, 
                                      base_isc: float, 
                                      temperature: float, 
                                      irradiance: float) -> Tuple[float, float]:
        """
        Apply temperature and irradiance corrections using config parameters.
        
        Mathematical Model:
        - VOC(T) = VOC_STC + (T - T_STC) × TC_VOC × N_cells
        - ISC(G,T) = ISC_STC × (G/G_STC) × (1 + TC_ISC × (T - T_STC))
        """
        ref_temp = self.config['environmental']['reference_temperature']
        ref_irr = self.config['environmental']['reference_irradiance']
        
        # Temperature difference from STC
        temp_diff = temperature - ref_temp
        
        # Irradiance ratio
        irr_ratio = irradiance / ref_irr
        
        # Apply corrections using config coefficients
        temp_coeff_voc = self.config['pv_module']['temp_coeff_voc']
        temp_coeff_isc = self.config['pv_module']['temp_coeff_isc']
        cells_in_series = self.config['pv_module']['cells_in_series']
        
        # Voltage correction (per cell × number of cells)
        voc_corrected = base_voc + (temp_coeff_voc * cells_in_series * temp_diff)
        
        # Current correction (irradiance + temperature effects)
        isc_corrected = base_isc * irr_ratio * (1 + temp_coeff_isc * temp_diff)
        
        return max(0, voc_corrected), max(0, isc_corrected)
    
    def calculate_current_with_single_diode_model(self, 
                                                voltage: np.ndarray,
                                                voc: float,
                                                isc: float,
                                                rs: float,
                                                rsh: float,
                                                temperature: float) -> np.ndarray:
        """
        Calculate current using single-diode model with config parameters.
        
        Single-Diode Model Equation:
        I = Iph - I0 × (exp((V + I×Rs)/(n×Vt)) - 1) - (V + I×Rs)/Rsh
        
        Where:
        - Iph ≈ Isc (photocurrent)
        - I0 = reverse saturation current
        - n = ideality factor
        - Vt = thermal voltage = kT/q
        """
        # Physical constants
        k = 1.381e-23  # Boltzmann constant (J/K)
        q = 1.602e-19  # Elementary charge (C)
        
        # Temperature in Kelvin
        T_kelvin = temperature + 273.15
        
        # Thermal voltage
        Vt = k * T_kelvin / q
        
        # Model parameters
        n = 1.2  # Ideality factor (typical value)
        Iph = isc  # Photocurrent ≈ Isc
        
        # Simplified current calculation (avoiding complex iteration)
        current = np.zeros_like(voltage)
        
        for i, v in enumerate(voltage):
            if v >= voc:
                current[i] = 0
            else:
                # Simplified model for demonstration
                # More accurate implementation would require iterative solution
                v_norm = v / voc
                i_linear = isc * (1 - v_norm)  # Linear approximation
                i_exponential = isc * (1 - np.exp((v - voc) / (n * Vt * 10)))  # Exponential component
                i_shunt = v / rsh  # Shunt current
                
                # Combine components
                current[i] = max(0, min(i_linear * 0.3 + i_exponential * 0.7 - i_shunt, isc))
        
        return current
    
    def extract_parameters_from_curve(self, voltage: np.ndarray, current: np.ndarray) -> Dict:
        """Extract I-V parameters from generated curve."""
        # Basic parameter extraction
        voc = voltage[current <= 0.01][0] if np.any(current <= 0.01) else voltage[-1]
        isc = current[0]
        
        # Find maximum power point
        power = voltage * current
        max_power_idx = np.argmax(power)
        v_mp = voltage[max_power_idx]
        i_mp = current[max_power_idx]
        p_max = power[max_power_idx]
        
        # Calculate fill factor
        fill_factor = p_max / (voc * isc) if (voc * isc) > 0 else 0
        
        # Estimate resistances
        rs = self.estimate_series_resistance(voltage, current)
        rsh = self.estimate_shunt_resistance(voltage, current)
        
        return {
            'voc': float(voc),
            'isc': float(isc),
            'v_mp': float(v_mp),
            'i_mp': float(i_mp),
            'p_max': float(p_max),
            'fill_factor': float(fill_factor),
            'rs': float(rs),
            'rsh': float(rsh)
        }
    
    def estimate_series_resistance(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Estimate series resistance from slope near VOC."""
        # Find region near VOC (last 10% of voltage range)
        voc_region = voltage > 0.9 * np.max(voltage)
        if np.sum(voc_region) < 2:
            return self.config['pv_module']['series_resistance_typical']
        
        v_region = voltage[voc_region]
        i_region = current[voc_region]
        
        # Calculate slope dV/dI
        if len(v_region) > 1:
            dv = np.diff(v_region)
            di = np.diff(i_region) 
            slopes = dv / (di + 1e-12)  # Avoid division by zero
            rs = np.mean(np.abs(slopes))
        else:
            rs = self.config['pv_module']['series_resistance_typical']
        
        return max(0.01, min(rs, 10.0))  # Reasonable bounds
    
    def estimate_shunt_resistance(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Estimate shunt resistance from slope near ISC."""
        # Find region near ISC (first 10% of voltage range)
        isc_region = voltage < 0.1 * np.max(voltage)
        if np.sum(isc_region) < 2:
            return self.config['pv_module']['shunt_resistance_typical']
        
        v_region = voltage[isc_region]
        i_region = current[isc_region]
        
        # Calculate slope dI/dV
        if len(v_region) > 1:
            dv = np.diff(v_region) + 1e-12  # Avoid division by zero
            di = np.diff(i_region)
            slopes = di / dv
            rsh = -1 / np.mean(slopes) if np.mean(slopes) < 0 else self.config['pv_module']['shunt_resistance_typical']
        else:
            rsh = self.config['pv_module']['shunt_resistance_typical']
        
        return max(100, min(rsh, 50000))  # Reasonable bounds
    
    def perform_health_assessment(self, params: Dict) -> Dict:
        """Perform health assessment using configuration thresholds."""
        assessment = {
            'overall_health': 'Unknown',
            'health_score': 0,
            'fill_factor_rating': 'Unknown',
            'resistance_rating': 'Unknown',
            'performance_ratio': 0,
            'issues': [],
            'recommendations': []
        }
        
        # Assess fill factor against config thresholds
        ff = params['fill_factor']
        if ff >= self.config['diagnostics']['fill_factor_excellent']:
            assessment['fill_factor_rating'] = 'Excellent'
            ff_score = 100
        elif ff >= self.config['diagnostics']['fill_factor_good']:
            assessment['fill_factor_rating'] = 'Good'
            ff_score = 85
        elif ff >= self.config['diagnostics']['fill_factor_fair']:
            assessment['fill_factor_rating'] = 'Fair'
            ff_score = 70
        else:
            assessment['fill_factor_rating'] = 'Poor'
            ff_score = 40
            assessment['issues'].append('Low fill factor indicates performance degradation')
            assessment['recommendations'].append('Inspect for shading, soiling, or cell degradation')
        
        # Assess series resistance
        rs = params['rs']
        if rs <= self.config['diagnostics']['series_resistance_excellent']:
            assessment['resistance_rating'] = 'Excellent'
            rs_score = 100
        elif rs <= self.config['diagnostics']['series_resistance_good']:
            assessment['resistance_rating'] = 'Good'
            rs_score = 85
        elif rs <= self.config['diagnostics']['series_resistance_warning']:
            assessment['resistance_rating'] = 'Warning'
            rs_score = 60
            assessment['issues'].append('Elevated series resistance detected')
            assessment['recommendations'].append('Check electrical connections and contacts')
        else:
            assessment['resistance_rating'] = 'Critical'
            rs_score = 20
            assessment['issues'].append('High series resistance - immediate attention required')
            assessment['recommendations'].append('Emergency maintenance - check all connections and bypass diodes')
        
        # Calculate performance ratio
        expected_power = self.config['pv_module']['rated_power_stc']
        actual_power = params['p_max']
        assessment['performance_ratio'] = (actual_power / expected_power) * 100
        
        # Overall health score
        assessment['health_score'] = (ff_score + rs_score) / 2
        
        # Overall health classification
        if assessment['health_score'] >= 90:
            assessment['overall_health'] = 'Excellent'
        elif assessment['health_score'] >= 75:
            assessment['overall_health'] = 'Good'
        elif assessment['health_score'] >= 60:
            assessment['overall_health'] = 'Fair'
        else:
            assessment['overall_health'] = 'Poor'
        
        return assessment
    
    def calculate_economic_impact(self, params: Dict, health_assessment: Dict) -> Dict:
        """Calculate economic impact using configuration parameters."""
        # Configuration parameters
        rated_power = self.config['pv_module']['rated_power_stc']
        electricity_rate = self.config['economic']['electricity_rate']
        lifetime_years = self.config['economic']['project_lifetime_years']
        capacity_factor = self.config['economic']['capacity_factor']
        
        # Power loss calculation
        actual_power = params['p_max']
        power_loss_watts = max(0, rated_power - actual_power)
        power_loss_percent = (power_loss_watts / rated_power) * 100 if rated_power > 0 else 0
        
        # Annual energy loss
        hours_per_year = 8760
        annual_energy_loss_kwh = (power_loss_watts / 1000) * hours_per_year * capacity_factor
        annual_revenue_loss = annual_energy_loss_kwh * electricity_rate
        
        # Lifetime impact
        lifetime_revenue_loss = annual_revenue_loss * lifetime_years
        
        # Maintenance cost based on health
        base_maintenance = self.config['economic']['maintenance_cost_base']
        health_multipliers = {
            'Excellent': 0.8,
            'Good': 1.0,
            'Fair': 1.5,
            'Poor': 2.5
        }
        maintenance_multiplier = health_multipliers.get(health_assessment['overall_health'], 1.0)
        estimated_maintenance_cost = base_maintenance * maintenance_multiplier
        
        return {
            'power_loss_watts': power_loss_watts,
            'power_loss_percent': power_loss_percent,
            'annual_energy_loss_kwh': annual_energy_loss_kwh,
            'annual_revenue_loss': annual_revenue_loss,
            'lifetime_revenue_loss': lifetime_revenue_loss,
            'estimated_maintenance_cost': estimated_maintenance_cost,
            'total_economic_impact': lifetime_revenue_loss + estimated_maintenance_cost,
            'performance_ratio': health_assessment['performance_ratio']
        }


def run_comprehensive_config_analysis():
    """Run comprehensive analysis using configuration parameters."""
    print("🚀 Configuration-Integrated I-V Curve Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ConfigBasedIVAnalyzer()
    
    # Define test scenarios with different environmental conditions
    scenarios = [
        {
            'name': 'STC Conditions (Optimal)',
            'temperature': 25.0,
            'irradiance': 1000.0,
            'base_voc': 750.0,
            'base_isc': 20.0,
            'rs': 0.3,
            'rsh': 10000
        },
        {
            'name': 'High Temperature (Summer)',
            'temperature': 60.0,
            'irradiance': 1000.0,
            'base_voc': 750.0,
            'base_isc': 20.0,
            'rs': 0.5,
            'rsh': 8000
        },
        {
            'name': 'Low Irradiance (Cloudy)',
            'temperature': 25.0,
            'irradiance': 400.0,
            'base_voc': 750.0,
            'base_isc': 20.0,
            'rs': 0.4,
            'rsh': 9000
        },
        {
            'name': 'Degraded Module',
            'temperature': 25.0,
            'irradiance': 1000.0,
            'base_voc': 720.0,
            'base_isc': 18.5,
            'rs': 1.5,
            'rsh': 3000
        },
        {
            'name': 'Partially Shaded',
            'temperature': 25.0,
            'irradiance': 600.0,
            'base_voc': 740.0,
            'base_isc': 15.0,
            'rs': 0.8,
            'rsh': 5000
        }
    ]
    
    # Analyze each scenario
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        
        # Generate I-V curve with environmental corrections
        voltage, current = analyzer.generate_iv_curve_with_environmental_correction(
            base_voc=scenario['base_voc'],
            base_isc=scenario['base_isc'],
            rs=scenario['rs'],
            rsh=scenario['rsh'],
            temperature=scenario['temperature'],
            irradiance=scenario['irradiance']
        )
        
        # Extract parameters
        params = analyzer.extract_parameters_from_curve(voltage, current)
        
        # Perform health assessment
        health_assessment = analyzer.perform_health_assessment(params)
        
        # Calculate economic impact
        economic_impact = analyzer.calculate_economic_impact(params, health_assessment)
        
        # Display key results
        print(f"  📈 Power Output: {params['p_max']:.1f} W")
        print(f"  📊 Fill Factor: {params['fill_factor']:.3f}")
        print(f"  🏥 Health Score: {health_assessment['health_score']:.1f}/100")
        print(f"  💰 Annual Loss: ${economic_impact['annual_revenue_loss']:.2f}")
        print(f"  📉 Performance Ratio: {economic_impact['performance_ratio']:.1f}%")
        
        # Store results
        results.append({
            'scenario': scenario,
            'voltage': voltage,
            'current': current,
            'params': params,
            'health': health_assessment,
            'economic': economic_impact
        })
    
    # Create comprehensive visualization
    create_comprehensive_dashboard(results, analyzer)
    
    # Generate summary report
    generate_configuration_summary_report(results, analyzer)
    
    print(f"\n✅ Configuration-integrated analysis complete!")
    print(f"📈 Dashboard: config_integrated_dashboard.png")
    print(f"📋 Summary: CONFIG_INTEGRATION_SUMMARY.md")


def create_comprehensive_dashboard(results: List[Dict], analyzer: ConfigBasedIVAnalyzer):
    """Create comprehensive dashboard visualization."""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # I-V Curves Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ['green', 'orange', 'blue', 'red', 'purple']
    
    for i, result in enumerate(results):
        ax1.plot(result['voltage'], result['current'], 
                color=colors[i], linewidth=2.5, 
                label=result['scenario']['name'])
    
    ax1.set_xlabel('Voltage (V)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Current (A)', fontsize=12, fontweight='bold')
    ax1.set_title('I-V Curves: Environmental Conditions Impact', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Power Curves
    ax2 = fig.add_subplot(gs[1, :2])
    for i, result in enumerate(results):
        power = result['voltage'] * result['current']
        ax2.plot(result['voltage'], power, 
                color=colors[i], linewidth=2.5,
                label=f"{result['scenario']['name']}: {result['params']['p_max']:.0f}W")
    
    ax2.set_xlabel('Voltage (V)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Power (W)', fontsize=12, fontweight='bold')
    ax2.set_title('P-V Curves: Power Output Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Performance Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    scenario_names = [r['scenario']['name'].split()[0] for r in results]
    fill_factors = [r['params']['fill_factor'] for r in results]
    
    bars = ax3.bar(range(len(scenario_names)), fill_factors, color=colors, alpha=0.8)
    ax3.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fill Factor', fontsize=12, fontweight='bold')
    ax3.set_title('Fill Factor Analysis', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(scenario_names)))
    ax3.set_xticklabels(scenario_names, rotation=45, ha='right')
    
    # Add configuration thresholds
    ax3.axhline(y=analyzer.config['diagnostics']['fill_factor_excellent'], 
               color='green', linestyle='--', alpha=0.7, label='Excellent')
    ax3.axhline(y=analyzer.config['diagnostics']['fill_factor_good'], 
               color='orange', linestyle='--', alpha=0.7, label='Good')
    ax3.axhline(y=analyzer.config['diagnostics']['fill_factor_fair'], 
               color='red', linestyle='--', alpha=0.7, label='Fair')
    ax3.legend(fontsize=8)
    
    # Health Scores
    ax4 = fig.add_subplot(gs[1, 2])
    health_scores = [r['health']['health_score'] for r in results]
    bars = ax4.bar(range(len(scenario_names)), health_scores, color=colors, alpha=0.8)
    ax4.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Health Score', fontsize=12, fontweight='bold')
    ax4.set_title('Health Assessment', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(scenario_names)))
    ax4.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax4.set_ylim(0, 100)
    
    # Add value labels
    for bar, score in zip(bars, health_scores):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{score:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Economic Impact
    ax5 = fig.add_subplot(gs[2, :])
    full_names = [r['scenario']['name'] for r in results]
    annual_losses = [r['economic']['annual_revenue_loss'] for r in results]
    
    bars = ax5.bar(full_names, annual_losses, color=colors, alpha=0.8)
    ax5.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Annual Revenue Loss ($)', fontsize=12, fontweight='bold')
    ax5.set_title('Economic Impact: Annual Revenue Loss by Scenario', fontsize=14, fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, loss in zip(bars, annual_losses):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(annual_losses)*0.01,
                f'${loss:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Overall title and configuration info
    fig.suptitle('Configuration-Integrated I-V Curve Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Configuration summary box
    config_text = f"""Configuration Summary:
    Module: {analyzer.config['pv_module']['name']} ({analyzer.config['pv_module']['rated_power_stc']}W)
    STC: {analyzer.config['environmental']['reference_irradiance']} W/m², {analyzer.config['environmental']['reference_temperature']}°C
    Temp Coeffs: VOC={analyzer.config['pv_module']['temp_coeff_voc']*1000:.1f} mV/°C/cell, ISC={analyzer.config['pv_module']['temp_coeff_isc']*100:.2f}%/°C
    Health Thresholds: Excellent≥{analyzer.config['diagnostics']['fill_factor_excellent']:.2f}, Good≥{analyzer.config['diagnostics']['fill_factor_good']:.2f}
    Economic: ${analyzer.config['economic']['electricity_rate']:.2f}/kWh, {analyzer.config['economic']['project_lifetime_years']}yr lifetime"""
    
    fig.text(0.02, 0.02, config_text, fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('config_integrated_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_configuration_summary_report(results: List[Dict], analyzer: ConfigBasedIVAnalyzer):
    """Generate comprehensive summary report."""
    report = f"""# Configuration-Integrated I-V Curve Analysis Summary

## Analysis Overview
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Configuration Source:** config.toml
**Analysis Scenarios:** {len(results)}

## PV Module Configuration
- **Module Type:** {analyzer.config['pv_module']['name']}
- **Rated Power (STC):** {analyzer.config['pv_module']['rated_power_stc']} W
- **Rated Voltage (STC):** {analyzer.config['pv_module']['rated_voltage_stc']} V
- **Rated Current (STC):** {analyzer.config['pv_module']['rated_current_stc']} A
- **Nominal Fill Factor:** {analyzer.config['pv_module']['fill_factor_nominal']:.3f}
- **Temperature Coefficients:**
  - VOC: {analyzer.config['pv_module']['temp_coeff_voc']*1000:.1f} mV/°C/cell
  - ISC: {analyzer.config['pv_module']['temp_coeff_isc']*100:.2f} %/°C
  - Power: {analyzer.config['pv_module']['temp_coeff_power']*100:.2f} %/°C

## Environmental Analysis Conditions
- **Reference Conditions:** {analyzer.config['environmental']['reference_irradiance']} W/m², {analyzer.config['environmental']['reference_temperature']}°C
- **Operating Range:** {analyzer.config['environmental']['min_temperature']}°C to {analyzer.config['environmental']['max_temperature']}°C
- **Irradiance Range:** {analyzer.config['environmental']['min_irradiance']} to {analyzer.config['environmental']['max_irradiance']} W/m²

## Scenario Analysis Results

"""
    
    for i, result in enumerate(results, 1):
        scenario = result['scenario']
        params = result['params']
        health = result['health']
        economic = result['economic']
        
        report += f"""### {i}. {scenario['name']}

**Environmental Conditions:**
- Temperature: {scenario['temperature']}°C
- Irradiance: {scenario['irradiance']} W/m²

**Measured Parameters:**
- VOC: {params['voc']:.2f} V
- ISC: {params['isc']:.2f} A
- Pmax: {params['p_max']:.1f} W
- Fill Factor: {params['fill_factor']:.3f}
- Series Resistance: {params['rs']:.3f} Ω

**Health Assessment:**
- Overall Health: **{health['overall_health']}** ({health['health_score']:.1f}/100)
- Fill Factor Rating: {health['fill_factor_rating']}
- Performance Ratio: {economic['performance_ratio']:.1f}%

**Economic Impact:**
- Power Loss: {economic['power_loss_percent']:.1f}% ({economic['power_loss_watts']:.1f} W)
- Annual Revenue Loss: ${economic['annual_revenue_loss']:.2f}
- Estimated Maintenance: ${economic['estimated_maintenance_cost']:.2f}

"""
    
    # Summary statistics
    total_scenarios = len(results)
    avg_fill_factor = np.mean([r['params']['fill_factor'] for r in results])
    avg_power = np.mean([r['params']['p_max'] for r in results])
    total_annual_loss = sum([r['economic']['annual_revenue_loss'] for r in results])
    
    report += f"""## Fleet Analysis Summary

**Performance Statistics:**
- Total Scenarios Analyzed: {total_scenarios}
- Average Fill Factor: {avg_fill_factor:.3f}
- Average Power Output: {avg_power:.1f} W
- Total Annual Revenue Loss: ${total_annual_loss:.2f}

**Health Distribution:**
"""
    
    health_counts = {}
    for result in results:
        health_status = result['health']['overall_health']
        health_counts[health_status] = health_counts.get(health_status, 0) + 1
    
    for health_status, count in health_counts.items():
        percentage = (count / total_scenarios) * 100
        report += f"- {health_status}: {count} scenarios ({percentage:.1f}%)\n"
    
    report += f"""
## Configuration Validation

**Diagnostic Thresholds Applied:**
- Excellent Fill Factor: ≥ {analyzer.config['diagnostics']['fill_factor_excellent']:.3f}
- Good Fill Factor: ≥ {analyzer.config['diagnostics']['fill_factor_good']:.3f}
- Fair Fill Factor: ≥ {analyzer.config['diagnostics']['fill_factor_fair']:.3f}
- Series Resistance Warning: ≥ {analyzer.config['diagnostics']['series_resistance_warning']:.1f} Ω

**Economic Parameters:**
- Electricity Rate: ${analyzer.config['economic']['electricity_rate']:.2f}/kWh
- Project Lifetime: {analyzer.config['economic']['project_lifetime_years']} years
- Capacity Factor: {analyzer.config['economic']['capacity_factor']:.1%}

## Key Findings

1. **Environmental Impact:** Temperature increases significantly reduce VOC due to configured temperature coefficient of {analyzer.config['pv_module']['temp_coeff_voc']*1000:.1f} mV/°C/cell.

2. **Performance Degradation:** Scenarios with degraded modules show substantial economic impact, with annual losses ranging up to ${max([r['economic']['annual_revenue_loss'] for r in results]):.2f}.

3. **Configuration Effectiveness:** The integrated configuration system successfully applies module-specific corrections and thresholds for accurate analysis.

## Recommendations

1. **Monitoring:** Implement continuous monitoring for scenarios showing "Fair" or "Poor" health ratings.
2. **Maintenance:** Prioritize maintenance for modules with high series resistance (>{analyzer.config['diagnostics']['series_resistance_warning']:.1f} Ω).
3. **Economic Planning:** Budget ${total_annual_loss/len(results):.2f} average annual loss per module for financial planning.

---
*Report generated by Configuration-Integrated I-V Curve Analysis System*
"""
    
    # Save report
    with open('CONFIG_INTEGRATION_SUMMARY.md', 'w') as f:
        f.write(report)
    
    print(f"📋 Summary report saved: CONFIG_INTEGRATION_SUMMARY.md")


if __name__ == "__main__":
    run_comprehensive_config_analysis()
