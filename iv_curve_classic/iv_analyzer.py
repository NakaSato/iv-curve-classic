"""
I-V Curve Analyzer Module

This module provides the main I-V curve analysis functionality including
curve generation, parameter extraction, and plotting capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Dict, Tuple, Optional
import warnings


class IVAnalyzer:
    """
    Main class for I-V curve analysis and visualization.
    
    This class provides methods for generating I-V curves from parameters,
    extracting parameters from measured data, and creating visualizations.
    """
    
    def __init__(self):
        """Initialize the IV Analyzer."""
        self.thermal_voltage = 0.0259  # kT/q at 25°C in volts
    
    def generate_iv_curve(self, voc: float, isc: float, rs: float = 0.5, 
                         rsh: float = 1000.0, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate I-V curve from key parameters using the single diode model.
        
        Args:
            voc: Open-circuit voltage (V)
            isc: Short-circuit current (A)
            rs: Series resistance (Ω)
            rsh: Shunt resistance (Ω)
            n_points: Number of points in the curve
            
        Returns:
            Tuple of (voltage, current) arrays
        """
        # Constrain parameters to reasonable ranges to avoid numerical issues
        voc = min(max(voc, 10.0), 100.0)  # Typical module Voc: 10-100V  
        isc = min(max(isc, 0.5), 15.0)    # Typical module Isc: 0.5-15A
        rs = min(max(rs, 0.1), 2.0)       # Reasonable Rs range
        rsh = min(max(rsh, 100.0), 5000.0) # Reasonable Rsh range
        
        # Generate voltage points from 0 to Voc
        voltage = np.linspace(0, voc * 0.98, n_points)  # Don't go exactly to Voc
        current = np.zeros_like(voltage)
        
        # Single diode model parameters with overflow protection
        n = 1.2  # Ideality factor
        thermal_voltage_scaled = self.thermal_voltage * n
        
        # Calculate saturation current with overflow protection
        exp_arg = voc / thermal_voltage_scaled
        if exp_arg > 50:  # Prevent overflow
            exp_arg = 50
            
        i_sat = isc / (np.exp(exp_arg) - 1)
        
        for i, v in enumerate(voltage):
            # Use simplified approach for better numerical stability
            if v < voc * 0.05:  # Near short circuit
                current[i] = isc - v / rsh
            elif v > voc * 0.9:  # Near open circuit
                current[i] = max(0, (voc - v) / rs * 0.1)
            else:
                # Newton-Raphson with better initial guess and overflow protection
                current_guess = isc * (1 - v / voc) ** 2  # Better initial guess
                
                for iteration in range(5):  # Fewer iterations for stability
                    try:
                        # Calculate with overflow protection
                        exp_term = (v + current_guess * rs) / thermal_voltage_scaled
                        if exp_term > 50:
                            exp_term = 50
                            
                        f = (current_guess - isc + 
                             i_sat * (np.exp(exp_term) - 1) +
                             (v + current_guess * rs) / rsh)
                        
                        if abs(f) < 1e-8:  # Good enough convergence
                            break
                            
                        df = (1 + 
                              i_sat * rs * np.exp(exp_term) / thermal_voltage_scaled +
                              rs / rsh)
                        
                        if abs(df) < 1e-12:
                            break
                            
                        current_new = current_guess - f / df
                        
                        if abs(current_new - current_guess) < 1e-8:
                            break
                            
                        current_guess = max(0, min(current_new, isc * 1.1))  # Bound the current
                        
                    except (OverflowError, RuntimeWarning):
                        # Fallback to simple linear model for this point
                        current_guess = max(0, isc * (1 - v / voc))
                        break
                
                current[i] = max(0, current_guess)
        
        return voltage, current
    
    def extract_basic_parameters(self, voltage: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        """
        Extract basic I-V parameters from measured data.
        
        Args:
            voltage: Voltage array (V)
            current: Current array (A)
            
        Returns:
            Dictionary of extracted parameters
        """
        # Find Voc (where current crosses zero)
        voc_idx = np.where(current <= 0)[0]
        if len(voc_idx) > 0:
            voc = float(voltage[voc_idx[0]])
        else:
            voc = float(voltage[-1])
        
        # Find Isc (current at V=0)
        isc_idx = np.argmin(np.abs(voltage))
        isc = float(current[isc_idx])
        
        # Calculate power
        power = voltage * current
        max_power_idx = np.argmax(power)
        
        p_max = float(power[max_power_idx])
        v_mp = float(voltage[max_power_idx])
        i_mp = float(current[max_power_idx])
        
        # Calculate fill factor
        fill_factor = p_max / (voc * isc) if (voc * isc) > 0 else 0.0
        
        return {
            'voc': voc,
            'isc': isc,
            'p_max': p_max,
            'v_mp': v_mp,
            'i_mp': i_mp,
            'fill_factor': fill_factor
        }
    
    def plot_iv_curve(self, voltage: np.ndarray, current: np.ndarray, 
                     title: str = "I-V Characteristic", show_parameters: bool = True,
                     params: Optional[Dict] = None) -> matplotlib.figure.Figure:
        """
        Plot I-V curve with optional parameter annotations.
        
        Args:
            voltage: Voltage array (V)
            current: Current array (A)
            title: Plot title
            show_parameters: Whether to show parameter annotations
            params: Parameters dictionary for annotations
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # I-V curve
        ax1.plot(voltage, current, 'b-', linewidth=2)
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current (A)')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        # P-V curve
        power = voltage * current
        ax2.plot(voltage, power, 'r-', linewidth=2)
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Power (W)')
        ax2.set_title('P-V Curve')
        ax2.grid(True, alpha=0.3)
        
        if show_parameters and params:
            # Add parameter annotations
            ax1.axhline(y=params.get('isc', 0), color='g', linestyle='--', alpha=0.7)
            ax1.axvline(x=params.get('voc', 0), color='r', linestyle='--', alpha=0.7)
            
            if 'v_mp' in params and 'i_mp' in params:
                ax1.scatter([params['v_mp']], [params['i_mp']], 
                           color='red', s=100, zorder=5)
                ax2.scatter([params['v_mp']], [params.get('p_max', 0)], 
                           color='red', s=100, zorder=5)
        
        plt.tight_layout()
        return fig
    
    def calculate_efficiency(self, p_max: float, area: float, irradiance: float = 1000.0) -> float:
        """
        Calculate module efficiency.
        
        Args:
            p_max: Maximum power (W)
            area: Module area (m²)
            irradiance: Irradiance level (W/m²)
            
        Returns:
            Efficiency as percentage
        """
        if area <= 0 or irradiance <= 0:
            return 0.0
        
        return (p_max / (area * irradiance)) * 100
    
    def compare_curves(self, curves_data: Dict[str, Dict], title: str = "I-V Curve Comparison"):
        """
        Compare multiple I-V curves on the same plot.
        
        Args:
            curves_data: Dictionary with curve names as keys and 
                        {'voltage': array, 'current': array} as values
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(curves_data)))
        
        for i, (name, data) in enumerate(curves_data.items()):
            voltage = data['voltage']
            current = data['current']
            power = voltage * current
            
            ax1.plot(voltage, current, color=colors[i], linewidth=2, label=name)
            ax2.plot(voltage, power, color=colors[i], linewidth=2, label=name)
        
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current (A)')
        ax1.set_title(f'{title} - I-V Curves')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Power (W)')
        ax2.set_title(f'{title} - P-V Curves')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
