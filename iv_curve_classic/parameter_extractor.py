"""
Parameter Extraction Module

This module provides advanced parameter extraction capabilities for I-V curves,
including series and shunt resistance calculations using various methods.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize_scalar, curve_fit
import warnings


class ParameterExtractor:
    """
    Advanced parameter extraction for I-V curves.
    
    This class provides methods for extracting all key parameters from I-V curves
    including parasitic resistances using various analytical and numerical methods.
    """
    
    def __init__(self):
        """Initialize the parameter extractor."""
        self.thermal_voltage = 0.0259  # kT/q at 25°C in volts
        
    def extract_all_parameters(self, voltage: np.ndarray, current: np.ndarray, 
                              temperature: float = 25.0) -> Dict[str, float]:
        """
        Extract comprehensive set of I-V parameters.
        
        Args:
            voltage: Voltage array (V)
            current: Current array (A)
            temperature: Cell temperature (°C)
            
        Returns:
            Dictionary containing all extracted parameters
        """
        # Update thermal voltage for temperature
        vt = 8.617e-5 * (temperature + 273.15) / 1.602e-19  # Thermal voltage
        self.thermal_voltage = vt
        
        # Basic parameters
        params = self._extract_basic_parameters(voltage, current)
        
        # Parasitic resistances
        rs = self._calculate_series_resistance(voltage, current, params)
        rsh = self._calculate_shunt_resistance(voltage, current, params)
        
        params.update({
            'rs': rs,
            'rsh': rsh,
            'temperature': temperature
        })
        
        # Recalculate fill factor with resistive losses
        ideal_ff = self._calculate_ideal_fill_factor(params['voc'] / self.thermal_voltage)
        params['ideal_fill_factor'] = ideal_ff
        params['ff_loss'] = ideal_ff - params['fill_factor']
        
        return params
    
    def _extract_basic_parameters(self, voltage: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        """Extract basic I-V parameters."""
        # Sort data by voltage
        sort_idx = np.argsort(voltage)
        v_sorted = voltage[sort_idx]
        i_sorted = current[sort_idx]
        
        # Find Voc (interpolate to I=0)
        if np.any(i_sorted <= 0):
            zero_idx = np.where(i_sorted <= 0)[0]
            if len(zero_idx) > 0:
                voc = float(v_sorted[zero_idx[0]])
            else:
                voc = float(v_sorted[-1])
        else:
            # Extrapolate to I=0
            voc = float(np.interp(0, i_sorted[::-1], v_sorted[::-1]))
        
        # Find Isc (interpolate to V=0)
        isc = float(np.interp(0, v_sorted, i_sorted))
        
        # Maximum power point
        power = v_sorted * i_sorted
        max_idx = np.argmax(power)
        
        p_max = float(power[max_idx])
        v_mp = float(v_sorted[max_idx])
        i_mp = float(i_sorted[max_idx])
        
        # Fill factor
        fill_factor = p_max / (voc * isc) if (voc * isc) > 0 else 0.0
        
        return {
            'voc': voc,
            'isc': isc,
            'p_max': p_max,
            'v_mp': v_mp,
            'i_mp': i_mp,
            'fill_factor': fill_factor
        }
    
    def _calculate_series_resistance(self, voltage: np.ndarray, current: np.ndarray, 
                                   params: Dict[str, float]) -> float:
        """
        Calculate series resistance using the slope method near Voc.
        
        Args:
            voltage: Voltage array
            current: Current array  
            params: Basic parameters dictionary
            
        Returns:
            Series resistance in ohms
        """
        try:
            # Use points near Voc where the effect of Rs is most pronounced
            voc = params['voc']
            
            # Find points in the range [0.8*Voc, Voc]
            mask = (voltage >= 0.8 * voc) & (voltage <= voc) & (current > 0)
            
            if np.sum(mask) < 3:
                # Fallback: use points near maximum power
                v_mp = params['v_mp']
                mask = (voltage >= 0.9 * v_mp) & (voltage <= voc) & (current > 0)
            
            if np.sum(mask) < 2:
                return 0.5  # Default value
            
            v_subset = voltage[mask]
            i_subset = current[mask]
            
            # Calculate derivative dV/dI
            if len(v_subset) >= 2:
                # Use finite differences
                dv_di = np.gradient(v_subset, i_subset)
                rs = -np.mean(dv_di)  # Negative because current decreases with voltage
                
                # Ensure reasonable bounds
                rs = max(0.01, min(rs, 10.0))
                return rs
            
        except Exception:
            pass
        
        return 0.5  # Default series resistance
    
    def _calculate_shunt_resistance(self, voltage: np.ndarray, current: np.ndarray,
                                  params: Dict[str, float]) -> float:
        """
        Calculate shunt resistance using the slope method near Isc.
        
        Args:
            voltage: Voltage array
            current: Current array
            params: Basic parameters dictionary
            
        Returns:
            Shunt resistance in ohms
        """
        try:
            # Use points near Isc where the effect of Rsh is most pronounced
            isc = params['isc']
            
            # Find points in the range [0, 0.2*Voc]
            voc = params['voc']
            mask = (voltage >= 0) & (voltage <= 0.2 * voc) & (current > 0.1 * isc)
            
            if np.sum(mask) < 2:
                return 1000.0  # Default value
            
            v_subset = voltage[mask]
            i_subset = current[mask]
            
            # Calculate derivative dI/dV
            if len(v_subset) >= 2:
                di_dv = np.gradient(i_subset, v_subset)
                conductance = np.mean(di_dv)
                
                if conductance > 1e-6:
                    rsh = 1.0 / conductance
                    # Ensure reasonable bounds
                    rsh = max(10.0, min(rsh, 10000.0))
                    return rsh
            
        except Exception:
            pass
        
        return 1000.0  # Default shunt resistance
    
    def _calculate_ideal_fill_factor(self, voc_over_vt: float) -> float:
        """
        Calculate ideal fill factor based on Voc/Vt ratio.
        
        Args:
            voc_over_vt: Ratio of Voc to thermal voltage
            
        Returns:
            Ideal fill factor
        """
        if voc_over_vt <= 0:
            return 0.0
        
        # Green's approximation for ideal fill factor
        ff_ideal = (voc_over_vt - np.log(voc_over_vt + 0.72)) / (voc_over_vt + 1)
        
        return max(0.5, min(ff_ideal, 0.89))  # Reasonable bounds
    
    def extract_diode_parameters(self, voltage: np.ndarray, current: np.ndarray,
                               temperature: float = 25.0) -> Dict[str, float]:
        """
        Extract single diode model parameters.
        
        Args:
            voltage: Voltage array (V)
            current: Current array (A)
            temperature: Temperature (°C)
            
        Returns:
            Dictionary with diode model parameters
        """
        # Basic parameters
        basic_params = self._extract_basic_parameters(voltage, current)
        
        # Estimate saturation current and ideality factor
        voc = basic_params['voc']
        isc = basic_params['isc']
        
        # Thermal voltage at operating temperature
        vt = 8.617e-5 * (temperature + 273.15) / 1.602e-19
        
        # Initial estimate assuming n=1
        n = 1.0
        i_sat = isc / (np.exp(voc / (n * vt)) - 1)
        
        # Refine using iterative approach
        for iteration in range(5):
            # Use the slope at Voc to estimate ideality factor
            try:
                # Find slope near Voc
                mask = (voltage >= 0.9 * voc) & (voltage <= voc)
                if np.sum(mask) >= 2:
                    v_near_voc = voltage[mask]
                    i_near_voc = current[mask]
                    
                    # log(I) vs V slope gives 1/(n*Vt)
                    log_i = np.log(np.maximum(i_near_voc, 1e-10))
                    slope = np.polyfit(v_near_voc, log_i, 1)[0]
                    
                    if slope < 0:
                        n = -1.0 / (slope * vt)
                        n = max(1.0, min(n, 2.5))  # Reasonable bounds
                        
                        # Recalculate saturation current
                        i_sat = isc / (np.exp(voc / (n * vt)) - 1)
                        break
            except:
                pass
        
        return {
            **basic_params,
            'ideality_factor': n,
            'saturation_current': i_sat,
            'thermal_voltage': vt,
            'temperature': temperature
        }
    
    def fit_single_diode_model(self, voltage: np.ndarray, current: np.ndarray,
                              temperature: float = 25.0) -> Dict[str, float]:
        """
        Fit complete single diode model to I-V data.
        
        Args:
            voltage: Voltage array (V)
            current: Current array (A)
            temperature: Temperature (°C)
            
        Returns:
            Dictionary with fitted model parameters
        """
        # Get initial parameter estimates
        initial_params = self.extract_all_parameters(voltage, current, temperature)
        
        vt = 8.617e-5 * (temperature + 273.15) / 1.602e-19
        
        def single_diode_model(v, iph, i0, rs, rsh, n):
            """Single diode model equation."""
            # Solve for current iteratively
            i_out = np.zeros_like(v)
            
            for i, v_val in enumerate(v):
                # Initial guess
                i_guess = iph * (1 - v_val / initial_params['voc'])
                
                # Newton-Raphson
                for _ in range(10):
                    f = (i_guess - iph + 
                         i0 * (np.exp((v_val + i_guess * rs) / (n * vt)) - 1) +
                         (v_val + i_guess * rs) / rsh)
                    
                    df = (1 + 
                          i0 * rs * np.exp((v_val + i_guess * rs) / (n * vt)) / (n * vt) +
                          rs / rsh)
                    
                    if abs(df) < 1e-12:
                        break
                        
                    i_new = i_guess - f / df
                    
                    if abs(i_new - i_guess) < 1e-6:
                        break
                        
                    i_guess = max(0, i_new)
                
                i_out[i] = max(0, i_guess)
            
            return i_out
        
        # Initial parameter guesses
        iph_init = initial_params['isc']
        i0_init = 1e-9
        rs_init = initial_params['rs']
        rsh_init = initial_params['rsh']
        n_init = 1.2
        
        try:
            # Fit the model
            popt, pcov = curve_fit(
                single_diode_model,
                voltage, current,
                p0=[iph_init, i0_init, rs_init, rsh_init, n_init],
                bounds=([0, 1e-12, 0.01, 10, 0.8], 
                       [iph_init*1.2, 1e-6, 10, 10000, 2.5]),
                maxfev=1000
            )
            
            fitted_params = {
                'photocurrent': popt[0],
                'saturation_current': popt[1],
                'rs': popt[2],
                'rsh': popt[3],
                'ideality_factor': popt[4],
                'thermal_voltage': vt,
                'temperature': temperature
            }
            
            # Add basic parameters
            fitted_params.update(initial_params)
            
            return fitted_params
            
        except Exception:
            # Return initial estimates if fitting fails
            return initial_params
