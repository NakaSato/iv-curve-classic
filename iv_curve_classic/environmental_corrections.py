"""
Environmental Corrections Module

This module provides functionality to correct I-V curve measurements for 
temperature and irradiance variations, normalizing to Standard Test Conditions (STC).
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings


class EnvironmentalCorrections:
    """
    Environmental correction methods for I-V curve analysis.
    
    This class provides methods to correct I-V measurements for temperature
    and irradiance variations, allowing normalization to Standard Test Conditions.
    """
    
    def __init__(self):
        """Initialize with standard temperature coefficients."""
        # Standard temperature coefficients for crystalline silicon
        self.temp_coefficients = {
            'voc': -0.0032,  # V/°C per cell (typical: -0.32%/°C)
            'isc': 0.0005,   # A/°C per module (typical: +0.05%/°C)  
            'pmax': -0.004,  # W/°C per module (typical: -0.4%/°C)
            'ff': -0.002     # /°C (typical: -0.2%/°C)
        }
        
        # Standard Test Conditions
        self.stc = {
            'temperature': 25.0,  # °C
            'irradiance': 1000.0,  # W/m²
            'air_mass': 1.5
        }
    
    def correct_to_stc(self, voltage: np.ndarray, current: np.ndarray,
                      cell_temperature: float, irradiance: float,
                      temp_coeffs: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct I-V curve to Standard Test Conditions.
        
        Args:
            voltage: Measured voltage array (V)
            current: Measured current array (A)
            cell_temperature: Cell temperature during measurement (°C)
            irradiance: Irradiance during measurement (W/m²)
            temp_coeffs: Custom temperature coefficients (optional)
            
        Returns:
            Tuple of corrected (voltage, current) arrays at STC
        """
        if temp_coeffs is None:
            temp_coeffs = self.temp_coefficients
        
        # Temperature corrections
        temp_diff = self.stc['temperature'] - cell_temperature
        
        # Irradiance correction
        irr_ratio = self.stc['irradiance'] / irradiance if irradiance > 0 else 1.0
        
        # Correct current for irradiance (linear relationship)
        current_corrected = current * irr_ratio
        
        # Correct voltage for temperature
        # For each voltage point, apply temperature correction
        voltage_corrected = voltage + (temp_diff * temp_coeffs['voc'] * len(voltage) / 36)  # Approximate cells
        
        # Additional correction for series resistance temperature dependence
        if hasattr(self, 'series_resistance'):
            rs_temp_coeff = 0.005  # Typical Rs temperature coefficient
            rs_correction = 1 + rs_temp_coeff * temp_diff
            # Apply small voltage correction for Rs change
            voltage_corrected = voltage_corrected * rs_correction
        
        return voltage_corrected, current_corrected
    
    def correct_parameters_to_stc(self, parameters: Dict[str, float],
                                 cell_temperature: float, irradiance: float,
                                 temp_coeffs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Correct extracted parameters to Standard Test Conditions.
        
        Args:
            parameters: Extracted I-V parameters
            cell_temperature: Cell temperature during measurement (°C)
            irradiance: Irradiance during measurement (W/m²)
            temp_coeffs: Custom temperature coefficients (optional)
            
        Returns:
            Dictionary of corrected parameters at STC
        """
        if temp_coeffs is None:
            temp_coeffs = self.temp_coefficients
        
        corrected_params = parameters.copy()
        
        # Temperature difference from STC
        temp_diff = self.stc['temperature'] - cell_temperature
        
        # Irradiance ratio
        irr_ratio = self.stc['irradiance'] / irradiance if irradiance > 0 else 1.0
        
        # Correct each parameter
        if 'voc' in parameters:
            # Voc correction: linear with temperature
            voc_corrected = parameters['voc'] + (temp_diff * temp_coeffs['voc'] * 36)  # Assuming 36 cells
            corrected_params['voc'] = voc_corrected
        
        if 'isc' in parameters:
            # Isc correction: irradiance (linear) and slight temperature dependence
            isc_temp_corrected = parameters['isc'] * (1 + temp_coeffs['isc'] * temp_diff)
            isc_corrected = isc_temp_corrected * irr_ratio
            corrected_params['isc'] = isc_corrected
        
        if 'p_max' in parameters:
            # Power correction: combined temperature and irradiance effects
            p_temp_corrected = parameters['p_max'] * (1 + temp_coeffs['pmax'] * temp_diff)
            p_corrected = p_temp_corrected * irr_ratio
            corrected_params['p_max'] = p_corrected
        
        if 'fill_factor' in parameters:
            # Fill factor: primarily temperature dependent
            ff_corrected = parameters['fill_factor'] * (1 + temp_coeffs['ff'] * temp_diff)
            corrected_params['fill_factor'] = ff_corrected
        
        # Recalculate MPP parameters if possible
        if all(k in corrected_params for k in ['voc', 'isc', 'p_max']):
            # Estimate corrected MPP voltage and current
            corrected_params['v_mp'] = corrected_params['voc'] * 0.82  # Typical ratio
            corrected_params['i_mp'] = corrected_params['p_max'] / corrected_params['v_mp']
        
        # Add correction metadata
        corrected_params['correction_applied'] = True
        corrected_params['reference_temperature'] = cell_temperature
        corrected_params['reference_irradiance'] = irradiance
        corrected_params['stc_temperature'] = self.stc['temperature']
        corrected_params['stc_irradiance'] = self.stc['irradiance']
        
        return corrected_params
    
    def estimate_cell_temperature(self, ambient_temp: float, irradiance: float,
                                 wind_speed: float = 1.0, 
                                 mounting_type: str = 'open_rack') -> float:
        """
        Estimate cell temperature from environmental conditions.
        
        Args:
            ambient_temp: Ambient air temperature (°C)
            irradiance: Solar irradiance (W/m²)
            wind_speed: Wind speed (m/s)
            mounting_type: Mounting configuration ('open_rack', 'close_roof', 'ground')
            
        Returns:
            Estimated cell temperature (°C)
        """
        # Nominal Operating Cell Temperature (NOCT) based on mounting
        noct_values = {
            'open_rack': 45.0,
            'close_roof': 49.0,
            'ground': 47.0,
            'building_integrated': 51.0
        }
        
        noct = noct_values.get(mounting_type, 45.0)
        
        # Basic NOCT model
        cell_temp = ambient_temp + (noct - 20) * (irradiance / 800)
        
        # Wind speed correction (simplified)
        wind_correction = -2.0 * np.log10(wind_speed + 0.1)  # Cooling effect
        cell_temp += wind_correction
        
        return max(ambient_temp, cell_temp)  # Cell temp cannot be below ambient
    
    def calculate_performance_ratio(self, measured_params: Dict[str, float],
                                   nameplate_power: float,
                                   irradiance: float) -> float:
        """
        Calculate Performance Ratio (PR) - a key PV performance metric.
        
        Args:
            measured_params: Measured I-V parameters
            nameplate_power: Module nameplate power rating (W)
            irradiance: Measured irradiance (W/m²)
            
        Returns:
            Performance Ratio (dimensionless)
        """
        if 'p_max' not in measured_params or nameplate_power <= 0 or irradiance <= 0:
            return 0.0
        
        # Expected power at current irradiance
        expected_power = nameplate_power * (irradiance / 1000.0)
        
        # Performance Ratio
        pr = measured_params['p_max'] / expected_power if expected_power > 0 else 0.0
        
        return pr
    
    def irradiance_correction_factor(self, measured_irradiance: float) -> float:
        """
        Calculate irradiance correction factor for power measurements.
        
        Args:
            measured_irradiance: Measured irradiance (W/m²)
            
        Returns:
            Correction factor to normalize to STC irradiance
        """
        return self.stc['irradiance'] / measured_irradiance if measured_irradiance > 0 else 1.0
    
    def temperature_correction_factor(self, cell_temperature: float, 
                                    parameter: str = 'power') -> float:
        """
        Calculate temperature correction factor for a specific parameter.
        
        Args:
            cell_temperature: Cell temperature (°C)
            parameter: Parameter to correct ('voc', 'isc', 'power', 'fill_factor')
            
        Returns:
            Temperature correction factor
        """
        temp_diff = self.stc['temperature'] - cell_temperature
        
        if parameter in self.temp_coefficients:
            return 1 + self.temp_coefficients[parameter] * temp_diff
        else:
            warnings.warn(f"Unknown parameter: {parameter}")
            return 1.0
    
    def apply_spectral_correction(self, current: np.ndarray, 
                                 air_mass: float = 1.5) -> np.ndarray:
        """
        Apply spectral correction for different air mass conditions.
        
        Args:
            current: Current array (A)
            air_mass: Air mass value
            
        Returns:
            Spectrally corrected current array
        """
        # Simplified spectral correction (more complex models exist)
        # This is a basic approximation
        stc_air_mass = 1.5
        
        if air_mass <= 0:
            return current
        
        # Spectral correction factor (simplified)
        spectral_factor = 1.0 + 0.02 * (air_mass - stc_air_mass)
        spectral_factor = max(0.9, min(spectral_factor, 1.1))  # Reasonable bounds
        
        return current * spectral_factor
    
    def comprehensive_correction(self, voltage: np.ndarray, current: np.ndarray,
                               environmental_data: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Apply comprehensive environmental corrections.
        
        Args:
            voltage: Voltage array (V)
            current: Current array (A)
            environmental_data: Dictionary with environmental conditions
                               Expected keys: 'cell_temp', 'irradiance', 'air_mass' (optional)
            
        Returns:
            Tuple of (corrected_voltage, corrected_current, correction_summary)
        """
        correction_summary = {
            'temperature_correction': False,
            'irradiance_correction': False,
            'spectral_correction': False,
            'original_conditions': environmental_data.copy(),
            'target_conditions': self.stc.copy()
        }
        
        corrected_voltage = voltage.copy()
        corrected_current = current.copy()
        
        # Apply temperature and irradiance corrections
        if 'cell_temp' in environmental_data and 'irradiance' in environmental_data:
            corrected_voltage, corrected_current = self.correct_to_stc(
                corrected_voltage, corrected_current,
                environmental_data['cell_temp'],
                environmental_data['irradiance']
            )
            correction_summary['temperature_correction'] = True
            correction_summary['irradiance_correction'] = True
        
        # Apply spectral correction if air mass is provided
        if 'air_mass' in environmental_data:
            corrected_current = self.apply_spectral_correction(
                corrected_current, environmental_data['air_mass']
            )
            correction_summary['spectral_correction'] = True
        
        return corrected_voltage, corrected_current, correction_summary
