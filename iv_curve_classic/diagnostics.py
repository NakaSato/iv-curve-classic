"""
Diagnostic Analysis Module

This module provides comprehensive diagnostic capabilities for photovoltaic systems
including degradation analysis, fault detection, and performance assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class DiagnosticAnalyzer:
    """
    Comprehensive diagnostic analyzer for PV systems.
    
    This class provides methods for analyzing I-V curve parameters to identify
    degradation mechanisms, faults, and performance issues in photovoltaic systems.
    """
    
    def __init__(self):
        """Initialize the diagnostic analyzer with reference values."""
        # Reference values for healthy PV modules (typical ranges)
        self.reference_values = {
            'fill_factor': {'min': 0.75, 'max': 0.85, 'ideal': 0.80},
            'series_resistance': {'max': 1.0, 'warning': 0.7},  # Ohms
            'shunt_resistance': {'min': 500.0, 'warning': 1000.0},  # Ohms
            'voc_degradation': {'max': 0.02},  # 2% per year typical
            'isc_degradation': {'max': 0.015},  # 1.5% per year typical
            'power_degradation': {'max': 0.008}  # 0.8% per year typical
        }
    
    def analyze_degradation(self, parameters: Dict[str, float], 
                           reference_params: Optional[Dict[str, float]] = None) -> Dict:
        """
        Analyze degradation mechanisms from I-V parameters.
        
        Args:
            parameters: Extracted I-V parameters
            reference_params: Reference parameters for comparison (optional)
            
        Returns:
            Dictionary containing diagnostic results
        """
        diagnosis = {
            'overall_health': 'Good',
            'severity': 'Low',
            'primary_issues': [],
            'secondary_issues': [],
            'recommendations': [],
            'parameter_analysis': {},
            'degradation_mechanisms': []
        }
        
        # Analyze individual parameters
        fill_factor_analysis = self._analyze_fill_factor(parameters)
        resistance_analysis = self._analyze_resistances(parameters)
        power_analysis = self._analyze_power_performance(parameters)
        
        # Combine analyses
        diagnosis['parameter_analysis'] = {
            'fill_factor': fill_factor_analysis,
            'resistances': resistance_analysis,
            'power': power_analysis
        }
        
        # Determine primary issues
        all_issues = []
        if fill_factor_analysis['issues']:
            all_issues.extend(fill_factor_analysis['issues'])
        if resistance_analysis['issues']:
            all_issues.extend(resistance_analysis['issues'])
        if power_analysis['issues']:
            all_issues.extend(power_analysis['issues'])
        
        # Classify issues by severity
        severe_issues = [issue for issue in all_issues if 'severe' in issue.lower() or 'critical' in issue.lower()]
        moderate_issues = [issue for issue in all_issues if 'moderate' in issue.lower() or 'elevated' in issue.lower()]
        minor_issues = [issue for issue in all_issues if issue not in severe_issues + moderate_issues]
        
        diagnosis['primary_issues'] = severe_issues + moderate_issues[:2]
        diagnosis['secondary_issues'] = moderate_issues[2:] + minor_issues
        
        # Determine overall health
        if severe_issues:
            diagnosis['overall_health'] = 'Poor'
            diagnosis['severity'] = 'High'
        elif moderate_issues:
            diagnosis['overall_health'] = 'Fair'
            diagnosis['severity'] = 'Medium'
        else:
            diagnosis['overall_health'] = 'Good'
            diagnosis['severity'] = 'Low'
        
        # Generate recommendations
        diagnosis['recommendations'] = self._generate_recommendations(diagnosis)
        
        # Identify degradation mechanisms
        diagnosis['degradation_mechanisms'] = self._identify_degradation_mechanisms(parameters)
        
        return diagnosis
    
    def _analyze_fill_factor(self, params: Dict[str, float]) -> Dict:
        """Analyze fill factor for degradation indicators."""
        ff = params.get('fill_factor', 0)
        ref_ff = self.reference_values['fill_factor']
        
        analysis = {
            'value': ff,
            'status': 'Good',
            'issues': [],
            'severity': 'Low'
        }
        
        if ff < 0.65:
            analysis['status'] = 'Critical'
            analysis['severity'] = 'High'
            analysis['issues'].append('Severe fill factor degradation detected')
        elif ff < 0.70:
            analysis['status'] = 'Poor'
            analysis['severity'] = 'High'
            analysis['issues'].append('Significant fill factor reduction')
        elif ff < ref_ff['min']:
            analysis['status'] = 'Fair'
            analysis['severity'] = 'Medium'
            analysis['issues'].append('Moderate fill factor degradation')
        elif ff < ref_ff['warning'] if 'warning' in ref_ff else ref_ff['min'] + 0.05:
            analysis['issues'].append('Minor fill factor reduction observed')
        
        return analysis
    
    def _analyze_resistances(self, params: Dict[str, float]) -> Dict:
        """Analyze series and shunt resistances."""
        rs = params.get('rs', 0)
        rsh = params.get('rsh', float('inf'))
        
        analysis = {
            'series_resistance': {'value': rs, 'status': 'Good', 'issues': []},
            'shunt_resistance': {'value': rsh, 'status': 'Good', 'issues': []},
            'issues': []
        }
        
        # Series resistance analysis
        if rs > 2.0:
            analysis['series_resistance']['status'] = 'Critical'
            analysis['issues'].append('Severe series resistance increase (possible interconnect failure)')
        elif rs > self.reference_values['series_resistance']['max']:
            analysis['series_resistance']['status'] = 'Poor'
            analysis['issues'].append('Elevated series resistance detected')
        elif rs > self.reference_values['series_resistance']['warning']:
            analysis['series_resistance']['status'] = 'Fair'
            analysis['issues'].append('Moderate series resistance increase')
        
        # Shunt resistance analysis
        if rsh < 100:
            analysis['shunt_resistance']['status'] = 'Critical'
            analysis['issues'].append('Severe shunt resistance degradation (possible cell cracking)')
        elif rsh < self.reference_values['shunt_resistance']['min']:
            analysis['shunt_resistance']['status'] = 'Poor'
            analysis['issues'].append('Low shunt resistance detected')
        elif rsh < self.reference_values['shunt_resistance']['warning']:
            analysis['shunt_resistance']['status'] = 'Fair'
            analysis['issues'].append('Moderate shunt resistance reduction')
        
        return analysis
    
    def _analyze_power_performance(self, params: Dict[str, float]) -> Dict:
        """Analyze power performance indicators."""
        p_max = params.get('p_max', 0)
        voc = params.get('voc', 0)
        isc = params.get('isc', 0)
        
        analysis = {
            'max_power': p_max,
            'voc': voc,
            'isc': isc,
            'status': 'Good',
            'issues': []
        }
        
        # Basic power check (assuming standard module ratings)
        if p_max < 200:  # Very low for typical modules
            analysis['status'] = 'Poor'
            analysis['issues'].append('Very low power output detected')
        elif p_max < 250:
            analysis['status'] = 'Fair'
            analysis['issues'].append('Below-average power performance')
        
        # Voltage analysis
        if voc < 30:  # Low for typical crystalline silicon
            analysis['issues'].append('Low open-circuit voltage')
        elif voc > 70:  # High, possible measurement error
            analysis['issues'].append('Unusually high open-circuit voltage')
        
        # Current analysis
        if isc < 5:  # Low for typical modules
            analysis['issues'].append('Low short-circuit current')
        elif isc > 12:  # High for single module
            analysis['issues'].append('High short-circuit current (check measurement)')
        
        return analysis
    
    def _identify_degradation_mechanisms(self, params: Dict[str, float]) -> List[str]:
        """Identify likely degradation mechanisms based on parameter patterns."""
        mechanisms = []
        
        ff = params.get('fill_factor', 1.0)
        rs = params.get('rs', 0)
        rsh = params.get('rsh', float('inf'))
        voc = params.get('voc', 0)
        isc = params.get('isc', 0)
        
        # Series resistance related degradation
        if rs > 1.0:
            if ff < 0.70:
                mechanisms.append('Interconnect degradation')
            if rs > 2.0:
                mechanisms.append('Solder bond degradation')
        
        # Shunt resistance related degradation
        if rsh < 500:
            mechanisms.append('Cell cracking or micro-cracks')
            if rsh < 200:
                mechanisms.append('Potential Induced Degradation (PID)')
        
        # Combined effects
        if ff < 0.65 and rs > 0.8 and rsh < 800:
            mechanisms.append('Multiple degradation factors')
        
        # Light induced degradation
        if 0.70 < ff < 0.75 and 40 < voc < 60:
            mechanisms.append('Light-induced degradation (LID)')
        
        # Corrosion effects
        if isc/voc < 0.15 and ff < 0.72:  # Low current density with poor FF
            mechanisms.append('Possible corrosion effects')
        
        return mechanisms if mechanisms else ['Normal aging']
    
    def _generate_recommendations(self, diagnosis: Dict) -> List[str]:
        """Generate maintenance and remediation recommendations."""
        recommendations = []
        
        # Based on severity
        if diagnosis['severity'] == 'High':
            recommendations.append('Immediate inspection and maintenance required')
            recommendations.append('Consider module replacement if degradation is severe')
        elif diagnosis['severity'] == 'Medium':
            recommendations.append('Schedule maintenance within next 3 months')
            recommendations.append('Monitor performance trends closely')
        
        # Specific issue-based recommendations
        primary_issues = diagnosis.get('primary_issues', [])
        
        for issue in primary_issues:
            if 'series resistance' in issue.lower():
                recommendations.append('Inspect electrical connections and solder joints')
                recommendations.append('Check for corrosion at junction boxes')
            
            if 'shunt resistance' in issue.lower():
                recommendations.append('Inspect modules for physical damage or cracks')
                recommendations.append('Check for moisture ingress')
            
            if 'fill factor' in issue.lower():
                recommendations.append('Comprehensive electrical testing recommended')
                recommendations.append('Consider thermal imaging inspection')
            
            if 'power' in issue.lower():
                recommendations.append('Verify irradiance and temperature conditions')
                recommendations.append('Clean modules if soiling is suspected')
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append('Continue regular monitoring')
            recommendations.append('Maintain cleaning schedule')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:5]  # Return top 5 recommendations
    
    def compare_with_reference(self, current_params: Dict[str, float], 
                              reference_params: Dict[str, float]) -> Dict:
        """
        Compare current parameters with reference values.
        
        Args:
            current_params: Current measured parameters
            reference_params: Reference parameters (e.g., from commissioning)
            
        Returns:
            Comparison analysis
        """
        comparison = {
            'parameter_changes': {},
            'degradation_rates': {},
            'overall_degradation': 0.0,
            'status': 'Good'
        }
        
        key_params = ['voc', 'isc', 'p_max', 'fill_factor']
        
        for param in key_params:
            if param in current_params and param in reference_params:
                current_val = current_params[param]
                ref_val = reference_params[param]
                
                if ref_val != 0:
                    change_percent = ((current_val - ref_val) / ref_val) * 100
                    comparison['parameter_changes'][param] = {
                        'current': current_val,
                        'reference': ref_val,
                        'change_percent': change_percent,
                        'change_absolute': current_val - ref_val
                    }
        
        # Calculate overall degradation (typically based on power)
        if 'p_max' in comparison['parameter_changes']:
            comparison['overall_degradation'] = abs(comparison['parameter_changes']['p_max']['change_percent'])
            
            if comparison['overall_degradation'] > 20:
                comparison['status'] = 'Critical'
            elif comparison['overall_degradation'] > 10:
                comparison['status'] = 'Poor'
            elif comparison['overall_degradation'] > 5:
                comparison['status'] = 'Fair'
        
        return comparison
    
    def assess_environmental_impact(self, params: Dict[str, float], 
                                  temperature: float = 25.0, 
                                  irradiance: float = 1000.0) -> Dict:
        """
        Assess the impact of environmental conditions on performance.
        
        Args:
            params: I-V parameters
            temperature: Cell temperature (°C)
            irradiance: Irradiance level (W/m²)
            
        Returns:
            Environmental impact analysis
        """
        analysis = {
            'temperature_effect': 'Normal',
            'irradiance_effect': 'Normal',
            'environmental_factors': [],
            'corrections_needed': []
        }
        
        # Temperature analysis
        if temperature > 45:
            analysis['temperature_effect'] = 'High temperature impact'
            analysis['environmental_factors'].append('Elevated operating temperature detected')
            analysis['corrections_needed'].append('Temperature correction recommended')
        elif temperature < 15:
            analysis['temperature_effect'] = 'Low temperature impact'
            analysis['environmental_factors'].append('Low operating temperature')
        
        # Irradiance analysis
        if irradiance < 800:
            analysis['irradiance_effect'] = 'Low irradiance conditions'
            analysis['environmental_factors'].append('Low irradiance measurement conditions')
            analysis['corrections_needed'].append('Irradiance correction recommended')
        elif irradiance > 1200:
            analysis['irradiance_effect'] = 'High irradiance conditions'
            analysis['environmental_factors'].append('High irradiance conditions')
        
        return analysis
