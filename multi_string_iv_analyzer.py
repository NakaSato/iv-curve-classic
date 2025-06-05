#!/usr/bin/env python3
"""
Multi-String I-V Curve Analysis System (32-String Configuration)
Advanced photovoltaic analysis for large-scale solar installations

This system handles comprehensive analysis of up to 32 parallel strings:
- Individual string I-V characteristic analysis
- Cross-string performance comparison
- String imbalance detection and quantification
- Advanced electrical parameter extraction
- Performance optimization recommendations

Author: Advanced PV Analysis System
Version: 4.0.0 (Multi-String Enhanced)
Date: June 2025
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import optimize
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StringIVCharacteristics:
    """Individual string I-V curve characteristics"""
    string_id: int
    voc: float  # Open circuit voltage (V)
    isc: float  # Short circuit current (A)
    vmp: float  # Maximum power point voltage (V)
    imp: float  # Maximum power point current (A)
    pmp: float  # Maximum power (W)
    fill_factor: float  # Fill factor
    series_resistance: float  # Series resistance (Ohm)
    shunt_resistance: float  # Shunt resistance (Ohm)
    efficiency: float  # String efficiency (%)
    degradation_factor: float  # Relative performance factor
    
@dataclass
class MultiStringAnalysisResult:
    """Comprehensive multi-string analysis results"""
    timestamp: str
    total_strings: int
    active_strings: int
    string_characteristics: List[StringIVCharacteristics]
    system_performance: Dict[str, float]
    imbalance_analysis: Dict[str, Any]
    optimization_recommendations: List[str]
    performance_ranking: List[Tuple[int, float]]

class MultiStringIVAnalyzer:
    """Advanced 32-String I-V Curve Analysis System"""
    
    def __init__(self, max_strings: int = 32):
        """Initialize multi-string analyzer"""
        self.max_strings = max_strings
        self.string_columns = self._generate_column_names()
        
        logger.info(f"🔌 Multi-String I-V Analyzer Initialized")
        logger.info(f"📊 Maximum Strings: {max_strings}")
        logger.info(f"🎯 Advanced I-V Analysis | Cross-String Comparison | Performance Optimization")
    
    def _generate_column_names(self) -> Dict[str, List[str]]:
        """Generate voltage and current column names for all strings"""
        voltage_cols = [f"Vstr{i}(V)" for i in range(1, self.max_strings + 1)]
        current_cols = [f"Istr{i}(A)" for i in range(1, self.max_strings + 1)]
        
        return {
            'voltage': voltage_cols,
            'current': current_cols
        }
    
    def load_multi_string_data(self, data_file: str) -> pd.DataFrame:
        """Load and validate multi-string I-V data"""
        try:
            logger.info(f"📂 Loading multi-string data from: {data_file}")
            
            # Try different separators and encodings
            separators = [',', '\t', ';']
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            df = None
            for sep in separators:
                for encoding in encodings:
                    try:
                        df = pd.read_csv(data_file, sep=sep, encoding=encoding)
                        if len(df.columns) > 10:  # Multi-string data should have many columns
                            break
                    except:
                        continue
                if df is not None and len(df.columns) > 10:
                    break
            
            if df is None:
                raise ValueError("Could not load data with any separator/encoding combination")
            
            logger.info(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
            
            # Identify available strings
            available_v_cols = [col for col in df.columns if col in self.string_columns['voltage']]
            available_i_cols = [col for col in df.columns if col in self.string_columns['current']]
            
            self.active_strings = min(len(available_v_cols), len(available_i_cols))
            
            logger.info(f"🔌 Active strings detected: {self.active_strings}")
            
            # Clean and validate data
            df = self._clean_multi_string_data(df, available_v_cols, available_i_cols)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading multi-string data: {e}")
            raise
    
    def _clean_multi_string_data(self, df: pd.DataFrame, v_cols: List[str], i_cols: List[str]) -> pd.DataFrame:
        """Clean and validate multi-string data"""
        try:
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Convert to numeric, replacing non-numeric with NaN
            for col in v_cols + i_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows where all voltage or current values are NaN
            df = df.dropna(subset=v_cols + i_cols, how='all')
            
            # Remove negative values (physical impossibility for PV)
            for col in v_cols + i_cols:
                if col in df.columns:
                    df[col] = df[col].abs()  # Take absolute value
            
            logger.info(f"🧹 Data cleaned: {len(df)} valid rows remaining")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cleaning data: {e}")
            raise
    
    def analyze_string_characteristics(self, df: pd.DataFrame) -> List[StringIVCharacteristics]:
        """Analyze I-V characteristics for each string"""
        string_results = []
        
        logger.info(f"🔍 Analyzing {self.active_strings} string characteristics...")
        
        for string_num in range(1, self.active_strings + 1):
            v_col = f"Vstr{string_num}(V)"
            i_col = f"Istr{string_num}(A)"
            
            if v_col in df.columns and i_col in df.columns:
                try:
                    # Extract valid data points for this string
                    string_data = df[[v_col, i_col]].dropna()
                    
                    if len(string_data) < 5:  # Need minimum points for analysis
                        logger.warning(f"⚠️ String {string_num}: Insufficient data points")
                        continue
                    
                    voltage = string_data[v_col].values
                    current = string_data[i_col].values
                    
                    # Calculate I-V characteristics
                    characteristics = self._calculate_iv_characteristics(
                        voltage, current, string_num
                    )
                    
                    string_results.append(characteristics)
                    
                except Exception as e:
                    logger.error(f"❌ Error analyzing string {string_num}: {e}")
                    continue
        
        logger.info(f"✅ Successfully analyzed {len(string_results)} strings")
        return string_results
    
    def _calculate_iv_characteristics(self, voltage: np.ndarray, current: np.ndarray, string_id: int) -> StringIVCharacteristics:
        """Calculate comprehensive I-V characteristics for a single string"""
        try:
            # Sort by voltage for proper I-V curve analysis
            sort_idx = np.argsort(voltage)
            v_sorted = voltage[sort_idx]
            i_sorted = current[sort_idx]
            
            # Remove duplicate voltage points
            unique_mask = np.diff(v_sorted, prepend=-1) > 1e-6
            v_unique = v_sorted[unique_mask]
            i_unique = i_sorted[unique_mask]
            
            # Calculate basic parameters
            voc = self._find_voc(v_unique, i_unique)
            isc = self._find_isc(v_unique, i_unique)
            
            # Find maximum power point
            power = v_unique * i_unique
            max_power_idx = np.argmax(power)
            vmp = v_unique[max_power_idx]
            imp = i_unique[max_power_idx]
            pmp = power[max_power_idx]
            
            # Calculate fill factor
            fill_factor = pmp / (voc * isc) if (voc * isc) > 0 else 0
            
            # Estimate resistances
            series_resistance = self._estimate_series_resistance(v_unique, i_unique, voc, isc)
            shunt_resistance = self._estimate_shunt_resistance(v_unique, i_unique, voc, isc)
            
            # Calculate efficiency (assuming standard test conditions)
            # Standard irradiance: 1000 W/m², cell area estimation based on typical values
            estimated_area = 2.0  # m² (typical string area)
            standard_irradiance = 1000  # W/m²
            efficiency = (pmp / (estimated_area * standard_irradiance)) * 100
            
            # Calculate degradation factor (relative to ideal performance)
            ideal_power = voc * isc * 0.85  # Typical ideal fill factor
            degradation_factor = pmp / ideal_power if ideal_power > 0 else 0
            
            return StringIVCharacteristics(
                string_id=string_id,
                voc=round(voc, 2),
                isc=round(isc, 3),
                vmp=round(vmp, 2),
                imp=round(imp, 3),
                pmp=round(pmp, 1),
                fill_factor=round(fill_factor, 3),
                series_resistance=round(series_resistance, 4),
                shunt_resistance=round(shunt_resistance, 1),
                efficiency=round(efficiency, 2),
                degradation_factor=round(degradation_factor, 3)
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating characteristics for string {string_id}: {e}")
            # Return default characteristics
            return StringIVCharacteristics(
                string_id=string_id, voc=0, isc=0, vmp=0, imp=0, pmp=0,
                fill_factor=0, series_resistance=0, shunt_resistance=0,
                efficiency=0, degradation_factor=0
            )
    
    def _find_voc(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Find open circuit voltage (Voc)"""
        try:
            # Find voltage where current is closest to zero
            min_current_idx = np.argmin(np.abs(current))
            return voltage[min_current_idx]
        except:
            return np.max(voltage) if len(voltage) > 0 else 0
    
    def _find_isc(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Find short circuit current (Isc)"""
        try:
            # Find current where voltage is closest to zero
            min_voltage_idx = np.argmin(np.abs(voltage))
            return current[min_voltage_idx]
        except:
            return np.max(current) if len(current) > 0 else 0
    
    def _estimate_series_resistance(self, voltage: np.ndarray, current: np.ndarray, voc: float, isc: float) -> float:
        """Estimate series resistance from high-current region"""
        try:
            # Use high-current region (I > 0.8 * Isc)
            high_current_mask = current > 0.8 * isc
            if np.sum(high_current_mask) < 2:
                return 0.01  # Default small value
            
            v_high = voltage[high_current_mask]
            i_high = current[high_current_mask]
            
            # Linear fit: V = Voc - Rs * I
            if len(v_high) > 1:
                slope = np.polyfit(i_high, v_high, 1)[0]
                return abs(slope)
            else:
                return 0.01
        except:
            return 0.01
    
    def _estimate_shunt_resistance(self, voltage: np.ndarray, current: np.ndarray, voc: float, isc: float) -> float:
        """Estimate shunt resistance from low-voltage region"""
        try:
            # Use low-voltage region (V < 0.2 * Voc)
            low_voltage_mask = voltage < 0.2 * voc
            if np.sum(low_voltage_mask) < 2:
                return 1000.0  # Default high value
            
            v_low = voltage[low_voltage_mask]
            i_low = current[low_voltage_mask]
            
            # Linear fit: I = V/Rsh + Iph
            if len(v_low) > 1 and np.std(v_low) > 1e-6:
                slope = np.polyfit(v_low, i_low, 1)[0]
                return 1/slope if slope > 1e-6 else 1000.0
            else:
                return 1000.0
        except:
            return 1000.0
    
    def analyze_string_imbalance(self, string_results: List[StringIVCharacteristics]) -> Dict[str, Any]:
        """Analyze imbalance across strings"""
        if len(string_results) < 2:
            return {"error": "Need at least 2 strings for imbalance analysis"}
        
        # Extract key parameters
        powers = [s.pmp for s in string_results]
        voltages = [s.voc for s in string_results]
        currents = [s.isc for s in string_results]
        efficiencies = [s.efficiency for s in string_results]
        
        # Calculate statistics
        power_stats = {
            'mean': np.mean(powers),
            'std': np.std(powers),
            'cv': np.std(powers) / np.mean(powers) * 100 if np.mean(powers) > 0 else 0,
            'range': np.max(powers) - np.min(powers),
            'min': np.min(powers),
            'max': np.max(powers)
        }
        
        voltage_stats = {
            'mean': np.mean(voltages),
            'std': np.std(voltages),
            'cv': np.std(voltages) / np.mean(voltages) * 100 if np.mean(voltages) > 0 else 0
        }
        
        current_stats = {
            'mean': np.mean(currents),
            'std': np.std(currents),
            'cv': np.std(currents) / np.mean(currents) * 100 if np.mean(currents) > 0 else 0
        }
        
        # Identify problematic strings
        power_threshold = power_stats['mean'] - 2 * power_stats['std']
        underperforming_strings = [
            s.string_id for s in string_results 
            if s.pmp < power_threshold
        ]
        
        # Calculate imbalance severity
        imbalance_severity = "Low"
        if power_stats['cv'] > 15:
            imbalance_severity = "High"
        elif power_stats['cv'] > 8:
            imbalance_severity = "Moderate"
        
        return {
            'power_statistics': power_stats,
            'voltage_statistics': voltage_stats,
            'current_statistics': current_stats,
            'underperforming_strings': underperforming_strings,
            'imbalance_severity': imbalance_severity,
            'total_system_power': sum(powers),
            'average_efficiency': np.mean(efficiencies)
        }
    
    def generate_optimization_recommendations(self, string_results: List[StringIVCharacteristics], 
                                           imbalance_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # Power imbalance recommendations
        if imbalance_analysis['imbalance_severity'] == "High":
            recommendations.append(
                f"🔧 CRITICAL: High power imbalance detected (CV: {imbalance_analysis['power_statistics']['cv']:.1f}%). "
                f"Inspect strings {imbalance_analysis['underperforming_strings']} for shading, soiling, or hardware issues."
            )
        elif imbalance_analysis['imbalance_severity'] == "Moderate":
            recommendations.append(
                f"⚠️ WARNING: Moderate power imbalance (CV: {imbalance_analysis['power_statistics']['cv']:.1f}%). "
                f"Consider string-level monitoring and cleaning schedule optimization."
            )
        
        # Efficiency recommendations
        avg_efficiency = imbalance_analysis['average_efficiency']
        if avg_efficiency < 15:
            recommendations.append(
                f"📉 LOW EFFICIENCY: System efficiency {avg_efficiency:.1f}% is below optimal. "
                f"Consider module replacement or system reconfiguration."
            )
        elif avg_efficiency < 18:
            recommendations.append(
                f"📊 MODERATE EFFICIENCY: System efficiency {avg_efficiency:.1f}% has improvement potential. "
                f"Regular maintenance and cleaning recommended."
            )
        
        # String-specific recommendations
        low_performance_strings = [s for s in string_results if s.degradation_factor < 0.8]
        if low_performance_strings:
            string_ids = [str(s.string_id) for s in low_performance_strings]
            recommendations.append(
                f"🔍 INDIVIDUAL STRING ISSUES: Strings {', '.join(string_ids)} show significant degradation. "
                f"Detailed inspection and potential module replacement recommended."
            )
        
        # Fill factor analysis
        low_ff_strings = [s for s in string_results if s.fill_factor < 0.7]
        if low_ff_strings:
            recommendations.append(
                f"⚡ ELECTRICAL ISSUES: {len(low_ff_strings)} strings have low fill factors. "
                f"Check for high series resistance or bypass diode issues."
            )
        
        # System-level recommendations
        total_power = imbalance_analysis['total_system_power']
        if total_power > 0:
            power_per_string = total_power / len(string_results)
            if power_per_string < 250:  # Typical string power threshold
                recommendations.append(
                    f"📊 SYSTEM OPTIMIZATION: Average string power {power_per_string:.0f}W suggests "
                    f"potential for system reconfiguration or technology upgrade."
                )
        
        # Add general recommendations if none specific
        if not recommendations:
            recommendations.append("✅ GOOD PERFORMANCE: System shows balanced operation. Continue regular monitoring.")
        
        return recommendations
    
    def create_comprehensive_dashboard(self, string_results: List[StringIVCharacteristics], 
                                     imbalance_analysis: Dict[str, Any],
                                     output_file: Optional[str] = None) -> str:
        """Create comprehensive multi-string analysis dashboard"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"multi_string_analysis_dashboard_{timestamp}.html"
        
        logger.info(f"🎨 Creating comprehensive multi-string dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "String Power Comparison",
                "Voltage vs Current Distribution", 
                "Fill Factor Analysis",
                "String Efficiency Comparison",
                "Power Imbalance Heatmap",
                "System Performance Summary"
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "table"}]
            ]
        )
        
        # Extract data for plotting
        string_ids = [s.string_id for s in string_results]
        powers = [s.pmp for s in string_results]
        voltages = [s.voc for s in string_results]
        currents = [s.isc for s in string_results]
        fill_factors = [s.fill_factor for s in string_results]
        efficiencies = [s.efficiency for s in string_results]
        
        # 1. String Power Comparison
        fig.add_trace(
            go.Bar(
                x=string_ids, y=powers,
                name="String Power",
                marker_color='lightblue',
                text=[f"{p:.0f}W" for p in powers],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Voltage vs Current Distribution
        fig.add_trace(
            go.Scatter(
                x=voltages, y=currents,
                mode='markers+text',
                text=[f"S{s}" for s in string_ids],
                textposition='top center',
                marker=dict(size=10, color=powers, colorscale='Viridis', showscale=True),
                name="V-I Distribution"
            ),
            row=1, col=2
        )
        
        # 3. Fill Factor Analysis
        colors = ['red' if ff < 0.7 else 'orange' if ff < 0.75 else 'green' for ff in fill_factors]
        fig.add_trace(
            go.Bar(
                x=string_ids, y=fill_factors,
                name="Fill Factor",
                marker_color=colors,
                text=[f"{ff:.3f}" for ff in fill_factors],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. String Efficiency Comparison
        fig.add_trace(
            go.Bar(
                x=string_ids, y=efficiencies,
                name="Efficiency (%)",
                marker_color='lightgreen',
                text=[f"{eff:.1f}%" for eff in efficiencies],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # 5. Power Imbalance Heatmap
        power_matrix = np.array(powers).reshape(1, -1)
        fig.add_trace(
            go.Heatmap(
                z=power_matrix,
                x=string_ids,
                y=["Power (W)"],
                colorscale="RdYlGn",
                showscale=True
            ),
            row=3, col=1
        )
        
        # 6. System Performance Summary Table
        summary_data = [
            ["Total Strings", len(string_results)],
            ["Total Power", f"{sum(powers):.0f} W"],
            ["Average Power", f"{np.mean(powers):.0f} W"],
            ["Power CV", f"{imbalance_analysis['power_statistics']['cv']:.1f}%"],
            ["Imbalance Level", imbalance_analysis['imbalance_severity']],
            ["Average Efficiency", f"{np.mean(efficiencies):.1f}%"],
            ["Best String", f"String {string_ids[np.argmax(powers)]} ({max(powers):.0f}W)"],
            ["Worst String", f"String {string_ids[np.argmin(powers)]} ({min(powers):.0f}W)"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=["Parameter", "Value"], fill_color="lightblue"),
                cells=dict(values=list(zip(*summary_data)), fill_color="white")
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Multi-String PV System Analysis - {len(string_results)} Strings",
            showlegend=False,
            height=1200,
            font=dict(size=12)
        )
        
        # Save dashboard
        html_content = fig.to_html(include_plotlyjs='cdn')
        
        # Add custom CSS and additional information
        html_header = f"""
        <html>
        <head>
            <title>Multi-String PV Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🔌 Multi-String PV System Analysis Dashboard</h1>
            <div class="summary">
                <h3>📊 System Overview</h3>
                <ul>
                    <li><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    <li><strong>Total Strings Analyzed:</strong> {len(string_results)}</li>
                    <li><strong>Total System Power:</strong> {sum(powers):.0f} W</li>
                    <li><strong>Power Imbalance:</strong> {imbalance_analysis['imbalance_severity']} (CV: {imbalance_analysis['power_statistics']['cv']:.1f}%)</li>
                    <li><strong>Average System Efficiency:</strong> {np.mean(efficiencies):.1f}%</li>
                </ul>
            </div>
        """
        
        # Add recommendations
        recommendations = self.generate_optimization_recommendations(string_results, imbalance_analysis)
        recommendations_html = """
            <div class="recommendations">
                <h3>🎯 Optimization Recommendations</h3>
                <ul>
        """
        for rec in recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += """
                </ul>
            </div>
        """
        
        # Combine all HTML
        full_html = html_header + recommendations_html + html_content + "</body></html>"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        logger.info(f"✅ Multi-string dashboard saved: {output_file}")
        return output_file
    
    def run_comprehensive_analysis(self, data_file: str) -> MultiStringAnalysisResult:
        """Run comprehensive multi-string I-V analysis"""
        
        logger.info("🚀 Starting Comprehensive Multi-String I-V Analysis...")
        
        try:
            # Load and process data
            df = self.load_multi_string_data(data_file)
            
            # Analyze individual string characteristics  
            string_results = self.analyze_string_characteristics(df)
            
            if not string_results:
                raise ValueError("No valid string data found for analysis")
            
            # Analyze string imbalance
            imbalance_analysis = self.analyze_string_imbalance(string_results)
            
            # Generate optimization recommendations
            recommendations = self.generate_optimization_recommendations(string_results, imbalance_analysis)
            
            # Create performance ranking
            performance_ranking = sorted(
                [(s.string_id, s.pmp) for s in string_results],
                key=lambda x: x[1], reverse=True
            )
            
            # Calculate system performance metrics
            total_power = sum(s.pmp for s in string_results)
            avg_efficiency = np.mean([s.efficiency for s in string_results])
            
            system_performance = {
                'total_power_w': total_power,
                'average_efficiency_pct': avg_efficiency,
                'power_density_w_per_string': total_power / len(string_results),
                'system_capacity_factor': avg_efficiency / 20 * 100,  # Assuming 20% as reference
                'imbalance_coefficient': imbalance_analysis['power_statistics']['cv']
            }
            
            # Create comprehensive dashboard
            dashboard_file = self.create_comprehensive_dashboard(string_results, imbalance_analysis)
            
            # Create analysis result
            result = MultiStringAnalysisResult(
                timestamp=datetime.now().isoformat(),
                total_strings=self.max_strings,
                active_strings=len(string_results),
                string_characteristics=string_results,
                system_performance=system_performance,
                imbalance_analysis=imbalance_analysis,
                optimization_recommendations=recommendations,
                performance_ranking=performance_ranking
            )
            
            logger.info("✅ Comprehensive Multi-String I-V Analysis Complete!")
            logger.info(f"📊 Dashboard: {dashboard_file}")
            logger.info(f"🔌 Analyzed Strings: {len(string_results)}")
            logger.info(f"⚡ Total Power: {total_power:.0f} W")
            logger.info(f"📈 Average Efficiency: {avg_efficiency:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Multi-string analysis failed: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    analyzer = MultiStringIVAnalyzer(max_strings=32)
    
    # Run analysis (replace with actual data file)
    # result = analyzer.run_comprehensive_analysis("multi_string_data.csv")
    print("🔌 Multi-String I-V Analyzer Ready for 32-String Analysis")
