#!/usr/bin/env python3
"""
Advanced Inverter Data Cleaner and Parameter Extractor
======================================================

Comprehensive data cleaning and parameter extraction system for real-world
inverter log data with 32-string configuration and 222+ parameters.

Features:
- Intelligent data cleaning and validation
- Multi-string parameter extraction (Vstr1-32, Istr1-32)
- System performance analytics
- Fault detection and analysis
- Temperature monitoring
- Power quality assessment
- Data quality scoring
- Export to multiple formats

Author: AI Assistant
Created: June 5, 2025
"""

import pandas as pd
import numpy as np
import logging
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class InverterDataQuality:
    """Data quality assessment results"""
    total_records: int
    valid_records: int
    data_quality_score: float
    missing_data_percentage: float
    duplicate_records: int
    timestamp_gaps: int
    invalid_values: Dict[str, int]
    outliers_detected: Dict[str, int]
    recommendations: List[str]

@dataclass
class StringPerformance:
    """Individual string performance metrics"""
    string_id: int
    avg_voltage: float
    avg_current: float
    avg_power: float
    max_voltage: float
    max_current: float
    peak_power: float
    operating_hours: float
    energy_generated: float
    efficiency: float
    performance_ratio: float
    issues_detected: List[str]

@dataclass
class SystemPerformance:
    """Overall system performance metrics"""
    total_power_generated: float
    total_energy: float
    system_efficiency: float
    capacity_factor: float
    performance_ratio: float
    uptime_percentage: float
    fault_count: int
    warning_count: int
    temperature_metrics: Dict[str, float]
    power_quality_metrics: Dict[str, float]

@dataclass
class CleanedInverterData:
    """Cleaned and processed inverter data"""
    timestamp: str
    data_quality: InverterDataQuality
    string_performance: List[StringPerformance]
    system_performance: SystemPerformance
    cleaned_dataset: pd.DataFrame
    summary_report: Dict[str, Any]
    export_files: List[str]

class InverterDataCleaner:
    """Advanced inverter data cleaning and parameter extraction system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data cleaner with configuration"""
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Parameter mappings for easy access
        self.string_voltage_cols = [f'Vstr{i}(V)' for i in range(1, 33)]
        self.string_current_cols = [f'Istr{i}(A)' for i in range(1, 33)]
        self.pv_voltage_cols = [f'Vpv{i}(V)' for i in range(1, 17)]
        self.pv_current_cols = [f'Ipv{i}(A)' for i in range(1, 17)]
        self.pv_power_cols = [f'Ppv{i}(W)' for i in range(1, 17)]
        self.temperature_cols = ['INVTemp(℃)', 'AMTemp1(℃)', 'BTTemp(℃)', 'OUTTemp(℃)', 'AMTemp2(℃)']
        self.ac_voltage_cols = ['VacR(V)', 'VacS(V)', 'VacT(V)', 'VacRS(V)', 'VacST(V)', 'VacTR(V)']
        self.ac_current_cols = ['IacR(A)', 'IacS(A)', 'IacT(A)']
        self.ac_power_cols = ['PacR(VA)', 'PacS(VA)', 'PacT(VA)']
        
        self.logger.info("🧹 Advanced Inverter Data Cleaner Initialized")
        self.logger.info(f"📊 Configuration: {self.config['cleaning']['aggressive_cleaning']}")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for data cleaning"""
        return {
            'cleaning': {
                'remove_duplicates': True,
                'handle_missing_values': True,
                'aggressive_cleaning': True,
                'outlier_detection': True,
                'validate_ranges': True
            },
            'validation': {
                'voltage_range': {'min': 0, 'max': 1000},  # Volts
                'current_range': {'min': 0, 'max': 50},    # Amperes
                'power_range': {'min': 0, 'max': 50000},   # Watts
                'temperature_range': {'min': -40, 'max': 85},  # Celsius
                'frequency_range': {'min': 45, 'max': 65}  # Hz
            },
            'analysis': {
                'calculate_derived_metrics': True,
                'performance_analysis': True,
                'fault_detection': True,
                'string_comparison': True
            },
            'export': {
                'save_cleaned_data': True,
                'generate_reports': True,
                'create_visualizations': True,
                'export_formats': ['csv', 'json', 'excel']
            }
        }
    
    def clean_inverter_data(self, file_path: str) -> CleanedInverterData:
        """
        Comprehensive cleaning and analysis of inverter data
        
        Args:
            file_path: Path to the inverter CSV data file
            
        Returns:
            CleanedInverterData object with all results
        """
        self.logger.info(f"🚀 Starting comprehensive inverter data cleaning...")
        self.logger.info(f"📂 Processing file: {file_path}")
        
        try:
            # Load raw data
            raw_data = self._load_raw_data(file_path)
            self.logger.info(f"📊 Loaded {len(raw_data)} records with {len(raw_data.columns)} columns")
            
            # Data quality assessment
            data_quality = self._assess_data_quality(raw_data)
            self.logger.info(f"📈 Data quality score: {data_quality.data_quality_score:.1f}%")
            
            # Clean the dataset
            cleaned_data = self._perform_data_cleaning(raw_data)
            self.logger.info(f"✅ Cleaned dataset: {len(cleaned_data)} records remaining")
            
            # Extract string performance
            string_performance = self._analyze_string_performance(cleaned_data)
            self.logger.info(f"🔌 Analyzed {len(string_performance)} strings")
            
            # System performance analysis
            system_performance = self._analyze_system_performance(cleaned_data)
            self.logger.info(f"⚡ System efficiency: {system_performance.system_efficiency:.1f}%")
            
            # Generate summary report
            summary_report = self._generate_summary_report(
                raw_data, cleaned_data, data_quality, string_performance, system_performance
            )
            
            # Export results
            export_files = self._export_results(
                cleaned_data, summary_report, file_path
            )
            
            # Create result object
            result = CleanedInverterData(
                timestamp=datetime.now().isoformat(),
                data_quality=data_quality,
                string_performance=string_performance,
                system_performance=system_performance,
                cleaned_dataset=cleaned_data,
                summary_report=summary_report,
                export_files=export_files
            )
            
            self.logger.info("🎉 Inverter data cleaning completed successfully!")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error during data cleaning: {str(e)}")
            raise
    
    def _load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load and perform initial processing of raw inverter data"""
        try:
            # Load CSV with error handling
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert timestamp
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M', errors='coerce')
                df = df.sort_values('Time').reset_index(drop=True)
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load data: {str(e)}")
            raise
    
    def _assess_data_quality(self, df: pd.DataFrame) -> InverterDataQuality:
        """Comprehensive data quality assessment"""
        total_records = len(df)
        
        # Count missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        total_cells = df.size
        missing_percentage = (total_missing / total_cells) * 100
        
        # Detect duplicates
        duplicates = df.duplicated().sum()
        
        # Timestamp gap analysis
        timestamp_gaps = 0
        if 'Time' in df.columns:
            time_diffs = df['Time'].diff()
            expected_interval = pd.Timedelta(minutes=3)  # Based on data pattern
            timestamp_gaps = (time_diffs > expected_interval * 2).sum()
        
        # Validate ranges and detect invalid values
        invalid_values = {}
        outliers_detected = {}
        
        # Voltage validation
        for col in self.string_voltage_cols + self.pv_voltage_cols + self.ac_voltage_cols:
            if col in df.columns:
                valid_range = self.config['validation']['voltage_range']
                invalid = ((df[col] < valid_range['min']) | 
                          (df[col] > valid_range['max'])).sum()
                if invalid > 0:
                    invalid_values[col] = invalid
                
                # Outlier detection using IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | 
                           (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    outliers_detected[col] = outliers
        
        # Current validation
        for col in self.string_current_cols + self.pv_current_cols + self.ac_current_cols:
            if col in df.columns:
                valid_range = self.config['validation']['current_range']
                invalid = ((df[col] < valid_range['min']) | 
                          (df[col] > valid_range['max'])).sum()
                if invalid > 0:
                    invalid_values[col] = invalid
        
        # Calculate overall data quality score
        quality_factors = [
            100 - missing_percentage,  # Less missing = better
            max(0, 100 - (duplicates / total_records) * 100),  # Fewer duplicates = better
            max(0, 100 - (len(invalid_values) / len(df.columns)) * 100),  # Fewer invalid = better
            max(0, 100 - (timestamp_gaps / total_records) * 100)  # Fewer gaps = better
        ]
        data_quality_score = np.mean(quality_factors)
        
        # Generate recommendations
        recommendations = []
        if missing_percentage > 10:
            recommendations.append("High missing data percentage - consider interpolation")
        if duplicates > 0:
            recommendations.append("Duplicate records detected - removal recommended")
        if len(invalid_values) > 0:
            recommendations.append("Invalid values detected - validation and cleaning needed")
        if timestamp_gaps > total_records * 0.05:
            recommendations.append("Significant timestamp gaps - check data continuity")
        
        valid_records = total_records - duplicates
        
        return InverterDataQuality(
            total_records=total_records,
            valid_records=valid_records,
            data_quality_score=data_quality_score,
            missing_data_percentage=missing_percentage,
            duplicate_records=duplicates,
            timestamp_gaps=timestamp_gaps,
            invalid_values=invalid_values,
            outliers_detected=outliers_detected,
            recommendations=recommendations
        )
    
    def _perform_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive data cleaning"""
        cleaned_df = df.copy()
        
        # Remove duplicates
        if self.config['cleaning']['remove_duplicates']:
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            self.logger.info(f"🔄 Removed {initial_count - len(cleaned_df)} duplicate records")
        
        # Handle missing values
        if self.config['cleaning']['handle_missing_values']:
            # Fill numeric columns with interpolation or forward fill
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    # Use forward fill for most inverter parameters
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                    # Then backward fill for remaining
                    cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
                    # Finally fill with 0 for string currents (likely disconnected)
                    if 'Istr' in col or 'Ipv' in col:
                        cleaned_df[col] = cleaned_df[col].fillna(0)
        
        # Validate and correct ranges
        if self.config['cleaning']['validate_ranges']:
            # Voltage validation and clipping
            voltage_cols = [col for col in self.string_voltage_cols + self.pv_voltage_cols 
                           if col in cleaned_df.columns]
            for col in voltage_cols:
                valid_range = self.config['validation']['voltage_range']
                cleaned_df[col] = cleaned_df[col].clip(
                    lower=valid_range['min'], upper=valid_range['max']
                )
            
            # Current validation (negative currents set to 0)
            current_cols = [col for col in self.string_current_cols + self.pv_current_cols 
                           if col in cleaned_df.columns]
            for col in current_cols:
                cleaned_df[col] = cleaned_df[col].clip(lower=0)
        
        # Handle status and error columns
        if 'Status' in cleaned_df.columns:
            # Standardize status values
            status_mapping = {
                'Waiting': 'Standby',
                'Normal': 'Operating',
                'Warning': 'Warning',
                'Error': 'Fault'
            }
            cleaned_df['Status'] = cleaned_df['Status'].map(status_mapping).fillna(cleaned_df['Status'])
        
        # Calculate derived parameters if missing
        if self.config['analysis']['calculate_derived_metrics']:
            cleaned_df = self._calculate_derived_parameters(cleaned_df)
        
        # Remove rows with all zero string values (likely no production)
        string_cols = [col for col in self.string_voltage_cols + self.string_current_cols 
                      if col in cleaned_df.columns]
        if string_cols:
            # Keep rows where at least one string has voltage > 10V or current > 0.1A
            voltage_mask = False
            current_mask = False
            
            for col in self.string_voltage_cols:
                if col in cleaned_df.columns:
                    voltage_mask |= (cleaned_df[col] > 10)
            
            for col in self.string_current_cols:
                if col in cleaned_df.columns:
                    current_mask |= (cleaned_df[col] > 0.1)
            
            valid_mask = voltage_mask | current_mask
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df[valid_mask].reset_index(drop=True)
            self.logger.info(f"🔄 Filtered out {initial_count - len(cleaned_df)} non-productive records")
        
        return cleaned_df
    
    def _calculate_derived_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived parameters from raw measurements"""
        derived_df = df.copy()
        
        # Calculate string power (P = V * I) for each string
        for i in range(1, 33):
            v_col = f'Vstr{i}(V)'
            i_col = f'Istr{i}(A)'
            p_col = f'Pstr{i}(W)'
            
            if v_col in derived_df.columns and i_col in derived_df.columns:
                derived_df[p_col] = derived_df[v_col] * derived_df[i_col]
        
        # Total string power
        string_power_cols = [f'Pstr{i}(W)' for i in range(1, 33) 
                            if f'Pstr{i}(W)' in derived_df.columns]
        if string_power_cols:
            derived_df['Total_String_Power(W)'] = derived_df[string_power_cols].sum(axis=1)
        
        # Power efficiency calculation
        if 'Pac(W)' in derived_df.columns and 'Total_String_Power(W)' in derived_df.columns:
            derived_df['Conversion_Efficiency(%)'] = (
                derived_df['Pac(W)'] / derived_df['Total_String_Power(W)'] * 100
            ).clip(upper=100)
        
        # String imbalance calculation
        if string_power_cols:
            derived_df['String_Power_CV(%)'] = (
                derived_df[string_power_cols].std(axis=1) / 
                derived_df[string_power_cols].mean(axis=1) * 100
            )
        
        # Temperature differential
        temp_cols = [col for col in self.temperature_cols if col in derived_df.columns]
        if len(temp_cols) >= 2:
            derived_df['Temp_Range(℃)'] = (
                derived_df[temp_cols].max(axis=1) - derived_df[temp_cols].min(axis=1)
            )
        
        return derived_df
    
    def _analyze_string_performance(self, df: pd.DataFrame) -> List[StringPerformance]:
        """Analyze individual string performance"""
        string_performances = []
        
        for i in range(1, 33):
            v_col = f'Vstr{i}(V)'
            i_col = f'Istr{i}(A)'
            
            if v_col not in df.columns or i_col not in df.columns:
                continue
            
            # Filter valid measurements (voltage > 0, current >= 0)
            valid_mask = (df[v_col] > 0) & (df[i_col] >= 0)
            valid_data = df[valid_mask]
            
            if len(valid_data) == 0:
                continue
            
            # Calculate power
            power = valid_data[v_col] * valid_data[i_col]
            
            # Performance metrics
            avg_voltage = valid_data[v_col].mean()
            avg_current = valid_data[i_col].mean()
            avg_power = power.mean()
            max_voltage = valid_data[v_col].max()
            max_current = valid_data[i_col].max()
            peak_power = power.max()
            
            # Operating hours (assuming 3-minute intervals)
            operating_hours = len(valid_data) * 3 / 60  # Convert to hours
            
            # Energy generated (Wh)
            energy_generated = power.sum() * 3 / 60  # 3-minute intervals to Wh
            
            # Calculate efficiency and performance ratio
            # Assuming rated power per string is around 300W (typical)
            rated_power = 300
            efficiency = (avg_power / rated_power) * 100 if rated_power > 0 else 0
            performance_ratio = peak_power / rated_power if rated_power > 0 else 0
            
            # Issue detection
            issues = []
            if avg_voltage < 30:  # Low voltage threshold
                issues.append("Low voltage detected")
            if avg_current < 1:  # Low current threshold
                issues.append("Low current output")
            if max_voltage > 60:  # High voltage threshold
                issues.append("High voltage detected")
            if max_current > 15:  # High current threshold
                issues.append("High current detected")
            
            # Check for irregular patterns
            voltage_cv = valid_data[v_col].std() / valid_data[v_col].mean()
            if voltage_cv > 0.1:  # High coefficient of variation
                issues.append("Unstable voltage pattern")
                
            current_cv = valid_data[i_col].std() / valid_data[i_col].mean()
            if current_cv > 0.3:  # High coefficient of variation for current
                issues.append("Irregular current pattern")
            
            string_performances.append(StringPerformance(
                string_id=i,
                avg_voltage=avg_voltage,
                avg_current=avg_current,
                avg_power=avg_power,
                max_voltage=max_voltage,
                max_current=max_current,
                peak_power=peak_power,
                operating_hours=operating_hours,
                energy_generated=energy_generated,
                efficiency=efficiency,
                performance_ratio=performance_ratio,
                issues_detected=issues
            ))
        
        return string_performances
    
    def _analyze_system_performance(self, df: pd.DataFrame) -> SystemPerformance:
        """Analyze overall system performance"""
        
        # Total power and energy
        if 'Pac(W)' in df.columns:
            total_power = df['Pac(W)'].sum() * 3 / 60 / 1000  # Convert to kWh
            avg_power = df['Pac(W)'].mean()
        else:
            total_power = 0
            avg_power = 0
        
        # System efficiency
        if 'Total_String_Power(W)' in df.columns and 'Pac(W)' in df.columns:
            valid_power_mask = (df['Total_String_Power(W)'] > 0) & (df['Pac(W)'] > 0)
            if valid_power_mask.sum() > 0:
                efficiency_data = df[valid_power_mask]
                system_efficiency = (
                    efficiency_data['Pac(W)'] / efficiency_data['Total_String_Power(W)']
                ).mean() * 100
            else:
                system_efficiency = 0
        else:
            system_efficiency = 95  # Default assumption
        
        # Uptime calculation
        total_records = len(df)
        if 'Status' in df.columns:
            operating_records = (df['Status'].isin(['Operating', 'Normal'])).sum()
            uptime_percentage = (operating_records / total_records) * 100
        else:
            # Assume operating if power > 100W
            if 'Pac(W)' in df.columns:
                operating_records = (df['Pac(W)'] > 100).sum()
                uptime_percentage = (operating_records / total_records) * 100
            else:
                uptime_percentage = 100
        
        # Fault and warning counts
        fault_count = 0
        warning_count = 0
        if 'FaultCode' in df.columns:
            fault_count = (df['FaultCode'] != 0).sum()
        if 'WarnCode' in df.columns:
            warning_count = (df['WarnCode'] != 0).sum()
        
        # Temperature metrics
        temperature_metrics = {}
        for temp_col in self.temperature_cols:
            if temp_col in df.columns:
                temp_data = df[temp_col].dropna()
                if len(temp_data) > 0:
                    temperature_metrics[temp_col] = {
                        'avg': temp_data.mean(),
                        'max': temp_data.max(),
                        'min': temp_data.min()
                    }
        
        # Power quality metrics
        power_quality_metrics = {}
        if 'Fac(Hz)' in df.columns:
            freq_data = df['Fac(Hz)'].dropna()
            if len(freq_data) > 0:
                power_quality_metrics['frequency'] = {
                    'avg': freq_data.mean(),
                    'std': freq_data.std(),
                    'min': freq_data.min(),
                    'max': freq_data.max()
                }
        
        # AC voltage analysis
        for ac_col in self.ac_voltage_cols:
            if ac_col in df.columns:
                ac_data = df[ac_col].dropna()
                if len(ac_data) > 0:
                    power_quality_metrics[ac_col] = {
                        'avg': ac_data.mean(),
                        'std': ac_data.std()
                    }
        
        # Capacity factor (assuming 10kW rated capacity)
        rated_capacity = 10000  # 10kW
        capacity_factor = (avg_power / rated_capacity) * 100 if rated_capacity > 0 else 0
        
        # Performance ratio (normalized performance)
        performance_ratio = min(100, capacity_factor / 20 * 100)  # Assuming 20% is good performance
        
        return SystemPerformance(
            total_power_generated=total_power,
            total_energy=total_power,
            system_efficiency=system_efficiency,
            capacity_factor=capacity_factor,
            performance_ratio=performance_ratio,
            uptime_percentage=uptime_percentage,
            fault_count=fault_count,
            warning_count=warning_count,
            temperature_metrics=temperature_metrics,
            power_quality_metrics=power_quality_metrics
        )
    
    def _generate_summary_report(self, raw_data: pd.DataFrame, cleaned_data: pd.DataFrame,
                               data_quality: InverterDataQuality, string_performance: List[StringPerformance],
                               system_performance: SystemPerformance) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        
        # Time period analysis
        if 'Time' in cleaned_data.columns:
            start_time = cleaned_data['Time'].min()
            end_time = cleaned_data['Time'].max()
            duration = end_time - start_time
        else:
            start_time = "Unknown"
            end_time = "Unknown"
            duration = "Unknown"
        
        # String summary statistics
        string_summary = {
            'total_strings': len(string_performance),
            'active_strings': len([s for s in string_performance if s.avg_power > 10]),
            'avg_string_power': np.mean([s.avg_power for s in string_performance]),
            'total_string_energy': sum([s.energy_generated for s in string_performance]),
            'strings_with_issues': len([s for s in string_performance if s.issues_detected])
        }
        
        # Performance ranking
        performance_ranking = sorted(string_performance, key=lambda x: x.avg_power, reverse=True)
        top_performers = performance_ranking[:5]
        bottom_performers = performance_ranking[-5:]
        
        # Critical issues summary
        all_issues = []
        for string in string_performance:
            all_issues.extend(string.issues_detected)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_period': {
                'start': str(start_time),
                'end': str(end_time),
                'duration': str(duration)
            },
            'data_summary': {
                'raw_records': len(raw_data),
                'cleaned_records': len(cleaned_data),
                'data_quality_score': data_quality.data_quality_score,
                'cleaning_efficiency': (len(cleaned_data) / len(raw_data)) * 100
            },
            'string_summary': string_summary,
            'system_summary': {
                'total_energy_kwh': system_performance.total_energy,
                'system_efficiency_pct': system_performance.system_efficiency,
                'uptime_pct': system_performance.uptime_percentage,
                'capacity_factor_pct': system_performance.capacity_factor,
                'fault_incidents': system_performance.fault_count,
                'warning_incidents': system_performance.warning_count
            },
            'performance_ranking': {
                'top_performers': [{'string_id': s.string_id, 'avg_power': s.avg_power} 
                                 for s in top_performers],
                'bottom_performers': [{'string_id': s.string_id, 'avg_power': s.avg_power} 
                                    for s in bottom_performers]
            },
            'issues_summary': issue_counts,
            'recommendations': data_quality.recommendations + self._generate_performance_recommendations(
                string_performance, system_performance
            )
        }
    
    def _generate_performance_recommendations(self, string_performance: List[StringPerformance],
                                           system_performance: SystemPerformance) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # String-level recommendations
        low_performers = [s for s in string_performance if s.avg_power < 50]
        if low_performers:
            recommendations.append(f"Investigate {len(low_performers)} underperforming strings")
        
        high_voltage_strings = [s for s in string_performance if s.max_voltage > 55]
        if high_voltage_strings:
            recommendations.append(f"Check {len(high_voltage_strings)} strings with high voltage")
        
        # System-level recommendations
        if system_performance.system_efficiency < 90:
            recommendations.append("System efficiency below 90% - check inverter performance")
        
        if system_performance.uptime_percentage < 95:
            recommendations.append("System uptime below 95% - investigate downtime causes")
        
        if system_performance.fault_count > 0:
            recommendations.append(f"Address {system_performance.fault_count} fault incidents")
        
        if system_performance.warning_count > 10:
            recommendations.append(f"Review {system_performance.warning_count} warning incidents")
        
        # Temperature-based recommendations
        for temp_sensor, metrics in system_performance.temperature_metrics.items():
            if metrics['max'] > 70:
                recommendations.append(f"High temperature detected in {temp_sensor} - check cooling")
        
        return recommendations
    
    def _export_results(self, cleaned_data: pd.DataFrame, summary_report: Dict[str, Any],
                       original_file: str) -> List[str]:
        """Export cleaning results in multiple formats"""
        export_files = []
        
        if not self.config['export']['save_cleaned_data']:
            return export_files
        
        # Generate base filename
        base_name = Path(original_file).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Export cleaned CSV
            if 'csv' in self.config['export']['export_formats']:
                csv_file = f"cleaned_{base_name}_{timestamp}.csv"
                cleaned_data.to_csv(csv_file, index=False)
                export_files.append(csv_file)
                self.logger.info(f"📁 Exported cleaned data: {csv_file}")
            
            # Export summary JSON
            if 'json' in self.config['export']['export_formats']:
                json_file = f"summary_{base_name}_{timestamp}.json"
                with open(json_file, 'w') as f:
                    json.dump(summary_report, f, indent=2, default=str)
                export_files.append(json_file)
                self.logger.info(f"📁 Exported summary report: {json_file}")
            
            # Export Excel with multiple sheets
            if 'excel' in self.config['export']['export_formats']:
                excel_file = f"inverter_analysis_{base_name}_{timestamp}.xlsx"
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    # Main cleaned data
                    cleaned_data.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                    
                    # Summary statistics
                    summary_df = pd.DataFrame([summary_report['system_summary']])
                    summary_df.to_excel(writer, sheet_name='System_Summary', index=False)
                    
                    # String performance
                    if 'string_summary' in summary_report:
                        string_df = pd.DataFrame([summary_report['string_summary']])
                        string_df.to_excel(writer, sheet_name='String_Summary', index=False)
                
                export_files.append(excel_file)
                self.logger.info(f"📁 Exported Excel analysis: {excel_file}")
        
        except Exception as e:
            self.logger.error(f"❌ Export failed: {str(e)}")
        
        return export_files
    
    def create_analysis_dashboard(self, result: CleanedInverterData, output_file: str = None) -> str:
        """Create interactive dashboard for cleaned inverter data"""
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"inverter_analysis_dashboard_{timestamp}.html"
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'System Power Output', 'String Performance Comparison',
                'Temperature Monitoring', 'System Efficiency',
                'String Voltage Distribution', 'String Current Distribution',
                'Power Quality (Frequency)', 'Data Quality Metrics'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"type": "box"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08
        )
        
        df = result.cleaned_dataset
        
        # System power output over time
        if 'Time' in df.columns and 'Pac(W)' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Time'], y=df['Pac(W)'], name='AC Power Output',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # String performance comparison
        string_powers = []
        string_labels = []
        for perf in result.string_performance:
            string_powers.append(perf.avg_power)
            string_labels.append(f'String {perf.string_id}')
        
        fig.add_trace(
            go.Bar(x=string_labels[:10], y=string_powers[:10], name='String Power',
                   marker_color='green'),
            row=1, col=2
        )
        
        # Temperature monitoring
        temp_cols = [col for col in ['INVTemp(℃)', 'AMTemp1(℃)', 'BTTemp(℃)'] 
                    if col in df.columns]
        if temp_cols:
            for temp_col in temp_cols:
                fig.add_trace(
                    go.Scatter(x=df['Time'], y=df[temp_col], name=temp_col,
                              mode='lines'),
                    row=2, col=1
                )
        
        # System efficiency over time
        if 'Conversion_Efficiency(%)' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Time'], y=df['Conversion_Efficiency(%)'],
                          name='Conversion Efficiency', line=dict(color='orange')),
                row=2, col=2
            )
        
        # String voltage distribution
        voltage_data = []
        for i in range(1, 11):  # First 10 strings
            col = f'Vstr{i}(V)'
            if col in df.columns:
                voltage_data.append(df[col].dropna().values)
        
        if voltage_data:
            for i, data in enumerate(voltage_data):
                fig.add_trace(
                    go.Box(y=data, name=f'String {i+1}', showlegend=False),
                    row=3, col=1
                )
        
        # String current distribution
        current_data = []
        for i in range(1, 11):  # First 10 strings
            col = f'Istr{i}(A)'
            if col in df.columns:
                current_data.append(df[col].dropna().values)
        
        if current_data:
            for i, data in enumerate(current_data):
                fig.add_trace(
                    go.Box(y=data, name=f'String {i+1}', showlegend=False),
                    row=3, col=2
                )
        
        # Frequency monitoring
        if 'Fac(Hz)' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Time'], y=df['Fac(Hz)'], name='Grid Frequency',
                          line=dict(color='red')),
                row=4, col=1
            )
        
        # Data quality indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=result.data_quality.data_quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Quality Score"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "darkblue"},
                      'steps': [{'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 80], 'color': "yellow"},
                               {'range': [80, 100], 'color': "green"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title={
                'text': f"Comprehensive Inverter Data Analysis Dashboard<br>"
                       f"Data Quality: {result.data_quality.data_quality_score:.1f}% | "
                       f"System Efficiency: {result.system_performance.system_efficiency:.1f}% | "
                       f"Active Strings: {len(result.string_performance)}",
                'x': 0.5,
                'font': {'size': 16}
            },
            showlegend=True,
            template="plotly_white"
        )
        
        # Save dashboard
        fig.write_html(output_file)
        self.logger.info(f"📊 Interactive dashboard created: {output_file}")
        
        return output_file

def main():
    """Main execution function for testing"""
    # Example usage
    cleaner = InverterDataCleaner()
    
    # Process a real inverter data file
    file_path = "inverter/INVERTER_01_2025-04-04_2025-04-05.csv"
    
    try:
        result = cleaner.clean_inverter_data(file_path)
        
        print(f"\n🎉 Data Cleaning Completed Successfully!")
        print(f"📊 Data Quality Score: {result.data_quality.data_quality_score:.1f}%")
        print(f"🔌 Active Strings: {len(result.string_performance)}")
        print(f"⚡ System Efficiency: {result.system_performance.system_efficiency:.1f}%")
        print(f"📁 Export Files: {len(result.export_files)}")
        
        # Create dashboard
        dashboard_file = cleaner.create_analysis_dashboard(result)
        print(f"📈 Dashboard Created: {dashboard_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()
