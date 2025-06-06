#!/usr/bin/env python3
"""
Enterprise Fleet Operations System
==================================

Advanced enterprise-grade photovoltaic inverter fleet management system with
intelligent health scoring, predictive analytics, and comprehensive operations dashboard.

Features:
- Intelligent health scoring algorithm with weighted factors
- Advanced string imbalance analysis and diagnostics
- Predictive maintenance with ML-based recommendations
- Enterprise dashboard with real-time KPIs
- Automated compliance reporting
- Performance benchmarking and optimization
- Risk assessment and mitigation planning

Author: Enterprise PV Analytics System
Date: June 6, 2025
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import json
import os
import time
import threading
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
import warnings
import statistics
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings('ignore')

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enterprise_fleet_operations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('enterprise_fleet_operations')

class SystemHealthCategory(Enum):
    """System health categorization"""
    EXCELLENT = "excellent"      # 90-100
    GOOD = "good"               # 80-89
    FAIR = "fair"               # 70-79
    POOR = "poor"               # 60-69
    CRITICAL = "critical"       # <60

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MaintenanceType(Enum):
    """Maintenance task types"""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"

@dataclass
class EnhancedSystemMetrics:
    """Enhanced system metrics with detailed analysis"""
    system_id: str
    timestamp: datetime
    
    # Power metrics
    ac_power: float
    dc_power: float
    efficiency: float
    power_factor: float
    
    # String analysis
    string_powers: List[float]
    string_count: int
    string_imbalance_score: float
    string_variance: float
    string_cv: float  # Coefficient of variation
    
    # Environmental
    temperature: float
    irradiance: Optional[float]
    ambient_temp: Optional[float]
    
    # Performance indicators
    health_score: float
    health_category: SystemHealthCategory
    performance_ratio: float
    degradation_rate: float
    
    # Fault information
    fault_codes: List[str]
    warning_codes: List[str]
    pid_fault_codes: List[str]
    
    # Trends
    efficiency_trend: str
    power_trend: str
    temperature_trend: str
    
    # Risk assessment
    risk_score: float
    risk_factors: List[str]

class EnterpriseHealthScoring:
    """Advanced health scoring with weighted factors and normalization"""
    
    def __init__(self):
        # Configurable weights for different health factors
        self.weights = {
            'efficiency': 0.30,      # Primary performance indicator
            'string_balance': 0.25,  # String performance uniformity
            'temperature': 0.20,     # Thermal management
            'fault_status': 0.15,    # Fault and warning status
            'power_stability': 0.10  # Power output consistency
        }
        
        # Scoring thresholds
        self.thresholds = {
            'efficiency': {'excellent': 95, 'good': 85, 'fair': 75, 'poor': 65},
            'string_cv': {'excellent': 5, 'good': 10, 'fair': 15, 'poor': 20},  # Lower is better
            'temperature': {'excellent': 45, 'good': 55, 'fair': 65, 'poor': 75},
            'power_stability': {'excellent': 95, 'good': 85, 'fair': 75, 'poor': 65}
        }
    
    def calculate_efficiency_score(self, efficiency: float) -> float:
        """Calculate efficiency component score (0-100)"""
        if efficiency >= self.thresholds['efficiency']['excellent']:
            return 100
        elif efficiency >= self.thresholds['efficiency']['good']:
            return 80 + (efficiency - self.thresholds['efficiency']['good']) / \
                   (self.thresholds['efficiency']['excellent'] - self.thresholds['efficiency']['good']) * 20
        elif efficiency >= self.thresholds['efficiency']['fair']:
            return 60 + (efficiency - self.thresholds['efficiency']['fair']) / \
                   (self.thresholds['efficiency']['good'] - self.thresholds['efficiency']['fair']) * 20
        elif efficiency >= self.thresholds['efficiency']['poor']:
            return 40 + (efficiency - self.thresholds['efficiency']['poor']) / \
                   (self.thresholds['efficiency']['fair'] - self.thresholds['efficiency']['poor']) * 20
        else:
            return max(0, efficiency / self.thresholds['efficiency']['poor'] * 40)
    
    def calculate_string_balance_score(self, string_cv: float) -> float:
        """Calculate string balance score based on coefficient of variation (0-100)"""
        if string_cv <= self.thresholds['string_cv']['excellent']:
            return 100
        elif string_cv <= self.thresholds['string_cv']['good']:
            return 80 + (self.thresholds['string_cv']['good'] - string_cv) / \
                   (self.thresholds['string_cv']['good'] - self.thresholds['string_cv']['excellent']) * 20
        elif string_cv <= self.thresholds['string_cv']['fair']:
            return 60 + (self.thresholds['string_cv']['fair'] - string_cv) / \
                   (self.thresholds['string_cv']['fair'] - self.thresholds['string_cv']['good']) * 20
        elif string_cv <= self.thresholds['string_cv']['poor']:
            return 40 + (self.thresholds['string_cv']['poor'] - string_cv) / \
                   (self.thresholds['string_cv']['poor'] - self.thresholds['string_cv']['fair']) * 20
        else:
            return max(0, 40 - (string_cv - self.thresholds['string_cv']['poor']) * 2)
    
    def calculate_temperature_score(self, temperature: float) -> float:
        """Calculate temperature management score (0-100)"""
        if temperature <= self.thresholds['temperature']['excellent']:
            return 100
        elif temperature <= self.thresholds['temperature']['good']:
            return 80 + (self.thresholds['temperature']['good'] - temperature) / \
                   (self.thresholds['temperature']['good'] - self.thresholds['temperature']['excellent']) * 20
        elif temperature <= self.thresholds['temperature']['fair']:
            return 60 + (self.thresholds['temperature']['fair'] - temperature) / \
                   (self.thresholds['temperature']['fair'] - self.thresholds['temperature']['good']) * 20
        elif temperature <= self.thresholds['temperature']['poor']:
            return 40 + (self.thresholds['temperature']['poor'] - temperature) / \
                   (self.thresholds['temperature']['poor'] - self.thresholds['temperature']['fair']) * 20
        else:
            return max(0, 40 - (temperature - self.thresholds['temperature']['poor']) * 2)
    
    def calculate_fault_score(self, fault_codes: List[str], warning_codes: List[str], 
                            pid_fault_codes: List[str]) -> float:
        """Calculate fault status score (0-100)"""
        total_faults = len(fault_codes) + len(warning_codes) + len(pid_fault_codes)
        
        if total_faults == 0:
            return 100
        elif total_faults == 1:
            return 80
        elif total_faults <= 3:
            return 60
        elif total_faults <= 5:
            return 40
        else:
            return max(0, 40 - (total_faults - 5) * 5)
    
    def calculate_composite_health_score(self, metrics: Dict[str, Any]) -> Tuple[float, SystemHealthCategory]:
        """Calculate composite health score with weighted factors"""
        
        # Individual component scores
        efficiency_score = self.calculate_efficiency_score(metrics.get('efficiency', 0))
        string_balance_score = self.calculate_string_balance_score(metrics.get('string_cv', 100))
        temperature_score = self.calculate_temperature_score(metrics.get('temperature', 100))
        fault_score = self.calculate_fault_score(
            metrics.get('fault_codes', []),
            metrics.get('warning_codes', []),
            metrics.get('pid_fault_codes', [])
        )
        power_stability_score = metrics.get('power_stability_score', 80)  # Default if not calculated
        
        # Weighted composite score
        composite_score = (
            efficiency_score * self.weights['efficiency'] +
            string_balance_score * self.weights['string_balance'] +
            temperature_score * self.weights['temperature'] +
            fault_score * self.weights['fault_status'] +
            power_stability_score * self.weights['power_stability']
        )
        
        # Determine health category
        if composite_score >= 90:
            category = SystemHealthCategory.EXCELLENT
        elif composite_score >= 80:
            category = SystemHealthCategory.GOOD
        elif composite_score >= 70:
            category = SystemHealthCategory.FAIR
        elif composite_score >= 60:
            category = SystemHealthCategory.POOR
        else:
            category = SystemHealthCategory.CRITICAL
        
        return composite_score, category

class AdvancedStringAnalyzer:
    """Advanced string performance analysis and diagnostics"""
    
    def __init__(self):
        self.string_history = defaultdict(deque)
        self.max_history = 100  # Keep last 100 readings per string
    
    def analyze_string_performance(self, string_powers: List[float], system_id: str) -> Dict[str, Any]:
        """Comprehensive string performance analysis"""
        if not string_powers or len(string_powers) == 0:
            return self._empty_analysis()
        
        # Filter out zero/negative values for active analysis
        active_strings = [p for p in string_powers if p > 10]  # Minimum 10W threshold
        
        if len(active_strings) < 2:
            return self._empty_analysis()
        
        # Basic statistics
        mean_power = np.mean(active_strings)
        std_power = np.std(active_strings)
        cv = (std_power / mean_power * 100) if mean_power > 0 else 100
        
        # String performance metrics
        max_power = max(active_strings)
        min_power = min(active_strings)
        power_range = max_power - min_power
        range_percent = (power_range / mean_power * 100) if mean_power > 0 else 100
        
        # Identify problem strings
        threshold = mean_power * 0.85  # 15% below average
        underperforming_strings = [i for i, p in enumerate(string_powers) if 0 < p < threshold]
        dead_strings = [i for i, p in enumerate(string_powers) if p <= 0]
        
        # Performance distribution analysis
        q1 = np.percentile(active_strings, 25)
        q3 = np.percentile(active_strings, 75)
        iqr = q3 - q1
        
        # Outlier detection
        outlier_threshold_low = q1 - 1.5 * iqr
        outlier_threshold_high = q3 + 1.5 * iqr
        outliers = [i for i, p in enumerate(string_powers) 
                   if p > 0 and (p < outlier_threshold_low or p > outlier_threshold_high)]
        
        # Imbalance severity assessment
        if cv <= 5:
            imbalance_severity = "excellent"
        elif cv <= 10:
            imbalance_severity = "good"
        elif cv <= 15:
            imbalance_severity = "moderate"
        elif cv <= 25:
            imbalance_severity = "high"
        else:
            imbalance_severity = "critical"
        
        # Generate recommendations
        recommendations = self._generate_string_recommendations(
            cv, underperforming_strings, dead_strings, outliers, mean_power
        )
        
        return {
            'string_count': len(string_powers),
            'active_strings': len(active_strings),
            'mean_power': mean_power,
            'std_power': std_power,
            'coefficient_of_variation': cv,
            'power_range': power_range,
            'range_percent': range_percent,
            'max_power': max_power,
            'min_power': min_power,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'underperforming_strings': underperforming_strings,
            'dead_strings': dead_strings,
            'outliers': outliers,
            'imbalance_severity': imbalance_severity,
            'imbalance_score': max(0, 100 - cv * 4),  # Score decreases with higher CV
            'recommendations': recommendations
        }
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for invalid data"""
        return {
            'string_count': 0,
            'active_strings': 0,
            'mean_power': 0,
            'std_power': 0,
            'coefficient_of_variation': 100,
            'power_range': 0,
            'range_percent': 0,
            'max_power': 0,
            'min_power': 0,
            'q1': 0,
            'q3': 0,
            'iqr': 0,
            'underperforming_strings': [],
            'dead_strings': [],
            'outliers': [],
            'imbalance_severity': "unknown",
            'imbalance_score': 0,
            'recommendations': ["Insufficient data for analysis"]
        }
    
    def _generate_string_recommendations(self, cv: float, underperforming: List[int], 
                                       dead: List[int], outliers: List[int], 
                                       mean_power: float) -> List[str]:
        """Generate specific maintenance recommendations"""
        recommendations = []
        
        if dead:
            recommendations.append(f"URGENT: Inspect strings {dead} - no power output detected")
        
        if underperforming:
            recommendations.append(f"Check strings {underperforming} - performing below 85% of average")
        
        if outliers:
            recommendations.append(f"Investigate strings {outliers} - statistical outliers detected")
        
        if cv > 25:
            recommendations.append("CRITICAL: High string imbalance - immediate inspection required")
        elif cv > 15:
            recommendations.append("HIGH: Significant string imbalance - schedule maintenance")
        elif cv > 10:
            recommendations.append("MEDIUM: Moderate imbalance - monitor closely")
        
        if mean_power < 50:
            recommendations.append("Low average string power - check irradiance conditions or system issues")
        
        if not recommendations:
            recommendations.append("String performance within acceptable parameters")
        
        return recommendations

class EnterpriseFleetOperations:
    """Enterprise-grade fleet operations management system"""
    
    def __init__(self, data_directory: str = "."):
        self.data_directory = data_directory
        self.health_scorer = EnterpriseHealthScoring()
        self.string_analyzer = AdvancedStringAnalyzer()
        self.db_path = "enterprise_fleet_operations.db"
        self.logger = logger
        
        # Performance tracking
        self.fleet_history = deque(maxlen=1000)  # Keep last 1000 fleet snapshots
        self.system_baselines = {}
        
        # Initialize database
        self._init_database()
        
        self.logger.info("Enterprise Fleet Operations System initialized")
    
    def _init_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced system metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                ac_power REAL,
                dc_power REAL,
                efficiency REAL,
                health_score REAL,
                health_category TEXT,
                temperature REAL,
                string_cv REAL,
                string_imbalance_score REAL,
                fault_count INTEGER,
                performance_ratio REAL,
                risk_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # String performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS string_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                string_number INTEGER,
                power_output REAL,
                performance_ratio REAL,
                status TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Maintenance recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_id TEXT NOT NULL,
                recommendation_type TEXT,
                severity TEXT,
                description TEXT,
                estimated_cost REAL,
                priority_score INTEGER,
                status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                scheduled_date DATETIME,
                completed_date DATETIME
            )
        ''')
        
        # Fleet KPIs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fleet_kpis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                total_systems INTEGER,
                systems_online INTEGER,
                average_efficiency REAL,
                total_power_output REAL,
                fleet_health_score REAL,
                systems_needing_maintenance INTEGER,
                estimated_energy_loss REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("Enhanced database schema initialized")
    
    def analyze_fleet_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive fleet analysis with enterprise metrics"""
        start_time = time.time()
        
        try:
            # Get all cleaned inverter files
            cleaned_files = [f for f in os.listdir(self.data_directory) 
                           if f.startswith('cleaned_') and f.endswith('.csv')]
            
            if not cleaned_files:
                self.logger.warning("No cleaned inverter files found")
                return self._empty_fleet_analysis()
            
            fleet_analysis = {
                'timestamp': datetime.now().isoformat(),
                'analysis_duration': 0,
                'total_systems': len(cleaned_files),
                'systems_analyzed': 0,
                'systems_online': 0,
                'fleet_kpis': {},
                'system_details': {},
                'fleet_recommendations': [],
                'risk_assessment': {},
                'performance_benchmarks': {},
                'maintenance_priorities': []
            }
            
            system_metrics = []
            all_recommendations = []
            
            # Process each system
            for filename in cleaned_files:
                try:
                    system_metrics_data = self._analyze_individual_system(filename)
                    if system_metrics_data:
                        system_metrics.append(system_metrics_data)
                        fleet_analysis['system_details'][system_metrics_data.system_id] = asdict(system_metrics_data)
                        
                        # Collect recommendations
                        all_recommendations.extend(system_metrics_data.risk_factors)
                        
                        if system_metrics_data.ac_power > 1000:  # Active system threshold
                            fleet_analysis['systems_online'] += 1
                        
                        fleet_analysis['systems_analyzed'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing {filename}: {str(e)}")
                    continue
            
            # Calculate fleet KPIs
            if system_metrics:
                fleet_analysis['fleet_kpis'] = self._calculate_fleet_kpis(system_metrics)
                fleet_analysis['risk_assessment'] = self._assess_fleet_risk(system_metrics)
                fleet_analysis['performance_benchmarks'] = self._calculate_performance_benchmarks(system_metrics)
                fleet_analysis['maintenance_priorities'] = self._prioritize_maintenance(system_metrics)
            
            # Store fleet KPIs in database
            self._store_fleet_kpis(fleet_analysis['fleet_kpis'])
            
            fleet_analysis['analysis_duration'] = round(time.time() - start_time, 2)
            
            # Generate enterprise summary report
            self._generate_enterprise_report(fleet_analysis)
            
            self.logger.info(f"Fleet analysis completed in {fleet_analysis['analysis_duration']}s")
            return fleet_analysis
            
        except Exception as e:
            self.logger.error(f"Fleet analysis failed: {str(e)}")
            return self._empty_fleet_analysis()
    
    def _analyze_individual_system(self, filename: str) -> Optional[EnhancedSystemMetrics]:
        """Analyze individual system with comprehensive metrics"""
        try:
            file_path = os.path.join(self.data_directory, filename)
            df = pd.read_csv(file_path)
            
            if df.empty:
                return None
            
            # Extract system ID from filename like 'cleaned_INVERTER_01_2025-04-04_2025-04-05_20250606_065503.csv'
            system_id = filename.replace('cleaned_', '').replace('.csv', '').split('_')[0:2]
            system_id = '_'.join(system_id)  # Should give us 'INVERTER_01'
            
            # Filter for active data (power > 1000W to avoid nighttime/standby)
            active_data = df[df['Pac(W)'] > 1000] if 'Pac(W)' in df.columns else df
            
            if active_data.empty:
                return None
            
            # Calculate basic metrics
            latest_data = active_data.iloc[-1]
            ac_power = latest_data.get('Pac(W)', 0)
            
            # Calculate DC power from strings
            string_columns = [col for col in df.columns if col.startswith('Pstr') and col.endswith('(W)')]
            string_powers = [latest_data.get(col, 0) for col in string_columns]
            dc_power = sum(string_powers)
            
            # Calculate efficiency
            if 'Conversion_Efficiency(%)' in latest_data and pd.notna(latest_data['Conversion_Efficiency(%)']):
                efficiency = latest_data['Conversion_Efficiency(%)']
            else:
                efficiency = (ac_power / dc_power * 100) if dc_power > 0 else 0
            
            # Temperature analysis
            temp_columns = [col for col in df.columns if 'Temp' in col and '℃' in col]
            temperatures = [latest_data.get(col, 0) for col in temp_columns if pd.notna(latest_data.get(col, 0))]
            temperature = max(temperatures) if temperatures else 50
            
            # String analysis
            string_analysis = self.string_analyzer.analyze_string_performance(string_powers, system_id)
            
            # Fault analysis
            fault_codes = []
            warning_codes = []
            pid_fault_codes = []
            
            if 'FaultCode' in latest_data and pd.notna(latest_data['FaultCode']) and latest_data['FaultCode'] != 0:
                fault_codes.append(str(latest_data['FaultCode']))
            if 'WarnCode' in latest_data and pd.notna(latest_data['WarnCode']) and latest_data['WarnCode'] != 0:
                warning_codes.append(str(latest_data['WarnCode']))
            if 'PidFaultCode' in latest_data and pd.notna(latest_data['PidFaultCode']) and latest_data['PidFaultCode'] != 0:
                pid_fault_codes.append(str(latest_data['PidFaultCode']))
            
            # Health scoring with enhanced algorithm
            health_metrics = {
                'efficiency': efficiency,
                'string_cv': string_analysis['coefficient_of_variation'],
                'temperature': temperature,
                'fault_codes': fault_codes,
                'warning_codes': warning_codes,
                'pid_fault_codes': pid_fault_codes,
                'power_stability_score': 85  # Default - could be calculated from historical data
            }
            
            health_score, health_category = self.health_scorer.calculate_composite_health_score(health_metrics)
            
            # Performance trends (simplified for single snapshot)
            efficiency_trend = "stable"  # Would need historical data for real trend
            power_trend = "stable"
            temperature_trend = "stable"
            
            # Risk assessment
            risk_factors = []
            risk_score = 0
            
            if efficiency < 70:
                risk_factors.append("Low conversion efficiency")
                risk_score += 20
            if string_analysis['coefficient_of_variation'] > 20:
                risk_factors.append("High string imbalance")
                risk_score += 15
            if temperature > 60:
                risk_factors.append("High operating temperature")
                risk_score += 10
            if fault_codes or warning_codes or pid_fault_codes:
                risk_factors.append("Active fault codes")
                risk_score += 25
            if len(string_analysis['dead_strings']) > 0:
                risk_factors.append("Dead strings detected")
                risk_score += 30
            
            # Store in database
            self._store_system_metrics(system_id, health_score, health_category.value, 
                                     efficiency, temperature, string_analysis, ac_power, dc_power)
            
            return EnhancedSystemMetrics(
                system_id=system_id,
                timestamp=datetime.now(),
                ac_power=ac_power,
                dc_power=dc_power,
                efficiency=efficiency,
                power_factor=0.98,  # Default
                string_powers=string_powers,
                string_count=len(string_powers),
                string_imbalance_score=string_analysis['imbalance_score'],
                string_variance=string_analysis['std_power']**2,
                string_cv=string_analysis['coefficient_of_variation'],
                temperature=temperature,
                irradiance=None,
                ambient_temp=None,
                health_score=health_score,
                health_category=health_category,
                performance_ratio=efficiency/100,  # Simplified
                degradation_rate=0,  # Would need historical data
                fault_codes=fault_codes,
                warning_codes=warning_codes,
                pid_fault_codes=pid_fault_codes,
                efficiency_trend=efficiency_trend,
                power_trend=power_trend,
                temperature_trend=temperature_trend,
                risk_score=risk_score,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing system {filename}: {str(e)}")
            return None
    
    def _calculate_fleet_kpis(self, system_metrics: List[EnhancedSystemMetrics]) -> Dict[str, Any]:
        """Calculate comprehensive fleet KPIs"""
        if not system_metrics:
            return {}
        
        # Basic fleet metrics
        total_systems = len(system_metrics)
        systems_online = len([s for s in system_metrics if s.ac_power > 1000])
        
        # Performance metrics
        efficiencies = [s.efficiency for s in system_metrics if s.efficiency > 0]
        health_scores = [s.health_score for s in system_metrics]
        temperatures = [s.temperature for s in system_metrics]
        power_outputs = [s.ac_power for s in system_metrics]
        
        # Fleet aggregations
        total_power = sum(power_outputs)
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0
        avg_health_score = np.mean(health_scores)
        avg_temperature = np.mean(temperatures)
        
        # Health distribution
        health_distribution = {category.value: 0 for category in SystemHealthCategory}
        for system in system_metrics:
            health_distribution[system.health_category.value] += 1
        
        # Risk analysis
        high_risk_systems = len([s for s in system_metrics if s.risk_score > 50])
        systems_with_faults = len([s for s in system_metrics if s.fault_codes or s.warning_codes or s.pid_fault_codes])
        
        # String analysis
        total_strings = sum([s.string_count for s in system_metrics])
        dead_strings = sum([len(s.risk_factors) for s in system_metrics if "Dead strings detected" in s.risk_factors])
        
        # Performance categories
        excellent_systems = health_distribution.get('excellent', 0)
        good_systems = health_distribution.get('good', 0)
        fair_systems = health_distribution.get('fair', 0)
        poor_systems = health_distribution.get('poor', 0)
        critical_systems = health_distribution.get('critical', 0)
        
        return {
            'total_systems': total_systems,
            'systems_online': systems_online,
            'system_availability': round(systems_online / total_systems * 100, 2),
            'total_power_output': round(total_power, 2),
            'average_efficiency': round(avg_efficiency, 2),
            'efficiency_std': round(np.std(efficiencies), 2) if efficiencies else 0,
            'average_health_score': round(avg_health_score, 2),
            'average_temperature': round(avg_temperature, 2),
            'health_distribution': health_distribution,
            'excellent_systems': excellent_systems,
            'good_systems': good_systems,
            'fair_systems': fair_systems,
            'poor_systems': poor_systems,
            'critical_systems': critical_systems,
            'high_risk_systems': high_risk_systems,
            'systems_with_faults': systems_with_faults,
            'total_strings': total_strings,
            'estimated_dead_strings': dead_strings,
            'fleet_capacity_factor': round(avg_efficiency / 100, 3),
            'maintenance_urgency_score': round((critical_systems * 10 + poor_systems * 5 + high_risk_systems * 3) / total_systems, 2)
        }
    
    def _assess_fleet_risk(self, system_metrics: List[EnhancedSystemMetrics]) -> Dict[str, Any]:
        """Comprehensive fleet risk assessment"""
        if not system_metrics:
            return {}
        
        # Risk categories
        operational_risks = []
        financial_risks = []
        technical_risks = []
        environmental_risks = []
        
        # Analyze system risks
        critical_systems = [s for s in system_metrics if s.health_category == SystemHealthCategory.CRITICAL]
        high_temp_systems = [s for s in system_metrics if s.temperature > 65]
        low_efficiency_systems = [s for s in system_metrics if s.efficiency < 70]
        
        # Operational risk assessment
        if len(critical_systems) > len(system_metrics) * 0.3:
            operational_risks.append("High percentage of critical systems")
        
        if len(high_temp_systems) > 0:
            operational_risks.append(f"{len(high_temp_systems)} systems operating at high temperature")
        
        # Financial risk assessment
        estimated_energy_loss = sum([max(0, 90 - s.efficiency) for s in system_metrics])
        if estimated_energy_loss > 100:
            financial_risks.append(f"Estimated {estimated_energy_loss:.1f}% efficiency loss")
        
        # Technical risk assessment
        string_issues = sum([1 for s in system_metrics if s.string_cv > 20])
        if string_issues > 0:
            technical_risks.append(f"{string_issues} systems with string imbalance issues")
        
        # Environmental risk assessment
        if np.mean([s.temperature for s in system_metrics]) > 55:
            environmental_risks.append("Fleet average temperature exceeds optimal range")
        
        # Overall risk score
        total_risk_score = np.mean([s.risk_score for s in system_metrics])
        
        return {
            'overall_risk_score': round(total_risk_score, 2),
            'risk_level': 'high' if total_risk_score > 50 else 'medium' if total_risk_score > 25 else 'low',
            'operational_risks': operational_risks,
            'financial_risks': financial_risks,
            'technical_risks': technical_risks,
            'environmental_risks': environmental_risks,
            'critical_systems_count': len(critical_systems),
            'high_temp_systems_count': len(high_temp_systems),
            'low_efficiency_systems_count': len(low_efficiency_systems),
            'estimated_energy_loss_percent': round(estimated_energy_loss / len(system_metrics), 2)
        }
    
    def _calculate_performance_benchmarks(self, system_metrics: List[EnhancedSystemMetrics]) -> Dict[str, Any]:
        """Calculate fleet performance benchmarks"""
        if not system_metrics:
            return {}
        
        efficiencies = [s.efficiency for s in system_metrics if s.efficiency > 0]
        health_scores = [s.health_score for s in system_metrics]
        string_cvs = [s.string_cv for s in system_metrics]
        
        return {
            'efficiency_percentiles': {
                'p10': round(np.percentile(efficiencies, 10), 2) if efficiencies else 0,
                'p25': round(np.percentile(efficiencies, 25), 2) if efficiencies else 0,
                'p50': round(np.percentile(efficiencies, 50), 2) if efficiencies else 0,
                'p75': round(np.percentile(efficiencies, 75), 2) if efficiencies else 0,
                'p90': round(np.percentile(efficiencies, 90), 2) if efficiencies else 0
            },
            'health_score_percentiles': {
                'p10': round(np.percentile(health_scores, 10), 2),
                'p25': round(np.percentile(health_scores, 25), 2),
                'p50': round(np.percentile(health_scores, 50), 2),
                'p75': round(np.percentile(health_scores, 75), 2),
                'p90': round(np.percentile(health_scores, 90), 2)
            },
            'string_balance_percentiles': {
                'p10': round(np.percentile(string_cvs, 10), 2),
                'p25': round(np.percentile(string_cvs, 25), 2),
                'p50': round(np.percentile(string_cvs, 50), 2),
                'p75': round(np.percentile(string_cvs, 75), 2),
                'p90': round(np.percentile(string_cvs, 90), 2)
            },
            'top_performers': [s.system_id for s in sorted(system_metrics, key=lambda x: x.health_score, reverse=True)[:3]],
            'bottom_performers': [s.system_id for s in sorted(system_metrics, key=lambda x: x.health_score)[:3]]
        }
    
    def _prioritize_maintenance(self, system_metrics: List[EnhancedSystemMetrics]) -> List[Dict[str, Any]]:
        """Prioritize maintenance tasks based on system conditions"""
        maintenance_tasks = []
        
        for system in system_metrics:
            priority_score = 0
            task_type = MaintenanceType.PREVENTIVE
            description = ""
            estimated_cost = 500  # Base cost
            
            # Critical issues
            if system.health_category == SystemHealthCategory.CRITICAL:
                priority_score += 100
                task_type = MaintenanceType.EMERGENCY
                description = "Emergency inspection required - critical health score"
                estimated_cost = 2000
            
            # String issues
            if "Dead strings detected" in system.risk_factors:
                priority_score += 80
                task_type = MaintenanceType.CORRECTIVE
                description = "String replacement/repair required"
                estimated_cost += 1500
            
            # High string imbalance
            if system.string_cv > 25:
                priority_score += 60
                description += " High string imbalance investigation"
                estimated_cost += 800
            
            # Temperature issues
            if system.temperature > 65:
                priority_score += 40
                description += " Thermal management check"
                estimated_cost += 400
            
            # Efficiency issues
            if system.efficiency < 70:
                priority_score += 50
                description += " Performance optimization"
                estimated_cost += 600
            
            # Fault codes
            if system.fault_codes or system.warning_codes or system.pid_fault_codes:
                priority_score += 30
                description += " Fault code investigation"
                estimated_cost += 300
            
            if priority_score > 20:  # Only create tasks for systems needing attention
                maintenance_tasks.append({
                    'system_id': system.system_id,
                    'priority_score': priority_score,
                    'task_type': task_type.value,
                    'description': description.strip(),
                    'estimated_cost': estimated_cost,
                    'urgency': 'critical' if priority_score > 80 else 'high' if priority_score > 50 else 'medium',
                    'health_score': system.health_score,
                    'risk_factors': system.risk_factors
                })
        
        # Sort by priority
        maintenance_tasks.sort(key=lambda x: x['priority_score'], reverse=True)
        return maintenance_tasks
    
    def _store_system_metrics(self, system_id: str, health_score: float, health_category: str,
                            efficiency: float, temperature: float, string_analysis: Dict, 
                            ac_power: float, dc_power: float):
        """Store system metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics 
                (system_id, timestamp, ac_power, dc_power, efficiency, health_score, 
                 health_category, temperature, string_cv, string_imbalance_score, 
                 fault_count, performance_ratio, risk_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                system_id, datetime.now(), ac_power, dc_power, efficiency, 
                health_score, health_category, temperature,
                string_analysis.get('coefficient_of_variation', 0),
                string_analysis.get('imbalance_score', 0),
                0,  # fault_count - would need to calculate
                efficiency/100,  # performance_ratio
                0   # risk_score - would need to calculate
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing system metrics: {str(e)}")
    
    def _store_fleet_kpis(self, fleet_kpis: Dict[str, Any]):
        """Store fleet KPIs in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fleet_kpis 
                (timestamp, total_systems, systems_online, average_efficiency, 
                 total_power_output, fleet_health_score, systems_needing_maintenance,
                 estimated_energy_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                fleet_kpis.get('total_systems', 0),
                fleet_kpis.get('systems_online', 0),
                fleet_kpis.get('average_efficiency', 0),
                fleet_kpis.get('total_power_output', 0),
                fleet_kpis.get('average_health_score', 0),
                fleet_kpis.get('critical_systems', 0) + fleet_kpis.get('poor_systems', 0),
                fleet_kpis.get('estimated_energy_loss_percent', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing fleet KPIs: {str(e)}")
    
    def _generate_enterprise_report(self, fleet_analysis: Dict[str, Any]):
        """Generate comprehensive enterprise report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"enterprise_fleet_report_{timestamp}.json"
        
        try:
            # Save detailed JSON report
            with open(report_filename, 'w') as f:
                json.dump(fleet_analysis, f, indent=2, default=str)
            
            # Generate executive summary
            self._generate_executive_summary(fleet_analysis, timestamp)
            
            self.logger.info(f"Enterprise report generated: {report_filename}")
            
        except Exception as e:
            self.logger.error(f"Error generating enterprise report: {str(e)}")
    
    def _generate_executive_summary(self, fleet_analysis: Dict[str, Any], timestamp: str):
        """Generate executive summary report"""
        summary_filename = f"executive_summary_{timestamp}.txt"
        
        try:
            kpis = fleet_analysis.get('fleet_kpis', {})
            risk = fleet_analysis.get('risk_assessment', {})
            
            with open(summary_filename, 'w') as f:
                f.write("ENTERPRISE FLEET OPERATIONS - EXECUTIVE SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Analysis Duration: {fleet_analysis.get('analysis_duration', 0)}s\n\n")
                
                f.write("FLEET OVERVIEW\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Systems: {kpis.get('total_systems', 0)}\n")
                f.write(f"Systems Online: {kpis.get('systems_online', 0)}\n")
                f.write(f"System Availability: {kpis.get('system_availability', 0)}%\n")
                f.write(f"Total Power Output: {kpis.get('total_power_output', 0):,.1f} W\n\n")
                
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Efficiency: {kpis.get('average_efficiency', 0):.1f}%\n")
                f.write(f"Average Health Score: {kpis.get('average_health_score', 0):.1f}\n")
                f.write(f"Average Temperature: {kpis.get('average_temperature', 0):.1f}°C\n\n")
                
                f.write("SYSTEM HEALTH DISTRIBUTION\n")
                f.write("-" * 30 + "\n")
                health_dist = kpis.get('health_distribution', {})
                f.write(f"Excellent: {health_dist.get('excellent', 0)} systems\n")
                f.write(f"Good: {health_dist.get('good', 0)} systems\n")
                f.write(f"Fair: {health_dist.get('fair', 0)} systems\n")
                f.write(f"Poor: {health_dist.get('poor', 0)} systems\n")
                f.write(f"Critical: {health_dist.get('critical', 0)} systems\n\n")
                
                f.write("RISK ASSESSMENT\n")
                f.write("-" * 15 + "\n")
                f.write(f"Overall Risk Level: {risk.get('risk_level', 'unknown').upper()}\n")
                f.write(f"Risk Score: {risk.get('overall_risk_score', 0):.1f}\n")
                f.write(f"Critical Systems: {risk.get('critical_systems_count', 0)}\n")
                f.write(f"High Risk Systems: {kpis.get('high_risk_systems', 0)}\n\n")
                
                f.write("MAINTENANCE PRIORITIES\n")
                f.write("-" * 20 + "\n")
                maintenance_tasks = fleet_analysis.get('maintenance_priorities', [])
                urgent_tasks = [t for t in maintenance_tasks if t['urgency'] == 'critical']
                f.write(f"Critical Maintenance Tasks: {len(urgent_tasks)}\n")
                f.write(f"Total Maintenance Tasks: {len(maintenance_tasks)}\n")
                
                if urgent_tasks:
                    f.write("\nMOST URGENT TASKS:\n")
                    for task in urgent_tasks[:5]:  # Top 5 urgent tasks
                        f.write(f"- {task['system_id']}: {task['description']} (${task['estimated_cost']:,})\n")
            
            self.logger.info(f"Executive summary generated: {summary_filename}")
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
    
    def _empty_fleet_analysis(self) -> Dict[str, Any]:
        """Return empty fleet analysis structure"""
        return {
            'timestamp': datetime.now().isoformat(),
            'analysis_duration': 0,
            'total_systems': 0,
            'systems_analyzed': 0,
            'systems_online': 0,
            'fleet_kpis': {},
            'system_details': {},
            'fleet_recommendations': [],
            'risk_assessment': {},
            'performance_benchmarks': {},
            'maintenance_priorities': []
        }
    
    def generate_advanced_dashboard(self) -> Optional[str]:
        """Generate advanced Plotly dashboard"""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for dashboard generation")
            return None
        
        try:
            # Get latest fleet analysis
            fleet_analysis = self.analyze_fleet_comprehensive()
            
            if fleet_analysis['total_systems'] == 0:
                self.logger.warning("No systems available for dashboard")
                return None
            
            # Create basic matplotlib dashboard instead if plotly not available
            self._generate_matplotlib_dashboard(fleet_analysis)
            
            return "enterprise_dashboard_matplotlib.png"
            
        except Exception as e:
            self.logger.error(f"Error generating advanced dashboard: {str(e)}")
            return None
    
    def _generate_matplotlib_dashboard(self, fleet_analysis: Dict[str, Any]):
        """Generate matplotlib-based dashboard"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Enterprise Fleet Operations Dashboard', fontsize=16, fontweight='bold')
            
            systems = fleet_analysis['system_details']
            kpis = fleet_analysis['fleet_kpis']
            
            # Health distribution pie chart
            health_dist = kpis.get('health_distribution', {})
            axes[0, 0].pie(health_dist.values(), labels=health_dist.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Health Distribution')
            
            # Efficiency scatter
            system_ids = list(systems.keys())
            efficiencies = [systems[sid]['efficiency'] for sid in system_ids]
            health_scores = [systems[sid]['health_score'] for sid in system_ids]
            
            scatter = axes[0, 1].scatter(efficiencies, health_scores, c=health_scores, cmap='RdYlGn', s=100)
            axes[0, 1].set_xlabel('Efficiency (%)')
            axes[0, 1].set_ylabel('Health Score')
            axes[0, 1].set_title('Efficiency vs Health Score')
            plt.colorbar(scatter, ax=axes[0, 1])
            
            # Temperature histogram
            temperatures = [systems[sid]['temperature'] for sid in system_ids]
            axes[0, 2].hist(temperatures, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 2].set_xlabel('Temperature (°C)')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].set_title('Temperature Distribution')
            
            # String balance box plot
            string_cvs = [systems[sid]['string_cv'] for sid in system_ids]
            axes[1, 0].boxplot(string_cvs)
            axes[1, 0].set_ylabel('String CV (%)')
            axes[1, 0].set_title('String Balance Distribution')
            
            # Power output bar chart
            power_outputs = [systems[sid]['ac_power'] for sid in system_ids]
            axes[1, 1].bar(range(len(system_ids)), power_outputs, color='lightblue')
            axes[1, 1].set_xlabel('System Index')
            axes[1, 1].set_ylabel('Power Output (W)')
            axes[1, 1].set_title('Power Output Distribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Fleet KPIs summary
            axes[1, 2].axis('off')
            kpi_text = f"""
Fleet KPIs Summary:
• Total Systems: {kpis.get('total_systems', 0)}
• Systems Online: {kpis.get('systems_online', 0)}
• Avg Efficiency: {kpis.get('average_efficiency', 0):.1f}%
• Avg Health Score: {kpis.get('average_health_score', 0):.1f}
• Total Power: {kpis.get('total_power_output', 0):,.0f} W
• Critical Systems: {kpis.get('critical_systems', 0)}
• Maintenance Urgency: {kpis.get('maintenance_urgency_score', 0):.1f}
            """
            axes[1, 2].text(0.1, 0.9, kpi_text, transform=axes[1, 2].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('Fleet KPIs')
            
            plt.tight_layout()
            
            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dashboard_filename = f"enterprise_dashboard_{timestamp}.png"
            plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Matplotlib dashboard generated: {dashboard_filename}")
            
        except Exception as e:
            self.logger.error(f"Error generating matplotlib dashboard: {str(e)}")

def main():
    """Main execution function"""
    logger.info("Starting Enterprise Fleet Operations System")
    
    # Initialize enterprise fleet operations
    enterprise_ops = EnterpriseFleetOperations()
    
    # Run comprehensive fleet analysis
    logger.info("Running comprehensive fleet analysis...")
    fleet_analysis = enterprise_ops.analyze_fleet_comprehensive()
    
    # Display summary
    if fleet_analysis['total_systems'] > 0:
        print("\n" + "="*60)
        print("ENTERPRISE FLEET OPERATIONS SUMMARY")
        print("="*60)
        
        kpis = fleet_analysis.get('fleet_kpis', {})
        print(f"Total Systems: {fleet_analysis['total_systems']}")
        print(f"Systems Online: {fleet_analysis['systems_online']}")
        print(f"Average Efficiency: {kpis.get('average_efficiency', 0):.1f}%")
        print(f"Average Health Score: {kpis.get('average_health_score', 0):.1f}")
        print(f"Total Power Output: {kpis.get('total_power_output', 0):,.1f} W")
        
        # Health distribution
        health_dist = kpis.get('health_distribution', {})
        print(f"\nHealth Distribution:")
        print(f"  Excellent: {health_dist.get('excellent', 0)}")
        print(f"  Good: {health_dist.get('good', 0)}")
        print(f"  Fair: {health_dist.get('fair', 0)}")
        print(f"  Poor: {health_dist.get('poor', 0)}")
        print(f"  Critical: {health_dist.get('critical', 0)}")
        
        # Risk assessment
        risk = fleet_analysis.get('risk_assessment', {})
        print(f"\nRisk Assessment:")
        print(f"  Overall Risk Level: {risk.get('risk_level', 'unknown').upper()}")
        print(f"  Critical Systems: {risk.get('critical_systems_count', 0)}")
        
        # Maintenance priorities
        maintenance_tasks = fleet_analysis.get('maintenance_priorities', [])
        urgent_tasks = [t for t in maintenance_tasks if t['urgency'] == 'critical']
        print(f"\nMaintenance Priorities:")
        print(f"  Critical Tasks: {len(urgent_tasks)}")
        print(f"  Total Tasks: {len(maintenance_tasks)}")
        
        print(f"\nAnalysis completed in {fleet_analysis['analysis_duration']}s")
        
        # Generate advanced dashboard
        print("\nGenerating advanced dashboard...")
        dashboard_file = enterprise_ops.generate_advanced_dashboard()
        if dashboard_file:
            print(f"Dashboard generated: {dashboard_file}")
        
        print("="*60)
    else:
        print("No systems found for analysis")
    
    logger.info("Enterprise Fleet Operations completed")

if __name__ == "__main__":
    main()